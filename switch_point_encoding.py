"""
switch_point_encoding.py — Language switch-point encoding for Hinglish text.

Hard requirements:
    (NLTK words corpus via _base.py)

What it does
────────────
Detects the language of each token using the four-way framework (EN, HI_ROM,
HI_DEV, UNK) from _base.py, then inserts a boundary marker between adjacent
tokens whenever the language label changes.

Distinction from LanguageIdentificationTagging
──────────────────────────────────────────────
LanguageIdentificationTagging  →  annotates individual tokens with their
                                   language identity.
SwitchPointEncoding            →  annotates the *boundary* between two
                                   consecutive linguistic segments.
Example:
    LIT output:  [EN]yaar  [HI_ROM]bahut  [EN]good
    SPE output:  yaar  [SWITCH_EN_HI_ROM]  bahut  [SWITCH_HI_ROM_EN]  good

Switch-density definition (aligned with LanguageIdentificationTagging):
    switch_density = switch_count / max(1, non_punctuation_tokens − 1)

Idempotency guarantee:
    process(process(text)) == process(text)

Existing switch markers are detected and skipped during re-processing so
markers are never duplicated.

Encoding strategies (req 8):
    SPECIAL_TOKEN  →  [SWITCH_EN_HI_ROM]            (default)
    XML            →  <SWITCH from="EN" to="HI_ROM"/>
    INLINE         →  ⟨EN→HI_ROM⟩
    GENERIC        →  [SWITCH]

No silent fallbacks: the module raises ValueError for unrecognised
strategy names at construction time.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase


# ---------------------------------------------------------------------------
# Encoding strategy type
# ---------------------------------------------------------------------------

EncodingStrategy = Literal["special_token", "xml", "inline", "generic"]

_VALID_STRATEGIES: frozenset[str] = frozenset(
    {"special_token", "xml", "inline", "generic"}
)

# ---------------------------------------------------------------------------
# Four-way label set
# ---------------------------------------------------------------------------

_CONTENT_LABELS: frozenset[str] = frozenset({"EN", "HI_ROM", "HI_DEV", "UNK"})

# ---------------------------------------------------------------------------
# Marker detection — used for idempotency
# ---------------------------------------------------------------------------

# Matches any marker that this module could have emitted in any strategy.
_MARKER_RE = re.compile(
    r"\[SWITCH(?:_[A-Z0-9_]+)?\]"           # SPECIAL_TOKEN / GENERIC
    r"|<SWITCH\b[^>]*/>"                     # XML
    r"|⟨[A-Z_]+→[A-Z_]+⟩"                   # INLINE
)


def _is_switch_marker(token: str) -> bool:
    """Return True if the token is a previously emitted switch marker."""
    return bool(_MARKER_RE.fullmatch(token))


# ---------------------------------------------------------------------------
# Switch statistics container
# ---------------------------------------------------------------------------

@dataclass
class SwitchStats:
    """
    Cumulative switch statistics accumulated across all process() calls.

    Fields
    ------
    total_tokens            All surface tokens seen (including punct/markers).
    non_punctuation_tokens  Content tokens (excluding punct, digits, markers).
    switch_count            Number of boundary markers inserted.
    transition_counts       {(from_label, to_label): count} mapping.

    Derived property
    ----------------
    switch_density  = switch_count / max(1, non_punctuation_tokens − 1)
    """
    total_tokens:           int                        = 0
    non_punctuation_tokens: int                        = 0
    switch_count:           int                        = 0
    transition_counts:      Dict[Tuple[str, str], int] = field(
        default_factory=dict
    )

    @property
    def switch_density(self) -> float:
        denom = max(1, self.non_punctuation_tokens - 1)
        return self.switch_count / denom

    def record_switch(self, from_label: str, to_label: str) -> None:
        key = (from_label, to_label)
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1
        self.switch_count += 1

    def transition_report(self) -> str:
        if not self.transition_counts:
            return "  (no switches)"
        lines = [
            f"  {f}→{t}: {c}"
            for (f, t), c in sorted(
                self.transition_counts.items(),
                key=lambda kv: -kv[1],
            )
        ]
        return "\n".join(lines)

    def report(self) -> str:
        return (
            "SwitchStats(\n"
            f"  total_tokens            = {self.total_tokens}\n"
            f"  non_punctuation_tokens  = {self.non_punctuation_tokens}\n"
            f"  switch_count            = {self.switch_count}\n"
            f"  switch_density          = {self.switch_density:.4f}\n"
            f"  transition_counts:\n"
            + "\n".join(f"    {f}→{t}: {c}"
                        for (f, t), c in sorted(self.transition_counts.items()))
            + "\n)"
        )


# ---------------------------------------------------------------------------
# Marker formatting
# ---------------------------------------------------------------------------

def _format_marker(
    from_label: str,
    to_label: str,
    strategy: EncodingStrategy,
    generic_marker: str,
    directional: bool,
) -> str:
    """
    Produce a switch marker string for the given strategy.

    Parameters
    ----------
    from_label, to_label : str
        Language labels of the two adjacent tokens.
    strategy : EncodingStrategy
        One of "special_token", "xml", "inline", "generic".
    generic_marker : str
        The value to use for strategy="generic".
    directional : bool
        When False, always use the generic marker regardless of strategy.
    """
    if not directional:
        return generic_marker

    if strategy == "special_token":
        return f"[SWITCH_{from_label}_{to_label}]"

    if strategy == "xml":
        return f'<SWITCH from="{from_label}" to="{to_label}"/>'

    if strategy == "inline":
        return f"⟨{from_label}→{to_label}⟩"

    # strategy == "generic"
    return generic_marker


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class SwitchPointEncoding(HinglishBase):
    """
    Inserts language-boundary markers between adjacent tokens whose language
    labels differ.

    Identity guarantee
    ------------------
    Original tokens are never modified.  Only boundary markers are inserted
    between them.  Markers are not themselves tokens in the original vocabulary.

    Idempotency guarantee
    ---------------------
    Re-processing an already-encoded sequence does not duplicate markers.
    Existing markers are detected and classified as non-content tokens,
    resetting the language-chain state so they are skipped cleanly.

    Parameters
    ----------
    strategy : "special_token" | "xml" | "inline" | "generic"
        Controls the surface form of inserted markers.
        Default "special_token" is backward-compatible with the previous
        directional_markers=True behaviour.

    directional_markers : bool
        When True (default), markers encode direction (from → to).
        When False, the generic_marker is always used.
        Exists for backward compatibility; prefer setting strategy="generic"
        for new code.

    mark_unknown : bool
        When False (default), UNK tokens do not participate in switch
        detection — they are emitted as-is without resetting the language
        chain or triggering a marker.
        When True, UNK tokens participate and can trigger markers.

    en_hi_marker, hi_en_marker : str
        Backward-compatible generic markers for the two primary directions.
        Used only when directional_markers=False and strategy="generic".

    switch_marker : str
        The generic marker token for strategy="generic" or
        directional_markers=False.

    stats : SwitchStats
        Cumulative session statistics; reset by calling
        ``processor.stats = SwitchStats()``.
    """

    strategy:            EncodingStrategy = "special_token"
    directional_markers: bool             = True
    mark_unknown:        bool             = False
    # Backward-compatible marker strings.
    en_hi_marker:        str              = "[SWITCH_EN_HI]"
    hi_en_marker:        str              = "[SWITCH_HI_EN]"
    switch_marker:       str              = "[SWITCH]"

    # Session statistics — reset between experiments if needed.
    stats: SwitchStats = field(default_factory=SwitchStats, init=False)

    def _setup(self) -> None:
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)}; "
                f"got '{self.strategy}'"
            )

    # ── Public process() — overrides HinglishBase; requires cross-token state ─

    def process(self, text: str) -> str:
        """
        Insert switch-point markers between adjacent tokens whose language
        labels differ.

        Punctuation is transparent: it is emitted (if preserve_punctuation is
        True) but does not reset the language chain or trigger markers.

        Digits are transparent: emitted as-is without affecting the chain.

        Existing switch markers (from a prior call) are recognised and skipped,
        guaranteeing idempotency.
        """
        if not self.enabled or not isinstance(text, str):
            return text

        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()

        tokens: List[str] = self._tokenize(text)
        result: List[str] = []
        prev_lang: Optional[str] = None

        self.stats.total_tokens += len(tokens)

        for tok in tokens:
            # ── Already a switch marker: skip without affecting the chain ──
            if _is_switch_marker(tok):
                # Don't append — idempotency means we re-emit the correct
                # marker ourselves if a switch still exists.
                continue

            # ── Punctuation: transparent ───────────────────────────────────
            if self._is_punctuation(tok):
                if self.preserve_punctuation:
                    result.append(tok)
                continue

            # ── Digits: transparent ────────────────────────────────────────
            if tok.isdigit():
                result.append(tok)
                continue

            # ── Content token: classify and (possibly) insert marker ───────
            lang = self._classify(tok)
            self.stats.non_punctuation_tokens += 1

            if lang is None:
                # UNK with mark_unknown=False: emit without affecting chain.
                result.append(tok)
                continue

            if prev_lang is not None and lang != prev_lang:
                marker = self._make_marker(prev_lang, lang)
                result.append(marker)
                self.stats.record_switch(prev_lang, lang)

            result.append(tok)
            prev_lang = lang

        return self._reconstruct(result)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _classify(self, token: str) -> Optional[str]:
        """
        Return the four-way language label, or None if the token should not
        participate in switch detection.

        None is returned for:
            - UNK tokens when mark_unknown=False.
        """
        lang = self._detect_language(token)
        if lang == "UNK" and not self.mark_unknown:
            return None
        return lang

    def _make_marker(self, from_label: str, to_label: str) -> str:
        """
        Produce the correctly formatted marker for this transition.

        Delegates to _format_marker with the instance's strategy settings.
        Backward-compatible shortcut: when directional_markers=False, always
        returns switch_marker regardless of strategy.
        """
        return _format_marker(
            from_label=from_label,
            to_label=to_label,
            strategy=self.strategy,
            generic_marker=self.switch_marker,
            directional=self.directional_markers,
        )

    # ── _process_token: not meaningful for this module ────────────────────────

    def _process_token(self, token: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "SwitchPointEncoding overrides process() directly; "
            "_process_token is not called.  Use process(text) instead."
        )


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[SwitchPointEncoding] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = SwitchPointEncoding()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: "str | Path",
    output_csv: "str | Path",
    text_col: str = "text",
    processor: Optional[SwitchPointEncoding] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = SwitchPointEncoding()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _run_smoke_tests() -> None:
    """
    Smoke tests covering:
      1.  EN → HI_ROM switch: marker inserted, tokens unchanged.
      2.  EN → HI_DEV switch: Devanagari token triggers marker.
      3.  HI_ROM → EN switch: reverse direction.
      4.  HI_ROM → HI_DEV inter-Hindi switch.
      5.  Punctuation transparency: commas/periods don't trigger markers.
      6.  UNK tokens ignored by default (mark_unknown=False).
      7.  UNK tokens participate when mark_unknown=True.
      8.  strategy="special_token" (default) marker format.
      9.  strategy="xml" marker format.
      10. strategy="inline" marker format.
      11. strategy="generic" / directional_markers=False.
      12. Statistics: total_tokens, non_punct_tokens, switch_count.
      13. Switch density: switch_count / max(1, non_punct - 1).
      14. Transition distribution: correct (from, to) counts.
      15. Idempotency: process(process(text)) == process(text).
      16. Lexical identity: original tokens survive unchanged.
      17. Invalid strategy raises ValueError at construction.
    """
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))
        status = PASS if condition else FAIL
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    def make(**kw) -> SwitchPointEncoding:
        return SwitchPointEncoding(**kw)

    # ── 1. EN → HI_ROM switch ─────────────────────────────────────────────
    spe = make()
    # "good" is EN; "bahut" should detect as HI_ROM.
    out1 = spe.process("good bahut")
    check("en_to_hi_rom_marker_present",
          "SWITCH" in out1,
          f"'good bahut' → '{out1}'")
    check("en_token_unchanged",
          "good" in out1,
          f"'good' must survive in '{out1}'")
    check("hi_rom_token_unchanged",
          "bahut" in out1,
          f"'bahut' must survive in '{out1}'")

    # ── 2. EN → HI_DEV switch ─────────────────────────────────────────────
    spe2 = make()
    out2 = spe2.process("good \u092c\u0939\u0941\u0924")    # good बहुत
    check("en_to_hi_dev_marker_present",
          "HI_DEV" in out2 or "SWITCH" in out2,
          f"'good बहुत' → '{out2}'")
    check("devanagari_token_unchanged",
          "\u092c\u0939\u0941\u0924" in out2,
          f"बहुत must survive in '{out2}'")

    # ── 3. HI_ROM → EN switch ─────────────────────────────────────────────
    spe3 = make()
    out3 = spe3.process("bahut good")
    check("hi_rom_to_en_marker_present",
          "SWITCH" in out3,
          f"'bahut good' → '{out3}'")

    # ── 4. HI_ROM → HI_DEV inter-Hindi switch ─────────────────────────────
    spe4 = make()
    # "accha" (HI_ROM) followed by "बहुत" (HI_DEV).
    out4 = spe4.process("accha \u092c\u0939\u0941\u0924")
    check("hi_rom_to_hi_dev_marker_present",
          "SWITCH" in out4,
          f"'accha बहुत' → '{out4}'")

    # ── 5. Punctuation transparency ───────────────────────────────────────
    spe5 = make(preserve_punctuation=True)
    # Comma between two EN tokens must not trigger a switch marker.
    out5a = spe5.process("good, work")
    check("punct_no_spurious_marker",
          out5a.count("SWITCH") == 0,
          f"'good, work' → '{out5a}'")
    check("punct_preserved",
          "," in out5a,
          f"comma must survive in '{out5a}'")
    # Comma between EN and HI_ROM must not block the switch marker.
    out5b = spe5.process("good, bahut")
    check("punct_transparent_across_switch",
          "SWITCH" in out5b,
          f"'good, bahut' → '{out5b}'")

    # ── 6. UNK tokens ignored by default ─────────────────────────────────
    spe6 = make(mark_unknown=False)
    # "xyzqrs" is UNK; no marker before or after it.
    out6 = spe6.process("good xyzqrs bahut")
    # The EN→HI_ROM switch should still fire for good→bahut crossing xyzqrs.
    # Depending on implementation, xyzqrs is transparent — check no UNK marker.
    check("unk_default_no_unk_in_marker",
          "UNK" not in out6,
          f"'good xyzqrs bahut' (mark_unknown=False) → '{out6}'")

    # ── 7. UNK participates when mark_unknown=True ────────────────────────
    spe7 = make(mark_unknown=True)
    out7 = spe7.process("good xyzqrs")
    check("unk_marked_when_enabled",
          "UNK" in out7 or "SWITCH" in out7,
          f"'good xyzqrs' (mark_unknown=True) → '{out7}'")

    # ── 8. strategy="special_token" format ───────────────────────────────
    spe8 = make(strategy="special_token")
    out8 = spe8.process("good bahut")
    check("special_token_format",
          out8.startswith("good") and "[SWITCH_" in out8 and "]" in out8,
          f"'{out8}'")

    # ── 9. strategy="xml" format ──────────────────────────────────────────
    spe9 = make(strategy="xml")
    out9 = spe9.process("good bahut")
    check("xml_format",
          '<SWITCH from=' in out9 and '/>' in out9,
          f"'{out9}'")

    # ── 10. strategy="inline" format ─────────────────────────────────────
    spe10 = make(strategy="inline")
    out10 = spe10.process("good bahut")
    check("inline_format",
          "⟨" in out10 and "→" in out10 and "⟩" in out10,
          f"'{out10}'")

    # ── 11. strategy="generic" / directional_markers=False ───────────────
    spe11a = make(strategy="generic", switch_marker="[SW]")
    out11a = spe11a.process("good bahut")
    check("generic_strategy",
          "[SW]" in out11a,
          f"'{out11a}'")

    spe11b = make(directional_markers=False, switch_marker="[SWITCH]")
    out11b = spe11b.process("good bahut")
    check("directional_false_uses_generic",
          "[SWITCH]" in out11b and "[SWITCH_" not in out11b,
          f"'{out11b}'")

    # ── 12. Statistics accumulation ───────────────────────────────────────
    spe12 = make()
    spe12.process("good bahut accha")
    check("stats_total_tokens",
          spe12.stats.total_tokens > 0,
          f"total_tokens={spe12.stats.total_tokens}")
    check("stats_non_punct_tokens",
          spe12.stats.non_punctuation_tokens >= 3,
          f"non_punct={spe12.stats.non_punctuation_tokens}")
    check("stats_switch_count",
          spe12.stats.switch_count >= 1,
          f"switch_count={spe12.stats.switch_count}")

    # ── 13. Switch density ────────────────────────────────────────────────
    spe13 = make()
    spe13.process("good bahut good bahut")   # 4 content tokens, 3 pairs, 3 switches
    expected_density = spe13.stats.switch_count / max(
        1, spe13.stats.non_punctuation_tokens - 1
    )
    check("switch_density_formula",
          abs(spe13.stats.switch_density - expected_density) < 1e-9,
          f"density={spe13.stats.switch_density:.4f}")

    # ── 14. Transition distribution ───────────────────────────────────────
    spe14 = make()
    spe14.process("good bahut good bahut")
    tc = spe14.stats.transition_counts
    check("transition_counts_nonempty",
          len(tc) > 0,
          f"transitions={tc}")
    # Every key must be a 2-tuple of strings.
    check("transition_counts_format",
          all(isinstance(k, tuple) and len(k) == 2 for k in tc),
          f"keys={list(tc.keys())}")

    # ── 15. Idempotency ───────────────────────────────────────────────────
    # Re-running on already-encoded text must not duplicate markers.
    text_idem = "good bahut accha"
    spe_idem = make()
    pass1 = spe_idem.process(text_idem)
    spe_idem2 = make()  # fresh instance — clean stats
    pass2 = spe_idem2.process(pass1)
    check("idempotency",
          pass1 == pass2,
          f"pass1='{pass1}' | pass2='{pass2}'")

    # ── 16. Lexical identity ──────────────────────────────────────────────
    original_tokens = ["good", "bahut", "accha"]
    encoded = spe.process(" ".join(original_tokens))
    for tok in original_tokens:
        check(f"lexical_identity_{tok}",
              tok in encoded,
              f"'{tok}' must survive in '{encoded}'")

    # ── 17. Invalid strategy raises ValueError ────────────────────────────
    try:
        _ = make(strategy="unknown_strategy")
        check("invalid_strategy_raises", False, "should have raised ValueError")
    except ValueError as exc:
        check("invalid_strategy_raises", True, str(exc))

    # Summary.
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"\n  {passed}/{total} smoke tests passed.")
    if passed < total:
        failed = [name for name, ok, _ in results if not ok]
        raise AssertionError(f"Failed: {failed}")


if __name__ == "__main__":
    print("Running smoke tests for switch_point_encoding.py …\n")
    _run_smoke_tests()
    print("\nDone.")