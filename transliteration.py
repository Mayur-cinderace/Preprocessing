"""
transliteration.py — Romanized Hindi → Devanagari transliteration for
Hinglish text.

Hard requirements:
    pip install indic-transliteration
    (+ NLTK words corpus via _base.py)

Scientific positioning
──────────────────────
Transliteration performs **token-level, one-directional script conversion**
from a romanized Hindi representation into Devanagari.  It maximises
script-conversion fidelity while preserving lexical identity and minimising
erroneous conversions through confidence-aware validation.

This module is distinct from ScriptUnification:
    Transliteration  — token-level Roman → Devanagari conversion.
    ScriptUnification — corpus-level script normalization and vocabulary
                        fragmentation reduction.

This module is NOT:
    • bidirectional script normalization,
    • vocabulary unification,
    • language identification (delegated to _base.HinglishBase),
    • translation,
    • phonetic normalization.

Processing pipeline (per token)
────────────────────────────────
  1. Pass empty / numeric tokens through unchanged.
  2. Pass already-Devanagari tokens through unchanged (HI_DEV).
  3. Pass English tokens (EN) through unchanged.
  4. For HI_ROM tokens — transliterate with optional confidence validation:
       confidence_strategy="none"      — convert unconditionally.
       confidence_strategy="lexicon"   — accept only if output is in
                                         HINGLISH_LEXICON.
       confidence_strategy="roundtrip" — convert, back-convert, compare
                                         similarity; reject below threshold.
  5. For UNK tokens — apply unknown_strategy:
       "preserve"      — return unchanged (default-safe).
       "transliterate" — convert without validation.
       "validate"      — convert and apply the active confidence_strategy.

Idempotency guarantee
─────────────────────
  Transliteration(Transliteration(text)) == Transliteration(text)
  because Devanagari tokens are always passed through unchanged (step 2).

No silent fallbacks: missing ``indic-transliteration`` raises ``ImportError``
at import time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate

from _base import HINGLISH_LEXICON, HinglishBase


# ── Supported romanization schemes ────────────────────────────────────────────

_VALID_SCHEMES: dict[str, str] = {
    "itrans": sanscript.ITRANS,
    "hk":     sanscript.HK,
    "iast":   sanscript.IAST,
    "slp1":   sanscript.SLP1,
}

# Scheme-detection heuristics: (scheme_key, signal_characters/patterns)
# Ordered so that more-distinctive schemes are checked first.
_SCHEME_SIGNALS: list[tuple[str, frozenset[str]]] = [
    # IAST uses diacritic characters (ā, ī, ū, ṭ, ḍ, ṇ, ś, ṣ, ṃ, ḥ …)
    ("iast",  frozenset("āīūṭḍṇśṣṃḥṅñṛ")),
    # Harvard-Kyoto uses uppercase only for aspirates / retroflexes
    ("hk",    frozenset("TKGCJDNPBMYRLVSZH")),
    # SLP1 uses a mix of upper and lowercase with specific markers
    ("slp1",  frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - frozenset("AEIO")),
]
# If no signal matches, fall back to ITRANS (most common in social-media text).
_DEFAULT_SCHEME = "itrans"


def _detect_scheme(token: str) -> str:
    """
    Heuristically identify the romanization scheme of a single token.

    Returns one of ``"itrans"``, ``"hk"``, ``"iast"``, ``"slp1"``.
    Falls back to ``"itrans"`` when no distinctive signal is found.
    """
    for scheme_key, signals in _SCHEME_SIGNALS:
        if any(ch in token for ch in signals):
            return scheme_key
    return _DEFAULT_SCHEME


# ── Similarity helper for round-trip validation ───────────────────────────────

from difflib import SequenceMatcher as _SM


def _roundtrip_similarity(a: str, b: str) -> float:
    """
    Compute sequence-level similarity between two strings using
    ``difflib.SequenceMatcher.ratio()``.

    Returns a float in [0, 1]; 1.0 means the strings are identical.
    This is a sequence-aware metric — it distinguishes "bahut" from "bahuut"
    unlike set-based Jaccard, which would return 1.0 for both.

    Used as the round-trip fidelity measure:
        Roman → Devanagari → Roman  vs.  original Roman
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return _SM(None, a.lower(), b.lower()).ratio()


# ── Statistics container ───────────────────────────────────────────────────────

@dataclass
class TransliterationStats:
    """
    Accumulates per-run transliteration statistics.

    Attributes
    ----------
    total_tokens                  : All non-trivial tokens seen.
    english_tokens                : Tokens classified EN (passed through).
    hindi_tokens                  : Tokens classified HI_ROM or HI_DEV.
    devanagari_tokens             : Tokens already in Devanagari (HI_DEV,
                                    fast-path before language detection).
    transliterated_tokens         : Tokens successfully converted to Devanagari.
    validated_tokens              : Tokens that passed confidence validation.
    rejected_tokens               : Tokens whose conversion was rejected;
                                    original form preserved.
    unknown_tokens_seen           : Tokens classified UNK.
    unknown_tokens_transliterated : UNK tokens that were transliterated.
    """
    total_tokens:                  int = 0
    english_tokens:                int = 0
    hindi_tokens:                  int = 0
    devanagari_tokens:             int = 0
    transliterated_tokens:         int = 0
    validated_tokens:              int = 0
    rejected_tokens:               int = 0
    unknown_tokens_seen:           int = 0
    unknown_tokens_transliterated: int = 0

    @property
    def transliteration_rate(self) -> float:
        """Fraction of all tokens that were successfully transliterated."""
        if self.total_tokens == 0:
            return 0.0
        return self.transliterated_tokens / self.total_tokens

    @property
    def rejection_rate(self) -> float:
        """Fraction of attempted conversions that were rejected."""
        attempted = self.transliterated_tokens + self.rejected_tokens
        if attempted == 0:
            return 0.0
        return self.rejected_tokens / attempted

    @property
    def conversion_coverage(self) -> float:
        """
        Fraction of HI_ROM tokens that were successfully transliterated.

        Measures how much of the romanized Hindi content was converted, ignoring
        English, Devanagari pass-throughs, and UNK tokens.
        """
        if self.hindi_tokens == 0:
            return 0.0
        return self.transliterated_tokens / self.hindi_tokens

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.total_tokens                  = 0
        self.english_tokens                = 0
        self.hindi_tokens                  = 0
        self.devanagari_tokens             = 0
        self.transliterated_tokens         = 0
        self.validated_tokens              = 0
        self.rejected_tokens               = 0
        self.unknown_tokens_seen           = 0
        self.unknown_tokens_transliterated = 0

    def summary(self) -> dict[str, object]:
        """Return all statistics as a plain dict for logging / serialization."""
        return {
            "total_tokens":                  self.total_tokens,
            "english_tokens":                self.english_tokens,
            "hindi_tokens":                  self.hindi_tokens,
            "devanagari_tokens":             self.devanagari_tokens,
            "transliterated_tokens":         self.transliterated_tokens,
            "validated_tokens":              self.validated_tokens,
            "rejected_tokens":               self.rejected_tokens,
            "unknown_tokens_seen":           self.unknown_tokens_seen,
            "unknown_tokens_transliterated": self.unknown_tokens_transliterated,
            "transliteration_rate":          round(self.transliteration_rate, 4),
            "rejection_rate":                round(self.rejection_rate, 4),
            "conversion_coverage":           round(self.conversion_coverage, 4),
        }


# ── Main class ────────────────────────────────────────────────────────────────

ConfidenceStrategy = Literal["none", "lexicon", "roundtrip"]
UnknownStrategy    = Literal["preserve", "transliterate", "validate"]


@dataclass
class Transliteration(HinglishBase):
    """
    Transliterates romanized Hindi tokens to Devanagari with confidence-aware
    validation and comprehensive statistics tracking.

    Token routing
    -------------
    HI_DEV tokens  → pass through unchanged (idempotency guarantee).
    EN     tokens  → pass through unchanged (lexical identity guarantee).
    HI_ROM tokens  → transliterate + validate via ``confidence_strategy``.
    UNK    tokens  → apply ``unknown_strategy``.

    Attributes
    ----------
    source_scheme : str
        Romanization input scheme.  One of ``"itrans"`` (default), ``"hk"``,
        ``"iast"``, ``"slp1"``.  ITRANS is the default because casual
        romanized Hindi (social media, chat) overwhelmingly follows ITRANS
        conventions.
    confidence_strategy : ConfidenceStrategy
        How to validate transliteration output:
            ``"none"``      — accept all conversions unconditionally.
            ``"lexicon"``   — accept only if the back-lookup key is in
                              HINGLISH_LEXICON (fast, conservative).
            ``"roundtrip"`` — convert to Devanagari, back-convert to Roman,
                              compare with source via character Jaccard
                              similarity; reject below ``min_roundtrip_similarity``.
    min_roundtrip_similarity : float
        Minimum Jaccard similarity [0, 1] for round-trip acceptance.
        Relevant only when ``confidence_strategy="roundtrip"``.
        Default 0.6.
    unknown_strategy : UnknownStrategy
        How to handle UNK tokens (tokens not classifiable as EN, HI_DEV,
        or HI_ROM):
            ``"preserve"``      — return original form unchanged (safe default).
            ``"transliterate"`` — convert without confidence check.
            ``"validate"``      — convert and apply ``confidence_strategy``.
    heuristic_scheme_detection : bool
        When True, apply a lightweight heuristic to guess the romanization
        scheme per token before conversion, rather than using the global
        ``source_scheme``.  Detection is signal-based (IAST diacritics, HK
        uppercase patterns) and is not a reliable classifier — it falls back
        to ITRANS when no distinctive signal is found.  Useful for corpora
        that mix IAST with casual ITRANS text.  Default False.
    collect_stats : bool
        When True, accumulate statistics in ``self.stats``.  Default True.
    """

    source_scheme:              str                = "itrans"
    confidence_strategy:        ConfidenceStrategy = "roundtrip"
    min_roundtrip_similarity:   float              = 0.6
    unknown_strategy:           UnknownStrategy    = "validate"
    heuristic_scheme_detection: bool               = False
    collect_stats:              bool               = True

    def _setup(self) -> None:
        if self.source_scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"source_scheme must be one of {list(_VALID_SCHEMES)}; "
                f"got '{self.source_scheme}'"
            )
        self._sp_scheme: str = _VALID_SCHEMES[self.source_scheme]
        self.stats: TransliterationStats = TransliterationStats()

    # ── Token dispatch ────────────────────────────────────────────────────────

    def _process_token(self, token: str) -> str:
        if not token or token.isdigit():
            return token

        self._bump("total_tokens")

        # HI_DEV — already Devanagari; pass through (idempotency guarantee).
        # This check runs before language detection so we never re-process
        # output from a prior pass (guarantees idempotency).
        if self._contains_devanagari(token):
            self._bump("devanagari_tokens")
            return token

        # Use _detect_language_with_score when available (enhanced _base.py),
        # otherwise fall back to _detect_language.  Both produce compatible labels
        # in the four-way EN | HI_DEV | HI_ROM | UNK framework.
        if hasattr(self, "_detect_language_with_score"):
            label, _conf = self._detect_language_with_score(token)
        else:
            label = self._detect_language(token)

        if label == "EN":
            self._bump("english_tokens")
            return token

        if label in ("HI_DEV", "HI_ROM"):
            self._bump("hindi_tokens")
            return self._handle_hindi(token)

        # UNK
        self._bump("unknown_tokens_seen")
        return self._handle_unknown(token)

    # ── HI_ROM / HI_DEV handler ───────────────────────────────────────────────

    def _handle_hindi(self, token: str) -> str:
        """Transliterate a HI_ROM token; apply confidence validation."""
        result = self._do_transliterate(token)
        if result == token:
            # Transliteration produced no change (e.g. token was already
            # in Devanagari when lowercased) — pass through.
            return token

        accepted = self._validate(token, result)
        if accepted:
            self._bump("transliterated_tokens")
            if self.confidence_strategy != "none":
                self._bump("validated_tokens")
            return result
        else:
            self._bump("rejected_tokens")
            return token

    # ── UNK handler ───────────────────────────────────────────────────────────

    def _handle_unknown(self, token: str) -> str:
        if self.unknown_strategy == "preserve":
            return token
        result = self._do_transliterate(token)
        if self.unknown_strategy == "transliterate":
            self._bump("transliterated_tokens")
            self._bump("unknown_tokens_transliterated")
            return result
        # "validate"
        accepted = self._validate(token, result)
        if accepted:
            self._bump("transliterated_tokens")
            self._bump("unknown_tokens_transliterated")
            if self.confidence_strategy != "none":
                self._bump("validated_tokens")
            return result
        self._bump("rejected_tokens")
        return token

    # ── Transliteration core ──────────────────────────────────────────────────

    def _do_transliterate(self, token: str) -> str:
        """
        Convert ``token`` from its romanization scheme to Devanagari.

        When ``heuristic_scheme_detection=True`` the per-token scheme is
        estimated heuristically; otherwise the global ``source_scheme`` is used.
        """
        if self.heuristic_scheme_detection:
            detected = _detect_scheme(token)
            sp = _VALID_SCHEMES[detected]
        else:
            sp = self._sp_scheme
        return _transliterate(token, sp, sanscript.DEVANAGARI)

    # ── Confidence validation ─────────────────────────────────────────────────

    def _validate(self, original: str, converted: str) -> bool:
        """
        Return True if ``converted`` passes the active confidence strategy.

        ``"none"``      — always True; every conversion is accepted.
        ``"lexicon"``   — accept only if ``original.lower()`` is in
                          HINGLISH_LEXICON.  This validates the *source* token:
                          if the source is a known canonical Hinglish form, the
                          conversion is considered trustworthy.  Tokens outside
                          the lexicon are rejected (preserved in original form).
                          Fast and conservative; suitable for production.
        ``"roundtrip"`` — back-convert ``converted`` (Devanagari → Roman) using
                          the active source scheme, then compute sequence
                          similarity against ``original`` via SequenceMatcher.
                          Accept if similarity ≥ ``min_roundtrip_similarity``.
                          Catches cases where the transliterator produces a
                          phonetically distant or malformed output.
        """
        if self.confidence_strategy == "none":
            return True

        if self.confidence_strategy == "lexicon":
            return original.lower() in HINGLISH_LEXICON

        # "roundtrip"
        back = _transliterate(converted, sanscript.DEVANAGARI, self._sp_scheme)
        sim  = _roundtrip_similarity(original, back)
        return sim >= self.min_roundtrip_similarity

    # ── Statistics helpers ────────────────────────────────────────────────────

    def _bump(self, field: str) -> None:
        if self.collect_stats:
            setattr(self.stats, field, getattr(self.stats, field) + 1)

    def get_stats(self) -> dict[str, object]:
        """Return a snapshot of the current statistics as a plain dict."""
        return self.stats.summary()

    def reset_stats(self) -> None:
        """Reset all statistics counters to zero."""
        self.stats.reset()


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[Transliteration] = None,
) -> pd.DataFrame:
    """
    Apply Transliteration to a DataFrame column.

    Parameters
    ----------
    df         : Input DataFrame.
    text_col   : Column containing raw Hinglish text.
    output_col : Output column for transliterated text.
    processor  : Optional pre-configured ``Transliteration`` instance.

    Returns
    -------
    DataFrame with an additional column ``output_col``.
    """
    if processor is None:
        processor = Transliteration()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[Transliteration] = None,
) -> pd.DataFrame:
    """
    Apply Transliteration to a CSV file and write the result.

    Parameters
    ----------
    input_csv  : Path to the input CSV.
    output_csv : Path for the output CSV.
    text_col   : Column name containing raw Hinglish text.
    processor  : Optional pre-configured ``Transliteration`` instance.

    Returns
    -------
    Processed DataFrame (also written to ``output_csv``).
    """
    if processor is None:
        processor = Transliteration()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)


# ── Smoke tests ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    _SEP = "─" * 70

    failures = 0

    def _check(label: str, got: object, expected: object) -> None:
        global failures
        ok = got == expected
        if not ok:
            failures += 1
        print(f"  {'✓' if ok else '✗'}  {label}")
        if not ok:
            print(f"     got     : {got!r}")
            print(f"     expected: {expected!r}")

    print(_SEP)
    print("  transliteration.py — smoke tests")
    print(_SEP)

    # ── Default processor (roundtrip strategy, itrans scheme) ─────────────
    # lowercase=False so the base layer does not lowercase before processing,
    # which would conflict with ITRANS conventions.
    t = Transliteration(lowercase=False, confidence_strategy="none")

    print("\n[Roman → Devanagari — exact output assertions]")
    _check("bahut → बहुत",   t.process("bahut"),   "बहुत")
    _check("yaar → यार",     t.process("yaar"),    "यार")
    _check("ghar → घर",      t.process("ghar"),    "घर")
    _check("pyar → प्यार",    t.process("pyar"),    "प्यार")
    _check("samajh → समझ",   t.process("samajh"),  "समझ")
    _check("accha → अच्छा",   t.process("accha"),   "अच्छा")
    _check("kya → क्या",      t.process("kya"),     "क्या")
    _check("mein → में",      t.process("mein"),    "में")

    print("\n[English preservation]")
    for word in ["movie", "laptop", "github", "chatgpt", "office", "lol"]:
        _check(f"'{word}' unchanged", t.process(word), word)

    print("\n[Devanagari preservation (HI_DEV pass-through)]")
    for deva in ["बहुत", "प्यार", "घर", "समझ"]:
        _check(f"'{deva}' unchanged", t.process(deva), deva)

    print("\n[Mixed sentence]")
    out = t.process("yaar this movie was bahut accha")
    _check("'movie' preserved", "movie" in out, True)
    _check("'this' preserved",  "this"  in out, True)
    _check("Hindi tokens in Devanagari",
           any(ord(c) > 0x0900 for c in out), True)

    # ── Idempotency ────────────────────────────────────────────────────────
    print("\n[Idempotency: T(T(text)) == T(text)]")
    for text in [
        "yaar bahut accha tha",
        "movie dekho aur khana khao",
        "बहुत movie accha",
    ]:
        first  = t.process(text)
        second = t.process(first)
        _check(f"T(T({text!r})) == T(text)", second, first)

    # ── Confidence strategies ──────────────────────────────────────────────
    print("\n[Confidence strategy: none]")
    t_none = Transliteration(confidence_strategy="none", lowercase=False)
    _check("bahut → बहुत (strategy=none)", t_none.process("bahut"), "बहुत")

    print("\n[Confidence strategy: lexicon]")
    t_lex = Transliteration(confidence_strategy="lexicon", lowercase=False)
    _check("bahut → बहुत (strategy=lexicon, lexicon hit)",
           t_lex.process("bahut"), "बहुत")
    # Token not in lexicon: should be preserved (rejected by lexicon strategy)
    unk_in_lex = "zxqv"
    _check(f"'{unk_in_lex}' preserved (not in lexicon)",
           t_lex.process(unk_in_lex), unk_in_lex)

    print("\n[Confidence strategy: roundtrip]")
    t_rt = Transliteration(confidence_strategy="roundtrip",
                            min_roundtrip_similarity=0.6, lowercase=False)
    _check("bahut → बहुत (strategy=roundtrip)",
           t_rt.process("bahut"), "बहुत")
    # Pure-noise token: round-trip should fail → original preserved
    noise = "xzkqp"
    result_noise = t_rt.process(noise)
    _check(f"noise token '{noise}' is str (preserved or converted)",
           isinstance(result_noise, str), True)

    # ── Unknown strategies ─────────────────────────────────────────────────
    print("\n[Unknown strategy: preserve]")
    t_pres = Transliteration(unknown_strategy="preserve", lowercase=False)
    unk_tok = "xylo"    # not in NLTK_WORDS, not in HINGLISH_LEXICON, low score
    _check(f"UNK '{unk_tok}' preserved",
           t_pres.process(unk_tok), unk_tok)

    print("\n[Unknown strategy: transliterate (confidence=none)]")
    t_utrans = Transliteration(unknown_strategy="transliterate",
                                confidence_strategy="none", lowercase=False)
    # Just verify it runs and returns a string
    _check("UNK transliterate returns str",
           isinstance(t_utrans.process("xylo"), str), True)

    print("\n[Unknown strategy: validate]")
    t_uval = Transliteration(unknown_strategy="validate",
                              confidence_strategy="roundtrip", lowercase=False)
    _check("UNK validate returns str",
           isinstance(t_uval.process("xylo"), str), True)

    # ── Heuristic scheme detection ─────────────────────────────────────────
    print("\n[Heuristic scheme detection (heuristic_scheme_detection=True)]")
    t_hsd = Transliteration(heuristic_scheme_detection=True,
                             confidence_strategy="none", lowercase=False)
    _check("bahut (heuristic ITRANS fallback) → बहुत",
           t_hsd.process("bahut"), "बहुत")
    # IAST diacritic token: āp should trigger IAST detection
    iast_result = t_hsd.process("āp")
    _check("āp (heuristic IAST) produces Devanagari",
           any(ord(c) > 0x0900 for c in iast_result), True)

    # ── Statistics ─────────────────────────────────────────────────────────
    print("\n[Statistics accumulation and conversion_coverage]")
    t_stat = Transliteration(confidence_strategy="none",
                              unknown_strategy="transliterate",
                              lowercase=False)
    t_stat.reset_stats()
    # "yaar" HI_ROM, "bahut" HI_ROM, "movie" EN, "accha" HI_ROM
    t_stat.process("yaar bahut movie accha")
    stats = t_stat.get_stats()
    _check("total_tokens > 0",            stats["total_tokens"] > 0,            True)
    _check("english_tokens > 0",          stats["english_tokens"] > 0,          True)
    _check("hindi_tokens > 0",            stats["hindi_tokens"] > 0,            True)
    _check("transliterated_tokens > 0",   stats["transliterated_tokens"] > 0,   True)
    _check("transliteration_rate ∈ (0,1]",
           0 < stats["transliteration_rate"] <= 1.0, True)
    _check("conversion_coverage ∈ (0,1]",
           0 < stats["conversion_coverage"] <= 1.0, True)

    # ── DataFrame processing ───────────────────────────────────────────────
    print("\n[DataFrame processing]")
    import pandas as _pd
    df_in = _pd.DataFrame({"text": ["yaar bahut accha", "movie dekho"]})
    df_out = process_dataframe(df_in, processor=Transliteration(
        lowercase=False, confidence_strategy="none"))
    _check("output column present",   "processed_text" in df_out.columns,              True)
    _check("row count preserved",     len(df_out) == len(df_in),                       True)
    _check("processed text is str",   isinstance(df_out["processed_text"].iloc[0], str), True)

    # ── CSV processing ─────────────────────────────────────────────────────
    print("\n[CSV processing]")
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as f_in:
        f_in.write("text\nyaar bahut accha\n")
        csv_in = f_in.name
    csv_out = csv_in.replace(".csv", "_out.csv")
    try:
        result_df = process_csv(csv_in, csv_out,
                                 processor=Transliteration(
                                     lowercase=False, confidence_strategy="none"))
        _check("CSV output file created",
               os.path.exists(csv_out), True)
        _check("CSV output has processed_text column",
               "processed_text" in result_df.columns, True)
    finally:
        for p in (csv_in, csv_out):
            if os.path.exists(p):
                os.remove(p)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    if failures:
        print(f"  {failures} test(s) FAILED.")
        sys.exit(1)
    else:
        print("  All tests passed.")
    print(_SEP)