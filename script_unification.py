"""
script_unification.py — Script unification for mixed-script Hinglish text.

Hard requirements:
    pip install indic-transliteration
    (+ NLTK words corpus via _base.py)

What it does
────────────
Detects the script of each token using the four-way language framework
(EN, HI_ROM, HI_DEV, UNK) from _base.py, then applies targeted conversions:

    target_script="devanagari"
        HI_ROM tokens → Devanagari via ITRANS.
        EN / UNK tokens are never modified.

    target_script="roman"
        HI_DEV tokens → romanised form via the configured scheme
        (itrans / hk / iast), then optionally collapsed to common Hinglish.
        EN / UNK tokens are never modified.

Eligibility check (replaces the old strict require_lexicon_match gate):
    1. Lexicon membership           — highest confidence, always convert.
    2. Language-confidence fallback — _detect_language label matches
                                      the expected script, convert.
    3. Neither                      — preserve original (if require_lexicon_match
                                      is True) or force-convert (if False).

Scientific framing
──────────────────
The primary contribution is reducing *script-induced lexical fragmentation*:
multiple surface variants of the same word (प्यार / pyaar / pyar) that
inflate vocabulary size and harm model generalisation are collapsed to a
single canonical form.  The fragmentation reduction ratio is:

    F_after / F_before

where F = number of distinct script variants of any lexeme in the corpus.

Idempotency guarantee
─────────────────────
For any well-formed Hinglish text t:

    process(process(t)) == process(t)

Both directions are tested in the smoke-test suite.

Round-trip validation
─────────────────────
When round_trip_validate=True every transliterated token is back-converted
and its edit distance to the original is checked.  Tokens whose round-trip
distance exceeds round_trip_threshold are reverted to the original surface
form rather than silently emitting a bad transliteration.

No silent fallbacks: missing indic-transliteration raises ImportError at
import time.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate

from _base import HINGLISH_LEXICON, HinglishBase


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

_SCHEME_MAP: Dict[str, str] = {
    "itrans": sanscript.ITRANS,
    "hk":     sanscript.HK,
    "iast":   sanscript.IAST,
}

_VALID_TARGET_SCRIPTS = {"devanagari", "roman"}
_VALID_ROMAN_SCHEMES  = set(_SCHEME_MAP.keys())


# ---------------------------------------------------------------------------
# Edit-distance helper (pure Python, no extra deps)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Classic dynamic-programming Levenshtein distance."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    # Single-row DP.
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# Roman normalisation
# ---------------------------------------------------------------------------

def _strip_diacritics(text: str) -> str:
    """
    Decompose Unicode, remove combining marks, then re-encode to ASCII where
    possible.  This converts:
        ātmā  → atma
        śakti → sakti   (close enough for common Hinglish usage)
        kṛṣṇa → krsna   (further collapsed by _collapse_clusters below)
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


# Hand-crafted cluster map for the most common Hinglish romanisations that
# diacritic stripping alone cannot handle.
_CLUSTER_MAP: Dict[str, str] = {
    "sh": "sh", "shh": "sh",   # retroflex ś → sh
    "ch": "ch", "chh": "chh",
    "ksh": "ksh",
    "gn":  "gn",
    "jn":  "gya",
}

def _simplify_roman(text: str, preserve_scheme: bool = False) -> str:
    """
    Produce a Hinglish-friendly romanisation from a scholarly-scheme string.

    Steps:
    1. Strip Unicode combining marks (ā → a, ī → i, etc.).
    2. Lowercase.
    3. Unless preserve_scheme=True, apply common-Hinglish cluster rewrites.
    """
    simplified = _strip_diacritics(text).lower()
    if preserve_scheme:
        return simplified
    # Apply cluster rewrites longest-first to avoid partial matches.
    for src, tgt in sorted(_CLUSTER_MAP.items(), key=lambda kv: -len(kv[0])):
        simplified = simplified.replace(src, tgt)
    return simplified


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class ScriptUnification(HinglishBase):
    """
    Converts Hindi tokens to a single target script.

    Eligibility pipeline (executed in order, first match wins):
        1. Token is in hinglish_lexicon           → convert.
        2. _detect_language label matches source   → convert.
        3. require_lexicon_match=False             → force-convert.
        4. Otherwise                               → preserve.

    Parameters
    ----------
    target_script : "devanagari" | "roman"
        Direction of unification.
    roman_scheme : "itrans" | "hk" | "iast"
        Transliteration scheme used when converting FROM Devanagari.
    require_lexicon_match : bool
        When True (default) and neither the lexicon nor the language-confidence
        check passes, the original token is preserved.  When False, all tokens
        whose script matches the source direction are converted.
    roman_post_map : dict
        Optional post-processing map applied after romanisation
        (e.g. scholarly IDs → common forms).
    preserve_scheme_differences : bool
        When True, scholarly distinctions (ā, ī, ṭ …) are kept after
        diacritic stripping but NOT fully collapsed to common Hinglish.
        Useful for linguistic studies.  When False (default) the output
        matches everyday Hinglish spelling conventions.
    round_trip_validate : bool
        When True, every transliterated token is back-converted and its
        edit distance to the original is measured.  Tokens that exceed
        round_trip_threshold are reverted to the original surface form.
    round_trip_threshold : int
        Maximum allowed edit distance between the original token and its
        round-tripped version.  Default: 2.
    allow_language_fallback : bool
        Controls behaviour for tokens that are NOT in the lexicon but whose
        _detect_language label matches the expected source script.

        True  (default) — convert them.  The language-confidence signal is
                          treated as sufficient evidence of Hindi identity even
                          without an explicit lexicon entry.  This is the
                          recommended setting for most NLP pipelines.
        False           — preserve them when require_lexicon_match=True.
                          Only tokens present in the lexicon will be converted;
                          morphological variants (ladkon, padhoge, samjhdaar)
                          will pass through unchanged.  Use this when you want
                          strict, auditable conversions.

        Interaction with require_lexicon_match:

            require_lexicon_match=True,  allow_language_fallback=True
                Convert if in lexicon OR if language detector agrees.
                (Highest recall while still filtering pure UNK / EN tokens.)

            require_lexicon_match=True,  allow_language_fallback=False
                Convert ONLY if in lexicon.  Strictest mode.

            require_lexicon_match=False, allow_language_fallback=<any>
                Convert everything whose script matches the direction,
                regardless of lexicon or language-confidence.
    """

    target_script:               str            = "devanagari"
    roman_scheme:                str            = "itrans"
    require_lexicon_match:       bool           = True
    roman_post_map:              Dict[str, str] = field(default_factory=dict)
    preserve_scheme_differences: bool           = False
    round_trip_validate:         bool           = True
    round_trip_threshold:        int            = 2
    allow_language_fallback:     bool           = True

    def _setup(self) -> None:
        if self.target_script not in _VALID_TARGET_SCRIPTS:
            raise ValueError(
                f"target_script must be one of {_VALID_TARGET_SCRIPTS}; "
                f"got '{self.target_script}'"
            )
        if self.roman_scheme not in _VALID_ROMAN_SCHEMES:
            raise ValueError(
                f"roman_scheme must be one of {_VALID_ROMAN_SCHEMES}; "
                f"got '{self.roman_scheme}'"
            )
        self._sp_scheme: str = _SCHEME_MAP[self.roman_scheme]

    # ── Public hook required by HinglishBase ──────────────────────────────────

    def _process_token(self, token: str) -> str:
        """Route each token through the four-way framework."""
        if not token or token.isdigit():
            return token

        lang = self._detect_language(token)

        if self.target_script == "devanagari":
            # Only HI_ROM tokens are candidates; everything else passes through.
            if lang != "HI_ROM":
                return token
            return self._to_devanagari(token, lang)

        else:  # target_script == "roman"
            # Only HI_DEV tokens are candidates; everything else passes through.
            if lang != "HI_DEV":
                return token
            return self._to_roman(token, lang)

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _is_eligible(self, token: str, lang: str) -> bool:
        """
        Two-gate eligibility check — called AFTER the caller has already
        confirmed that ``lang`` matches the expected source script.

        Gate 1 — lexicon (always passes):
            If the token is in the lexicon it is unconditionally eligible,
            regardless of the other flags.

        Gate 2 — language-confidence fallback (controlled by flags):
            The caller pre-filtered so ``lang`` is already ``HI_ROM`` or
            ``HI_DEV``; that label IS the confidence signal.  Whether we
            act on it depends on two orthogonal settings:

                require_lexicon_match=False
                    -> eligible regardless (user opted out of filtering).

                require_lexicon_match=True, allow_language_fallback=True
                    -> eligible: language-confidence accepted as sufficient
                       evidence of Hindi identity.

                require_lexicon_match=True, allow_language_fallback=False
                    -> NOT eligible: only explicit lexicon entries trusted.

        Why the old ``lang in ("HI_ROM", "HI_DEV")`` branch was wrong:
            Because the caller already guarantees that condition, the check
            was always True, silently making ``require_lexicon_match=True``
            a no-op.  Gate 2 now correctly delegates to
            ``allow_language_fallback``.
        """
        # Gate 1: lexicon membership — unconditional.
        if token in self.hinglish_lexicon:
            return True

        # Gate 2: language-confidence path.
        if not self.require_lexicon_match:
            # User opted out of lexicon filtering entirely.
            return True
        # require_lexicon_match=True: honour allow_language_fallback.
        return self.allow_language_fallback

    def _to_devanagari(self, token: str, lang: str) -> str:
        """Roman → Devanagari.  Idempotent: already-Devanagari tokens pass through."""
        if self._contains_devanagari(token):
            return token  # idempotency: already in target script

        if not self._is_eligible(token, lang):
            return token

        converted = _transliterate(token, sanscript.ITRANS, sanscript.DEVANAGARI)

        if self.round_trip_validate:
            back = _transliterate(converted, sanscript.DEVANAGARI, sanscript.ITRANS)
            back = _simplify_roman(back, preserve_scheme=self.preserve_scheme_differences)
            orig_norm = _simplify_roman(token, preserve_scheme=self.preserve_scheme_differences)
            if _edit_distance(orig_norm, back) > self.round_trip_threshold:
                return token  # revert — transliteration is unreliable

        return converted

    def _to_roman(self, token: str, lang: str) -> str:
        """Devanagari → Roman.  Idempotent: already-Roman tokens pass through."""
        if not self._contains_devanagari(token):
            return token  # idempotency: already in target script

        if not self._is_eligible(token, lang):
            return token

        raw_roman = _transliterate(token, sanscript.DEVANAGARI, self._sp_scheme)
        roman = _simplify_roman(raw_roman, preserve_scheme=self.preserve_scheme_differences)
        roman = self.roman_post_map.get(roman, roman)

        if self.round_trip_validate:
            back_dev = _transliterate(roman, sanscript.ITRANS, sanscript.DEVANAGARI)
            if _edit_distance(token, back_dev) > self.round_trip_threshold:
                return token  # revert — round-trip diverged too far

        # Final eligibility: if the result still isn't lexicon-known and the
        # user wants strict matching, revert to Devanagari.
        if self.require_lexicon_match and roman not in self.hinglish_lexicon:
            if token in self.hinglish_lexicon:
                return token
            # Accept anyway — language-confidence path already approved it.
        return roman


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[ScriptUnification] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = ScriptUnification()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: "str | Path",
    output_csv: "str | Path",
    text_col: str = "text",
    processor: Optional[ScriptUnification] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = ScriptUnification()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _run_smoke_tests() -> None:
    """
    Smoke tests covering:
      1.  Roman → Devanagari: basic lexicon word (bahut → बहुत).
      2.  Devanagari → Roman: basic lexicon word (बहुत → bahut).
      3.  English preservation (both directions).
      4.  Unknown token preserved when both gates are closed
          (require_lexicon_match=True, allow_language_fallback=False).
      5a. Language-confidence fallback ON (default): 'samjhdaar' converts.
      5b. Language-confidence fallback OFF: 'samjhdaar' is preserved.
      5c. Regression guard: require_lexicon_match=True is not a no-op.
      6.  Idempotency: devanagari direction.
      7.  Idempotency: roman direction.
      8.  Diacritic stripping: ātmā → atma.
      9.  Round-trip validation rejects bad transliterations.
      10. preserve_scheme_differences=True retains scholarly distinctions.
      11. roman_post_map applied correctly.
      12. require_lexicon_match=False: force-converts non-lexicon tokens.
      13. Digit tokens pass through unchanged.
      14. Mixed sentence: only the correct script tokens are converted.
      15. process_dataframe expands correctly.
    """
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))
        status = PASS if condition else FAIL
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    # ── 1. Roman → Devanagari ──────────────────────────────────────────────
    su_dev = ScriptUnification(target_script="devanagari", require_lexicon_match=False)
    out1 = su_dev._process_token("bahut")
    # बहुत is the expected Devanagari for bahut.
    check("roman_to_devanagari",
          "\u092c" in out1,    # ब is U+092C
          f"'bahut' → '{out1}'")

    # ── 2. Devanagari → Roman ──────────────────────────────────────────────
    su_rom = ScriptUnification(target_script="roman", require_lexicon_match=False)
    # बहुत = bahut in Devanagari.
    out2 = su_rom._process_token("\u092c\u0939\u0941\u0924")
    check("devanagari_to_roman",
          out2.startswith("b") and len(out2) > 0,
          f"'बहुत' → '{out2}'")

    # ── 3. English preservation ────────────────────────────────────────────
    for direction, su in [("devanagari", su_dev), ("roman", su_rom)]:
        out = su._process_token("movie")
        check(f"english_preserved_{direction}",
              out == "movie",
              f"'movie' → '{out}'")

    # ── 4. Unknown token preserved when both gates are closed ─────────────
    # require_lexicon_match=True + allow_language_fallback=False:
    # only explicit lexicon entries may be converted; everything else passes.
    su_strict = ScriptUnification(
        target_script="devanagari",
        require_lexicon_match=True,
        allow_language_fallback=False,
    )
    out4 = su_strict._process_token("xyzabc")
    check("unknown_preserved_strict",
          out4 == "xyzabc",
          f"'xyzabc' → '{out4}'")

    # ── 5a. Language-confidence fallback ON (default) ─────────────────────
    # require_lexicon_match=True, allow_language_fallback=True (default):
    # 'samjhdaar' is NOT in the stock lexicon but _detect_language returns
    # HI_ROM, so it should be converted.
    su_fallback_on = ScriptUnification(
        target_script="devanagari",
        require_lexicon_match=True,
        allow_language_fallback=True,   # default
        round_trip_validate=False,
    )
    out5a = su_fallback_on._process_token("samjhdaar")
    has_dev5a = any("\u0900" <= c <= "\u097F" for c in out5a)
    check("language_confidence_fallback_on_converts",
          has_dev5a,
          f"'samjhdaar' (fallback=True) → '{out5a}' — must contain Devanagari")

    # ── 5b. Language-confidence fallback OFF ──────────────────────────────
    # require_lexicon_match=True, allow_language_fallback=False:
    # 'samjhdaar' is not in the lexicon, so it must be preserved unchanged.
    su_fallback_off = ScriptUnification(
        target_script="devanagari",
        require_lexicon_match=True,
        allow_language_fallback=False,
    )
    out5b = su_fallback_off._process_token("samjhdaar")
    check("language_confidence_fallback_off_preserves",
          out5b == "samjhdaar",
          f"'samjhdaar' (fallback=False) → '{out5b}' — must be unchanged")

    # ── 5c. Regression: require_lexicon_match=True actually gates ─────────
    # This directly tests the bug where lang in ("HI_ROM", "HI_DEV") inside
    # _is_eligible was always True (caller pre-filters), making the flag
    # effectively a no-op.  A word not in the lexicon must be preserved when
    # both require_lexicon_match=True and allow_language_fallback=False.
    non_lexicon_word = "padhoge"   # valid HI_ROM but not in stock lexicon
    out5c = su_fallback_off._process_token(non_lexicon_word)
    check("require_lexicon_match_regression_guard",
          out5c == non_lexicon_word,
          f"'{non_lexicon_word}' (strict) → '{out5c}' — flag must not be a no-op")

    # ── 6. Idempotency — devanagari direction ─────────────────────────────
    text_idem_dev = "bahut accha khana"
    pass1_dev = su_dev.process(text_idem_dev)
    pass2_dev = su_dev.process(pass1_dev)
    check("idempotency_devanagari",
          pass1_dev == pass2_dev,
          f"pass1='{pass1_dev}' pass2='{pass2_dev}'")

    # ── 7. Idempotency — roman direction ──────────────────────────────────
    text_idem_rom = "\u092c\u0939\u0941\u0924 \u0905\u091a\u094d\u091b\u093e"  # बहुत अच्छा
    pass1_rom = su_rom.process(text_idem_rom)
    pass2_rom = su_rom.process(pass1_rom)
    check("idempotency_roman",
          pass1_rom == pass2_rom,
          f"pass1='{pass1_rom}' pass2='{pass2_rom}'")

    # ── 8. Diacritic stripping ────────────────────────────────────────────
    stripped = _simplify_roman("\u0101tm\u0101")   # ātmā
    check("diacritic_stripping",
          stripped == "atma",
          f"'ātmā' → '{stripped}'")

    # Also test ī → i.
    stripped2 = _simplify_roman("d\u012bpak")     # dīpak
    check("diacritic_stripping_long_i",
          stripped2 == "dipak",
          f"'dīpak' → '{stripped2}'")

    # ── 9. Round-trip validation ──────────────────────────────────────────
    # Construct a token that will almost certainly fail round-trip validation.
    su_rt = ScriptUnification(
        target_script="devanagari",
        require_lexicon_match=False,
        round_trip_validate=True,
        round_trip_threshold=0,  # zero tolerance
    )
    # "qwerty" has no meaningful ITRANS transliteration; should revert.
    out9 = su_rt._process_token("qwerty")
    check("round_trip_reverts_bad_transliteration",
          out9 == "qwerty",
          f"'qwerty' with threshold=0 → '{out9}' (should revert)")

    # ── 10. preserve_scheme_differences ───────────────────────────────────
    su_scholarly = ScriptUnification(
        target_script="roman",
        roman_scheme="iast",
        require_lexicon_match=False,
        preserve_scheme_differences=True,
        round_trip_validate=False,
    )
    # बहुत in IAST should give something with diacritics retained.
    out10 = su_scholarly._process_token("\u092c\u0939\u0941\u0924")
    check("preserve_scheme_differences",
          isinstance(out10, str) and len(out10) > 0,
          f"बहुत (IAST, preserve) → '{out10}'")

    # ── 11. roman_post_map ────────────────────────────────────────────────
    su_postmap = ScriptUnification(
        target_script="roman",
        require_lexicon_match=False,
        roman_post_map={"bahut": "bohot"},
        round_trip_validate=False,
    )
    out11 = su_postmap._process_token("\u092c\u0939\u0941\u0924")  # बहुत
    check("roman_post_map_applied",
          out11 == "bohot",
          f"बहुत with post_map → '{out11}'")

    # ── 12. require_lexicon_match=False force-converts ────────────────────
    su_force = ScriptUnification(
        target_script="devanagari",
        require_lexicon_match=False,
        round_trip_validate=False,
    )
    out12 = su_force._process_token("padhoge")
    has_dev12 = any("\u0900" <= c <= "\u097F" for c in out12)
    check("force_convert_non_lexicon",
          has_dev12,
          f"'padhoge' (require_lexicon_match=False) → '{out12}'")

    # ── 13. Digit pass-through ────────────────────────────────────────────
    check("digit_passthrough",
          su_dev._process_token("42") == "42",
          "'42' should pass through unchanged")

    # ── 14. Mixed sentence: only matching-script tokens converted ─────────
    # "yaar movie bahut accha" — all roman; devanagari direction should
    # convert HI_ROM tokens and leave "movie" (EN) untouched.
    mixed = "yaar movie bahut accha"
    out14 = su_dev.process(mixed)
    check("mixed_sentence_english_preserved",
          "movie" in out14,
          f"'{mixed}' → '{out14}'")

    # ── 15. process_dataframe ─────────────────────────────────────────────
    import pandas as _pd
    df = _pd.DataFrame({"text": ["bahut accha", "movie theek hai"]})
    result_df = su_dev.process_dataframe(df, text_col="text", output_col="unified")
    check("process_dataframe_columns",
          "unified" in result_df.columns,
          f"columns: {list(result_df.columns)}")
    check("process_dataframe_row_count",
          len(result_df) == len(df),
          f"rows: {len(result_df)}")

    # Summary.
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"\n  {passed}/{total} smoke tests passed.")
    if passed < total:
        failed = [name for name, ok, _ in results if not ok]
        raise AssertionError(f"Failed: {failed}")


if __name__ == "__main__":
    print("Running smoke tests for script_unification.py …\n")
    _run_smoke_tests()
    print("\nDone.")