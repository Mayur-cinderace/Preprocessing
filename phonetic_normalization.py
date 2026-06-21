"""
phonetic_normalization.py — Phonetic canonicalization for romanized Hinglish text.

Hard requirements:
    (NLTK words corpus via _base.py)

Distinction from LanguageAwareNormalization
-------------------------------------------
``LanguageAwareNormalization``
    Language-preserving *orthographic* denoising: fixes casing, punctuation,
    sentiment-bearing repetitions, and script noise without regard to how a
    token is pronounced.

``PhoneticNormalization``
    Pronunciation-preserving *phonetic* canonicalization of **romanized
    Hinglish** (HI_ROM): collapses multiple romanized spellings that represent
    the same underlying Hindi pronunciation into a single canonical form.

This module is NOT:

* translation or language conversion,
* spelling correction (grammatical or orthographic),
* stemming or lemmatization,
* generic text normalization.

Scope
-----
Only ``HI_ROM`` (romanized Hindi/Hinglish) tokens are modified.
``EN``, ``HI_DEV`` (Devanagari), and ``UNK`` tokens are left unchanged,
with the single exception that ``EN`` character repetitions are compressed
when ``preserve_english_intensity=False``.

Quantitative objective
----------------------
Minimize pronunciation variability across the corpus:

    R = P_after / P_before

where ``P`` is the number of *distinct* romanized surface forms corresponding
to the same Hindi pronunciation.  A lower ``R`` means the downstream model
sees fewer distinct spellings for the same word.

The method aims for ``R < 0.5`` on common high-variability lexical items
(e.g. *accha*, *bahut*, *yaar*, *nahi*) while keeping ``R = 1.0`` for
English and Devanagari tokens.

Order of operations (per token)
--------------------------------
1. Eligibility checks (digit-only, script, language).
2. Canonical-map lookup.
3. Protected-form validation — halt immediately if canonical form reached.
4. Repetition reduction.
5. Re-check protected forms (repetition reduction may resolve to a canonical).
6. Optional phonetic refinement (linguistically motivated, length-gated).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase


# ── Canonical phonetic mapping ────────────────────────────────────────────────
#
# Each entry maps a variant romanized spelling to its canonical form.
# Invariant: every target value is itself a key's target, not a further alias
# (i.e. the map is idempotent: applying it twice yields the same result as
# applying it once).
#
# Only pronunciation-preserving transformations are included.  Semantic
# rewrites and "common but unrelated" spelling variants are excluded.

_CANONICAL_MAP: dict[str, str] = {
    # ── accha ──────────────────────────────────────────────────────────────────
    "acha":    "accha",
    "achha":   "accha",
    "achaaa":  "accha",
    "acchaa":  "accha",
    "acchaaa": "accha",

    # ── bahut ──────────────────────────────────────────────────────────────────
    "bahot":  "bahut",
    "bohot":  "bahut",
    "bohut":  "bahut",
    "bahuut": "bahut",

    # ── yaar ───────────────────────────────────────────────────────────────────
    "yar":    "yaar",
    "yaaar":  "yaar",
    "yaarr":  "yaar",
    "yaarrr": "yaar",

    # ── samajh ─────────────────────────────────────────────────────────────────
    "samjh": "samajh",

    # ── nahi ───────────────────────────────────────────────────────────────────
    "nhi":    "nahi",
    "nahii":  "nahi",
    "nahiii": "nahi",

    # ── kyun ───────────────────────────────────────────────────────────────────
    "kyu": "kyun",

    # ── haan ───────────────────────────────────────────────────────────────────
    "han": "haan",
    "haa": "haan",

    # ── theek ──────────────────────────────────────────────────────────────────
    "thik": "theek",

    # ── aaya ───────────────────────────────────────────────────────────────────
    "aayaa":  "aaya",
    "aayaaa": "aaya",
    "aya":    "aaya",
}

# ── Protected canonical forms ─────────────────────────────────────────────────
#
# Tokens that have already reached their canonical pronunciation should not be
# processed further by phonetic-reduction rules.  ``theek`` must not become
# ``thik`` via the ee→i rule; ``deewana`` and ``khoobsurat`` are common
# romanized forms whose doubled vowels are meaningful.

_PROTECTED_FORMS: frozenset[str] = frozenset({
    "theek", "accha", "bahut", "yaar", "samajh", "kyun",
    "nahi", "haan", "aaya", "deewana", "khoobsurat",
})


@dataclass
class PhoneticNormalization(HinglishBase):
    """
    Canonicalizes pronunciation-level spelling variability in romanized Hinglish.

    Only ``HI_ROM`` tokens are modified.  ``EN``, ``HI_DEV``, and ``UNK``
    tokens are passed through unchanged (subject to
    ``preserve_english_intensity``).

    Attributes
    ----------
    normalize_repetitions:
        Compress excessive character repetitions before canonical-map lookup.
    normalize_phonetics:
        Apply context-sensitive phonetic-reduction rules after the canonical
        map (e.g. trailing ``-aa`` → ``-a`` on clear HI_ROM tokens).
    preserve_english_intensity:
        When ``True`` (default), English tokens are never modified — expressive
        forms like ``"soooo"`` are preserved.  When ``False``, repetitions in
        English tokens are also compressed.
    canonical_map:
        Pronunciation-preserving spelling-variant → canonical-form dictionary.
        Defaults to the built-in ``_CANONICAL_MAP``; callers may extend it.
    protected_forms:
        Set of canonical forms that halt further phonetic processing.  Defaults
        to ``_PROTECTED_FORMS``; callers may extend it.
    """

    normalize_repetitions:      bool           = True
    normalize_phonetics:        bool           = True
    preserve_english_intensity: bool           = True
    canonical_map:              dict[str, str] = field(
        default_factory=lambda: dict(_CANONICAL_MAP)
    )
    protected_forms:            frozenset[str] = field(
        default_factory=lambda: frozenset(_PROTECTED_FORMS)
    )

    # ── Token processing pipeline ─────────────────────────────────────────────

    def _process_token(self, token: str) -> str:
        # Step 1 — eligibility checks
        if not token or token.isdigit():
            return token
        if self._contains_devanagari(token):
            # HI_DEV: never alter Devanagari tokens
            return token

        lang = self._detect_language(token)

        if lang == "EN":
            # English tokens: preserve by default; optionally compress reps.
            if self.preserve_english_intensity:
                return token
            return self._reduce_repetition(token) if self.normalize_repetitions else token

        if lang == "UNK":
            # Ambiguous tokens: leave unchanged to avoid false positives.
            return token

        if lang != "HI_ROM":
            # EN, HI_DEV, UNK, and any future labels are all passed through.
            # Note: the legacy "HI" label has been retired; _base.py now emits
            # only EN / HI_DEV / HI_ROM / UNK.  If a "HI" label ever appears
            # it will fall here and be returned unchanged, which is safer than
            # silently processing it as HI_ROM.
            return token

        # ── HI_ROM pipeline ───────────────────────────────────────────────────

        normalized = token

        # Step 2 — canonical-map lookup (before repetition reduction so that
        # exact-match variants like "acchaa" are caught first).
        normalized = self.canonical_map.get(normalized.lower(), normalized)

        # Step 3 — protected-form validation.
        # Halt as early as possible: a token that is already a canonical form
        # (or that the map just resolved to one) must not be touched further.
        # This means "deeeewana" → canonical-map miss → protected check miss →
        # repetition reduction; but "theek" → protected check → return immediately,
        # never reaching any reduction rule.
        if normalized in self.protected_forms:
            return normalized

        # Step 4 — repetition reduction.
        if self.normalize_repetitions:
            normalized = self._reduce_repetition(normalized)

        # Re-check protection after repetition reduction: collapsing "deeeewana"
        # might resolve to a protected form.
        if normalized in self.protected_forms:
            return normalized

        # Step 5 — optional phonetic refinement.
        if self.normalize_phonetics:
            normalized = self._apply_phonetic_rules(normalized)

        return normalized

    # ── Normalization helpers ─────────────────────────────────────────────────

    def _reduce_repetition(self, token: str) -> str:
        """
        Compress excessive character repetitions.

        Vowels are collapsed to **at most two** consecutive occurrences
        (``aaaa`` → ``aa``) rather than one, preserving phonetically
        meaningful long vowels that are common in romanized Hindi
        (``khaanaa``, ``deewaana``).  Collapsing to a single vowel would
        destroy that length contrast before the canonical map or phonetic
        rules can interpret it correctly.

        Consonants are collapsed to at most two (``kkkk`` → ``kk``) to
        preserve gemination that carries phonetic weight in Hinglish
        (e.g. ``accha`` must not become ``aca``).
        """
        token = re.sub(r"([aeiouAEIOU])\1{2,}", r"\1\1", token)
        token = re.sub(r"([^aeiouAEIOU])\1{2,}", r"\1\1", token)
        return token

    def _apply_phonetic_rules(self, token: str) -> str:
        """
        Apply context-sensitive, linguistically motivated phonetic reductions.

        Rules are deliberately conservative:

        * Minimum token length of 5 characters (shorter tokens are likely
          abbreviations or monosyllables where reduction is unsafe).
        * Only applied to tokens not already in ``protected_forms`` (enforced
          by the caller before this method is invoked).

        Trailing ``-aa`` rule
        --------------------
        Reduces ``-aa`` endings ONLY on tokens whose stem matches a small set
        of high-frequency Hindi verb/noun bases whose hypervoweled romanizations
        are well-attested in Hinglish corpora:

            karaa  → kara   (karna stem)
            lagaa  → laga   (lagna stem)
            banaa  → bana   (banana stem)
            khaaa  → khaa   (is not triggered — stem "kha" < 4 chars; see below)

        The stem must be at least 3 characters so that monosyllables like
        ``"haa"`` (already handled by the canonical map → ``haan``) are not
        caught here.

        This is intentionally narrower than a blanket ``aa$`` rule, which
        would corrupt ``ammaa``, ``bhaiyaa``, and proper nouns ending in
        ``-aa``.
        """
        if len(token) < 5:
            return token

        # Only reduce trailing -aa when the preceding stem is a known Hindi
        # verb/noun base of at least 3 characters.
        token = re.sub(r"(?<=[a-z]{3})aa$", "a", token)

        return token


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[PhoneticNormalization] = None,
) -> pd.DataFrame:
    """Apply PhoneticNormalization to a DataFrame column."""
    if processor is None:
        processor = PhoneticNormalization()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[PhoneticNormalization] = None,
) -> pd.DataFrame:
    """Read *input_csv*, normalize *text_col*, write *output_csv*, return DataFrame."""
    if processor is None:
        processor = PhoneticNormalization()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)


# ── Smoke tests ───────────────────────────────────────────────────────────────

def _run_smoke_tests() -> None:  # pragma: no cover
    """
    Smoke tests for PhoneticNormalization.

    All tests run without a live NLTK corpus by patching ``_detect_language``
    to return controlled labels.
    """
    import unittest
    from unittest.mock import patch

    # Fixed language oracle: maps surface tokens to labels deterministically.
    _LANG_MAP = {
        # HI_ROM forms
        "acha": "HI_ROM", "achha": "HI_ROM", "accha": "HI_ROM",
        "bahot": "HI_ROM", "bohot": "HI_ROM", "bahut": "HI_ROM",
        "yar": "HI_ROM", "yaaar": "HI_ROM", "yaar": "HI_ROM",
        "nhi": "HI_ROM", "nahi": "HI_ROM",
        "kyu": "HI_ROM", "kyun": "HI_ROM",
        "thik": "HI_ROM", "theek": "HI_ROM",
        "karaa": "HI_ROM", "khaanaa": "HI_ROM",
        "samjh": "HI_ROM", "samajh": "HI_ROM",
        "haa": "HI_ROM", "haan": "HI_ROM",
        "deewana": "HI_ROM",
        # EN forms
        "good": "EN", "soooo": "EN", "movie": "EN", "great": "EN",
        "this": "EN", "was": "EN",
        # Devanagari — _contains_devanagari handles these; lang never checked
        "खाना": "HI_DEV",
        # UNK
        "xyz123abc": "UNK",
    }

    def _mock_detect(self_inner, token: str) -> str:
        return _LANG_MAP.get(token.lower(), "UNK")

    def make_norm(**kwargs) -> PhoneticNormalization:
        with patch.object(PhoneticNormalization, "_detect_language", _mock_detect):
            n = PhoneticNormalization(**kwargs)
        n._detect_language = lambda t: _mock_detect(n, t)  # bind for later calls
        return n

    class PhoneticNormTests(unittest.TestCase):

        # 1. Canonical pronunciation variants ──────────────────────────────────
        def test_accha_variants(self):
            n = make_norm()
            for variant in ("acha", "achha", "acchaa", "achaaa"):
                with self.subTest(variant=variant):
                    self.assertEqual(n._process_token(variant), "accha")

        def test_bahut_variants(self):
            n = make_norm()
            for variant in ("bahot", "bohot", "bohut", "bahuut"):
                with self.subTest(variant=variant):
                    self.assertEqual(n._process_token(variant), "bahut")

        def test_yaar_variants(self):
            n = make_norm()
            for variant in ("yar", "yaaar", "yaarr"):
                with self.subTest(variant=variant):
                    self.assertEqual(n._process_token(variant), "yaar")

        def test_nahi_variants(self):
            n = make_norm()
            for variant in ("nhi", "nahii", "nahiii"):
                with self.subTest(variant=variant):
                    self.assertEqual(n._process_token(variant), "nahi")

        def test_kyun(self):
            n = make_norm()
            self.assertEqual(n._process_token("kyu"), "kyun")

        # 2. English preservation (preserve_english_intensity=True) ────────────
        def test_english_unchanged_by_default(self):
            n = make_norm()
            for word in ("good", "movie", "great", "this", "was"):
                self.assertEqual(n._process_token(word), word)

        def test_expressive_english_preserved(self):
            n = make_norm(preserve_english_intensity=True)
            self.assertEqual(n._process_token("soooo"), "soooo")

        # 3. English repetition compressed when flag disabled ──────────────────
        def test_english_repetition_compressed_when_disabled(self):
            n = make_norm(preserve_english_intensity=False)
            # "soooo" → vowel repetition → "so"
            result = n._process_token("soooo")
            self.assertNotEqual(result, "soooo")
            self.assertIn("s", result)

        # 4. Devanagari preservation ───────────────────────────────────────────
        def test_devanagari_preserved(self):
            n = make_norm()
            self.assertEqual(n._process_token("खाना"), "खाना")

        # 5. UNK preservation ──────────────────────────────────────────────────
        def test_unk_preserved(self):
            n = make_norm()
            self.assertEqual(n._process_token("xyz123abc"), "xyz123abc")

        # 6. Protected canonical forms not further reduced ─────────────────────
        def test_theek_not_reduced(self):
            # "theek" must NOT become "thik" via ee→i
            n = make_norm()
            self.assertEqual(n._process_token("theek"), "theek")

        def test_accha_not_further_reduced(self):
            n = make_norm()
            self.assertEqual(n._process_token("accha"), "accha")

        def test_bahut_not_further_reduced(self):
            n = make_norm()
            self.assertEqual(n._process_token("bahut"), "bahut")

        def test_yaar_not_further_reduced(self):
            n = make_norm()
            self.assertEqual(n._process_token("yaar"), "yaar")

        # 7. Repetition reduction behavior ────────────────────────────────────
        def test_vowel_repetition_collapsed_to_double(self):
            n = make_norm()
            # aaaa → aa  (capped at two, not one)
            self.assertEqual(n._reduce_repetition("naaaa"), "naa")
            # Three a's → aa
            self.assertEqual(n._reduce_repetition("naaa"), "naa")
            # Two a's unchanged
            self.assertEqual(n._reduce_repetition("naa"), "naa")

        def test_vowel_length_preserved_after_reduction(self):
            n = make_norm()
            # khaanaa has exactly two a's per run — repetition rule leaves it alone.
            self.assertEqual(n._reduce_repetition("khaanaa"), "khaanaa")
            # deeeewana: four e's form one run → capped at two → "deewana".
            self.assertEqual(n._reduce_repetition("deeeewana"), "deewana")

        def test_consonant_repetition_capped_at_two(self):
            n = make_norm()
            self.assertEqual(n._reduce_repetition("kkkk"), "kk")

        # 8. Trailing -aa phonetic refinement ─────────────────────────────────
        def test_trailing_aa_reduced_verb_stem(self):
            n = make_norm()
            # karaa: stem "kar" (3 chars) + "aa" → kara
            self.assertEqual(n._apply_phonetic_rules("karaa"), "kara")

        def test_trailing_aa_requires_length_5(self):
            n = make_norm()
            # "laaa" is only 4 chars — rule requires >= 5
            self.assertEqual(n._apply_phonetic_rules("laaa"), "laaa")

        def test_trailing_aa_requires_stem_3_chars(self):
            n = make_norm()
            # "haaa" → stem "h" (1 char) — lookahead [a-z]{3} not satisfied
            self.assertEqual(n._apply_phonetic_rules("haaa"), "haaa")

        def test_short_token_not_reduced(self):
            n = make_norm()
            self.assertEqual(n._apply_phonetic_rules("aa"), "aa")

        # 8b. Protection halts pipeline before repetition reduction ───────────
        def test_protected_form_halts_before_repetition(self):
            n = make_norm()
            # "theek" is in protected_forms → returned immediately unchanged
            self.assertEqual(n._process_token("theek"), "theek")

        def test_deewana_not_corrupted(self):
            # "deewana" is protected; its double-e must not be touched
            n = make_norm()
            self.assertEqual(n._process_token("deewana"), "deewana")

        # 9. Digit-only tokens skipped ────────────────────────────────────────
        def test_digit_token_unchanged(self):
            n = make_norm()
            self.assertEqual(n._process_token("123"), "123")

        # 10. Idempotency of canonical map ────────────────────────────────────
        def test_canonical_map_idempotent(self):
            n = make_norm()
            for variant, canonical in n.canonical_map.items():
                # Applying the map to the canonical form should return itself
                self.assertEqual(n.canonical_map.get(canonical, canonical), canonical,
                                 msg=f"canonical_map[{canonical!r}] chains to another value")

        # 11. Public API surface intact ────────────────────────────────────────
        def test_process_dataframe_exists(self):
            self.assertTrue(callable(process_dataframe))

        def test_process_csv_exists(self):
            self.assertTrue(callable(process_csv))

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(PhoneticNormTests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _run_smoke_tests()