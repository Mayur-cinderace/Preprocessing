"""
lang_id_tagging.py — Token-level language identification tagging for Hinglish.

Hard requirements:
    (NLTK words corpus via _base.py)

What it does
------------
Annotates every non-punctuation token with a configurable language tag drawn
from the four-way label set used throughout this toolkit:

    EN      — English token
    HI_ROM  — Romanized Hindi/Hinglish token
    HI_DEV  — Devanagari-script Hindi token
    UNK     — Language-ambiguous token

Unlike language identification itself, which merely predicts labels, this
module transforms those labels into *sequence-level signals* that downstream
models can exploit.  It also optionally inserts explicit code-switch boundary
markers at language-transition points.

This module is NOT:

* a language identification model,
* translation,
* normalization,
* token replacement,
* sequence rewriting.

Tagging strategies
------------------
Four strategies are supported, selected via ``TagFormat``:

    XML (default, backward-compatible):
        <EN>movie</EN>     <HI_ROM>yaar</HI_ROM>

    INLINE:
        movie|EN           yaar|HI_ROM

    PREFIX:
        [EN] movie         [HI_ROM] yaar

    SUFFIX:
        movie [EN]         yaar [HI_ROM]

Code-switch boundary markers
-----------------------------
When ``insert_boundary_markers=True``, a boundary token is inserted between
any two adjacent non-punctuation tokens whose language labels differ.

Default boundary format (XML):
    <SWITCH HI_ROM→EN>

Transformer-compatible format (SPECIAL_TOKEN):
    <SWITCH_HI_ROM_EN>

Inline format:
    |||HI_ROM→EN|||

Boundary markers are inserted *between* tokens and never alter lexical
content.  Punctuation tokens are transparent to boundary detection — a
language change across a punctuation token still counts as one transition,
not two.

Quantitative objective
----------------------
Language Identification Tagging aims to maximise explicit language-boundary
encoding while preserving lexical identity:

    coverage = tagged_tokens / non_punctuation_tokens
    density  = switch_boundaries / max(1, non_punctuation_tokens - 1)

``coverage`` excludes punctuation and digit-only tokens from the denominator
so that punctuation-heavy strings do not produce misleadingly low values.
High coverage with high density indicates a heavily code-switched sequence;
high coverage with low density indicates a monolingual or mildly mixed one.

Order of operations (per sequence)
------------------------------------
1. Tokenise via ``self._tokenize(text)`` (HinglishBase — preserves URLs,
   mentions, hashtags, contractions).
2. Detect language per token.
3. Identify punctuation tokens (transparent to boundary detection).
4. For each non-punctuation token: format the tag, emit boundary marker if
   the language has changed since the last non-punctuation token.
5. Reconstruct via ``self._reconstruct(tokens)`` (HinglishBase).
6. Accumulate statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase


# ── Enumerations ──────────────────────────────────────────────────────────────

class TagFormat(str, Enum):
    """Controls how language labels are wrapped around tokens."""
    XML            = "xml"            # <EN>word</EN>        (default)
    INLINE         = "inline"         # word|EN
    PREFIX         = "prefix"         # [EN] word
    SUFFIX         = "suffix"         # word [EN]


class BoundaryFormat(str, Enum):
    """Controls the format of code-switch boundary markers."""
    XML            = "xml"            # <SWITCH HI_ROM→EN>
    INLINE         = "inline"         # |||HI_ROM→EN|||
    SPECIAL_TOKEN  = "special_token"  # <SWITCH_HI_ROM_EN>  (transformer-safe)


# ── Tag formatting helpers ────────────────────────────────────────────────────

def _format_tag(token: str, lang: str, fmt: TagFormat) -> str:
    """Wrap *token* with a language annotation in the requested format."""
    if fmt is TagFormat.XML:
        return f"<{lang}>{token}</{lang}>"
    if fmt is TagFormat.INLINE:
        return f"{token}|{lang}"
    if fmt is TagFormat.PREFIX:
        return f"[{lang}] {token}"
    if fmt is TagFormat.SUFFIX:
        return f"{token} [{lang}]"
    # Unreachable given exhaustive enum, but keeps mypy happy.
    return token  # pragma: no cover


def _format_boundary(from_lang: str, to_lang: str, fmt: BoundaryFormat) -> str:
    """Return a boundary marker string for a *from_lang* → *to_lang* switch."""
    if fmt is BoundaryFormat.XML:
        return f"<SWITCH {from_lang}→{to_lang}>"
    if fmt is BoundaryFormat.INLINE:
        return f"|||{from_lang}→{to_lang}|||"
    if fmt is BoundaryFormat.SPECIAL_TOKEN:
        return f"<SWITCH_{from_lang}_{to_lang}>"
    return ""  # pragma: no cover


# ── Statistics accumulator ────────────────────────────────────────────────────

@dataclass
class TaggingStats:
    """Mutable accumulator for tagging statistics."""
    total_tokens:          int = 0
    non_punctuation_tokens: int = 0   # excludes punctuation and digit-only tokens
    tagged_tokens:         int = 0
    switch_boundaries:     int = 0
    language_counts:       dict = field(default_factory=dict)

    def increment_lang(self, lang: str) -> None:
        self.language_counts[lang] = self.language_counts.get(lang, 0) + 1

    def reset(self) -> None:
        self.total_tokens           = 0
        self.non_punctuation_tokens = 0
        self.tagged_tokens          = 0
        self.switch_boundaries      = 0
        self.language_counts        = {}

    def as_dict(self) -> dict:
        return {
            "total_tokens":           self.total_tokens,
            "non_punctuation_tokens": self.non_punctuation_tokens,
            "tagged_tokens":          self.tagged_tokens,
            "switch_boundaries":      self.switch_boundaries,
            "language_counts":        dict(self.language_counts),
            # coverage = tagged / non-punctuation (not total), so that
            # punctuation-heavy strings don't produce misleadingly low values.
            "coverage": round(
                self.tagged_tokens / max(1, self.non_punctuation_tokens), 4
            ),
            # density = switches per adjacent non-punctuation token pair.
            "density": round(
                self.switch_boundaries / max(1, self.non_punctuation_tokens - 1), 4
            ),
        }


# ── Main class ────────────────────────────────────────────────────────────────

@dataclass
class LanguageIdentificationTagging(HinglishBase):
    """
    Wraps each token with a language tag and optionally inserts code-switch
    boundary markers into the token sequence.

    Attributes
    ----------
    tag_unknown:
        Tag ambiguous tokens with the UNK label.  When ``False``, unknown
        tokens are returned as-is (no tag applied).
    tag_format:
        Tagging strategy.  Defaults to ``TagFormat.XML`` for backward
        compatibility (``<EN>word</EN>``).
    insert_boundary_markers:
        Insert an explicit marker token between adjacent tokens whose language
        labels differ.  Punctuation tokens are transparent to this detection.
    boundary_format:
        Format of the inserted boundary marker.  Defaults to
        ``BoundaryFormat.XML`` (``<SWITCH HI_ROM→EN>``).
    """

    tag_unknown:              bool          = True
    tag_format:               TagFormat     = TagFormat.XML
    insert_boundary_markers:  bool          = False
    boundary_format:          BoundaryFormat = BoundaryFormat.XML

    def _setup(self) -> None:
        self._stats = TaggingStats()

    # ── Public statistics ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Read-only snapshot of tagging statistics accumulated so far."""
        return self._stats.as_dict()

    def reset_stats(self) -> None:
        """Reset all internal counters to zero."""
        self._stats.reset()

    # ── Sequence-level processing (overrides HinglishBase) ────────────────────

    def process(self, text: str) -> str:
        """
        Override ``HinglishBase.process`` to route through ``augment``.

        Without this override the inherited ``process`` iterates tokens via
        ``_process_token``, which has no access to neighbouring language labels
        and therefore cannot emit boundary markers or accumulate sequence-level
        statistics.  Delegating here ensures ``t.process(text)`` and
        ``t.augment(text)`` are always identical — no silent behavioural split
        depending on which entry point the caller uses.
        """
        return self.augment(text)

    def augment(self, text: str) -> str:
        """
        Tag all tokens in *text* and optionally insert boundary markers.

        Uses ``self._tokenize`` and ``self._reconstruct`` from HinglishBase so
        that hashtags, mentions, URLs, emails, and contractions are handled
        consistently with the rest of the toolkit.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return text

        output_tokens: list[str] = []
        prev_lang: Optional[str] = None  # last *non-punctuation* token's lang

        for token in tokens:
            self._stats.total_tokens += 1

            # Punctuation: pass through untouched, transparent to boundaries.
            if not token or token.isdigit() or self._is_punctuation_token(token):
                output_tokens.append(token)
                continue

            self._stats.non_punctuation_tokens += 1
            lang = self._detect_language(token)
            self._stats.increment_lang(lang)

            # Boundary detection: compare against last non-punctuation lang.
            if (
                self.insert_boundary_markers
                and prev_lang is not None
                and prev_lang != lang
            ):
                self._stats.switch_boundaries += 1
                output_tokens.append(
                    _format_boundary(prev_lang, lang, self.boundary_format)
                )

            # Tag the token.
            if lang == "UNK" and not self.tag_unknown:
                output_tokens.append(token)
            else:
                output_tokens.append(_format_tag(token, lang, self.tag_format))
                self._stats.tagged_tokens += 1

            prev_lang = lang

        return self._reconstruct(output_tokens)

    # ── Single-token path (satisfies HinglishBase._process_token contract) ────

    def _process_token(self, token: str) -> str:
        """
        Tag a single token.

        This satisfies the ``HinglishBase._process_token`` contract for callers
        that invoke tokens individually.  It does NOT emit boundary markers
        (those require sequence context) and does NOT update statistics.
        Use ``augment()`` for full sequence processing with boundary support.
        """
        if not token or token.isdigit() or self._is_punctuation_token(token):
            return token
        lang = self._detect_language(token)
        if lang == "UNK" and not self.tag_unknown:
            return token
        return _format_tag(token, lang, self.tag_format)


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[LanguageIdentificationTagging] = None,
) -> pd.DataFrame:
    """Apply LanguageIdentificationTagging to a DataFrame column."""
    if processor is None:
        processor = LanguageIdentificationTagging()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[LanguageIdentificationTagging] = None,
) -> pd.DataFrame:
    """Read *input_csv*, tag *text_col*, write *output_csv*, return DataFrame."""
    if processor is None:
        processor = LanguageIdentificationTagging()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)


# ── Smoke tests ───────────────────────────────────────────────────────────────

def _run_smoke_tests() -> None:  # pragma: no cover
    """
    Smoke tests for LanguageIdentificationTagging.

    All tests use a fixed language oracle so no live NLTK corpus is needed.
    """
    import unittest
    from unittest.mock import patch

    _LANG_MAP: dict[str, str] = {
        "movie":   "EN",
        "good":    "EN",
        "this":    "EN",
        "was":     "EN",
        "really":  "EN",
        "yaar":    "HI_ROM",
        "bahut":   "HI_ROM",
        "achha":   "HI_ROM",
        "खाना":    "HI_DEV",
        "पानी":    "HI_DEV",
        "xyz99":   "UNK",
    }

    def _mock_detect(self_inner, token: str) -> str:
        return _LANG_MAP.get(token.lower(), "UNK")

    def _mock_tokenize(self_inner, text: str) -> list[str]:
        # Minimal whitespace tokenizer — adequate for smoke tests.
        return text.split()

    def _mock_reconstruct(self_inner, tokens: list[str]) -> str:
        return " ".join(tokens)

    def make_tagger(**kwargs) -> LanguageIdentificationTagging:
        t = LanguageIdentificationTagging(**kwargs)
        t._detect_language   = lambda tok: _mock_detect(t, tok)
        t._tokenize          = lambda text: _mock_tokenize(t, text)
        t._reconstruct       = lambda toks: _mock_reconstruct(t, toks)
        t._stats             = TaggingStats()
        return t

    class LITTests(unittest.TestCase):

        # 1. Backward compatibility — default XML tags ─────────────────────────
        def test_default_en_tag(self):
            t = make_tagger()
            self.assertEqual(t._process_token("movie"), "<EN>movie</EN>")

        def test_default_hi_rom_tag(self):
            t = make_tagger()
            self.assertEqual(t._process_token("yaar"), "<HI_ROM>yaar</HI_ROM>")

        def test_default_hi_dev_tag(self):
            t = make_tagger()
            self.assertEqual(t._process_token("खाना"), "<HI_DEV>खाना</HI_DEV>")

        # 2. UNK behavior ──────────────────────────────────────────────────────
        def test_unk_tagged_when_flag_true(self):
            t = make_tagger(tag_unknown=True)
            self.assertEqual(t._process_token("xyz99"), "<UNK>xyz99</UNK>")

        def test_unk_passthrough_when_flag_false(self):
            t = make_tagger(tag_unknown=False)
            self.assertEqual(t._process_token("xyz99"), "xyz99")

        # 3. Punctuation preservation ──────────────────────────────────────────
        def test_punctuation_not_tagged(self):
            t = make_tagger()
            for punct in (".", ",", "!", "?", "...", "—"):
                with self.subTest(punct=punct):
                    self.assertEqual(t._process_token(punct), punct)

        def test_digit_not_tagged(self):
            t = make_tagger()
            self.assertEqual(t._process_token("42"), "42")

        # 4. Tagging strategies ────────────────────────────────────────────────
        def test_inline_format(self):
            t = make_tagger(tag_format=TagFormat.INLINE)
            self.assertEqual(t._process_token("movie"), "movie|EN")
            self.assertEqual(t._process_token("yaar"),  "yaar|HI_ROM")

        def test_prefix_format(self):
            t = make_tagger(tag_format=TagFormat.PREFIX)
            self.assertEqual(t._process_token("movie"), "[EN] movie")

        def test_suffix_format(self):
            t = make_tagger(tag_format=TagFormat.SUFFIX)
            self.assertEqual(t._process_token("movie"), "movie [EN]")

        # 5. Boundary markers — XML format ────────────────────────────────────
        def test_boundary_inserted_on_switch(self):
            t = make_tagger(insert_boundary_markers=True)
            result = t.augment("yaar movie")
            self.assertIn("<SWITCH HI_ROM→EN>", result)

        def test_no_boundary_in_monolingual_span(self):
            t = make_tagger(insert_boundary_markers=True)
            result = t.augment("this was really good")
            self.assertNotIn("SWITCH", result)

        def test_boundary_does_not_reorder_tokens(self):
            t = make_tagger(insert_boundary_markers=True)
            result = t.augment("yaar movie")
            # yaar must appear before movie regardless of boundary marker
            self.assertLess(result.index("yaar"), result.index("movie"))

        # 6. Boundary formats ──────────────────────────────────────────────────
        def test_boundary_inline_format(self):
            t = make_tagger(
                insert_boundary_markers=True,
                boundary_format=BoundaryFormat.INLINE,
            )
            result = t.augment("yaar movie")
            self.assertIn("|||HI_ROM→EN|||", result)

        def test_boundary_special_token_format(self):
            t = make_tagger(
                insert_boundary_markers=True,
                boundary_format=BoundaryFormat.SPECIAL_TOKEN,
            )
            result = t.augment("yaar movie")
            self.assertIn("<SWITCH_HI_ROM_EN>", result)

        # 7. Transformer-compatible tag names ─────────────────────────────────
        def test_xml_tags_are_transformer_compatible(self):
            # Tag names must contain only word characters and underscores
            # (no spaces, slashes, or special chars inside the tag name itself).
            t = make_tagger()
            tagged = t._process_token("yaar")
            # Extract tag name: first word inside < >
            import re
            m = re.match(r"<([A-Za-z_]+)>", tagged)
            self.assertIsNotNone(m)

        # 8. Lexical identity preserved ────────────────────────────────────────
        def test_lexical_identity_xml(self):
            t = make_tagger()
            original = "movie"
            tagged = t._process_token(original)
            self.assertIn(original, tagged)

        def test_lexical_identity_devanagari(self):
            t = make_tagger()
            original = "खाना"
            tagged = t._process_token(original)
            self.assertIn(original, tagged)

        # 9. Statistics reporting ─────────────────────────────────────────────
        def test_stats_populated(self):
            t = make_tagger(insert_boundary_markers=True)
            t.augment("yaar movie bahut good")
            s = t.stats
            self.assertIn("total_tokens",            s)
            self.assertIn("non_punctuation_tokens",  s)
            self.assertIn("tagged_tokens",           s)
            self.assertIn("switch_boundaries",       s)
            self.assertIn("language_counts",         s)
            self.assertIn("coverage",                s)
            self.assertIn("density",                 s)

        def test_stats_language_counts(self):
            t = make_tagger()
            t.augment("yaar movie bahut good")
            counts = t.stats["language_counts"]
            self.assertEqual(counts.get("HI_ROM", 0), 2)
            self.assertEqual(counts.get("EN",     0), 2)

        def test_stats_switch_boundaries(self):
            t = make_tagger(insert_boundary_markers=True)
            t.augment("yaar movie bahut good")
            # yaar→movie (HI_ROM→EN), movie→bahut (EN→HI_ROM), bahut→good (HI_ROM→EN)
            self.assertEqual(t.stats["switch_boundaries"], 3)

        def test_coverage_excludes_punctuation(self):
            # "yaar ! ! !" — 1 real token, 3 punctuation tokens
            # coverage should be 1/1 = 1.0, not 1/4 = 0.25
            t = make_tagger()
            t.augment("yaar ! ! !")
            s = t.stats
            self.assertEqual(s["non_punctuation_tokens"], 1)
            self.assertEqual(s["total_tokens"], 4)
            self.assertEqual(s["coverage"], 1.0)

        def test_stats_reset(self):
            t = make_tagger()
            t.augment("yaar movie")
            t.reset_stats()
            s = t.stats
            self.assertEqual(s["total_tokens"],           0)
            self.assertEqual(s["non_punctuation_tokens"], 0)
            self.assertEqual(s["tagged_tokens"],          0)
            self.assertEqual(s["switch_boundaries"],      0)

        # 12. process() and augment() are identical ────────────────────────────
        def test_process_equals_augment(self):
            # process() must route through augment() — same output, same stats.
            t1 = make_tagger(insert_boundary_markers=True)
            t2 = make_tagger(insert_boundary_markers=True)
            text = "yaar movie bahut good"
            self.assertEqual(t1.process(text), t2.augment(text))

        def test_process_accumulates_stats(self):
            # Calling process() should update statistics, not bypass them.
            t = make_tagger(insert_boundary_markers=True)
            t.process("yaar movie")
            self.assertGreater(t.stats["total_tokens"], 0)
            self.assertGreater(t.stats["switch_boundaries"], 0)

        # 10. Punctuation transparent to boundary detection ────────────────────
        def test_punctuation_transparent_to_boundary(self):
            t = make_tagger(insert_boundary_markers=True)
            # "yaar, movie" — comma between HI_ROM and EN should still fire one boundary
            result = t.augment("yaar , movie")
            self.assertEqual(result.count("SWITCH"), 1)

        # 11. Public API surface intact ────────────────────────────────────────
        def test_process_dataframe_exists(self):
            self.assertTrue(callable(process_dataframe))

        def test_process_csv_exists(self):
            self.assertTrue(callable(process_csv))

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(LITTests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _run_smoke_tests()