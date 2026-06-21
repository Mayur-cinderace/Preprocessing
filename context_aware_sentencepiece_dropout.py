"""
context_aware_sentencepiece_dropout.py — Context-Aware Stochastic SentencePiece
subword sampling for code-switched Hinglish text.

Hard requirements:
    pip install transformers sentencepiece
    (+ NLTK words corpus via _base.py)

The module requires a SentencePiece-backed tokenizer (e.g. ``xlm-roberta-base``)
that exposes ``.sp_model``.  Both the ``transformers`` import and the tokenizer
load are validated at construction time — no deferred warnings, no silent no-ops.

What it does
------------
Unlike standard SentencePiece dropout, which applies a fixed per-token sampling
probability, **Context-Aware SentencePiece Dropout** conditions augmentation
intensity on the *code-switching context* of each token:

* Tokens at code-switch **boundaries** (adjacent to a language transition) receive
  *increased* perturbation probability — subword ambiguity is highest at these
  seams.
* Tokens inside **stable monolingual spans** receive *reduced* perturbation
  probability — their segmentation is less ambiguous.
* Base probabilities remain language-specific (Hindi vs. English).

The effective probability follows:

    effective_probability = clip(base_prob × context_multiplier, 0.0, 1.0)

Default context multipliers:

    switch boundary  → 1.5
    stable span      → 0.7
    unknown context  → 1.0

This preserves semantic identity: only alternative subword segmentations are
generated; no words are replaced, deleted, reordered, or newly introduced.

Quantitative objectives
-----------------------
* Improve downstream model robustness to subword segmentation variability in
  code-switched Hinglish.
* Preserve code-switch structure so that language-boundary information is not
  destroyed during augmentation.
* Provide reproducible augmentation under fixed random seeds.
"""
from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import AutoTokenizer

from _base import HINGLISH_LEXICON, HinglishBase


# ── Context multiplier constants ──────────────────────────────────────────────

_MULTIPLIER_SWITCH_BOUNDARY: float = 1.5
_MULTIPLIER_STABLE_SPAN: float = 0.7
_MULTIPLIER_UNKNOWN: float = 1.0


@dataclass
class AugmentationStats:
    """Mutable accumulator for augmentation statistics."""
    total_tokens_seen: int = 0
    eligible_tokens: int = 0
    perturbed_tokens: int = 0
    switch_boundary_tokens: int = 0

    def reset(self) -> None:
        self.total_tokens_seen = 0
        self.eligible_tokens = 0
        self.perturbed_tokens = 0
        self.switch_boundary_tokens = 0

    def as_dict(self) -> dict:
        return {
            "total_tokens_seen": self.total_tokens_seen,
            "eligible_tokens": self.eligible_tokens,
            "perturbed_tokens": self.perturbed_tokens,
            "switch_boundary_tokens": self.switch_boundary_tokens,
        }


@dataclass
class ContextAwareSentencePieceDropout(HinglishBase):
    """
    Applies context-aware stochastic SentencePiece subword sampling to Hinglish.

    Unlike standard SentencePiece dropout this class conditions augmentation
    intensity on the *code-switching context* of each token, not just its
    language label.  Tokens at language-switch boundaries receive higher
    perturbation probability; tokens within stable monolingual runs receive
    lower perturbation probability.

    Attributes
    ----------
    tokenizer_name:
        HuggingFace model ID for a SentencePiece-backed tokenizer.
        Must expose ``.sp_model``; raises ``RuntimeError`` otherwise.
    hindi_dropout_prob:
        Base sampling probability for Hindi/Hinglish tokens.
    english_dropout_prob:
        Base sampling probability for English tokens.
    perturb_devanagari:
        Apply sampling to tokens written in Devanagari script.
    min_token_length:
        Skip tokens shorter than this (their subword split is trivially the
        same as the original token).
    sp_nbest_size:
        ``nbest_size`` argument forwarded to ``SampleEncodeAsPieces``.
    sp_alpha:
        ``alpha`` smoothing argument forwarded to ``SampleEncodeAsPieces``.
    subword_marker:
        Separator inserted between subword pieces in the output.
    random_seed:
        Seed for reproducibility; used to initialise a dedicated RNG instance.
    boundary_multiplier:
        Context multiplier applied to tokens at code-switch boundaries.
    stable_span_multiplier:
        Context multiplier applied to tokens inside stable monolingual spans.
    unknown_multiplier:
        Context multiplier applied when context cannot be determined.
    """

    tokenizer_name:        str   = "xlm-roberta-base"
    hindi_dropout_prob:    float = 0.3
    english_dropout_prob:  float = 0.05
    perturb_devanagari:    bool  = False
    min_token_length:      int   = 5
    sp_nbest_size:         int   = 64
    sp_alpha:              float = 0.1
    subword_marker:        str   = "▁"
    random_seed:           int   = 42
    boundary_multiplier:   float = _MULTIPLIER_SWITCH_BOUNDARY
    stable_span_multiplier: float = _MULTIPLIER_STABLE_SPAN
    unknown_multiplier:    float = _MULTIPLIER_UNKNOWN

    def _setup(self) -> None:
        # ── Validate probabilities ────────────────────────────────────────────
        for name, val, lo, hi in [
            ("hindi_dropout_prob",   self.hindi_dropout_prob,   0.0, 1.0),
            ("english_dropout_prob", self.english_dropout_prob, 0.0, 1.0),
            ("sp_alpha",             self.sp_alpha,             0.0, 1.0),
        ]:
            if not lo <= val <= hi:
                raise ValueError(f"{name} must be in [{lo}, {hi}]; got {val}")
        if self.min_token_length < 1:
            raise ValueError(
                f"min_token_length must be >= 1; got {self.min_token_length}"
            )

        self._rng = random.Random(self.random_seed)
        self._stats = AugmentationStats()
        self._tokenizer = self._load_tokenizer()

    # ── Tokenizer loading ─────────────────────────────────────────────────────

    def _load_tokenizer(self):
        try:
            tok = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer '{self.tokenizer_name}': {exc}\n"
                f"Ensure it is installed:\n"
                f"  python -c \"from transformers import AutoTokenizer; "
                f"AutoTokenizer.from_pretrained('{self.tokenizer_name}')\""
            ) from exc

        if not hasattr(tok, "sp_model"):
            raise RuntimeError(
                f"ContextAwareSentencePieceDropout requires a tokenizer that "
                f"exposes a '.sp_model' attribute (i.e. a SentencePiece-backed "
                f"tokenizer such as 'xlm-roberta-base'), but "
                f"'{self.tokenizer_name}' does not.  "
                f"This module will not fall back silently to alternative "
                f"tokenizers; please supply a compatible model ID."
            )

        return tok

    # ── Public statistics ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Read-only snapshot of augmentation statistics accumulated so far."""
        return self._stats.as_dict()

    def reset_stats(self) -> None:
        """Reset all internal counters to zero."""
        self._stats.reset()

    # ── Sequence-level processing (override from HinglishBase) ────────────────

    def augment(self, text: str) -> str:
        """
        Augment a single Hinglish string.

        Process steps:
          1. Tokenise via ``self._tokenize(text)`` (from HinglishBase) so that
             hashtags, mentions, URLs, emails, and contractions are handled
             correctly before any language detection or perturbation.
          2. Detect language per token.
          3. Compute context multiplier per token from the language sequence.
          4. Perturb eligible tokens stochastically.
          5. Reconstruct via ``self._reconstruct(tokens)`` (from HinglishBase)
             to ensure spacing and special-token handling are consistent with
             the rest of the toolkit.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return text

        lang_seq = [self._detect_language(t) for t in tokens]
        multipliers = self._compute_context_multipliers(lang_seq)

        output_tokens: list[str] = []
        for idx, (token, lang, mult) in enumerate(zip(tokens, lang_seq, multipliers)):
            self._stats.total_tokens_seen += 1

            if not self._is_eligible(token):
                output_tokens.append(token)
                continue

            self._stats.eligible_tokens += 1
            if mult >= self.boundary_multiplier:
                self._stats.switch_boundary_tokens += 1

            if lang == "UNK" or not self._should_perturb(lang, mult):
                output_tokens.append(token)
                continue

            perturbed = self._apply_sp_dropout(token)
            if perturbed != token:
                self._stats.perturbed_tokens += 1
            output_tokens.append(perturbed)

        return self._reconstruct(output_tokens)

    # ── Context multiplier computation ────────────────────────────────────────

    def _compute_context_multipliers(self, lang_seq: list[str]) -> list[float]:
        """
        Assign a context multiplier to each position based on its neighbourhood.

        A position is a *switch boundary* when its language differs from the
        immediately preceding or following token's language (ignoring UNK).
        Otherwise it is a *stable span* position when surrounded by the same
        language on both sides; positions at the edges of the sequence or
        adjacent to UNK use the unknown multiplier.
        """
        n = len(lang_seq)
        multipliers: list[float] = []

        for i, lang in enumerate(lang_seq):
            if lang == "UNK":
                multipliers.append(self.unknown_multiplier)
                continue

            prev_lang = self._prev_known(lang_seq, i)
            next_lang = self._next_known(lang_seq, i)

            at_boundary = (
                (prev_lang is not None and prev_lang != lang) or
                (next_lang is not None and next_lang != lang)
            )

            if at_boundary:
                multipliers.append(self.boundary_multiplier)
            elif prev_lang == lang and next_lang == lang:
                multipliers.append(self.stable_span_multiplier)
            else:
                # Edge token with only one known neighbour matching — neutral
                multipliers.append(self.unknown_multiplier)

        return multipliers

    @staticmethod
    def _prev_known(lang_seq: list[str], idx: int) -> Optional[str]:
        for i in range(idx - 1, -1, -1):
            if lang_seq[i] != "UNK":
                return lang_seq[i]
        return None

    @staticmethod
    def _next_known(lang_seq: list[str], idx: int) -> Optional[str]:
        for i in range(idx + 1, len(lang_seq)):
            if lang_seq[i] != "UNK":
                return lang_seq[i]
        return None

    # ── Token-level helpers ───────────────────────────────────────────────────

    def _is_eligible(self, token: str) -> bool:
        if not token or token.isdigit():
            return False
        if len(token) < self.min_token_length:
            return False
        if self._contains_devanagari(token) and not self.perturb_devanagari:
            return False
        return True

    def _should_perturb(self, lang: str, multiplier: float) -> bool:
        if lang in ("HI_DEV", "HI_ROM", "HI"):
            base = self.hindi_dropout_prob
        elif lang == "EN":
            base = self.english_dropout_prob
        else:
            # UNK and any future labels: do not perturb.
            return False
        effective = min(base * multiplier, 1.0)
        return self._rng.random() < effective

    def _apply_sp_dropout(self, token: str) -> str:
        pieces = self._sample_pieces(token)
        if len(pieces) <= 1:
            return token
        return self._reconstruct_subword_pieces(pieces)

    def _sample_pieces(self, token: str) -> list[str]:
        return self._tokenizer.sp_model.SampleEncodeAsPieces(
            token, self.sp_nbest_size, self.sp_alpha
        )

    def _reconstruct_subword_pieces(self, pieces: list[str]) -> str:
        """
        Reconstruct a token from SentencePiece pieces.

        SentencePiece uses ``▁`` as a word-initial marker.  We strip the leading
        ``▁`` from the first piece (it carries no intra-token information) and
        join remaining pieces with ``self.subword_marker`` so that the boundary
        is clearly visible and pieces are not accidentally merged.

        Example
        -------
        pieces = ['▁kha', 'na']  →  'khana'  (default marker = '▁')
        pieces = ['▁un', 'der', 'stand']  →  'un▁der▁stand'
        """
        cleaned: list[str] = []
        for i, piece in enumerate(pieces):
            if piece == "▁":
                # Bare separator piece — skip.
                continue
            p = piece.lstrip("▁") if i == 0 else piece
            if p:
                cleaned.append(p)

        if not cleaned:
            return "".join(p.lstrip("▁") for p in pieces if p != "▁")

        return self.subword_marker.join(cleaned)


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[ContextAwareSentencePieceDropout] = None,
) -> pd.DataFrame:
    """Apply ContextAwareSentencePieceDropout to a DataFrame column."""
    if processor is None:
        processor = ContextAwareSentencePieceDropout()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[ContextAwareSentencePieceDropout] = None,
) -> pd.DataFrame:
    """Read *input_csv*, augment *text_col*, write *output_csv*, return DataFrame."""
    if processor is None:
        processor = ContextAwareSentencePieceDropout()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)


# ── Smoke tests ───────────────────────────────────────────────────────────────

def _run_smoke_tests() -> None:  # pragma: no cover
    """
    Lightweight smoke tests that exercise the public surface of
    ContextAwareSentencePieceDropout without requiring a live tokenizer.

    Each test validates a distinct behavioural guarantee.  They are written to
    be readable rather than exhaustive; full integration tests should be run
    against a real ``xlm-roberta-base`` checkpoint.
    """
    import unittest
    from unittest.mock import MagicMock, patch

    class _MockSPModel:
        """Minimal mock of sentencepiece.SentencePieceProcessor."""
        def SampleEncodeAsPieces(self, text, nbest_size, alpha):
            # Deterministically split on vowels for testability.
            import re
            parts = re.split(r"(?<=[aeiou])", text)
            return ["▁" + parts[0]] + parts[1:] if len(parts) > 1 else ["▁" + text]

    def _make_processor(**kwargs) -> ContextAwareSentencePieceDropout:
        mock_tok = MagicMock()
        mock_tok.sp_model = _MockSPModel()
        with patch.object(
            ContextAwareSentencePieceDropout, "_load_tokenizer", return_value=mock_tok
        ):
            p = ContextAwareSentencePieceDropout(**kwargs)
        return p

    class SmokeTests(unittest.TestCase):

        # 1. Tokenizer validation failures ─────────────────────────────────────
        def test_missing_sp_model_raises(self):
            mock_tok = MagicMock(spec=[])  # no sp_model attribute
            with patch.object(
                ContextAwareSentencePieceDropout,
                "_load_tokenizer",
                return_value=mock_tok,
            ):
                with self.assertRaises(RuntimeError) as cm:
                    ContextAwareSentencePieceDropout()
            self.assertIn("sp_model", str(cm.exception))

        def test_bad_tokenizer_name_raises(self):
            with self.assertRaises(RuntimeError):
                ContextAwareSentencePieceDropout(
                    tokenizer_name="definitely-not-a-real-model-xyz-9999"
                )

        # 2. Reproducibility ───────────────────────────────────────────────────
        def test_reproducibility(self):
            text = "yaar this movie bahut good"
            p1 = _make_processor(random_seed=0, hindi_dropout_prob=1.0,
                                  english_dropout_prob=1.0, min_token_length=1)
            p2 = _make_processor(random_seed=0, hindi_dropout_prob=1.0,
                                  english_dropout_prob=1.0, min_token_length=1)
            self.assertEqual(p1.augment(text), p2.augment(text))

        def test_different_seeds_may_differ(self):
            text = "yaar this movie bahut good"
            p1 = _make_processor(random_seed=1, hindi_dropout_prob=1.0,
                                  english_dropout_prob=1.0, min_token_length=1)
            p2 = _make_processor(random_seed=99, hindi_dropout_prob=1.0,
                                  english_dropout_prob=1.0, min_token_length=1)
            # Not guaranteed to differ on every input, but extremely likely.
            # We only assert no crash.
            _ = p1.augment(text)
            _ = p2.augment(text)

        # 3. Monolingual English span ──────────────────────────────────────────
        def test_monolingual_english_no_crash(self):
            p = _make_processor(english_dropout_prob=1.0, min_token_length=1)
            result = p.augment("this movie was really good")
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

        # 4. Monolingual Hindi span ────────────────────────────────────────────
        def test_monolingual_hindi_no_crash(self):
            p = _make_processor(hindi_dropout_prob=1.0, min_token_length=1)
            result = p.augment("yaar bahut achha")
            self.assertIsInstance(result, str)

        # 5. Mixed Hinglish sequence ───────────────────────────────────────────
        def test_hinglish_no_crash(self):
            p = _make_processor(min_token_length=1)
            result = p.augment("yaar this movie bahut good")
            self.assertIsInstance(result, str)

        # 6. Word content preserved (no deletions / insertions of words) ──────
        def test_word_content_preserved(self):
            # Every original word should still appear in the output (possibly
            # internally split by subword_marker, but not deleted or replaced).
            text = "yaar this movie bahut good"
            p = _make_processor(hindi_dropout_prob=1.0,
                                  english_dropout_prob=1.0, min_token_length=1)
            result = p.augment(text)
            # strip subword markers to recover surface words for comparison
            marker = p.subword_marker
            result_words = [w.replace(marker, "") for w in result.split()]
            orig_words = [w.replace(marker, "") for w in text.split()]
            self.assertEqual(sorted(result_words), sorted(orig_words))

        # 7. Devanagari preservation ───────────────────────────────────────────
        def test_devanagari_preserved_when_flag_false(self):
            token = "खाना"
            p = _make_processor(perturb_devanagari=False, hindi_dropout_prob=1.0,
                                  min_token_length=1)
            result = p.augment(token)
            self.assertEqual(result, token)

        def test_devanagari_perturbed_when_flag_true(self):
            token = "खाना"
            p = _make_processor(perturb_devanagari=True, hindi_dropout_prob=1.0,
                                  min_token_length=1, random_seed=0)
            # We only assert no crash; perturbation depends on sp output.
            _ = p.augment(token)

        # 8. Switch boundary detection ─────────────────────────────────────────
        def test_switch_boundary_tokens_counted(self):
            p = _make_processor(min_token_length=1,
                                  hindi_dropout_prob=1.0,
                                  english_dropout_prob=1.0)
            p.augment("yaar this movie bahut good")
            # "this" follows "yaar" (EN follows HI) → boundary
            # "bahut" follows "movie" (HI follows EN) → boundary
            self.assertGreater(p.stats["switch_boundary_tokens"], 0)

        # 9. Augmentation statistics ───────────────────────────────────────────
        def test_stats_populated(self):
            p = _make_processor(min_token_length=1)
            p.augment("yaar this movie bahut good")
            s = p.stats
            self.assertIn("total_tokens_seen", s)
            self.assertIn("eligible_tokens", s)
            self.assertIn("perturbed_tokens", s)
            self.assertIn("switch_boundary_tokens", s)
            self.assertEqual(s["total_tokens_seen"], 5)

        def test_stats_reset(self):
            p = _make_processor(min_token_length=1)
            p.augment("yaar this movie bahut good")
            p.reset_stats()
            s = p.stats
            self.assertEqual(s["total_tokens_seen"], 0)
            self.assertEqual(s["perturbed_tokens"], 0)

        # 10. Short tokens skipped ─────────────────────────────────────────────
        def test_short_tokens_skipped(self):
            p = _make_processor(min_token_length=10,
                                  hindi_dropout_prob=1.0,
                                  english_dropout_prob=1.0)
            text = "hi ok go"
            result = p.augment(text)
            self.assertEqual(result, text)

        # 11. Context multipliers respected ────────────────────────────────────
        def test_context_multipliers_range(self):
            p = _make_processor()
            lang_seq = ["EN", "HI", "EN", "EN", "HI"]
            mults = p._compute_context_multipliers(lang_seq)
            for m in mults:
                self.assertGreaterEqual(m, 0.0)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(SmokeTests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _run_smoke_tests()