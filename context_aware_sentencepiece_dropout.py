"""
context_aware_sentencepiece_dropout.py — Stochastic SentencePiece subword
sampling for code-switched Hinglish text.

Hard requirements:
    pip install transformers sentencepiece
    (+ NLTK words corpus via _base.py)

The module requires a SentencePiece-backed tokenizer (e.g. ``xlm-roberta-base``)
that exposes ``.sp_model``.  Both the ``transformers`` import and the tokenizer
load are validated at construction time — no deferred warnings.

What it does:
    - Detects language per token (EN / HI / UNK).
    - Applies stochastic SentencePiece sampling (``SampleEncodeAsPieces``) to
      eligible tokens with language-specific dropout probabilities.
    - Short tokens and Devanagari tokens (when ``perturb_devanagari=False``) are
      left unchanged.
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


@dataclass
class ContextAwareSentencePieceDropout(HinglishBase):
    """
    Applies stochastic SentencePiece subword sampling to Hinglish tokens.

    Attributes:
        tokenizer_name:        HuggingFace model ID for a SentencePiece tokenizer.
        hindi_dropout_prob:    Sampling probability for Hindi/Hinglish tokens.
        english_dropout_prob:  Sampling probability for English tokens.
        perturb_devanagari:    Apply sampling to tokens in Devanagari script.
        min_token_length:      Skip tokens shorter than this (after subword split
                               they would be trivially the same).
        sp_nbest_size:         ``nbest_size`` argument to SampleEncodeAsPieces.
        sp_alpha:              ``alpha`` smoothing argument.
        subword_marker:        Separator inserted between subword pieces.
        random_seed:           Seed for reproducibility.
    """

    tokenizer_name:       str   = "xlm-roberta-base"
    hindi_dropout_prob:   float = 0.3
    english_dropout_prob: float = 0.05
    perturb_devanagari:   bool  = False
    min_token_length:     int   = 5
    sp_nbest_size:        int   = 64
    sp_alpha:             float = 0.1
    subword_marker:       str   = " "
    random_seed:          int   = 42

    def _setup(self) -> None:
        for name, val, lo, hi in [
            ("hindi_dropout_prob",   self.hindi_dropout_prob,   0.0, 1.0),
            ("english_dropout_prob", self.english_dropout_prob, 0.0, 1.0),
            ("sp_alpha",             self.sp_alpha,             0.0, 1.0),
        ]:
            if not lo <= val <= hi:
                raise ValueError(f"{name} must be in [{lo}, {hi}]; got {val}")
        if self.min_token_length < 1:
            raise ValueError(f"min_token_length must be >= 1; got {self.min_token_length}")

        self._rng = random.Random(self.random_seed)
        self._tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            tok = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer '{self.tokenizer_name}': {exc}\n"
                f"Ensure it is installed:  "
                f"python -c \"from transformers import AutoTokenizer; "
                f"AutoTokenizer.from_pretrained('{self.tokenizer_name}')\""
            ) from exc

        if not hasattr(tok, "sp_model"):
            self._sentencepiece_available = False
            warnings.warn(
                f"Tokenizer '{self.tokenizer_name}' has no '.sp_model' attribute; "
                "context-aware SentencePiece dropout will act as a no-op.",
                UserWarning,
                stacklevel=2,
            )
            return tok

        self._sentencepiece_available = True
        return tok

    # ── Token processing ──────────────────────────────────────────────────────

    def _process_token(self, token: str) -> str:
        if not token or token.isdigit():
            return token
        if len(token) < self.min_token_length:
            return token
        if self._contains_devanagari(token) and not self.perturb_devanagari:
            return token

        lang = self._detect_language(token)
        if lang == "UNK":
            return token
        if not self._should_perturb(lang):
            return token
        return self._apply_sp_dropout(token)

    def _should_perturb(self, lang: str) -> bool:
        prob = self.hindi_dropout_prob if lang == "HI" else self.english_dropout_prob
        return self._rng.random() < prob

    def _apply_sp_dropout(self, token: str) -> str:
        if not getattr(self, "_sentencepiece_available", False):
            return token
        pieces = self._sample_pieces(token)
        if len(pieces) <= 1:
            return token
        cleaned = [p.replace("▁", "") for p in pieces if p != "▁"]
        return self.subword_marker.join(cleaned)

    def _sample_pieces(self, token: str) -> list[str]:
        if not getattr(self, "_sentencepiece_available", False):
            return [token]
        return self._tokenizer.sp_model.SampleEncodeAsPieces(
            token, self.sp_nbest_size, self.sp_alpha
        )


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[ContextAwareSentencePieceDropout] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = ContextAwareSentencePieceDropout()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[ContextAwareSentencePieceDropout] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = ContextAwareSentencePieceDropout()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)
