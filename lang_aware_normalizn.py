"""
lang_aware_normalizn.py — Language-aware normalization for Hinglish text.

Hard requirements:
    (NLTK words corpus via _base.py)

What it does:
    - Detects language per token (EN / HI / UNK).
    - Applies English normalization: compresses character repetitions and
      optionally expands common slang abbreviations.
    - Applies Hindi normalization: compresses repetitions, reduces excess
      vowel elongation, maps spelling variants to canonical forms, and
      applies regex-based pattern fixes.
    - Unknown tokens are normalized only if ``normalize_unknown`` is enabled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase


# ── Canonical spelling-variant maps ───────────────────────────────────────────

_HINGLISH_VARIANTS: dict[str, str] = {
    "acha": "accha", "achha": "accha", "accha": "accha",
    "bohot": "bahut", "bohut": "bahut", "bahot": "bahut",
    "bahuut": "bahut", "bahut": "bahut",
    "yar": "yaar", "yaar": "yaar", "yaaar": "yaar",
    "yaarr": "yaar", "yaarrr": "yaar",
    "thik": "theek", "theek": "theek",
    "kyu": "kyun", "kyun": "kyun",
    "han": "haan", "haan": "haan",
    "nahi": "nahi", "nahii": "nahi",
    "samjh": "samajh",
}

_SLANG_EXPANSIONS: dict[str, str] = {
    "u":   "you",
    "ur":  "your",
    "pls": "please",
    "thx": "thanks",
    "idk": "i do not know",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "brb": "be right back",
    "btw": "by the way",
}


@dataclass
class LanguageAwareNormalization(HinglishBase):
    """
    Applies language-specific normalization rules without aggressive rewriting.

    Attributes:
        normalize_english:           Apply compression and optional slang
                                     expansion to English tokens.
        normalize_hindi:             Apply Hinglish-specific normalization.
        normalize_unknown:           Apply generic compression to UNK tokens.
        expand_slang:                Expand English slang contractions.
        preserve_sentiment_intensity: Allow up to 4 repeated chars (vs 2) to
                                     preserve expressive emphasis like "soooo".
        hinglish_variants:           Override or extend the built-in spelling-
                                     variant canonicalization map.
        slang_expansions:            Override or extend the slang expansion map.
    """

    normalize_english:             bool              = True
    normalize_hindi:               bool              = True
    normalize_unknown:             bool              = False
    expand_slang:                  bool              = False
    preserve_sentiment_intensity:  bool              = False
    hinglish_variants:             dict[str, str]    = field(
        default_factory=lambda: dict(_HINGLISH_VARIANTS)
    )
    slang_expansions:              dict[str, str]    = field(
        default_factory=lambda: dict(_SLANG_EXPANSIONS)
    )

    def _process_token(self, token: str) -> str:
        if not token or token.isdigit():
            return token

        lang = self._detect_language(token)
        if lang == "EN" and self.normalize_english:
            return self._normalize_english(token)
        if lang == "HI" and self.normalize_hindi:
            return self._normalize_hindi(token)
        if lang == "UNK" and self.normalize_unknown:
            return self._normalize_unknown(token)
        return token

    # ── Per-language normalizers ──────────────────────────────────────────────

    def _normalize_english(self, token: str) -> str:
        max_r = 4 if self.preserve_sentiment_intensity else 2
        normalized = self._compress_repetitions(token, max_r)
        if self.expand_slang:
            expanded = self.slang_expansions.get(normalized)
            if expanded is not None:
                return expanded
        return normalized

    def _normalize_hindi(self, token: str) -> str:
        max_r = 4 if self.preserve_sentiment_intensity else 2
        normalized = self._compress_repetitions(token, max_r)
        normalized = self._reduce_vowel_repetitions(normalized)
        normalized = self.hinglish_variants.get(normalized, normalized)
        normalized = self._apply_hinglish_patterns(normalized)
        return normalized

    def _normalize_unknown(self, token: str) -> str:
        max_r = 4 if self.preserve_sentiment_intensity else 2
        return self._compress_repetitions(token, max_r)

    # ── Normalization helpers ─────────────────────────────────────────────────

    def _compress_repetitions(self, token: str, max_repeats: int = 2) -> str:
        return re.sub(
            r"(.)\1{%d,}" % max_repeats,
            lambda m: m.group(1) * max_repeats,
            token,
        )

    def _reduce_vowel_repetitions(self, token: str) -> str:
        return re.sub(r"([aeiou])\1+", r"\1", token)

    def _apply_hinglish_patterns(self, token: str) -> str:
        token = re.sub(r"ach+a+",  "accha",  token)
        token = re.sub(r"nah+i+",  "nahi",   token)
        token = re.sub(r"yaa+r+",  "yaar",   token)
        token = re.sub(r"bohu+t",  "bahut",  token)
        token = re.sub(r"samj+h",  "samajh", token)
        return token


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[LanguageAwareNormalization] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = LanguageAwareNormalization()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[LanguageAwareNormalization] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = LanguageAwareNormalization()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)
