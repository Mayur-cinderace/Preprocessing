"""
phonetic_normalization.py — Phonetic normalization for romanized Hinglish text.

Hard requirements:
    (NLTK words corpus via _base.py)

What it does:
    - Keeps Devanagari and English tokens unchanged (unless
      ``preserve_english_intensity`` is disabled, in which case English
      character repetitions are also compressed).
    - For Hindi/Hinglish tokens: compresses repetitions, maps spelling
      variants to canonical forms, and applies phonetic-reduction rules
      (e.g. "ee" → "i", "oo" → "u" where not protected by the lexicon).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase


# ── Canonical phonetic mapping ────────────────────────────────────────────────

_CANONICAL_MAP: dict[str, str] = {
    "acha": "accha", "achha": "accha", "achaaa": "accha",
    "acchaa": "accha", "acchaaa": "accha",
    "bahot": "bahut", "bohot": "bahut", "bohut": "bahut",
    "bahuut": "bahut", "bahut": "bahut",
    "yar": "yaar", "yaaar": "yaar", "yaarr": "yaar", "yaarrr": "yaar",
    "samjh": "samajh",
    "nhi": "nahi", "nahii": "nahi", "nahiii": "nahi",
    "kyu": "kyun",
    "han": "haan", "haa": "haan",
    "thik": "theek",
    "aayaa": "aaya", "aayaaa": "aaya", "aya": "aaya",
}


@dataclass
class PhoneticNormalization(HinglishBase):
    """
    Normalizes noisy Hinglish phonetic spellings to canonical forms.

    Attributes:
        normalize_repetitions:     Compress excessive character repetitions.
        normalize_phonetics:       Apply phonetic reduction rules (ee→i, oo→u).
        preserve_english_intensity: Skip repetition compression on English
                                    tokens (preserves "soooo good" for English).
        canonical_map:             Override or extend the built-in spelling map.
    """

    normalize_repetitions:    bool             = True
    normalize_phonetics:      bool             = True
    preserve_english_intensity: bool           = True
    canonical_map:            dict[str, str]   = field(
        default_factory=lambda: dict(_CANONICAL_MAP)
    )

    def _process_token(self, token: str) -> str:
        if not token or token.isdigit():
            return token
        if self._contains_devanagari(token):
            return token

        lang = self._detect_language(token)

        if lang == "EN":
            if self.preserve_english_intensity:
                return token
            return self._reduce_repetition(token) if self.normalize_repetitions else token

        if lang != "HI":
            return token

        normalized = token
        if self.normalize_repetitions:
            normalized = self._reduce_repetition(normalized)
        normalized = self.canonical_map.get(normalized, normalized)
        if self.normalize_phonetics:
            normalized = self._apply_phonetic_rules(normalized)
        return normalized

    # ── Normalization helpers ─────────────────────────────────────────────────

    def _reduce_repetition(self, token: str) -> str:
        # Vowels: collapse to single; consonants: collapse to at most 2.
        token = re.sub(r"([aeiou])\1{2,}", r"\1", token)
        token = re.sub(r"([^aeiou])\1{2,}", r"\1\1", token)
        return token

    def _apply_phonetic_rules(self, token: str) -> str:
        if len(token) <= 3:
            return token
        # Only reduce ee/oo if the token is not in the protected lexicon,
        # to avoid corrupting words like "theek".
        if token not in self.hinglish_lexicon:
            token = re.sub(r"ee+", "i", token)
            token = re.sub(r"oo+", "u", token)
        return token


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[PhoneticNormalization] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = PhoneticNormalization()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[PhoneticNormalization] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = PhoneticNormalization()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)
