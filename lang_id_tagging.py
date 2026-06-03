"""
lang_id_tagging.py — Token-level language identification tagging for Hinglish.

Hard requirements:
    (NLTK words corpus via _base.py)

What it does:
    - Annotates every non-punctuation token with XML-style language tags:
      ``<EN>word</EN>``, ``<HI>word</HI>``, or ``<UNK>word</UNK>``.
    - Unknown-language tokens are tagged only if ``tag_unknown=True``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase


@dataclass
class LanguageIdentificationTagging(HinglishBase):
    """
    Wraps each token with a language tag for downstream sequence-labelling
    or diagnostics.

    Attributes:
        tag_unknown: Tag ambiguous tokens with ``<UNK>…</UNK>``; when
                     ``False``, unknown tokens are returned as-is.
    """

    tag_unknown: bool = True

    def _process_token(self, token: str) -> str:
        if not token or token.isdigit():
            return token

        lang = self._detect_language(token)
        if lang == "UNK" and not self.tag_unknown:
            return token
        return f"<{lang}>{token}</{lang}>"


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[LanguageIdentificationTagging] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = LanguageIdentificationTagging()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[LanguageIdentificationTagging] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = LanguageIdentificationTagging()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)
