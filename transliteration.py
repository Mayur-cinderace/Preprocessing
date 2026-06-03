"""
transliteration.py — Romanized Hindi → Devanagari transliteration for
Hinglish text.

Hard requirements:
    pip install indic-transliteration
    (+ NLTK words corpus via _base.py)

What it does:
    - Keeps English tokens and already-Devanagari tokens unchanged.
    - Transliterates recognized Hindi/Hinglish tokens from the configured
      romanization scheme (default: ITRANS) to Devanagari.
    - Optionally transliterates UNK tokens via ``transliterate_unknown``.

ITRANS is the default source scheme because casual romanized Hindi (social
media, chat) overwhelmingly follows ITRANS conventions.  Switch to
``source_scheme="hk"`` for academic/corpus text.

No silent fallbacks: missing ``indic-transliteration`` raises ``ImportError``
at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate

from _base import HINGLISH_LEXICON, HinglishBase


_VALID_SCHEMES: dict[str, str] = {
    "itrans": sanscript.ITRANS,
    "hk":     sanscript.HK,
    "iast":   sanscript.IAST,
    "slp1":   sanscript.SLP1,
}


@dataclass
class Transliteration(HinglishBase):
    """
    Transliterates romanized Hindi tokens to Devanagari.

    Attributes:
        source_scheme:        Romanization scheme of the input Hindi tokens.
                              One of ``"itrans"`` (default), ``"hk"``,
                              ``"iast"``, ``"slp1"``.
        transliterate_unknown: Also transliterate tokens whose language cannot
                              be determined (use with caution — may corrupt
                              English tokens not in the vocabulary).
    """

    source_scheme:         str  = "itrans"
    transliterate_unknown: bool = False

    def _setup(self) -> None:
        if self.source_scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"source_scheme must be one of {list(_VALID_SCHEMES)}; "
                f"got '{self.source_scheme}'"
            )
        self._sp_scheme: str = _VALID_SCHEMES[self.source_scheme]

    def _process_token(self, token: str) -> str:
        if not token or token.isdigit():
            return token

        # Already Devanagari — pass through.
        if self._contains_devanagari(token):
            return token

        lang = self._detect_language(token)
        if lang == "EN":
            return token
        if lang == "HI":
            return self._do_transliterate(token)
        if lang == "UNK" and self.transliterate_unknown:
            return self._do_transliterate(token)
        return token

    def _do_transliterate(self, token: str) -> str:
        return _transliterate(token, self._sp_scheme, sanscript.DEVANAGARI)


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[Transliteration] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = Transliteration()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[Transliteration] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = Transliteration()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)
