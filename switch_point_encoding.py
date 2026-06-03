"""
switch_point_encoding.py — Language switch-point encoding for Hinglish text.

Hard requirements:
    (NLTK words corpus via _base.py)

What it does:
    - Detects language per token (EN / HI / UNK).
    - Inserts a boundary marker between adjacent tokens whenever the language
      changes (EN→HI, HI→EN).
    - Supports directional markers (``[SWITCH_EN_HI]`` / ``[SWITCH_HI_EN]``)
      or a single generic marker (``[SWITCH]``).
    - Unknown-language tokens can optionally participate in switch detection
      via ``mark_unknown``.

The module overrides ``process()`` directly because it requires state across
consecutive tokens — a per-token hook is insufficient.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase, _PUNCT_RE, _TOKEN_RE


@dataclass
class SwitchPointEncoding(HinglishBase):
    """
    Inserts language-boundary markers between adjacent tokens that switch
    language.

    Attributes:
        directional_markers: Use ``en_hi_marker`` / ``hi_en_marker`` instead
                             of the generic ``switch_marker``.
        mark_unknown:        Treat UNK tokens as language participants for
                             switch detection.
        en_hi_marker:        Marker inserted before a Hindi token following an
                             English token.
        hi_en_marker:        Marker inserted before an English token following
                             a Hindi token.
        switch_marker:       Marker used when ``directional_markers=False``.
    """

    directional_markers: bool = True
    mark_unknown:        bool = False
    en_hi_marker:        str  = "[SWITCH_EN_HI]"
    hi_en_marker:        str  = "[SWITCH_HI_EN]"
    switch_marker:       str  = "[SWITCH]"

    # ── Override process() — needs cross-token state ──────────────────────────

    def process(self, text: str) -> str:
        if not self.enabled or not isinstance(text, str):
            return text

        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()

        tokens = _TOKEN_RE.findall(text)
        result: list[str] = []
        prev_lang: Optional[str] = None

        for tok in tokens:
            if _PUNCT_RE.fullmatch(tok):
                if self.preserve_punctuation:
                    result.append(tok)
                continue

            lang = self._classify(tok)
            if lang is None:
                result.append(tok)
                continue

            marker = self._switch_marker(prev_lang, lang)
            if marker is not None:
                result.append(marker)
            result.append(tok)
            prev_lang = lang

        return self._reconstruct(result)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _classify(self, token: str) -> Optional[str]:
        """Return language label, or None for digits (no language)."""
        if token.isdigit():
            return None
        lang = self._detect_language(token)
        if lang == "UNK" and not self.mark_unknown:
            return None
        return lang

    def _switch_marker(
        self, prev: Optional[str], curr: str
    ) -> Optional[str]:
        if prev is None or curr is None or prev == curr:
            return None

        if self.directional_markers and prev in {"EN", "HI"} and curr in {"EN", "HI"}:
            if prev == "EN" and curr == "HI":
                return self.en_hi_marker
            if prev == "HI" and curr == "EN":
                return self.hi_en_marker
            return None  # UNK involved — no directional marker

        return self.switch_marker

    # ── _process_token: not meaningful for this module ────────────────────────

    def _process_token(self, token: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "SwitchPointEncoding overrides process() directly; "
            "_process_token is not called."
        )


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[SwitchPointEncoding] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = SwitchPointEncoding()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[SwitchPointEncoding] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = SwitchPointEncoding()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)
