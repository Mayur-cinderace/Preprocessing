"""
script_unification.py — Script unification for mixed-script Hinglish text.

Hard requirements:
    pip install indic-transliteration
    (+ NLTK words corpus via _base.py)

What it does:
    - Detects the script of each token.
    - When ``target_script="devanagari"``: converts romanized Hindi tokens to
      Devanagari via ITRANS.  Tokens not in the Hinglish lexicon are left
      unchanged unless ``require_lexicon_match=False``.
    - When ``target_script="roman"``: converts Devanagari tokens to romanized
      form via the configured scheme (itrans / hk / iast), then simplifies
      to lowercase.  Resulting tokens not in the Hinglish lexicon are reverted
      to the original Devanagari unless ``require_lexicon_match=False``.
    - English tokens are never modified.

No silent fallbacks: missing ``indic-transliteration`` raises ``ImportError``
at import time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate

from _base import HINGLISH_LEXICON, HinglishBase


_SCHEME_MAP: dict[str, str] = {
    "itrans": sanscript.ITRANS,
    "hk":     sanscript.HK,
    "iast":   sanscript.IAST,
}

_VALID_TARGET_SCRIPTS = {"devanagari", "roman"}
_VALID_ROMAN_SCHEMES   = set(_SCHEME_MAP.keys())


@dataclass
class ScriptUnification(HinglishBase):
    """
    Converts Hindi tokens to a single target script.

    Attributes:
        target_script:       ``"devanagari"`` or ``"roman"``.
        roman_scheme:        Romanisation scheme when converting FROM Devanagari:
                             ``"itrans"`` (default), ``"hk"``, or ``"iast"``.
        require_lexicon_match: Only convert tokens found in the Hinglish lexicon;
                               unknown tokens are returned unchanged.
        roman_post_map:      Optional post-processing map applied after roman
                             conversion (e.g. scholarly IDs → common forms).
    """

    target_script:         str            = "devanagari"
    roman_scheme:          str            = "itrans"
    require_lexicon_match: bool           = True
    roman_post_map:        dict[str, str] = field(default_factory=dict)

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

    def _process_token(self, token: str) -> str:
        if not token or token.isdigit():
            return token
        if self._is_english_word(token):
            return token

        if self.target_script == "devanagari":
            return self._to_devanagari(token)
        return self._to_roman(token)

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _to_devanagari(self, token: str) -> str:
        if self._contains_devanagari(token):
            return token  # already Devanagari
        if self.require_lexicon_match and token not in self.hinglish_lexicon:
            return token
        if not self.require_lexicon_match and not self._is_hindi_token(token):
            return token
        return _transliterate(token, sanscript.ITRANS, sanscript.DEVANAGARI)

    def _to_roman(self, token: str) -> str:
        if not self._contains_devanagari(token):
            return token  # already roman
        roman = _transliterate(token, sanscript.DEVANAGARI, self._sp_scheme)
        roman = self._simplify_roman(roman)
        roman = self.roman_post_map.get(roman, roman)
        if self.require_lexicon_match and roman not in self.hinglish_lexicon:
            return token  # revert — unknown result
        return roman

    @staticmethod
    def _simplify_roman(text: str) -> str:
        """Lowercase and strip diacritics common in academic schemes."""
        return (
            text.replace("A", "a").replace("I", "i").replace("U", "u")
                .replace("M", "m").replace("H", "h")
                .lower()
        )


# ── Module-level convenience wrappers ─────────────────────────────────────────

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
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[ScriptUnification] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = ScriptUnification()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)
