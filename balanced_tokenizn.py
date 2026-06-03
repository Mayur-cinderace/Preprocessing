"""
balanced_tokenizn.py — Balanced tokenization for code-switched Hinglish text.

Hard requirements:
    pip install indic-transliteration
    (+ NLTK words corpus via _base.py)

What it does:
    - Keeps English words unchanged.
    - Maps common romanized Hindi function words to Devanagari via a built-in
      lookup table (extensible via ``map_path``).
    - Optionally transliterates any remaining Hindi-like tokens to Devanagari
      using ITRANS → Devanagari conversion.
    - Normalizes elongated character repetitions (e.g. "bahutttt" → "bahutt").

No silent fallbacks: missing ``indic-transliteration`` raises ``ImportError``
at import time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate

from _base import HINGLISH_LEXICON, HinglishBase


# ── Built-in romanized → Devanagari map ──────────────────────────────────────

_DEFAULT_HINGLISH_MAP: dict[str, str] = {
    "hai": "है", "haan": "हाँ", "han": "हाँ", "nahi": "नहीं", "nahin": "नहीं",
    "na": "ना", "kya": "क्या", "kyun": "क्यों", "kyu": "क्यों", "ka": "का",
    "ki": "की", "ke": "के", "ko": "को", "se": "से", "mein": "में", "me": "में",
    "men": "में", "main": "मैं", "mai": "मैं", "hum": "हम", "ham": "हम",
    "tum": "तुम", "aap": "आप", "aur": "और", "ya": "या", "toh": "तो",
    "to": "तो", "par": "पर", "pe": "पे", "ab": "अब", "phir": "फिर",
    "fir": "फिर", "tab": "तब", "jab": "जब", "agar": "अगर", "bhi": "भी",
    "bhai": "भाई", "bahut": "बहुत", "bohot": "बहुत", "bahot": "बहुत",
    "yaar": "यार", "yar": "यार", "ji": "जी", "haanji": "हाँजी",
    "theek": "ठीक", "thik": "ठीक", "acha": "अच्छा", "accha": "अच्छा",
    "achha": "अच्छा", "bura": "बुरा", "sahi": "सही", "galat": "गलत",
    "badhiya": "बढ़िया", "mast": "मस्त", "thoda": "थोड़ा", "thodi": "थोड़ी",
    "thora": "थोड़ा", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे",
    "tera": "तेरा", "teri": "तेरी", "tere": "तेरे", "unka": "उनका",
    "unki": "उनकी", "unke": "उनके", "iska": "इसका", "iski": "इसकी",
    "iske": "इसके", "wo": "वो", "woh": "वो", "ye": "ये", "yeh": "यह",
    "is": "इस", "us": "उस", "unhonne": "उन्होंने", "inhonne": "इन्होंने",
    "hamne": "हमने", "maine": "मैंने", "kiye": "किए", "kiya": "किया",
    "hua": "हुआ", "hui": "हुई", "hue": "हुए", "tha": "था", "thi": "थी",
    "the": "थे", "raha": "रहा", "rahi": "रही", "rahe": "रहे",
    "gaya": "गया", "gayi": "गई", "gaye": "गए", "ho": "हो",
    "hun": "हूं", "hu": "हूं", "hoon": "हूं", "hoga": "होगा",
    "hogi": "होगी", "hoge": "होगे", "hota": "होता", "hoti": "होती",
    "hote": "होते", "karna": "करना", "karta": "करता", "karti": "करती",
    "karte": "करते", "karen": "करें", "karke": "करके", "dekh": "देख",
    "dekho": "देखो", "dekhna": "देखना", "bol": "बोल", "bolo": "बोलो",
    "sun": "सुन", "suno": "सुनो", "chal": "चल", "chalo": "चलो",
    "ja": "जा", "jao": "जाओ", "aao": "आओ", "aana": "आना", "jana": "जाना",
    "rakho": "रखो", "rakha": "रखा", "rakhna": "रखना", "karo": "करो",
    "kar": "कर", "samajh": "समझ", "samjh": "समझ", "ghar": "घर",
    "khana": "खाना", "padhai": "पढ़ाई", "pyar": "प्यार", "sach": "सच",
    "jaldi": "जल्दी",
}


@dataclass
class BalancedTokenization(HinglishBase):
    """
    Reduces tokenizer fragmentation asymmetry by converting common Hinglish
    tokens to Devanagari.

    Attributes:
        normalize_repetitions: Collapse 3+ repeated characters to 2
                               (e.g. "bahuuuut" → "bahuut").
        transliterate_unknown: ITRANS-transliterate Hindi-like tokens not
                               found in the built-in map.
        map_path:              Optional path to a 2-column CSV (roman,devanagari)
                               or JSON dict that extends the built-in map.
    """

    normalize_repetitions: bool          = True
    transliterate_unknown: bool          = True
    map_path:              Optional[Path] = None

    def _setup(self) -> None:
        self._hinglish_map: dict[str, str] = dict(_DEFAULT_HINGLISH_MAP)
        if self.map_path is not None:
            self._hinglish_map.update(self._load_map(Path(self.map_path)))

    # ── External map loading ──────────────────────────────────────────────────

    def _load_map(self, path: Path) -> dict[str, str]:
        if not path.exists():
            raise FileNotFoundError(f"Hinglish map file not found: {path}")
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as fh:
                raw: dict = json.load(fh)
            if not isinstance(raw, dict):
                raise ValueError(f"JSON map at '{path}' must be a flat {{roman: devanagari}} object.")
            return {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if str(k).strip()}
        # CSV fallback
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError(
                f"Map CSV '{path}' must have at least two columns: roman, devanagari"
            )
        return {
            str(k).strip().lower(): str(v).strip()
            for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])
            if str(k).strip()
        }

    # ── Core token processing ─────────────────────────────────────────────────

    def _process_token(self, token: str) -> str:
        if not token or token.isdigit():
            return token

        if self.normalize_repetitions:
            token = re.sub(r"(.)\1{2,}", r"\1\1", token)

        # English tokens pass through unchanged.
        if self._is_english_word(token):
            return token

        # Exact map lookup (fast path).
        mapped = self._hinglish_map.get(token)
        if mapped is not None:
            return mapped

        # ITRANS transliteration for remaining Hindi-like tokens.
        if self.transliterate_unknown and self._is_hindi_token(token):
            return _transliterate(token, sanscript.ITRANS, sanscript.DEVANAGARI)

        return token


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[BalancedTokenization] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = BalancedTokenization()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[BalancedTokenization] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = BalancedTokenization()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)


if __name__ == "__main__":
    p = BalancedTokenization(normalize_repetitions=True, transliterate_unknown=True)
    for text in [
        "yaar movie bahuuut acchi thi",
        "kya hua bhai, sab theek hai?",
        "main office mein hun",
    ]:
        print(f"  IN : {text}")
        print(f"  OUT: {p.process(text)}\n")
