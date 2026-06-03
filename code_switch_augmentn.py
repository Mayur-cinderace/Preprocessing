"""
code_switch_augmentn.py — Controlled code-switch augmentation for Hinglish text.

Hard requirements:
    (NLTK words corpus via _base.py)

What it does:
    - Detects English tokens and stochastically replaces them with Hinglish
      equivalents from a configurable lexicon.
    - Generates N synthetic Hinglish variants per original sample.
    - Optionally keeps the original alongside its augmented versions.

No silent fallbacks: if augmentation cannot produce a variant different from
the original after ``max_attempts`` tries, that augment slot is skipped
rather than silently returning the original marked as augmented.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase, _PUNCT_RE, _TOKEN_RE


# ── Default English → Hinglish replacement lexicon ───────────────────────────

_DEFAULT_REPLACEMENT_LEXICON: dict[str, list[str]] = {
    "very":   ["bahut"],
    "good":   ["accha"],
    "bad":    ["bura"],
    "love":   ["pyaar"],
    "happy":  ["khush"],
    "sad":    ["dukhi"],
    "today":  ["aaj"],
    "movie":  ["film"],
    "food":   ["khana"],
    "home":   ["ghar"],
    "work":   ["kaam"],
    "friend": ["dost"],
    "yes":    ["haan"],
    "no":     ["nahi"],
    "why":    ["kyun"],
    "what":   ["kya"],
    "how":    ["kaise"],
    "when":   ["kab"],
    "where":  ["kahan"],
    "come":   ["aao"],
    "go":     ["jao"],
    "see":    ["dekho"],
    "tell":   ["bolo"],
    "listen": ["suno"],
    "do":     ["karo"],
    "make":   ["banao"],
    "right":  ["sahi"],
    "wrong":  ["galat"],
}


@dataclass
class CodeSwitchAugmentation(HinglishBase):
    """
    Generates synthetic Hinglish variants via controlled lexical substitution.

    This module **expands** rows rather than transforming them 1-to-1.
    Use ``augment_dataframe`` for dataset processing; ``process`` applies a
    single stochastic augmentation pass on one string.

    Attributes:
        augmentation_probability: Probability of replacing each eligible
                                  English token with a Hinglish equivalent.
        n_augments:               Number of augmented variants to generate per
                                  original sample.
        max_attempts:             Maximum replacement attempts before skipping
                                  a variant that remains identical to input.
        preserve_original:        Keep original row alongside augmentations.
        random_seed:              Seed for reproducibility.
        replacement_lexicon:      Mapping of English word → list of Hinglish
                                  alternatives (merged with built-in defaults).
    """

    augmentation_probability: float               = 0.3
    n_augments:               int                 = 1
    max_attempts:             int                 = 5
    preserve_original:        bool                = True
    random_seed:              int                 = 42
    replacement_lexicon:      dict[str, list[str]] = field(
        default_factory=lambda: dict(_DEFAULT_REPLACEMENT_LEXICON)
    )

    def _setup(self) -> None:
        if self.augmentation_probability < 0.0 or self.augmentation_probability > 1.0:
            raise ValueError(
                f"augmentation_probability must be in [0, 1]; "
                f"got {self.augmentation_probability}"
            )
        if self.n_augments < 0:
            raise ValueError(f"n_augments must be >= 0; got {self.n_augments}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1; got {self.max_attempts}")
        self._rng = random.Random(self.random_seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, text: str) -> str:
        """Apply one stochastic augmentation pass.  Use for single-string calls."""
        if not self.enabled or not isinstance(text, str):
            return text
        return self._augment_once(text)

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        output_col: str = "processed_text",
        is_augmented_col: str = "is_augmented",
    ) -> pd.DataFrame:
        """
        Expand ``df`` with ``n_augments`` synthetic variants per row.

        Each augmented row has ``is_augmented_col = True``.  Original rows
        (if ``preserve_original``) have ``is_augmented_col = False``.
        """
        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found. Available: {list(df.columns)}"
            )
        rows: list[dict] = []
        for _, row in df.iterrows():
            text = str(row[text_col])
            if not self.enabled:
                new_row = row.to_dict()
                new_row[output_col]       = text
                new_row[is_augmented_col] = False
                rows.append(new_row)
                continue

            if self.preserve_original:
                new_row = row.to_dict()
                new_row[output_col]       = text
                new_row[is_augmented_col] = False
                rows.append(new_row)

            for _ in range(self.n_augments):
                variant: Optional[str] = None
                for _ in range(self.max_attempts):
                    candidate = self._augment_once(text)
                    if candidate != text:
                        variant = candidate
                        break
                if variant is None:
                    # No change achievable — skip this slot rather than fabricate.
                    continue
                new_row = row.to_dict()
                new_row[output_col]       = variant
                new_row[is_augmented_col] = True
                rows.append(new_row)
        return pd.DataFrame(rows, columns=list(df.columns) + [output_col, is_augmented_col])

    def augment_csv(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        text_col: str = "text",
        output_col: str = "processed_text",
    ) -> pd.DataFrame:
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        df = pd.read_csv(input_csv)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        processed = self.augment_dataframe(df, text_col=text_col, output_col=output_col)
        processed.to_csv(output_csv, index=False)
        return processed

    # ── Internal augmentation logic ───────────────────────────────────────────

    def _augment_once(self, text: str) -> str:
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()

        tokens = _TOKEN_RE.findall(text)
        result: list[str] = []
        for tok in tokens:
            if _PUNCT_RE.fullmatch(tok):
                if self.preserve_punctuation:
                    result.append(tok)
                continue
            if tok.isdigit():
                result.append(tok)
                continue
            lang = self._detect_language(tok)
            if lang == "EN":
                replacement = self._maybe_replace(tok)
                result.append(replacement)
            else:
                result.append(tok)
        return self._reconstruct(result)

    def _maybe_replace(self, token: str) -> str:
        candidates = self.replacement_lexicon.get(token)
        if candidates and self._rng.random() < self.augmentation_probability:
            return self._rng.choice(candidates)
        return token

    # ── _process_token: single-token interface used by base.process ───────────

    def _process_token(self, token: str) -> str:
        if token.isdigit():
            return token
        lang = self._detect_language(token)
        if lang == "EN":
            return self._maybe_replace(token)
        return token


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[CodeSwitchAugmentation] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = CodeSwitchAugmentation()
    return processor.augment_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[CodeSwitchAugmentation] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = CodeSwitchAugmentation()
    return processor.augment_csv(input_csv, output_csv, text_col=text_col)
