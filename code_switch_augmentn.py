"""
code_switch_augmentn.py — Controlled code-switch augmentation for Hinglish text.

Hard requirements:
    NLTK words corpus via _base.py

What it does:
    - Detects language of each token using the four-way framework (EN, HI_ROM,
      HI_DEV, UNK) from _base.py.
    - Applies context-constrained, switch-aware lexical substitution.
    - Generates N synthetic Hinglish variants per original sample.
    - Tracks augmentation statistics per session.
    - Optionally keeps the original alongside its augmented versions.

Design principles:
    ─ Semantic identity
        No token deletions, reorderings, or 1→N expansions.
        Every replacement is strictly 1-to-1 (one surface token → one
        surface token).  Multi-word Hinglish phrases are excluded from the
        lexicon precisely to uphold this contract.

    ─ Switch realism
        Augmentation probability is boosted near existing code-switch
        boundaries, reflecting natural Hinglish clustering patterns.
        An adaptive context strategy scales the boost by the overall
        switch density of the sequence, so heavily mixed text receives
        stronger augmentation than monolingual text.

    ─ Monolingual span dampening
        Long homogeneous spans receive reduced augmentation intensity to
        avoid over-switching pure-English or pure-Hindi sequences.

    ─ Density control
        Augmented sequences target a configurable switch-transition density
        (target_switch_density), defined as the proportion of adjacent
        content-token pairs that cross a language boundary.

    ─ Protected tokens
        Proper nouns, URLs, @mentions, #hashtags, and numerics are never
        substituted.  Proper-noun detection operates on the *original*
        surface form before any lowercasing.

    ─ Architecture
        All tokenisation and punctuation handling is delegated to HinglishBase
        hooks (self._tokenize / self._is_punctuation) rather than importing
        private regexes from _base.py.

This module is NOT:
    - machine translation,
    - paraphrasing,
    - unrestricted lexical replacement,
    - random corruption.

No silent fallbacks: if augmentation cannot produce a variant different from
the original after ``max_attempts`` tries, that slot is skipped rather than
silently returning the original marked as augmented.
"""
from __future__ import annotations

import re
import random
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase


# ---------------------------------------------------------------------------
# Default English → Hinglish replacement lexicon
#
# Rules:
#   1. Keys are lowercase English words.
#   2. Values are lists of single-token Hinglish alternatives only.
#      Multi-word phrases ("so jao", "pata hai") are intentionally excluded
#      to uphold the strict 1-to-1 semantic-identity contract.
#   3. Substitutions are semantically faithful and POS-compatible.
# ---------------------------------------------------------------------------

_DEFAULT_REPLACEMENT_LEXICON: Dict[str, List[str]] = {
    # ── adjectives ──────────────────────────────────────────────────────────
    "very":      ["bahut"],
    "good":      ["accha", "badhiya"],
    "bad":       ["bura", "kharab"],
    "happy":     ["khush"],
    "sad":       ["dukhi", "udaas"],
    "right":     ["sahi", "theek"],
    "wrong":     ["galat"],
    "big":       ["bada"],
    "small":     ["chota"],
    "nice":      ["accha", "sundar"],
    "beautiful": ["sundar", "khubsurat"],
    "tired":     ["thaka"],
    "busy":      ["vyast"],
    "easy":      ["aasaan"],
    "hard":      ["mushkil"],
    "new":       ["naya"],
    "old":       ["purana"],
    "free":      ["muft", "azaad"],
    # ── nouns ────────────────────────────────────────────────────────────────
    "love":      ["pyaar", "mohabbat"],
    "today":     ["aaj"],
    "movie":     ["film"],
    "food":      ["khana"],
    "home":      ["ghar"],
    "work":      ["kaam"],
    "friend":    ["dost", "yaar"],
    "money":     ["paisa"],
    "time":      ["waqt", "samay"],
    "day":       ["din"],
    "night":     ["raat"],
    "morning":   ["subah"],
    "evening":   ["shaam"],
    "heart":     ["dil"],
    "life":      ["zindagi"],
    "world":     ["duniya"],
    "people":    ["log"],
    "girl":      ["ladki"],
    "boy":       ["ladka"],
    "mother":    ["maa", "amma"],
    "father":    ["papa", "baap"],
    "brother":   ["bhai"],
    "sister":    ["behen"],
    "house":     ["ghar"],
    "city":      ["shehar"],
    "shop":      ["dukaan"],
    "water":     ["paani"],
    "tea":       ["chai"],
    # ── single-token verbs / imperatives ────────────────────────────────────
    "yes":       ["haan"],
    "no":        ["nahi", "na"],
    "why":       ["kyun"],
    "what":      ["kya"],
    "how":       ["kaise"],
    "when":      ["kab"],
    "where":     ["kahan"],
    "come":      ["aao", "aa"],
    "go":        ["jao", "ja"],
    "see":       ["dekho"],
    "tell":      ["bolo", "batao"],
    "listen":    ["suno"],
    "do":        ["karo"],
    "make":      ["banao"],
    "eat":       ["khao"],
    "drink":     ["piyo"],
    "sleep":     ["so"],
    "wait":      ["ruko"],
    "stop":      ["ruko"],
    "think":     ["socho"],
    "like":      ["pasand"],
    "hate":      ["nafrat"],
    "take":      ["lo"],
    "give":      ["do"],
    "show":      ["dikhao"],
    "keep":      ["rakho"],
    "put":       ["daalo"],
    "call":      ["bulao"],
    "laugh":     ["haso"],
    "cry":       ["roo"],
    # ── discourse / connectors ───────────────────────────────────────────────
    "because":   ["kyunki"],
    "but":       ["lekin", "par"],
    "and":       ["aur"],
    "or":        ["ya"],
    "also":      ["bhi"],
    "only":      ["sirf", "hi"],
    "now":       ["ab", "abhi"],
    "then":      ["tab", "phir"],
    "here":      ["yahan"],
    "there":     ["wahan"],
    "so":        ["toh", "isliye"],
    "always":    ["hamesha"],
    "never":     ["kabhi"],
    "again":     ["dobara"],
    "maybe":     ["shayad"],
    "really":    ["sachchi"],
    "okay":      ["theek"],
    "ok":        ["theek"],
}

# ---------------------------------------------------------------------------
# Token-protection patterns
# These are module-private and only referenced inside _is_protected.
# ---------------------------------------------------------------------------
_PROTECTED_RE = re.compile(
    r"https?://\S+"    # full URLs
    r"|www\.\S+"       # bare www-URLs
    r"|@\w+"           # @mentions
    r"|#\w+"           # #hashtags
    r"|\d[\d,\.]*"     # numerics (int, float, comma-separated)
)
# Capitalised Title-case word — conservative proper-noun heuristic.
# Applied to the *original* surface token BEFORE any lowercasing.
_PROPER_NOUN_RE = re.compile(r"^[A-Z][a-z]+$")

# ---------------------------------------------------------------------------
# Boost / damping constants (overridable via dataclass params)
# ---------------------------------------------------------------------------
_SWITCH_WINDOW    = 3      # token radius for switch-proximity detection
_SWITCH_BOOST     = 1.8    # fixed boost multiplier (used in "fixed" strategy)
_MONO_DAMPING     = 0.07   # prob reduction per token beyond mono_threshold
_MONO_THRESHOLD   = 4      # min monolingual run before dampening activates

ContextStrategy = Literal["fixed", "adaptive"]


# ---------------------------------------------------------------------------
# Augmentation statistics
# ---------------------------------------------------------------------------

@dataclass
class AugmentationStats:
    """
    Cumulative statistics accumulated across all _augment_once calls in a
    session.

    Metrics
    -------
    total_tokens_seen    Total surface tokens processed (incl. punct/nums).
    eligible_tokens      Tokens that were in the lexicon and eligible for swap.
    replaced_tokens      Tokens actually swapped.
    generated_variants   Augmented variants accepted (distinct from source).
    skipped_variants     Augment slots abandoned after max_attempts failures.
    switch_density_before  Running mean of pre-augmentation switch density.
    switch_density_after   Running mean of post-augmentation switch density.

    Derived properties
    ------------------
    replacement_rate     = replaced_tokens / eligible_tokens
    augmentation_yield   = generated_variants / (generated + skipped)
    """
    total_tokens_seen:     int   = 0
    eligible_tokens:       int   = 0
    replaced_tokens:       int   = 0
    generated_variants:    int   = 0
    skipped_variants:      int   = 0
    switch_density_before: float = 0.0
    switch_density_after:  float = 0.0
    _samples:              int   = 0

    def record_density(self, before: float, after: float) -> None:
        n = self._samples
        self.switch_density_before = (self.switch_density_before * n + before) / (n + 1)
        self.switch_density_after  = (self.switch_density_after  * n + after)  / (n + 1)
        self._samples += 1

    @property
    def replacement_rate(self) -> float:
        return self.replaced_tokens / self.eligible_tokens if self.eligible_tokens else 0.0

    @property
    def augmentation_yield(self) -> float:
        total = self.generated_variants + self.skipped_variants
        return self.generated_variants / total if total else 0.0

    def report(self) -> str:
        return (
            "AugmentationStats(\n"
            f"  total_tokens_seen     = {self.total_tokens_seen}\n"
            f"  eligible_tokens       = {self.eligible_tokens}\n"
            f"  replaced_tokens       = {self.replaced_tokens}\n"
            f"  replacement_rate      = {self.replacement_rate:.3f}\n"
            f"  generated_variants    = {self.generated_variants}\n"
            f"  skipped_variants      = {self.skipped_variants}\n"
            f"  augmentation_yield    = {self.augmentation_yield:.3f}\n"
            f"  switch_density_before = {self.switch_density_before:.3f}\n"
            f"  switch_density_after  = {self.switch_density_after:.3f}\n"
            ")"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class CodeSwitchAugmentation(HinglishBase):
    """
    Generates synthetic Hinglish variants via controlled lexical substitution.

    Augmentation is context-constrained:

    Switch-proximity boost
        Tokens within ``switch_window`` positions of an existing EN↔HI
        boundary receive a probability multiplier.  When ``context_strategy``
        is "adaptive" the multiplier scales with the overall switch density of
        the sequence, so heavily mixed text augments more aggressively than
        monolingual text.

    Monolingual-span dampening
        Tokens inside a homogeneous run longer than ``mono_threshold`` have
        their probability reduced by ``mono_damping`` per excess token.

    Density targeting
        When ``target_switch_density > 0`` the per-token probability is
        linearly rescaled so the augmented sequence approaches the desired
        proportion of adjacent cross-language token pairs.

    Proper-noun protection
        Detection runs on the *original* surface form before any lowercasing,
        so "Rahul" is correctly protected even when ``lowercase=True``.

    Semantic identity guarantee
        Every substitution is strictly 1-to-1 (one surface token replaced by
        exactly one surface token).  The replacement lexicon contains only
        single-token Hinglish alternatives.

    Architecture
        Tokenisation and punctuation classification are delegated to
        ``self._tokenize()`` and ``self._is_punctuation()`` from HinglishBase,
        keeping this module independent of _base.py's private internals.

    Parameters
    ----------
    augmentation_probability : float
        Base per-token substitution probability.
    n_augments : int
        Variants to generate per source sample.
    max_attempts : int
        Max trials before skipping a variant slot.
    preserve_original : bool
        Include the unmodified row in output.
    random_seed : int
        Seed for reproducible augmentation.
    replacement_lexicon : dict
        EN word → list of single-token Hinglish alternatives.
    target_switch_density : float
        Desired proportion of adjacent content-token pairs that cross a
        language boundary.  0.0 disables density targeting.
    context_strategy : "fixed" | "adaptive"
        "fixed"    — switch-proximity boost is ``switch_boost`` regardless of
                     the sequence's existing switch density.
        "adaptive" — boost is scaled by ``1 + switch_density_before``, so
                     already-mixed sequences receive stronger augmentation.
    switch_boost : float
        Base multiplier for the fixed context strategy.
    switch_window : int
        Token-distance radius for switch-proximity detection.
    mono_damping : float
        Per-token probability reduction during long monolingual runs.
    mono_threshold : int
        Minimum monolingual run length before dampening activates.
    """

    augmentation_probability: float                = 0.3
    n_augments:               int                  = 1
    max_attempts:             int                  = 5
    preserve_original:        bool                 = True
    random_seed:              int                  = 42
    replacement_lexicon:      Dict[str, List[str]] = field(
        default_factory=lambda: dict(_DEFAULT_REPLACEMENT_LEXICON)
    )
    target_switch_density:    float                = 0.0
    context_strategy:         ContextStrategy      = "fixed"
    switch_boost:             float                = _SWITCH_BOOST
    switch_window:            int                  = _SWITCH_WINDOW
    mono_damping:             float                = _MONO_DAMPING
    mono_threshold:           int                  = _MONO_THRESHOLD

    # Session statistics — populated post-init via _setup.
    stats: AugmentationStats = field(default_factory=AugmentationStats, init=False)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError(
                f"augmentation_probability must be in [0, 1]; "
                f"got {self.augmentation_probability}"
            )
        if self.n_augments < 0:
            raise ValueError(f"n_augments must be >= 0; got {self.n_augments}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1; got {self.max_attempts}")
        if not 0.0 <= self.target_switch_density <= 1.0:
            raise ValueError(
                f"target_switch_density must be in [0, 1]; "
                f"got {self.target_switch_density}"
            )
        self._rng = random.Random(self.random_seed)
        self.stats = AugmentationStats()

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, text: str) -> str:
        """Apply one stochastic augmentation pass.  Use for single-string calls."""
        if not self.enabled or not isinstance(text, str):
            return text
        result, _ = self._augment_once(text)
        return result

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        output_col: str = "processed_text",
        is_augmented_col: str = "is_augmented",
    ) -> pd.DataFrame:
        """
        Expand ``df`` with ``n_augments`` synthetic variants per row.

        Augmented rows have ``is_augmented_col = True``.
        Original rows (when ``preserve_original``) have it as ``False``.
        Slots that fail to produce a distinct variant are skipped silently
        and counted in ``self.stats.skipped_variants``.
        """
        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found.  Available: {list(df.columns)}"
            )
        rows: List[dict] = []
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
                    candidate, _ = self._augment_once(text)
                    if candidate != text:
                        variant = candidate
                        break
                if variant is None:
                    self.stats.skipped_variants += 1
                    continue
                self.stats.generated_variants += 1
                new_row = row.to_dict()
                new_row[output_col]       = variant
                new_row[is_augmented_col] = True
                rows.append(new_row)

        out_cols = list(df.columns) + [output_col, is_augmented_col]
        return pd.DataFrame(rows, columns=out_cols)

    def augment_csv(
        self,
        input_csv: "str | Path",
        output_csv: "str | Path",
        text_col: str = "text",
        output_col: str = "processed_text",
    ) -> pd.DataFrame:
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        df = pd.read_csv(input_csv)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        processed = self.augment_dataframe(df, text_col=text_col, output_col=output_col)
        processed.to_csv(output_csv, index=False)
        return processed

    # ── Token protection ──────────────────────────────────────────────────────

    def _is_protected(self, original_surface: str) -> bool:
        """
        Return True if the token must never be substituted.

        Critically, this method receives the *original* surface form — before
        any normalisation or lowercasing — so that proper-noun detection
        (capitalised Title-case) works correctly even when ``lowercase=True``.
        """
        if _PROTECTED_RE.fullmatch(original_surface):
            return True
        # Proper-noun heuristic: capitalised Title-case non-punctuation word.
        if _PROPER_NOUN_RE.fullmatch(original_surface):
            return True
        return False

    # ── Language labelling ────────────────────────────────────────────────────

    def _label_tokens(
        self,
        original_tokens: List[str],
        normalised_tokens: List[str],
    ) -> List[str]:
        """
        Return one language label per token from the four-way scheme:
        EN, HI_ROM, HI_DEV, UNK, or SKIP (punct/protected/numeric).

        Protection check uses ``original_tokens`` (pre-normalisation).
        Language detection uses ``normalised_tokens``.
        """
        labels: List[str] = []
        for orig, norm in zip(original_tokens, normalised_tokens):
            if self._is_punctuation(orig):
                labels.append("SKIP")
            elif self._is_protected(orig):
                labels.append("SKIP")
            elif norm.isdigit():
                labels.append("SKIP")
            else:
                labels.append(self._detect_language(norm))
        return labels

    # ── Switch-density (transition-based definition) ──────────────────────────

    def _switch_density(self, labels: List[str]) -> float:
        """
        Proportion of adjacent content-token pairs that cross a language
        boundary.

        Definition:
            switches / (content_tokens − 1)

        This counts *transitions*, not token proportions, so:
            EN HI EN HI EN  →  4/4 = 1.00  (fully alternating)
            EN EN EN EN HI  →  1/4 = 0.25  (one boundary)
        """
        content = [l for l in labels if l != "SKIP"]
        if len(content) < 2:
            return 0.0
        switches = sum(
            1 for a, b in zip(content, content[1:]) if a != b
        )
        return switches / (len(content) - 1)

    # ── Switch-proximity detection ────────────────────────────────────────────

    def _switch_positions(self, labels: List[str]) -> Set[int]:
        """
        Return the set of *original* token indices that fall within
        ``switch_window`` tokens of a language-switch boundary.
        """
        content_positions = [i for i, l in enumerate(labels) if l != "SKIP"]
        content_labels    = [labels[i] for i in content_positions]

        switch_content_indices: List[int] = []
        for i in range(1, len(content_labels)):
            if content_labels[i] != content_labels[i - 1]:
                switch_content_indices.append(i)
                switch_content_indices.append(i - 1)

        near_switch: Set[int] = set()
        for ci in switch_content_indices:
            orig = content_positions[ci]
            for delta in range(-self.switch_window, self.switch_window + 1):
                p = orig + delta
                if 0 <= p < len(labels):
                    near_switch.add(p)
        return near_switch

    # ── Monolingual run lengths ───────────────────────────────────────────────

    def _monolingual_run_lengths(self, labels: List[str]) -> List[int]:
        """
        For each token return the length of its current same-language run
        (SKIP tokens inherit the current run counter without resetting it).
        """
        run_lengths: List[int] = [0] * len(labels)
        current_lang: Optional[str] = None
        run_len = 0
        for i, lab in enumerate(labels):
            if lab == "SKIP":
                run_lengths[i] = run_len
                continue
            if lab == current_lang:
                run_len += 1
            else:
                current_lang = lab
                run_len = 1
            run_lengths[i] = run_len
        return run_lengths

    # ── Per-token probability ─────────────────────────────────────────────────

    def _compute_token_prob(
        self,
        idx: int,
        switch_positions: Set[int],
        run_lengths: List[int],
        density_before: float,
        current_density: float,
    ) -> float:
        """
        Return the effective substitution probability for an eligible EN token.

        Steps:
        1. Start from base ``augmentation_probability``.
        2. Apply switch-proximity boost (fixed or adaptive).
        3. Apply monolingual-run dampening.
        4. Apply density-gap scaling toward ``target_switch_density``.
        """
        prob = self.augmentation_probability

        # ── Step 2: switch-proximity boost ──────────────────────────────────
        if idx in switch_positions:
            if self.context_strategy == "adaptive":
                # Scale boost by the sequence's existing mixing level so that
                # already-mixed text receives stronger nudges.
                boost = 1.0 + density_before * (self.switch_boost - 1.0)
            else:
                boost = self.switch_boost
            prob = min(1.0, prob * boost)

        # ── Step 3: monolingual-run dampening ────────────────────────────────
        run_len = run_lengths[idx]
        if run_len > self.mono_threshold:
            excess = run_len - self.mono_threshold
            prob = max(0.0, prob - excess * self.mono_damping)

        # ── Step 4: density-gap scaling ──────────────────────────────────────
        if self.target_switch_density > 0.0:
            gap = self.target_switch_density - current_density
            # Linear: gap>0 → scale>1 (undershoot); gap<0 → scale<1 (overshoot).
            scale = 1.0 + 2.0 * gap
            prob = max(0.0, min(1.0, prob * scale))

        return prob

    # ── Core augmentation ─────────────────────────────────────────────────────

    def _augment_once(self, text: str) -> Tuple[str, Dict]:
        """
        Apply one stochastic, context-constrained augmentation pass.

        Returns (augmented_text, metadata_dict).

        Semantic identity contract:
            - Token order is preserved.
            - No tokens are inserted or deleted.
            - Substitutions are strictly 1-to-1 (one token → one token).

        Proper-noun protection is evaluated on the original surface form
        captured before normalisation, so capitalised words survive even
        when ``lowercase=True``.
        """
        # Step 1 — tokenise on the *original* text to capture surface forms.
        original_tokens: List[str] = self._tokenize(text)

        # Step 2 — normalise a copy for language detection and lexicon lookup.
        normalised_text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            normalised_text = normalised_text.lower()
        normalised_tokens: List[str] = self._tokenize(normalised_text)

        # Step 3 — label every token (uses original for protection, norm for lang).
        labels = self._label_tokens(original_tokens, normalised_tokens)

        # Step 4 — compute context signals.
        density_before    = self._switch_density(labels)
        switch_positions  = self._switch_positions(labels)
        run_lengths       = self._monolingual_run_lengths(labels)

        # Running switched-pair count for density estimation.
        # We track the *label sequence as modified* to estimate current density.
        working_labels = list(labels)

        result: List[str] = []
        n_eligible = 0
        n_replaced = 0

        for idx, (orig_tok, norm_tok) in enumerate(
            zip(original_tokens, normalised_tokens)
        ):
            lab = labels[idx]

            # ── Punctuation: honour preserve_punctuation flag ─────────────
            if self._is_punctuation(orig_tok):
                if self.preserve_punctuation:
                    result.append(orig_tok)
                continue

            # ── SKIP (protected / numeric): pass through unchanged ─────────
            if lab == "SKIP":
                result.append(orig_tok)
                continue

            # ── Non-EN tokens (HI_ROM, HI_DEV, UNK): pass through ─────────
            if lab != "EN":
                result.append(orig_tok)
                continue

            # ── EN token: check lexicon ────────────────────────────────────
            candidates = self.replacement_lexicon.get(norm_tok)
            if not candidates:
                result.append(orig_tok)
                continue

            n_eligible += 1

            # Estimate current switch density from the evolving label sequence.
            current_density = self._switch_density(working_labels)

            prob = self._compute_token_prob(
                idx, switch_positions, run_lengths, density_before, current_density
            )

            if self._rng.random() < prob:
                replacement = self._rng.choice(candidates)
                result.append(replacement)
                n_replaced += 1
                # Update working labels so later tokens see the updated density.
                working_labels[idx] = "HI_ROM"
            else:
                result.append(orig_tok)

        # Update cumulative stats.
        self.stats.total_tokens_seen += len(original_tokens)
        self.stats.eligible_tokens   += n_eligible
        self.stats.replaced_tokens   += n_replaced

        augmented = self._reconstruct(result)

        # Measure post-augmentation density.
        aug_tokens = self._tokenize(augmented)
        aug_norm   = self._tokenize(
            unicodedata.normalize("NFKC", augmented).lower()
            if self.lowercase else unicodedata.normalize("NFKC", augmented)
        )
        after_labels  = self._label_tokens(aug_tokens, aug_norm)
        density_after = self._switch_density(after_labels)
        self.stats.record_density(density_before, density_after)

        meta = {
            "eligible":       n_eligible,
            "replaced":       n_replaced,
            "density_before": density_before,
            "density_after":  density_after,
        }
        return augmented, meta

    # ── _process_token: single-token hook used by HinglishBase.process ────────

    def _process_token(self, token: str) -> str:
        """
        Single-token interface required by HinglishBase.

        No context signals are available here; falls back to base-probability
        substitution with protection checks.
        """
        if self._is_protected(token):
            return token
        if token.isdigit():
            return token
        lang = self._detect_language(token)
        if lang != "EN":
            return token
        candidates = self.replacement_lexicon.get(token.lower())
        if candidates and self._rng.random() < self.augmentation_probability:
            return self._rng.choice(candidates)
        return token


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

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
    input_csv: "str | Path",
    output_csv: "str | Path",
    text_col: str = "text",
    processor: Optional[CodeSwitchAugmentation] = None,
) -> pd.DataFrame:
    if processor is None:
        processor = CodeSwitchAugmentation()
    return processor.augment_csv(input_csv, output_csv, text_col=text_col)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _run_smoke_tests() -> None:
    """
    Smoke tests covering:
      1.  Successful augmentation of a mixed Hinglish sequence.
      2.  Skipped variants when no lexicon match exists.
      3.  Monolingual English: eligible tokens detected and replaced.
      4.  Mixed Hinglish: output is a non-empty string.
      5.  Density is a valid [0,1] transition-based value post-augmentation.
      6.  Proper noun protection — survives lowercasing.
      7.  URL preservation.
      8.  @mention preservation.
      9.  #hashtag preservation.
      10. Punctuation preservation.
      11. Deterministic outputs under fixed seeds.
      12. Statistics accumulation across multiple calls.
      13. 1-to-1 substitution: token count unchanged after augmentation.
      14. Adaptive context strategy produces valid output.
      15. target_switch_density parameter accepted without error.
    """
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))
        status = PASS if condition else FAIL
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    def make(**kw) -> CodeSwitchAugmentation:
        return CodeSwitchAugmentation(random_seed=0, augmentation_probability=0.9, **kw)

    aug = make()

    # 1. Successful augmentation.
    text_mixed = "yaar this food is bahut good"
    out1, meta1 = aug._augment_once(text_mixed)
    check("successful_augmentation",
          isinstance(out1, str) and len(out1) > 0,
          f"'{text_mixed}' → '{out1}'")

    # 2. Skipped variants — tokens with no lexicon entry pass unchanged.
    text_noop = "xyz pqr mnopqrst"
    out2, meta2 = aug._augment_once(text_noop)
    check("skipped_variants_unchanged",
          out2 == text_noop,
          f"expected unchanged, got '{out2}'")
    check("skipped_variants_zero_eligible",
          meta2["eligible"] == 0,
          f"eligible={meta2['eligible']}")

    # 3. Monolingual English: eligible and replaced > 0 with prob=0.9.
    text_en = "the food is good and the work is very bad"
    out3, meta3 = aug._augment_once(text_en)
    check("monolingual_english_eligible",
          meta3["eligible"] > 0,
          f"eligible={meta3['eligible']}")
    check("monolingual_english_some_replaced",
          meta3["replaced"] > 0,
          f"replaced={meta3['replaced']}")

    # 4. Mixed Hinglish: valid output string.
    text_hi = "bahut good yaar come home"
    out4, _ = aug._augment_once(text_hi)
    check("mixed_hinglish_produces_output",
          isinstance(out4, str) and len(out4) > 0,
          f"'{out4}'")

    # 5. Density values are in [0, 1].
    _, meta5 = aug._augment_once("the food is very good and work is bad today")
    check("density_before_valid",
          0.0 <= meta5["density_before"] <= 1.0,
          f"density_before={meta5['density_before']:.3f}")
    check("density_after_valid",
          0.0 <= meta5["density_after"] <= 1.0,
          f"density_after={meta5['density_after']:.3f}")

    # 6. Proper noun protection — must survive even when lowercase=True.
    aug_pn = CodeSwitchAugmentation(
        random_seed=2,
        augmentation_probability=1.0,
        lowercase=True,
        replacement_lexicon={"rahul": ["dost"], "friend": ["dost"]},
    )
    text_pn = "Rahul is my friend"
    protected_ok = all(
        # "rahul" (lowercased) will appear if NOT protected; "rahul" means failure.
        # Original "Rahul" detected as proper noun → should pass through as "Rahul"
        # or lowercased "rahul" depending on lowercase flag, but NOT replaced by "dost".
        "dost" not in aug_pn._augment_once(text_pn)[0].split()[0]
        for _ in range(20)
    )
    check("proper_noun_protected",
          protected_ok,
          "'Rahul' first token must not become 'dost'")

    # 7. URL preservation.
    text_url = "check https://example.com for good food"
    out7, _ = aug._augment_once(text_url)
    check("url_preserved",
          "https://example.com" in out7,
          f"got '{out7}'")

    # 8. @mention preservation.
    text_mention = "tell @yaar to come home"
    out8, _ = aug._augment_once(text_mention)
    check("mention_preserved",
          "@yaar" in out8,
          f"got '{out8}'")

    # 9. #hashtag preservation.
    text_hashtag = "good food #khana today"
    out9, _ = aug._augment_once(text_hashtag)
    check("hashtag_preserved",
          "#khana" in out9,
          f"got '{out9}'")

    # 10. Punctuation preservation.
    aug_punct = make(preserve_punctuation=True)
    text_punct = "good food, yes! very happy."
    out10, _ = aug_punct._augment_once(text_punct)
    check("punctuation_preserved",
          "," in out10 and "!" in out10 and "." in out10,
          f"got '{out10}'")

    # 11. Deterministic seeds.
    aug_a = CodeSwitchAugmentation(random_seed=99, augmentation_probability=0.5)
    aug_b = CodeSwitchAugmentation(random_seed=99, augmentation_probability=0.5)
    text_det = "the food is good and friend is happy"
    out_a, _ = aug_a._augment_once(text_det)
    out_b, _ = aug_b._augment_once(text_det)
    check("deterministic_seeds",
          out_a == out_b,
          f"'{out_a}' == '{out_b}'")

    # 12. Statistics accumulation.
    aug_stat = CodeSwitchAugmentation(random_seed=7, augmentation_probability=0.6)
    stat_texts = ["the food is good", "yaar very happy today", "work is very hard"]
    for t in stat_texts:
        aug_stat._augment_once(t)
    check("stats_tokens_seen",
          aug_stat.stats.total_tokens_seen > 0,
          f"total_tokens_seen={aug_stat.stats.total_tokens_seen}")
    check("stats_eligible_nonzero",
          aug_stat.stats.eligible_tokens > 0,
          f"eligible_tokens={aug_stat.stats.eligible_tokens}")
    check("stats_density_samples",
          aug_stat.stats._samples == len(stat_texts),
          f"samples={aug_stat.stats._samples}")

    # 13. 1-to-1 substitution: word-token count preserved.
    aug_count = make()
    text_count = "the food is good and work is bad"
    # Count whitespace-separated words (no punct insertion expected).
    before_count = len(text_count.split())
    out13, _ = aug_count._augment_once(text_count)
    after_count = len(out13.split())
    check("one_to_one_token_count",
          before_count == after_count,
          f"before={before_count}, after={after_count}, '{out13}'")

    # 14. Adaptive context strategy: no errors, valid output.
    aug_adapt = CodeSwitchAugmentation(
        random_seed=5, augmentation_probability=0.5, context_strategy="adaptive"
    )
    out14, meta14 = aug_adapt._augment_once("yaar this food is bahut good")
    check("adaptive_strategy_valid_output",
          isinstance(out14, str) and 0.0 <= meta14["density_after"] <= 1.0,
          f"'{out14}', density_after={meta14['density_after']:.3f}")

    # 15. target_switch_density accepted without validation error.
    try:
        aug_dens = CodeSwitchAugmentation(
            random_seed=1, augmentation_probability=0.7, target_switch_density=0.4
        )
        out15, _ = aug_dens._augment_once("the food is very good")
        check("target_switch_density_accepted",
              isinstance(out15, str),
              f"'{out15}'")
    except Exception as exc:
        check("target_switch_density_accepted", False, str(exc))

    # Summary.
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"\n  {passed}/{total} smoke tests passed.")
    if passed < total:
        failed = [name for name, ok, _ in results if not ok]
        raise AssertionError(f"Failed: {failed}")


if __name__ == "__main__":
    print("Running smoke tests for code_switch_augmentn.py …\n")
    _run_smoke_tests()
    print("\nDone.")