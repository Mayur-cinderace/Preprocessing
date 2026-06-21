"""
text_sanitizer.py — Strict separation between analytical annotations and
text that is actually fed into a transformer model.

Why this module exists
------------------------
The preprocessing pipeline can annotate text with control/diagnostic markers
that are valuable for *analysis* (language tags, switch-point markers,
transliteration provenance markers) but are NOT natural language the
backbone models were pretrained on:

    [SWITCH]  [SWITCH_EN_HI]  [SWITCH_HI_EN]
    <EN> </EN>  <HI> </HI>  <UNK> </UNK>
    DEV:   TR:

Feeding these directly into a tokenizer means the backbone sees them as
out-of-vocabulary noise (typically shredded into several wordpiece/BPE
fragments), which both wastes context budget and can distort attention
patterns in ways that have nothing to do with the actual code-switching
phenomenon being studied — i.e. exactly the kind of confound this whole
research framework exists to measure and avoid.

This module enforces: every record keeps BOTH
    processed_text     — the full annotated text, used for analytical metrics
                          (H_F, CRD, Proxy-SPRC, SPRI, switch-token counts)
    model_input_text   — the same text with all control markers stripped,
                          used for everything that is an actual forward pass
                          through a transformer

and ``clean_for_model()`` is the single function responsible for producing
the latter from the former, so there is exactly one place in the codebase
that defines what counts as a "control marker" versus real text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Pattern, Tuple

# ── Marker patterns ────────────────────────────────────────────────────────
#
# Each entry: (human-readable name, compiled regex). Order matters only in
# that bracketed/special-token markers are removed before the bare prefix
# markers (DEV:, TR:) so a marker like "[SWITCH] DEV:word" doesn't leave a
# stray "DEV:" fragment after the surrounding switch marker is stripped —
# in practice the patterns don't overlap, but documenting the ordering
# assumption here means a future marker addition can check it explicitly
# instead of relying on accidental non-interference.

_MARKER_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("switch_bracket",       re.compile(r"\[SWITCH(?:_[A-Z]+_[A-Z]+)?\]")),
    ("switch_xml",           re.compile(r"<SWITCH[^>]*>")),
    ("lang_tag_xml_open",    re.compile(r"<(?:EN|HI|HI_ROM|HI_DEV|UNK)>")),
    ("lang_tag_xml_close",   re.compile(r"</(?:EN|HI|HI_ROM|HI_DEV|UNK)>")),
    ("lang_tag_inline",      re.compile(r"\|(?:EN|HI|HI_ROM|HI_DEV|UNK)\b")),
    ("lang_tag_prefix",      re.compile(r"\[(?:EN|HI|HI_ROM|HI_DEV|UNK)\]\s*")),
    ("lang_tag_suffix",      re.compile(r"\s*\[(?:EN|HI|HI_ROM|HI_DEV|UNK)\]")),
    ("dev_prefix",           re.compile(r"\bDEV:")),
    ("tr_prefix",            re.compile(r"\bTR:")),
]

_WHITESPACE_RE = re.compile(r"\s{2,}")


@dataclass
class SanitizeResult:
    """Result of cleaning one text string, with a record of what was removed."""
    model_input_text: str
    markers_removed: dict  # marker name -> count removed
    original_length: int
    cleaned_length: int

    @property
    def any_markers_removed(self) -> bool:
        return any(v > 0 for v in self.markers_removed.values())


def clean_for_model(text: str) -> SanitizeResult:
    """
    Strip every known analytical/control marker from *text*, returning the
    model-ready string plus a record of what was removed.

    This function is intentionally conservative: it only removes markers
    matching the explicit patterns above. It does NOT attempt generic
    "looks like a tag" heuristics, because a false-positive removal would
    silently corrupt real text (e.g. a literal "<HI>" appearing in a user's
    quoted message would be indistinguishable from our own marker without
    this being a closed, explicit list).
    """
    if not isinstance(text, str):
        return SanitizeResult(
            model_input_text="", markers_removed={}, original_length=0, cleaned_length=0,
        )

    cleaned = text
    removed_counts: dict = {}
    for name, pattern in _MARKER_PATTERNS:
        cleaned, n = pattern.subn(" ", cleaned)
        if n:
            removed_counts[name] = n

    # Collapse whitespace left behind by marker removal, then trim.
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    return SanitizeResult(
        model_input_text=cleaned,
        markers_removed=removed_counts,
        original_length=len(text),
        cleaned_length=len(cleaned),
    )


def clean_for_model_text(text: str) -> str:
    """Convenience wrapper returning only the cleaned string."""
    return clean_for_model(text).model_input_text


def contains_control_markers(text: str) -> bool:
    """True if *text* contains any recognized analytical/control marker."""
    if not isinstance(text, str):
        return False
    return any(pattern.search(text) for _, pattern in _MARKER_PATTERNS)


def add_model_input_column(
    df,
    processed_col: str = "processed_text",
    model_input_col: str = "model_input_text",
):
    """
    Add a *model_input_col* to *df* derived from *processed_col* via
    ``clean_for_model``, leaving *processed_col* untouched.

    This is the dataframe-level entry point used by job pipelines: after
    running the preprocessing pipeline (which produces ``processed_text``
    full of analytical markers), call this once before any model inference
    step, and pass ``model_input_col`` — never ``processed_col`` — into the
    tokenizer/model.
    """
    df = df.copy()
    df[model_input_col] = df[processed_col].astype(str).apply(clean_for_model_text)
    return df
