"""
cached_analytical_evaluator.py — Cache-aware, model-manager-aware
AnalyticalEvaluator.

Why a subclass rather than editing analytical_evaluator.py directly
----------------------------------------------------------------------
``AnalyticalEvaluator`` (see analytical_evaluator.py) is a research artifact
with its own careful honesty-in-naming and metric-correctness work already
done (token-level CRD, bootstrap CIs, MPP composite score, SPRI, etc.). This
module does NOT touch that file's metric logic. It overrides exactly one
seam — ``_batch_encode`` — which every embedding-consuming metric (spectral
analysis, embedding stability, CRD, Proxy-SPRC) already calls through. That
makes this the single, minimal, low-risk integration point for:

  1. routing embedding computation through ``ModelManager`` (so the encoder
     is loaded once, from the unified HF cache, on the right device) instead
     of each AnalyticalEvaluator instance lazily constructing its own
     ``SentenceTransformer``,
  2. routing every embedding lookup through the persistent SQLite
     ``EmbeddingCache`` (Requirement 4), so re-running metrics — even across
     server restarts — does not recompute embeddings for text already seen,
  3. enforcing the analytical/model-input separation (Requirement 6 /
     the project's closing mandate): by default this evaluator embeds
     ``model_input_text`` (control markers stripped), not raw
     ``processed_text``, for every embedding-bearing metric. The *non*
     -embedding metrics (information theory, fertility entropy, switch-point
     retention) still operate on the original ``processed_col`` text, because
     those metrics are specifically measuring properties of the annotated
     text itself (e.g. SPRI counts switch markers; stripping them first
     would make the metric measure nothing).

Known caveat — marker-wrapped tokens and the parent class's tokenizer
------------------------------------------------------------------------
``AnalyticalEvaluator``'s non-embedding metrics (fertility entropy, SPRI,
Proxy-SPRC's internal tokenization) tokenize text via a plain
``text.split()`` on whitespace. If a preprocessing module wraps each token
individually in XML-style tags — e.g. ``"<HI>yaar</HI> <EN>movie</EN>"`` —
each split-on-whitespace "token" becomes the string ``"<HI>yaar</HI>"``
rather than ``"yaar"``, which will not match the Hinglish/English lexicons
used by ``assign_token_language``, and those metrics will report ``None`` or
near-zero values rather than raising an error. This is a pre-existing
property of ``AnalyticalEvaluator``'s tokenizer, not something introduced by
this subclass, and is intentionally NOT patched here — this module's
mandate is to fix the *integration* seam (caching, model-manager routing,
analytical/model-input separation), not to silently change inherited metric
logic. If your pipeline's switch_point_encoding or language_identification_
tagging modules wrap every token in markers, run fertility entropy / SPRI on
a marker-free annotation scheme (e.g. inline boundary markers only, not
per-token wrapping), or address tokenizer awareness directly in
``analytical_evaluator.py``.

Usage
-----
    from cached_analytical_evaluator import CachedAnalyticalEvaluator

    evaluator = CachedAnalyticalEvaluator(
        df_original=df, df_processed=processed_df,
        text_col="text", processed_col="processed_text",
        label_col="label", batch_size=batch_size,
    )
    results = evaluator.evaluate_all(progress_callback=cb)
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from analytical_evaluator import AnalyticalEvaluator
from embedding_cache import EmbeddingCache, get_embedding_cache
from logging_config import get_logger
from model_manager import ModelManager, get_model_manager
from text_sanitizer import add_model_input_column

logger = get_logger(__name__)

_DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class CachedAnalyticalEvaluator(AnalyticalEvaluator):
    """
    Drop-in replacement for ``AnalyticalEvaluator`` with cached, model-manager-
    routed embedding computation and strict analytical/model-input separation.

    All public methods (``evaluate_information_theory``,
    ``evaluate_fertility_entropy``, ``evaluate_spectral_analysis``,
    ``evaluate_embedding_stability``, ``evaluate_cross_lingual_consistency``,
    ``evaluate_switch_point_retention``,
    ``evaluate_switch_point_representation_collapse``, ``evaluate_all``,
    ``compute_composite_score``, ``compare_profiles``) are inherited
    unchanged. Only embedding acquisition is intercepted.
    """

    def __init__(
        self,
        df_original: pd.DataFrame,
        df_processed: pd.DataFrame,
        text_col: str = "text",
        processed_col: str = "processed_text",
        label_col: str = "label",
        batch_size: int = 32,
        svd_threshold: int = 2000,
        embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL,
        model_manager: ModelManager | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        # Derive model_input_text columns for both frames BEFORE calling
        # super().__init__, since the parent constructor only stores column
        # names/dataframes and doesn't read text content itself — but we want
        # the cleaned column to exist on self.df_proc / self.df_orig from the
        # very first access.
        df_processed = add_model_input_column(
            df_processed, processed_col=processed_col, model_input_col="model_input_text",
        )

        super().__init__(
            df_original=df_original, df_processed=df_processed,
            text_col=text_col, processed_col=processed_col, label_col=label_col,
            batch_size=batch_size, svd_threshold=svd_threshold,
        )

        # df_original doesn't have a "processed" column to clean, but several
        # embedding-bearing metrics (embedding_stability, CRD's token
        # extraction) read self.df_orig[self.text_col] directly — that raw
        # source text is not expected to contain our pipeline's control
        # markers in the first place (markers are introduced BY preprocessing,
        # not present in raw input), so no cleaning is applied there. This is
        # documented rather than silently assumed: if a future preprocessing
        # stage starts annotating raw input before it reaches this evaluator,
        # this assumption should be revisited.

        self.embedding_model_name = embedding_model_name
        self._manager = model_manager or get_model_manager()
        self._cache = embedding_cache or get_embedding_cache()

    # ── The single overridden seam ────────────────────────────────────────────

    def _ensure_embed_model(self, model_name: str = _DEFAULT_EMBEDDING_MODEL):
        """
        Override: load via ``ModelManager`` (unified cache_dir, GPU placement,
        singleton reuse across the whole process) instead of constructing a
        fresh ``SentenceTransformer`` per ``AnalyticalEvaluator`` instance.
        """
        return self._manager.get_sentence_transformer(model_name or self.embedding_model_name)

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Override: serve from the persistent embedding cache where possible,
        only invoking the model for cache misses.

        This is the exact same call signature and return type as the parent
        implementation, so every metric method in ``AnalyticalEvaluator``
        (which only ever calls ``self._batch_encode(...)``) works completely
        unmodified.
        """
        model = self._ensure_embed_model(self.embedding_model_name)

        def _encode_fn(missing_texts: List[str]) -> np.ndarray:
            return np.array(
                model.encode(
                    missing_texts, batch_size=self.batch_size,
                    show_progress_bar=False, convert_to_numpy=True,
                )
            )

        return self._cache.encode_with_cache(self.embedding_model_name, texts, _encode_fn)

    # ── Model-input-text routing for embedding metrics ────────────────────────
    #
    # The parent class's embedding-bearing methods read self.df_proc[self.
    # proc_col] directly. Rather than overriding each of those methods
    # (spectral analysis, embedding stability, CRD, Proxy-SPRC) individually —
    # which would mean re-implementing their logic and risking drift from the
    # parent's carefully-fixed metric correctness — we expose model_input_text
    # as the column those methods should read by temporarily pointing
    # self.proc_col at it for the duration of an embedding-bearing call.
    #
    # This is implemented via a context manager rather than a permanent
    # attribute swap so that non-embedding metrics (information_theory,
    # fertility_entropy, switch_point_retention) continue reading the
    # annotated processed_col as normal when called directly.

    class _UseModelInputColumn:
        def __init__(self, evaluator: "CachedAnalyticalEvaluator"):
            self.evaluator = evaluator
            self._original_proc_col = None

        def __enter__(self):
            self._original_proc_col = self.evaluator.proc_col
            self.evaluator.proc_col = "model_input_text"
            return self.evaluator

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.evaluator.proc_col = self._original_proc_col

    def evaluate_spectral_analysis(self, model_name: str = _DEFAULT_EMBEDDING_MODEL):
        with self._UseModelInputColumn(self):
            return super().evaluate_spectral_analysis(model_name=model_name)

    def evaluate_embedding_stability(self, model_name: str = _DEFAULT_EMBEDDING_MODEL):
        with self._UseModelInputColumn(self):
            return super().evaluate_embedding_stability(model_name=model_name)

    def evaluate_switch_point_representation_collapse(self):
        # Proxy-SPRC specifically needs to embed individual TOKENS from the
        # annotated text to locate switch points (it tokenizes proc_col
        # itself internally) — it must keep reading the annotated
        # processed_col, NOT model_input_text, because the switch-point
        # positions it detects are defined in terms of the annotated tokens.
        # No column swap here; inherited unchanged.
        return super().evaluate_switch_point_representation_collapse()

    # evaluate_cross_lingual_consistency (CRD) tokenizes self.df_orig[self.
    # text_col] (raw original text), not self.df_proc — see the constructor
    # note above on why df_orig is not given a model_input_text column. No
    # override needed; inherited unchanged.
