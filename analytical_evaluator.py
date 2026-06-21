"""
analytical_evaluator.py — Stage 4 Analytical Evaluation Framework.

Implements the full Multilingual Preprocessing Profile described in:
  "Preprocessing is Not Language-Agnostic: An Information-Theoretic and
   Spectral Study of Code-Switched Text Preprocessing on Transformer Performance"

Seven scores forming the Multilingual Preprocessing Profile:

Standard scores:
  ΔI    — Task-Relevant Information Proxy (info theory; see note below)
  κ(E)  — Embedding Matrix Condition Number (spectral)
  Stability — Embedding Stability (cosine / euclidean / angular)

Novel multilingual scores (from the paper):
  H_F   — Fertility Entropy: tokeniser balance across languages
  Proxy-SPRC — Switch-Point Representation Collapse (embedding-similarity proxy)
  CRD   — Cross-lingual Representation Distance (token-level)
  SPRI  — Switch-Point Retention Index (how many CS boundaries survive
          preprocessing)

Honesty notes on naming
------------------------
* ``ΔI`` is NOT a strict mutual-information difference I(X;Y) − I(f(X);Y).
  ``mutual_info_classif`` on TF-IDF features estimates *feature–label*
  mutual information, which is a reasonable but approximate proxy for
  task-relevant information content. All outputs and documentation use the
  phrase "task-relevant information proxy" rather than claiming an exact
  information-theoretic quantity.
* The metric formerly called "SPAC" (Switch-Point Attention Collapse) does
  NOT inspect a transformer's actual attention matrices — no forward pass
  through an attention mechanism occurs anywhere in this module. It computes
  entropy over a cosine-similarity matrix of token embeddings, which is a
  structurally different (and weaker) signal than attention entropy. This
  module calls it **Proxy-SPRC** (Switch-Point Representation Collapse) to
  avoid implying a stronger claim than the computation supports. A
  backward-compatible alias method is kept for callers using the old name.
* ``CRD`` (Cross-lingual Representation Distance) is computed from
  **token-level** language assignments, not sentence-level Devanagari
  detection. A code-switched sentence such as "yaar movie bahut accha" has
  no Devanagari characters at all but contains both Hindi (romanized) and
  English tokens; sentence-level script detection would have misclassified
  the entire sentence as "English," corrupting the centroid comparison.

Hard requirements:
    pip install scikit-learn numpy pandas scipy
    pip install sentence-transformers torch   # for embedding-based metrics
"""
from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import pairwise, silhouette_score

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_stats = None  # type: ignore
    _SCIPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False


# ── Devanagari detector ───────────────────────────────────────────────────────

_DEV_RE = re.compile(r"[\u0900-\u097F]")


def _contains_devanagari(s: str) -> bool:
    return bool(_DEV_RE.search(s))


# ── Shared token-language assignment ──────────────────────────────────────────
#
# Used by fertility entropy, CRD, Proxy-SPRC, and SPRI so that all four
# multilingual metrics agree on what counts as an EN vs. HI token. Promoting
# this to a module-level function (rather than duplicating it per-metric, as
# the original draft did) means a future improvement to the heuristic
# automatically propagates to every metric that depends on it.

_PHONETIC_HINDI_CLUES: Tuple[str, ...] = (
    "aa", "ai", "au", "bh", "dh", "kh", "ph", "sh", "th",
    "ch", "gh", "jh", "ky", "py", "ny", "ri", "ya",
)


def _load_lexicons() -> Tuple[frozenset, frozenset]:
    """Load HINGLISH_LEXICON / ENGLISH_VOCAB from _base.py, with safe fallback."""
    try:
        from _base import HINGLISH_LEXICON, ENGLISH_VOCAB
        return HINGLISH_LEXICON, ENGLISH_VOCAB
    except ImportError:
        return frozenset(), frozenset()


def assign_token_language(
    tok: str,
    hinglish_lexicon: Optional[frozenset] = None,
    english_vocab: Optional[frozenset] = None,
) -> str:
    """
    Heuristically assign a coarse language label to a single token.

    Returns one of ``"HI"``, ``"EN"``, ``"UNK"``.

    This is a lightweight heuristic, not a calibrated classifier — it is
    documented as such everywhere it is used. Token-level language ID is
    inherently noisy for short or ambiguous romanized tokens; metrics built
    on top of this function (CRD, H_F, Proxy-SPRC, SPRI) inherit that noise
    and should be read as directional signals, not exact measurements.
    """
    if hinglish_lexicon is None or english_vocab is None:
        hinglish_lexicon, english_vocab = _load_lexicons()

    t = tok.lower()
    if _contains_devanagari(t):
        return "HI"
    if t in hinglish_lexicon:
        return "HI"
    if t in english_vocab:
        return "EN"
    if t.isalpha() and any(c in t for c in _PHONETIC_HINDI_CLUES):
        return "HI"
    return "UNK"


# ── Bootstrap helper ──────────────────────────────────────────────────────────

def bootstrap_ci(
    values: Iterable[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> Dict[str, Optional[float]]:
    """
    Percentile bootstrap confidence interval for a 1-D sample of metric values.

    Returns ``{"point_estimate", "ci_low", "ci_high", "n_resamples", "n"}``.
    If fewer than 2 values are supplied, CI bounds are ``None`` (a bootstrap
    over 0 or 1 points is not meaningful) but the point estimate is still
    reported when possible.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)] if arr.size else arr

    if arr.size == 0:
        return {
            "point_estimate": None, "ci_low": None, "ci_high": None,
            "n_resamples": n_resamples, "n": 0,
        }
    if arr.size < 2:
        return {
            "point_estimate": float(statistic(arr)), "ci_low": None,
            "ci_high": None, "n_resamples": n_resamples, "n": int(arr.size),
        }

    rng = np.random.RandomState(random_state)
    boot_stats = np.empty(n_resamples, dtype=float)
    n = arr.size
    for i in range(n_resamples):
        sample = arr[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic(sample)

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": float(statistic(arr)),
        "ci_low": lo,
        "ci_high": hi,
        "n_resamples": n_resamples,
        "n": int(n),
    }


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Structured container for a full Multilingual Preprocessing Profile run.

    Mirrors the dict keys returned by ``evaluate_all`` so existing dict-based
    consumers keep working (``evaluate_all`` still returns a dict by default);
    pass ``as_dataclass=True`` to ``evaluate_all`` to receive this instead.
    """
    information_theory: Dict[str, Any] = field(default_factory=dict)
    fertility_entropy: Dict[str, Any] = field(default_factory=dict)
    spectral_analysis: Dict[str, Any] = field(default_factory=dict)
    embedding_stability: Dict[str, Any] = field(default_factory=dict)
    cross_lingual: Dict[str, Any] = field(default_factory=dict)
    switch_point_retention: Dict[str, Any] = field(default_factory=dict)
    switch_point_representation_collapse: Optional[Dict[str, Any]] = None
    composite_score: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "information_theory": self.information_theory,
            "fertility_entropy": self.fertility_entropy,
            "spectral_analysis": self.spectral_analysis,
            "embedding_stability": self.embedding_stability,
            "cross_lingual": self.cross_lingual,
            "switch_point_retention": self.switch_point_retention,
            "composite_score": self.composite_score,
        }
        if self.switch_point_representation_collapse is not None:
            d["switch_point_representation_collapse"] = (
                self.switch_point_representation_collapse
            )
        return d


# ─────────────────────────────────────────────────────────────────────────────

class AnalyticalEvaluator:
    """
    Computes the full Multilingual Preprocessing Profile over a dataset.

    All metrics are available independently and via ``evaluate_all``.

    Parameters
    ----------
    df_original   : DataFrame with the raw text column.
    df_processed  : DataFrame with the preprocessed text column.
    text_col      : Column name for raw text.
    processed_col : Column name for preprocessed text.
    label_col     : Column name for integer class labels (0/1/2).
    batch_size    : Embedding batch size (increase for GPU).
    svd_threshold : If n_samples > this value, spectral analysis uses
                    TruncatedSVD instead of full eigendecomposition
                    (see evaluate_spectral_analysis for rationale).
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
    ) -> None:
        self.df_orig       = df_original.reset_index(drop=True)
        self.df_proc       = df_processed.reset_index(drop=True)
        self.text_col      = text_col
        self.proc_col      = processed_col
        self.label_col     = label_col
        self.batch_size    = batch_size
        self.svd_threshold = svd_threshold
        self._embed_model  = None
        self._hinglish_lexicon, self._english_vocab = _load_lexicons()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        return [t for t in text.lower().split() if t]

    def _assign_lang(self, tok: str) -> str:
        """Instance-bound wrapper around the shared heuristic (reuses lexicons)."""
        return assign_token_language(tok, self._hinglish_lexicon, self._english_vocab)

    def _token_entropy(self, texts: Iterable[str]) -> float:
        counter: Counter = Counter()
        total = 0
        for t in texts:
            toks = self._tokenize(t)
            counter.update(toks)
            total += len(toks)
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in counter.values()], dtype=float)
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def _ensure_embed_model(
        self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ) -> "SentenceTransformer":
        if not _ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed.\n"
                "Run:  pip install sentence-transformers torch"
            )
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(model_name)
        return self._embed_model

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_embed_model()
        return np.array(
            model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 1. INFORMATION THEORY  (ΔI — task-relevant information proxy)
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_information_theory(self) -> Dict[str, Any]:
        """
        Computes:
          • Token-distribution entropy H before and after preprocessing.
          • ΔI = a *proxy* for information discarded by preprocessing,
            estimated as the drop in TF-IDF feature–label mutual information
            (via ``sklearn.feature_selection.mutual_info_classif``).
          • Vocabulary size change.

        Important caveat
        -----------------
        ``mutual_info_classif`` estimates mutual information between
        *engineered TF-IDF features* and the label, not the strict
        information-theoretic quantity I(X; Y) between the raw text random
        variable and the label. It is a reasonable, widely-used proxy for
        "how much task-relevant signal survives in the feature representation,"
        but it is NOT a direct measurement of I(X;Y) and should not be reported
        as one. All output keys below are named accordingly
        (``mutual_information_*`` rather than claiming exact MI).
        """
        df = self.df_orig
        if self.label_col not in df.columns:
            raise ValueError(
                f"Label column '{self.label_col}' not in dataframe."
            )

        labels = df[self.label_col].astype(int).values

        entropy_before = self._token_entropy(
            df[self.text_col].astype(str).values
        )
        entropy_after = self._token_entropy(
            self.df_proc[self.proc_col].astype(str).values
        )
        entropy_delta = entropy_after - entropy_before

        texts = self.df_proc[self.proc_col].astype(str).values
        mi_mean: Optional[float] = None
        mi_topk: List[float] = []
        mi_method = "tfidf"
        mi_error: Optional[str] = None

        try:
            vec = TfidfVectorizer(max_features=20000)
            X = vec.fit_transform(texts)
            X_dense = X.toarray()
            mi = mutual_info_classif(
                X_dense, labels, discrete_features=False, random_state=0
            )
            mi_mean = float(np.nanmean(mi))
            mi_topk = sorted(mi.tolist(), reverse=True)[:10]
        except ValueError as e:
            msg = str(e)
            mi_error = msg
            if "sparse" in msg.lower():
                try:
                    mi_method = "countvec_fallback"
                    vec2 = CountVectorizer(max_features=5000)
                    X2 = vec2.fit_transform(texts)
                    mi2 = mutual_info_classif(
                        X2, labels, discrete_features=True, random_state=0
                    )
                    mi_mean = float(np.nanmean(mi2))
                    mi_topk = sorted(mi2.tolist(), reverse=True)[:10]
                    mi_error = None
                except Exception as e2:
                    mi_method = "countvec_fallback_failed"
                    mi_error = str(e2)
            else:
                mi_method = "tfidf_failed"
        except Exception as e:
            mi_method = "tfidf_failed"
            mi_error = str(e)

        # ΔI proxy = task-relevant information apparently discarded by
        # preprocessing. Positive → preprocessing reduced the feature–label
        # MI proxy. This is NOT a strict I(X;Y) − I(f(X);Y) computation;
        # see the docstring caveat above.
        delta_mi_proxy = None
        if mi_mean is not None:
            try:
                vec_orig = TfidfVectorizer(max_features=20000)
                orig_texts = df[self.text_col].astype(str).values
                X_orig = vec_orig.fit_transform(orig_texts).toarray()
                mi_orig = mutual_info_classif(
                    X_orig, labels, discrete_features=False, random_state=0
                )
                mi_orig_mean = float(np.nanmean(mi_orig))
                delta_mi_proxy = mi_orig_mean - mi_mean
            except Exception:
                pass

        return {
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "entropy_delta": entropy_delta,
            "mutual_information_proxy_mean": mi_mean,
            "mutual_information_proxy_top_k": mi_topk,
            "mutual_information_proxy_delta": delta_mi_proxy,
            "mutual_information_proxy_method": mi_method,
            "mutual_information_proxy_error": mi_error,
            "proxy_caveat": (
                "mutual_information_proxy_* values estimate feature-label MI "
                "on TF-IDF features via mutual_info_classif; they approximate "
                "but do not equal the strict information-theoretic I(X;Y)."
            ),
            "vocab_size_before": int(
                len(Counter(
                    " ".join(df[self.text_col].astype(str).values).split()
                ))
            ),
            "vocab_size_after": int(
                len(Counter(
                    " ".join(self.df_proc[self.proc_col].astype(str).values).split()
                ))
            ),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 2. SPECTRAL ANALYSIS  (κ(E))
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_spectral_analysis(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> Dict[str, Any]:
        """
        Computes embedding covariance eigenspectrum.
        κ(E) = σ₁(E) / σᵣ(E)  — condition number of embedding matrix.
        Spectral entropy of the eigenvalue distribution.

        Scalability
        -----------
        ``np.linalg.eigvalsh`` on a full d×d covariance matrix is exact but
        becomes slow and numerically unstable as the embedding dimension or
        sample count grows. When ``n_samples > self.svd_threshold``, this
        method switches to ``sklearn.decomposition.TruncatedSVD`` to obtain
        the leading singular values directly from the embedding matrix
        without forming the full covariance matrix. TruncatedSVD only
        recovers the top-k singular values (not the full spectrum), so
        ``all_eigenvalues`` is omitted and ``condition_number`` is computed
        as σ₁/σ_k over the k components retained rather than σ₁/σ_min over
        the full spectrum — this is reported explicitly in ``spectral_method``
        so results from the two code paths are not silently conflated.
        """
        try:
            texts = self.df_proc[self.proc_col].astype(str).tolist()
            emb = self._batch_encode(texts)
        except Exception as e:
            return {"error": f"Embedding failed: {e}"}

        n_samples, n_dim = emb.shape

        if n_samples > self.svd_threshold:
            # ── Scalable path: TruncatedSVD on the embedding matrix ──────────
            k = min(n_dim - 1, 50, n_samples - 1)
            k = max(k, 1)
            svd = TruncatedSVD(n_components=k, random_state=0)
            svd.fit(emb - emb.mean(axis=0, keepdims=True))
            singular_values = svd.singular_values_
            eig = (singular_values ** 2) / max(1, n_samples - 1)
            eig = np.sort(np.abs(eig))[::-1]
            total = eig.sum() if eig.sum() > 0 else 1.0
            p = eig / total

            spectral_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
            condition_number = float(eig[0] / (eig[-1] + 1e-12))
            spectral_decay = float(eig[0] / (total + 1e-12))
            cumulative_variance = np.cumsum(p).tolist()

            return {
                "spectral_method": "truncated_svd",
                "spectral_method_note": (
                    f"n_samples ({n_samples}) > svd_threshold "
                    f"({self.svd_threshold}); used TruncatedSVD with "
                    f"k={k} components. condition_number is sigma_1/sigma_k "
                    f"over retained components, not the full spectrum."
                ),
                "top_k_eigenvalues": eig[:10].tolist(),
                "all_eigenvalues": None,
                "spectral_entropy": spectral_entropy,
                "condition_number": condition_number,
                "spectral_decay": spectral_decay,
                "cumulative_variance_top10": cumulative_variance[:10],
                "embedding_dim": int(n_dim),
                "n_samples": int(n_samples),
                "n_components_retained": int(k),
            }

        # ── Exact path: full eigendecomposition of covariance matrix ─────────
        cov = np.cov(emb, rowvar=False)
        cov = (cov + cov.T) / 2.0
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.abs(eig))[::-1]
        total = eig.sum() if eig.sum() > 0 else 1.0
        p = eig / total

        spectral_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
        condition_number = float(eig[0] / (eig[-1] + 1e-12))
        spectral_decay = float(eig[0] / (total + 1e-12))
        cumulative_variance = np.cumsum(p).tolist()

        return {
            "spectral_method": "exact_eigh",
            "spectral_method_note": (
                f"n_samples ({n_samples}) <= svd_threshold "
                f"({self.svd_threshold}); used full eigendecomposition."
            ),
            "top_k_eigenvalues": eig[:10].tolist(),
            "all_eigenvalues": eig.tolist(),
            "spectral_entropy": spectral_entropy,
            "condition_number": condition_number,
            "spectral_decay": spectral_decay,
            "cumulative_variance_top10": cumulative_variance[:10],
            "embedding_dim": int(n_dim),
            "n_samples": int(n_samples),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 3. EMBEDDING STABILITY
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_embedding_stability(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> Dict[str, Any]:
        """
        Per-sample distance between original and preprocessed embeddings,
        measured three ways:

          • cosine_similarity   — direction-only; insensitive to magnitude
            changes. High mean → preprocessing preserved semantic direction.
          • euclidean_distance  — sensitive to magnitude; can reveal large
            shifts that cosine similarity hides (e.g. two vectors can point
            in nearly the same direction but differ greatly in norm after
            preprocessing strips content).
          • angular_distance    — arccos(cosine similarity) in radians;
            a metric (satisfies the triangle inequality) version of cosine
            similarity, useful when distances need to be averaged or
            compared across pairs in a way that respects metric properties.

        Reporting only cosine similarity (as the original implementation
        did) can hide cases where preprocessing causes a large drop in
        embedding norm — e.g. aggressive truncation or stop-word removal —
        while the surviving tokens still point in roughly the same direction.
        """
        try:
            orig_texts = self.df_orig[self.text_col].astype(str).tolist()
            proc_texts = self.df_proc[self.proc_col].astype(str).tolist()
            emb_orig = self._batch_encode(orig_texts)
            emb_proc = self._batch_encode(proc_texts)
        except Exception as e:
            return {"error": f"Embedding failed: {e}"}

        cos_sims = pairwise.cosine_similarity(emb_orig, emb_proc).diagonal()
        cos_sims = np.clip(cos_sims, -1.0, 1.0)

        euclidean = np.linalg.norm(emb_orig - emb_proc, axis=1)

        angular = np.arccos(cos_sims)  # radians, in [0, pi]

        hist, bins = np.histogram(cos_sims, bins=20, range=(-1, 1))

        return {
            "cosine_similarity": {
                "mean": float(np.mean(cos_sims)),
                "median": float(np.median(cos_sims)),
                "std": float(np.std(cos_sims)),
                "min": float(np.min(cos_sims)),
                "max": float(np.max(cos_sims)),
                "samples": cos_sims.tolist()[:200],
            },
            "euclidean_distance": {
                "mean": float(np.mean(euclidean)),
                "median": float(np.median(euclidean)),
                "std": float(np.std(euclidean)),
                "min": float(np.min(euclidean)),
                "max": float(np.max(euclidean)),
                "samples": euclidean.tolist()[:200],
            },
            "angular_distance_radians": {
                "mean": float(np.mean(angular)),
                "median": float(np.median(angular)),
                "std": float(np.std(angular)),
                "min": float(np.min(angular)),
                "max": float(np.max(angular)),
                "samples": angular.tolist()[:200],
            },
            # Backward-compatible top-level keys (old consumers expecting
            # cosine-only fields keep working).
            "mean_similarity": float(np.mean(cos_sims)),
            "median_similarity": float(np.median(cos_sims)),
            "std_similarity": float(np.std(cos_sims)),
            "min_similarity": float(np.min(cos_sims)),
            "max_similarity": float(np.max(cos_sims)),
            "histogram_counts": hist.tolist(),
            "histogram_bins": bins.tolist(),
            "samples": cos_sims.tolist()[:200],
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 4. CROSS-LINGUAL CONSISTENCY  (CRD) — token-level
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_cross_lingual_consistency(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """
        CRD = ||μ_EN − μ_HI||₂ — L2 distance between centroid embeddings of
        EN and HI **tokens**, not EN and HI **sentences**.

        Why token-level, not sentence-level
        -------------------------------------
        The original implementation classified an entire sentence as "HI" if
        it contained any Devanagari character and "EN" otherwise. For
        romanized code-switched text such as "yaar movie bahut accha" — all
        Latin script, no Devanagari — that sentence is 100% misclassified as
        English even though half its tokens are Hindi. The corpus this module
        targets is specifically romanized Hinglish, so sentence-level script
        detection would silently degrade CRD into "Devanagari-script sentences
        vs. everything else," which is a different and far less informative
        quantity than the paper's intended cross-lingual representation
        distance.

        This version assigns a language label to every individual token via
        ``self._assign_lang`` (HI / EN / UNK heuristic — see
        ``assign_token_language``), embeds each token, and computes centroids
        over the resulting per-language token embedding sets pooled across the
        whole corpus.

        Token-level embedding caveat
        ------------------------------
        ``SentenceTransformer`` models are trained to embed full sentences;
        embedding isolated single tokens is an out-of-distribution use of the
        model and the resulting vectors carry less contextual meaning than a
        token embedded in its original sentence. This is a known limitation
        shared with the token-level encoding used in ``evaluate_switch_point_
        representation_collapse``. Results should be read as a directional
        cross-lingual alignment signal, not a precise geometric measurement.
        """
        orig_texts = self.df_orig[self.text_col].astype(str).tolist()

        # Collect (token, language) pairs across the corpus, deduplicating
        # identical tokens to bound the number of embedding calls — repeated
        # common words (e.g. "the", "yaar") would otherwise be embedded many
        # times for no additional information.
        token_lang_pairs: List[Tuple[str, str]] = []
        seen_tokens: set = set()
        for text in orig_texts:
            for tok in self._tokenize(text):
                if tok in seen_tokens:
                    continue
                seen_tokens.add(tok)
                lang = self._assign_lang(tok)
                if lang in ("HI", "EN"):
                    token_lang_pairs.append((tok, lang))

        en_tokens = [t for t, l in token_lang_pairs if l == "EN"]
        hi_tokens = [t for t, l in token_lang_pairs if l == "HI"]

        result: Dict[str, Any] = {
            "alignment_count_en": len(en_tokens),
            "alignment_count_hi": len(hi_tokens),
            "assignment_level": "token",
            "assignment_method": "assign_token_language heuristic (see module docstring)",
        }

        if not en_tokens or not hi_tokens:
            result["note"] = (
                "Not enough cross-lingual tokens for centroid comparison "
                "(need at least one EN and one HI token)."
            )
            return result

        try:
            emb_en = self._batch_encode(en_tokens)
            emb_hi = self._batch_encode(hi_tokens)
        except Exception as e:
            return {"error": f"Embedding failed: {e}", **result}

        centroid_en = emb_en.mean(axis=0)
        centroid_hi = emb_hi.mean(axis=0)

        crd = float(np.linalg.norm(centroid_en - centroid_hi))
        cosine = float(
            pairwise.cosine_similarity(
                centroid_en.reshape(1, -1), centroid_hi.reshape(1, -1)
            )[0, 0]
        )

        result["cross_lingual_representation_distance"] = crd
        result["cross_lingual_centroid_cosine"] = cosine

        # Bootstrap CI on CRD: resample token sets with replacement and
        # recompute the centroid distance, to express uncertainty given
        # finite (and possibly imbalanced) EN/HI token counts.
        rng = np.random.RandomState(0)
        n_en, n_hi = emb_en.shape[0], emb_hi.shape[0]
        boot_crd = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            sample_en = emb_en[rng.randint(0, n_en, size=n_en)]
            sample_hi = emb_hi[rng.randint(0, n_hi, size=n_hi)]
            boot_crd[i] = np.linalg.norm(
                sample_en.mean(axis=0) - sample_hi.mean(axis=0)
            )
        alpha = 0.05
        result["crd_bootstrap"] = {
            "point_estimate": crd,
            "ci_low": float(np.percentile(boot_crd, 100 * alpha / 2)),
            "ci_high": float(np.percentile(boot_crd, 100 * (1 - alpha / 2))),
            "n_resamples": n_bootstrap,
        }

        # Silhouette score as cluster-cohesion proxy over the pooled token set.
        try:
            pooled_emb = np.vstack([emb_en, emb_hi])
            pooled_labels = np.array([0] * n_en + [1] * n_hi)
            result["silhouette_score"] = float(
                silhouette_score(pooled_emb, pooled_labels)
            )
        except Exception:
            result["silhouette_score"] = None

        if n_en > 1:
            cos_en = pairwise.cosine_similarity(emb_en)
            result["en_intra_cohesion"] = float(
                cos_en[np.triu_indices(n_en, k=1)].mean()
            )
        if n_hi > 1:
            cos_hi = pairwise.cosine_similarity(emb_hi)
            result["hi_intra_cohesion"] = float(
                cos_hi[np.triu_indices(n_hi, k=1)].mean()
            )

        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 5. FERTILITY ENTROPY  (H_F)  ← Novel score from the paper
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_fertility_entropy(self) -> Dict[str, Any]:
        """
        H_F(s) = −∑_l p_l(s) · log p_l(s)
        where p_l(s) = n_l(s) / ∑ n_{l'}(s)

        Measures how balanced the tokeniser's subword allocation is across
        the two languages in a CS sentence. H_F ∈ [0, log 2].

        Token-language assignment uses the shared ``self._assign_lang``
        heuristic (Devanagari → HI, lexicon membership, phonetic clues;
        see ``assign_token_language``).

        A high mean H_F (close to log 2) means both languages get equal
        token share — ideal. Low H_F indicates one language dominates.

        Distribution statistics
        ------------------------
        In addition to mean/std, this method reports median, variance,
        skewness, and kurtosis of the per-sentence H_F distribution. A
        single mean can hide a bimodal distribution (e.g. many purely
        monolingual sentences at H_F≈0 mixed with a few heavily
        code-switched sentences at H_F≈log 2) — skewness and kurtosis make
        that shape visible without inspecting the full histogram.
        """
        log2 = math.log(2)  # maximum H_F for two languages

        def _sentence_fertility_entropy(text: str) -> Optional[float]:
            toks = text.split()
            if not toks:
                return None
            counts: Counter = Counter()
            for t in toks:
                counts[self._assign_lang(t)] += 1
            n_en = counts.get("EN", 0)
            n_hi = counts.get("HI", 0)
            n_lang = n_en + n_hi
            if n_lang == 0:
                return None
            vals = []
            for n in (n_en, n_hi):
                if n > 0:
                    p = n / n_lang
                    vals.append(-p * math.log2(p))
            return sum(vals)

        proc_texts = self.df_proc[self.proc_col].astype(str).tolist()
        orig_texts = self.df_orig[self.text_col].astype(str).tolist()

        hf_proc = [
            h for t in proc_texts
            if (h := _sentence_fertility_entropy(t)) is not None
        ]
        hf_orig = [
            h for t in orig_texts
            if (h := _sentence_fertility_entropy(t)) is not None
        ]

        def _distribution_stats(values: List[float]) -> Dict[str, Optional[float]]:
            if not values:
                return {
                    "mean": None, "median": None, "std": None, "variance": None,
                    "skewness": None, "kurtosis": None,
                }
            arr = np.asarray(values, dtype=float)
            stats: Dict[str, Optional[float]] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "variance": float(np.var(arr)),
            }
            if _SCIPY_AVAILABLE and arr.size >= 2:
                # bias=False matches the common sample-skewness/kurtosis
                # convention used in most reporting; fisher=True (default)
                # reports excess kurtosis (normal distribution → 0).
                stats["skewness"] = float(_scipy_stats.skew(arr, bias=False))
                stats["kurtosis"] = float(_scipy_stats.kurtosis(arr, bias=False))
            else:
                stats["skewness"] = None
                stats["kurtosis"] = None
            return stats

        stats_before = _distribution_stats(hf_orig)
        stats_after = _distribution_stats(hf_proc)

        # Per-token language distributions
        all_langs_orig: Counter = Counter()
        all_langs_proc: Counter = Counter()
        for t in orig_texts:
            for tok in t.split():
                all_langs_orig[self._assign_lang(tok)] += 1
        for t in proc_texts:
            for tok in t.split():
                all_langs_proc[self._assign_lang(tok)] += 1

        result: Dict[str, Any] = {
            "max_fertility_entropy": float(log2),
            "fertility_entropy_before": stats_before,
            "fertility_entropy_after": stats_after,
            # Backward-compatible flat keys
            "mean_fertility_entropy_before": stats_before["mean"],
            "mean_fertility_entropy_after": stats_after["mean"],
            "std_fertility_entropy_before": stats_before["std"],
            "std_fertility_entropy_after": stats_after["std"],
            "histogram_before": np.histogram(hf_orig, bins=10, range=(0.0, log2))[0].tolist() if hf_orig else [],
            "histogram_after": np.histogram(hf_proc, bins=10, range=(0.0, log2))[0].tolist() if hf_proc else [],
            "token_lang_dist_before": dict(all_langs_orig),
            "token_lang_dist_after": dict(all_langs_proc),
        }

        if stats_before["mean"] is not None and stats_after["mean"] is not None:
            result["delta_fertility_entropy"] = (
                stats_after["mean"] - stats_before["mean"]
            )
            result["fertility_entropy_bootstrap_after"] = bootstrap_ci(hf_proc)
        else:
            result["delta_fertility_entropy"] = None

        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 6. SWITCH-POINT REPRESENTATION COLLAPSE  (Proxy-SPRC)
    #    ← Novel score from the paper; renamed from "SPAC" — see note below.
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_switch_point_representation_collapse(self) -> Dict[str, Any]:
        """
        Proxy-SPRC(l) = H̄_repr(SP) − H̄_repr(non-SP)

        Naming note — please read before citing this metric
        ------------------------------------------------------
        This method was originally named "SPAC" (Switch-Point Attention
        Collapse) and its output described as attention entropy. That name
        overclaims: nowhere in this computation does a transformer forward
        pass occur, and no attention weight matrix is read from any model.
        What is actually computed is entropy over a **cosine-similarity
        matrix of independently-encoded token embeddings** — a structurally
        different and considerably weaker signal than genuine attention
        entropy extracted from a model's attention heads. Real attention
        entropy would require register-and-extract hooks into a transformer's
        self-attention layers during a forward pass over the full sentence,
        which this module does not do.

        This method is renamed **Proxy-SPRC** (Switch-Point Representation
        Collapse) to reflect what is actually measured: whether token
        embeddings near code-switch boundaries become less mutually
        distinguishable (higher similarity-entropy) than embeddings in
        monolingual spans. That is still a meaningful and novel signal about
        representation geometry at switch points — it just is not attention.
        A backward-compatible alias,
        ``evaluate_switch_point_attention_collapse``, is kept below and
        forwards to this method with a deprecation note in its return value.

        Computation
        -----------
        For each sentence:
          1. Find switch-point positions (lang(i) ≠ lang(i-1), both known).
          2. Independently embed each token (see the token-level embedding
             caveat in ``evaluate_cross_lingual_consistency`` — applies here
             too).
          3. Compute token-level similarity-entropy as the normalized
             cosine-similarity row entropy against other tokens in the
             same sentence.
          4. Proxy-SPRC = mean entropy at switch positions − mean entropy
             elsewhere.

        A strongly negative score means token representations near language
        boundaries are less distinguishable from their neighbours (more
        diffuse / "collapsed") than representations in monolingual spans.
        """
        sp_entropies: List[float] = []
        non_sp_entropies: List[float] = []
        total_switch_points = 0
        total_sentences_with_switch = 0

        try:
            proc_texts = self.df_proc[self.proc_col].astype(str).tolist()
        except Exception as e:
            return {"error": str(e)}

        def _switch_points(toks: List[str]) -> List[bool]:
            langs = [self._assign_lang(t) for t in toks]
            sp = [False] * len(langs)
            for i in range(1, len(langs)):
                if langs[i] != langs[i - 1] and langs[i] != "UNK" and langs[i - 1] != "UNK":
                    sp[i] = True
            return sp

        for text in proc_texts:
            toks = text.split()
            if len(toks) < 2:
                continue
            is_sp = _switch_points(toks)
            sw_count = sum(is_sp)
            if sw_count == 0:
                continue
            total_switch_points += sw_count
            total_sentences_with_switch += 1

            try:
                tok_embs = self._batch_encode(toks)
                sim = pairwise.cosine_similarity(tok_embs)
                for i, tok in enumerate(toks):
                    row = sim[i].copy()
                    row[i] = 0.0  # exclude self
                    row = np.clip(row, 0.0, None)
                    s = row.sum()
                    if s < 1e-9:
                        continue
                    p = row / s
                    entropy = float(-np.sum(p * np.log2(p + 1e-12)))
                    if is_sp[i]:
                        sp_entropies.append(entropy)
                    else:
                        non_sp_entropies.append(entropy)
            except Exception:
                continue

        result: Dict[str, Any] = {
            "metric_name": "Proxy-SPRC (Switch-Point Representation Collapse)",
            "naming_note": (
                "Formerly called SPAC / 'attention collapse'; renamed because "
                "no transformer attention matrix is computed here, only "
                "cosine-similarity entropy over independently-encoded token "
                "embeddings. See method docstring."
            ),
            "total_switch_points": total_switch_points,
            "sentences_with_switch_points": total_sentences_with_switch,
            "mean_entropy_at_switch_points": float(np.mean(sp_entropies)) if sp_entropies else None,
            "mean_entropy_at_non_switch_points": float(np.mean(non_sp_entropies)) if non_sp_entropies else None,
        }

        if sp_entropies and non_sp_entropies:
            sprc = float(np.mean(sp_entropies)) - float(np.mean(non_sp_entropies))
            result["proxy_sprc_score"] = sprc
            result["interpretation"] = (
                "negative — switch-point representation collapse detected "
                "(embeddings near language boundaries are less mutually "
                "distinguishable than elsewhere)"
                if sprc < 0 else
                "positive — no collapse; switch-point embeddings are at "
                "least as distinguishable as elsewhere"
            )
        else:
            result["proxy_sprc_score"] = None
            result["interpretation"] = "Insufficient switch-point data."

        return result

    def evaluate_switch_point_attention_collapse(self) -> Dict[str, Any]:
        """
        Deprecated alias for ``evaluate_switch_point_representation_collapse``.

        Kept so existing callers using the old "SPAC" name do not break.
        The original name implied a transformer attention-based measurement;
        this method does not compute attention. See
        ``evaluate_switch_point_representation_collapse`` for the honest
        description of what is actually measured.
        """
        warnings.warn(
            "evaluate_switch_point_attention_collapse is a deprecated alias. "
            "The computation does not use transformer attention weights — "
            "use evaluate_switch_point_representation_collapse, which "
            "documents this honestly.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.evaluate_switch_point_representation_collapse()
        # Preserve the legacy key name for old consumers reading 'spac_score'.
        if "proxy_sprc_score" in result:
            result["spac_score"] = result["proxy_sprc_score"]
        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 7. SWITCH-POINT RETENTION INDEX  (SPRI)  ← New metric
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_switch_point_retention(self) -> Dict[str, Any]:
        """
        SPRI = retained_switch_points / original_switch_points

        Motivation
        ----------
        Almost every text-preprocessing pipeline risks destroying
        code-switch boundary information without anyone noticing: stripping
        a romanized Hindi clause as "noise," normalizing it into English-
        looking tokens, or dropping short tokens that happened to mark a
        language transition all silently erase the very signal that H_F,
        CRD, and Proxy-SPRC are trying to measure downstream. SPRI directly
        answers: of the code-switch boundaries present in the *original*
        text, how many still exist — at a comparable position — in the
        *preprocessed* text?

        Computation
        -----------
        For each (original, processed) sentence pair:
          1. Tokenize both versions; assign a coarse language label to every
             token via ``self._assign_lang``.
          2. Identify switch points in the original (positions where
             lang(i) != lang(i-1), both known).
          3. Identify switch points in the processed text the same way.
          4. A switch point is "retained" if the processed text has a switch
             point whose relative position (index / token count) lies within
             ``position_tolerance`` of the original switch point's relative
             position. Relative (not absolute) position is used because
             preprocessing routinely changes token counts (removing
             punctuation, splitting contractions, etc.), so comparing raw
             indices would understate retention for reasons unrelated to
             code-switch preservation.

        SPRI ∈ [0, 1] (can exceed 1 in principle if processing introduces
        spurious switch points faster than retaining originals — this is
        flagged in ``warning`` when it occurs, since SPRI > 1 indicates the
        denominator framing needs a second look for that corpus).

        SPRI is computed per-sentence and aggregated; both the corpus-level
        mean and a bootstrap CI are reported.
        """
        position_tolerance = 0.15  # fraction of sentence length

        def _switch_positions(toks: List[str]) -> List[float]:
            if len(toks) < 2:
                return []
            langs = [self._assign_lang(t) for t in toks]
            positions = []
            for i in range(1, len(langs)):
                if langs[i] != langs[i - 1] and langs[i] != "UNK" and langs[i - 1] != "UNK":
                    positions.append(i / len(langs))
            return positions

        orig_texts = self.df_orig[self.text_col].astype(str).tolist()
        proc_texts = self.df_proc[self.proc_col].astype(str).tolist()

        n = min(len(orig_texts), len(proc_texts))
        if len(orig_texts) != len(proc_texts):
            warnings.warn(
                f"df_original ({len(orig_texts)} rows) and df_processed "
                f"({len(proc_texts)} rows) have different lengths; SPRI is "
                f"computed over the first {n} aligned rows only.",
                stacklevel=2,
            )

        per_sentence_spri: List[float] = []
        total_original_switches = 0
        total_retained_switches = 0
        sentences_with_original_switches = 0

        for i in range(n):
            orig_toks = orig_texts[i].split()
            proc_toks = proc_texts[i].split()

            orig_sp = _switch_positions(orig_toks)
            proc_sp = _switch_positions(proc_toks)

            if not orig_sp:
                continue  # SPRI undefined for sentences with no CS boundary

            sentences_with_original_switches += 1
            total_original_switches += len(orig_sp)

            retained = 0
            remaining_proc_sp = list(proc_sp)
            for op in orig_sp:
                match_idx = None
                for j, pp in enumerate(remaining_proc_sp):
                    if abs(op - pp) <= position_tolerance:
                        match_idx = j
                        break
                if match_idx is not None:
                    retained += 1
                    remaining_proc_sp.pop(match_idx)  # one-to-one matching

            total_retained_switches += retained
            per_sentence_spri.append(retained / len(orig_sp))

        result: Dict[str, Any] = {
            "position_tolerance": position_tolerance,
            "sentences_with_original_switch_points": sentences_with_original_switches,
            "total_original_switch_points": total_original_switches,
            "total_retained_switch_points": total_retained_switches,
        }

        if total_original_switches > 0:
            corpus_spri = total_retained_switches / total_original_switches
            result["spri_corpus_level"] = corpus_spri
            if corpus_spri > 1.0:
                result["warning"] = (
                    "spri_corpus_level > 1.0: more switch points matched than "
                    "existed in the original at this tolerance, which can "
                    "happen if preprocessing introduced new spurious switch "
                    "points that happened to align with original positions. "
                    "Inspect token_lang_dist before trusting this value."
                )
        else:
            result["spri_corpus_level"] = None
            result["note"] = "No code-switch boundaries found in original text."

        if per_sentence_spri:
            result["spri_per_sentence_mean"] = float(np.mean(per_sentence_spri))
            result["spri_per_sentence_bootstrap"] = bootstrap_ci(per_sentence_spri)
        else:
            result["spri_per_sentence_mean"] = None
            result["spri_per_sentence_bootstrap"] = bootstrap_ci([])

        return result

    # ═════════════════════════════════════════════════════════════════════════
    # COMPOSITE SCORE  (MPP — Multilingual Preprocessing Profile)
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize(value: Optional[float], lo: float, hi: float, invert: bool = False) -> Optional[float]:
        """Clip-and-scale *value* into [0, 1] given an expected [lo, hi] range."""
        if value is None or lo == hi:
            return None
        v = (value - lo) / (hi - lo)
        v = float(np.clip(v, 0.0, 1.0))
        return 1.0 - v if invert else v

    def compute_composite_score(
        self,
        profile: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Combine the individual metrics in *profile* (as returned by
        ``evaluate_all``) into a single Multilingual Preprocessing Profile
        (MPP) score in [0, 1], higher = better preprocessing.

            MPP = w1*(1 − ΔI_norm)        # less task-relevant info lost is better
                + w2*HF_norm              # more balanced tokenization is better
                + w3*(1 − CRD_norm)       # smaller cross-lingual gap is better
                + w4*Stability_norm       # higher cosine stability is better
                + w5*SpectralEntropy_norm # higher spectral entropy (less
                                          #   collapse onto few directions) is better
                + w6*SPRI_norm            # more retained switch points is better

        Weights default to equal (1/6 each) and are renormalized to sum to 1
        if the caller supplies a partial or unnormalized weight dict.

        Normalization ranges are necessarily corpus-dependent rough defaults
        (documented inline below) rather than universal constants — this
        composite is intended for *relative* comparison between preprocessing
        pipelines run on the same dataset (see ``compare_profiles``), not as
        an absolute, cross-dataset benchmark number.
        """
        default_weights = {
            "delta_mi_proxy": 1 / 6,
            "fertility_entropy": 1 / 6,
            "crd": 1 / 6,
            "stability": 1 / 6,
            "spectral_entropy": 1 / 6,
            "spri": 1 / 6,
        }
        w = dict(default_weights)
        if weights:
            w.update({k: v for k, v in weights.items() if k in default_weights})
        w_sum = sum(w.values()) or 1.0
        w = {k: v / w_sum for k, v in w.items()}

        info = profile.get("information_theory", {}) or {}
        hf = profile.get("fertility_entropy", {}) or {}
        spectral = profile.get("spectral_analysis", {}) or {}
        stability = profile.get("embedding_stability", {}) or {}
        crosslingual = profile.get("cross_lingual", {}) or {}
        spri = profile.get("switch_point_retention", {}) or {}

        delta_mi = info.get("mutual_information_proxy_delta")
        hf_mean = hf.get("mean_fertility_entropy_after")
        crd_val = crosslingual.get("cross_lingual_representation_distance")
        stability_mean = stability.get("mean_similarity")
        spectral_entropy_val = spectral.get("spectral_entropy")
        spri_val = spri.get("spri_corpus_level")

        # Rough normalization ranges (documented assumptions, not universal
        # truths — recalibrate for your corpus if these don't fit):
        #   delta_mi_proxy ∈ [-0.05, 0.05]  (typical TF-IDF MI deltas are small)
        #   H_F            ∈ [0, log 2]      (theoretical bound, exact)
        #   CRD            ∈ [0, 2]          (cosine-similarity-scale L2 distance)
        #   stability      ∈ [-1, 1]         (cosine similarity, exact bound)
        #   spectral_entropy ∈ [0, log2(emb_dim)] if available, else [0, 10]
        #   SPRI           ∈ [0, 1]          (exact bound)
        log2 = math.log(2)
        emb_dim = spectral.get("embedding_dim")
        spectral_entropy_hi = math.log2(emb_dim) if emb_dim else 10.0

        norm = {
            "delta_mi_proxy": self._normalize(delta_mi, -0.05, 0.05, invert=True),
            "fertility_entropy": self._normalize(hf_mean, 0.0, log2),
            "crd": self._normalize(crd_val, 0.0, 2.0, invert=True),
            "stability": self._normalize(stability_mean, -1.0, 1.0),
            "spectral_entropy": self._normalize(spectral_entropy_val, 0.0, spectral_entropy_hi),
            "spri": self._normalize(spri_val, 0.0, 1.0),
        }

        contributions = {}
        mpp = 0.0
        total_weight_used = 0.0
        for key, n_val in norm.items():
            if n_val is None:
                continue
            contributions[key] = {"normalized_value": n_val, "weight": w[key],
                                   "contribution": n_val * w[key]}
            mpp += n_val * w[key]
            total_weight_used += w[key]

        # Renormalize over only the metrics that were actually available,
        # so a missing metric doesn't silently deflate the score toward 0.
        mpp_score = mpp / total_weight_used if total_weight_used > 0 else None

        return {
            "mpp_score": mpp_score,
            "weights_used": w,
            "metrics_available": list(contributions.keys()),
            "metrics_missing": [k for k in norm if k not in contributions],
            "contributions": contributions,
            "normalization_note": (
                "Normalization ranges are documented assumptions for this "
                "module, not universal constants. Treat mpp_score as a "
                "within-corpus relative ranking signal, not an absolute "
                "cross-dataset benchmark."
            ),
        }

    def compare_profiles(
        self,
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare two previously-computed profiles (each shaped like the dict
        returned by ``evaluate_all``) and report percentage change for each
        headline metric, plus the change in composite MPP score.

        Direction-of-improvement is metric-specific (e.g. lower CRD is
        better, higher H_F is better) — this method reports raw percentage
        change only; interpreting the sign correctly requires knowing each
        metric's "better" direction, which is documented in
        ``compute_composite_score``.
        """
        def _pct_change(old: Optional[float], new: Optional[float]) -> Optional[float]:
            if old is None or new is None or old == 0:
                return None
            return float((new - old) / abs(old) * 100.0)

        b_info = baseline.get("information_theory", {}) or {}
        c_info = candidate.get("information_theory", {}) or {}
        b_hf = baseline.get("fertility_entropy", {}) or {}
        c_hf = candidate.get("fertility_entropy", {}) or {}
        b_cl = baseline.get("cross_lingual", {}) or {}
        c_cl = candidate.get("cross_lingual", {}) or {}
        b_stab = baseline.get("embedding_stability", {}) or {}
        c_stab = candidate.get("embedding_stability", {}) or {}
        b_spri = baseline.get("switch_point_retention", {}) or {}
        c_spri = candidate.get("switch_point_retention", {}) or {}

        comparison = {
            "delta_mi_proxy_pct_change": _pct_change(
                b_info.get("mutual_information_proxy_delta"),
                c_info.get("mutual_information_proxy_delta"),
            ),
            "fertility_entropy_pct_change": _pct_change(
                b_hf.get("mean_fertility_entropy_after"),
                c_hf.get("mean_fertility_entropy_after"),
            ),
            "crd_pct_change": _pct_change(
                b_cl.get("cross_lingual_representation_distance"),
                c_cl.get("cross_lingual_representation_distance"),
            ),
            "stability_pct_change": _pct_change(
                b_stab.get("mean_similarity"), c_stab.get("mean_similarity"),
            ),
            "spri_pct_change": _pct_change(
                b_spri.get("spri_corpus_level"), c_spri.get("spri_corpus_level"),
            ),
        }

        b_mpp = baseline.get("composite_score", {}).get("mpp_score")
        c_mpp = candidate.get("composite_score", {}).get("mpp_score")
        comparison["mpp_score_baseline"] = b_mpp
        comparison["mpp_score_candidate"] = c_mpp
        comparison["mpp_score_pct_change"] = _pct_change(b_mpp, c_mpp)
        comparison["candidate_better_overall"] = (
            (c_mpp > b_mpp) if (b_mpp is not None and c_mpp is not None) else None
        )

        return comparison

    # ═════════════════════════════════════════════════════════════════════════
    # ALL
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_all(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        include_proxy_sprc: bool = False,
        include_spac: Optional[bool] = None,
        as_dataclass: bool = False,
        composite_weights: Optional[Dict[str, float]] = None,
    ) -> Any:
        """
        Run the full Multilingual Preprocessing Profile.

        ``include_proxy_sprc`` is False by default because Proxy-SPRC requires
        per-token embeddings and is expensive on long datasets. Enable it
        explicitly when you have GPU resources. ``include_spac`` is a
        deprecated alias for ``include_proxy_sprc`` kept for backward
        compatibility; if both are given, ``include_proxy_sprc`` wins.

        ``as_dataclass``: when True, returns an ``AnalysisResult`` instance
        instead of a plain dict. Default is False (dict) to preserve exact
        backward compatibility with existing callers; the dataclass exists
        so structured/typed consumers can opt in without a breaking change.

        Returns (dict by default; AnalysisResult if ``as_dataclass=True``)
        with keys/fields:
          information_theory, fertility_entropy, spectral_analysis,
          embedding_stability, cross_lingual, switch_point_retention,
          composite_score [, switch_point_representation_collapse]
        """
        if include_spac is not None:
            warnings.warn(
                "include_spac is a deprecated alias for include_proxy_sprc.",
                DeprecationWarning,
                stacklevel=2,
            )
            include_proxy_sprc = include_proxy_sprc or include_spac

        def _cb(p: int, msg: str) -> None:
            if progress_callback:
                progress_callback(p, msg)

        out: Dict[str, Any] = {}

        _cb(5, "Computing information theory (task-relevant info proxy)")
        out["information_theory"] = self.evaluate_information_theory()

        _cb(20, "Computing fertility entropy (H_F)")
        out["fertility_entropy"] = self.evaluate_fertility_entropy()

        _cb(35, "Computing spectral analysis (κ(E))")
        out["spectral_analysis"] = self.evaluate_spectral_analysis()

        _cb(55, "Computing embedding stability")
        out["embedding_stability"] = self.evaluate_embedding_stability()

        _cb(70, "Computing cross-lingual representation distance (CRD)")
        out["cross_lingual"] = self.evaluate_cross_lingual_consistency()

        _cb(82, "Computing switch-point retention index (SPRI)")
        out["switch_point_retention"] = self.evaluate_switch_point_retention()

        if include_proxy_sprc:
            _cb(90, "Computing switch-point representation collapse (Proxy-SPRC)")
            out["switch_point_representation_collapse"] = (
                self.evaluate_switch_point_representation_collapse()
            )

        _cb(96, "Computing composite MPP score")
        out["composite_score"] = self.compute_composite_score(
            out, weights=composite_weights
        )

        _cb(100, "Done")

        if as_dataclass:
            return AnalysisResult(
                information_theory=out["information_theory"],
                fertility_entropy=out["fertility_entropy"],
                spectral_analysis=out["spectral_analysis"],
                embedding_stability=out["embedding_stability"],
                cross_lingual=out["cross_lingual"],
                switch_point_retention=out["switch_point_retention"],
                switch_point_representation_collapse=out.get(
                    "switch_point_representation_collapse"
                ),
                composite_score=out["composite_score"],
            )
        return out