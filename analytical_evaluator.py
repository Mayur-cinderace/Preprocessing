"""
analytical_evaluator.py — Stage 4 Analytical Evaluation Framework.

Implements the full Multilingual Preprocessing Profile described in:
  "Preprocessing is Not Language-Agnostic: An Information-Theoretic and
   Spectral Study of Code-Switched Text Preprocessing on Transformer Performance"

Six scores forming the Multilingual Preprocessing Profile:

Standard scores:
  ΔI   — Mutual Information Preservation (info theory)
  κ(E) — Embedding Matrix Condition Number (spectral)
  S_att — Attention Spectral Entropy (spectral, requires transformer)

Novel multilingual scores (from the paper):
  H_F   — Fertility Entropy: tokeniser balance across languages
  SPAC  — Switch-Point Attention Collapse Score
  CRD   — Cross-lingual Representation Distance

Hard requirements:
    pip install scikit-learn numpy pandas
    pip install sentence-transformers torch   # for embedding-based metrics
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import pairwise, silhouette_score

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


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    info: Dict[str, Any]
    spectral: Dict[str, Any]
    stability: Dict[str, Any]
    crosslingual: Dict[str, Any]
    fertility: Dict[str, Any]
    spac: Dict[str, Any]
    crd: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────

class AnalyticalEvaluator:
    """
    Computes the full Multilingual Preprocessing Profile over a dataset.

    All six metrics are available independently and via ``evaluate_all``.

    Parameters
    ----------
    df_original   : DataFrame with the raw text column.
    df_processed  : DataFrame with the preprocessed text column.
    text_col      : Column name for raw text.
    processed_col : Column name for preprocessed text.
    label_col     : Column name for integer class labels (0/1/2).
    batch_size    : Embedding batch size (increase for GPU).
    """

    def __init__(
        self,
        df_original: pd.DataFrame,
        df_processed: pd.DataFrame,
        text_col: str = "text",
        processed_col: str = "processed_text",
        label_col: str = "label",
        batch_size: int = 32,
    ) -> None:
        self.df_orig      = df_original.reset_index(drop=True)
        self.df_proc      = df_processed.reset_index(drop=True)
        self.text_col     = text_col
        self.proc_col     = processed_col
        self.label_col    = label_col
        self.batch_size   = batch_size
        self._embed_model = None

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        return [t for t in text.lower().split() if t]

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
    # 1. INFORMATION THEORY  (ΔI)
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_information_theory(self) -> Dict[str, Any]:
        """
        Computes:
          • Token-distribution entropy H before and after preprocessing.
          • ΔI = I(X;Y) − I(f(X);Y)  via TF-IDF mutual information.
          • Vocabulary size change.
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

        # ΔI = information discarded by preprocessing
        # Positive → preprocessing reduced task-relevant information.
        delta_mi = None
        if mi_mean is not None:
            # We approximate I(X;Y) as the TF-IDF MI on original text.
            try:
                vec_orig = TfidfVectorizer(max_features=20000)
                orig_texts = df[self.text_col].astype(str).values
                X_orig = vec_orig.fit_transform(orig_texts).toarray()
                mi_orig = mutual_info_classif(
                    X_orig, labels, discrete_features=False, random_state=0
                )
                mi_orig_mean = float(np.nanmean(mi_orig))
                delta_mi = mi_orig_mean - mi_mean  # ΔI = I(X;Y) - I(f(X);Y)
            except Exception:
                pass

        return {
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "entropy_delta": entropy_delta,
            "mutual_information_mean": mi_mean,
            "mutual_information_top_k": mi_topk,
            "mutual_information_delta": delta_mi,
            "mutual_information_method": mi_method,
            "mutual_information_error": mi_error,
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
    # 2. SPECTRAL ANALYSIS  (κ(E), S_att)
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_spectral_analysis(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> Dict[str, Any]:
        """
        Computes embedding covariance eigenspectrum.
        κ(E) = σ₁(E) / σᵣ(E)  — condition number of embedding matrix.
        S_att — spectral entropy of eigenvalue distribution.
        """
        try:
            texts = self.df_proc[self.proc_col].astype(str).tolist()
            emb = self._batch_encode(texts)
        except Exception as e:
            return {"error": f"Embedding failed: {e}"}

        # Covariance of embedding matrix (d×d).
        cov = np.cov(emb, rowvar=False)
        cov = (cov + cov.T) / 2.0
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.abs(eig))[::-1]
        total = eig.sum() if eig.sum() > 0 else 1.0
        p = eig / total

        spectral_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
        condition_number = float(eig[0] / (eig[-1] + 1e-12))
        spectral_decay = float(eig[0] / (total + 1e-12))

        # Variance explained by top-k components.
        cumulative_variance = np.cumsum(p).tolist()

        return {
            "top_k_eigenvalues": eig[:10].tolist(),
            "all_eigenvalues": eig.tolist(),
            "spectral_entropy": spectral_entropy,
            "condition_number": condition_number,
            "spectral_decay": spectral_decay,
            "cumulative_variance_top10": cumulative_variance[:10],
            "embedding_dim": int(emb.shape[1]),
            "n_samples": int(emb.shape[0]),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 3. EMBEDDING STABILITY
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_embedding_stability(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> Dict[str, Any]:
        """
        Per-sample cosine similarity between original and preprocessed embeddings.
        High mean → preprocessing preserved semantic content.
        """
        try:
            orig_texts = self.df_orig[self.text_col].astype(str).tolist()
            proc_texts = self.df_proc[self.proc_col].astype(str).tolist()
            emb_orig = self._batch_encode(orig_texts)
            emb_proc = self._batch_encode(proc_texts)
        except Exception as e:
            return {"error": f"Embedding failed: {e}"}

        sims = pairwise.cosine_similarity(emb_orig, emb_proc).diagonal()
        sims = np.clip(sims, -1.0, 1.0)
        hist, bins = np.histogram(sims, bins=20, range=(-1, 1))

        return {
            "mean_similarity": float(np.mean(sims)),
            "median_similarity": float(np.median(sims)),
            "std_similarity": float(np.std(sims)),
            "min_similarity": float(np.min(sims)),
            "max_similarity": float(np.max(sims)),
            "histogram_counts": hist.tolist(),
            "histogram_bins": bins.tolist(),
            "samples": sims.tolist()[:200],
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 4. CROSS-LINGUAL CONSISTENCY  (CRD)
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_cross_lingual_consistency(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> Dict[str, Any]:
        """
        CRD = mean L2 distance between centroid embeddings of EN and HI tokens.
        Lower CRD → preprocessing aligned language representation spaces.

        Also computes centroid cosine similarity and silhouette score.
        """
        texts = self.df_proc[self.proc_col].astype(str).tolist()
        try:
            emb = self._batch_encode(texts)
        except Exception as e:
            return {"error": f"Embedding failed: {e}"}

        src_texts = self.df_orig[self.text_col].astype(str).tolist()
        masks_hi = np.array([_contains_devanagari(t) for t in src_texts])
        masks_en = ~masks_hi

        emb_en = emb[masks_en] if masks_en.any() else np.empty((0, emb.shape[1]))
        emb_hi = emb[masks_hi] if masks_hi.any() else np.empty((0, emb.shape[1]))

        result: Dict[str, Any] = {
            "alignment_count_en": int(emb_en.shape[0]),
            "alignment_count_hi": int(emb_hi.shape[0]),
        }

        if emb_en.shape[0] > 0 and emb_hi.shape[0] > 0:
            centroid_en = emb_en.mean(axis=0)
            centroid_hi = emb_hi.mean(axis=0)

            # CRD: L2 distance between language centroids.
            crd = float(np.linalg.norm(centroid_en - centroid_hi))

            # Centroid cosine similarity.
            cosine = float(
                pairwise.cosine_similarity(
                    centroid_en.reshape(1, -1), centroid_hi.reshape(1, -1)
                )[0, 0]
            )

            result["cross_lingual_representation_distance"] = crd
            result["cross_lingual_centroid_cosine"] = cosine

            # Silhouette score as cluster cohesion proxy.
            try:
                lang_labels = np.zeros(len(texts), dtype=int)
                lang_labels[masks_hi] = 1
                sil = float(silhouette_score(emb, lang_labels))
                result["silhouette_score"] = sil
            except Exception:
                result["silhouette_score"] = None

            # Within-language cohesion (mean intra-class cosine similarity).
            if emb_en.shape[0] > 1:
                cos_en = pairwise.cosine_similarity(emb_en)
                result["en_intra_cohesion"] = float(
                    cos_en[np.triu_indices(len(emb_en), k=1)].mean()
                )
            if emb_hi.shape[0] > 1:
                cos_hi = pairwise.cosine_similarity(emb_hi)
                result["hi_intra_cohesion"] = float(
                    cos_hi[np.triu_indices(len(emb_hi), k=1)].mean()
                )
        else:
            result["note"] = "Not enough cross-lingual samples for centroid comparison."

        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 5. FERTILITY ENTROPY  (H_F)  ← Novel score from the paper
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_fertility_entropy(self) -> Dict[str, Any]:
        """
        H_F(s) = −∑_l p_l(s) · log p_l(s)
        where p_l(s) = n_l(s) / ∑ n_{l'}(s)

        Measures how balanced the tokeniser's subword allocation is across
        the two languages in a CS sentence.  H_F ∈ [0, log 2].

        Here we approximate token-language assignment by:
          • Devanagari tokens → HI
          • Tokens in HINGLISH_LEXICON → HI
          • Tokens in ENGLISH_VOCAB (heuristic) → EN
          • Others → UNK

        A high mean H_F (close to log 2) means both languages get equal
        token share — ideal.  Low H_F indicates one language dominates.
        """
        try:
            from _base import HINGLISH_LEXICON, ENGLISH_VOCAB
        except ImportError:
            # Minimal fallbacks so the metric still runs standalone.
            HINGLISH_LEXICON = frozenset()
            ENGLISH_VOCAB = frozenset()

        log2 = math.log(2)  # maximum H_F for two languages

        def _assign_lang(tok: str) -> str:
            t = tok.lower()
            if _contains_devanagari(t):
                return "HI"
            if t in HINGLISH_LEXICON:
                return "HI"
            if t in ENGLISH_VOCAB:
                return "EN"
            # phonetic clue heuristic
            phonetic = ("aa", "ai", "au", "bh", "dh", "kh", "ph", "sh", "th",
                        "ch", "gh", "jh", "ky", "py", "ny", "ri", "ya")
            if t.isalpha() and any(c in t for c in phonetic):
                return "HI"
            return "UNK"

        def _sentence_fertility_entropy(text: str) -> Optional[float]:
            toks = text.split()
            if not toks:
                return None
            counts: Counter = Counter()
            for t in toks:
                counts[_assign_lang(t)] += 1
            total = sum(counts.values())
            # Only compute over EN + HI
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

        # Per-token language distributions
        all_langs_orig: Counter = Counter()
        all_langs_proc: Counter = Counter()
        for t in orig_texts:
            for tok in t.split():
                all_langs_orig[_assign_lang(tok)] += 1
        for t in proc_texts:
            for tok in t.split():
                all_langs_proc[_assign_lang(tok)] += 1

        result: Dict[str, Any] = {
            "max_fertility_entropy": float(log2),
            "mean_fertility_entropy_before": float(np.mean(hf_orig)) if hf_orig else None,
            "mean_fertility_entropy_after": float(np.mean(hf_proc)) if hf_proc else None,
            "std_fertility_entropy_before": float(np.std(hf_orig)) if hf_orig else None,
            "std_fertility_entropy_after": float(np.std(hf_proc)) if hf_proc else None,
            "histogram_before": np.histogram(hf_orig, bins=10, range=(0.0, log2))[0].tolist() if hf_orig else [],
            "histogram_after": np.histogram(hf_proc, bins=10, range=(0.0, log2))[0].tolist() if hf_proc else [],
            "token_lang_dist_before": dict(all_langs_orig),
            "token_lang_dist_after": dict(all_langs_proc),
        }

        # Δ H_F — improvement in balance
        if result["mean_fertility_entropy_before"] and result["mean_fertility_entropy_after"]:
            result["delta_fertility_entropy"] = (
                result["mean_fertility_entropy_after"]
                - result["mean_fertility_entropy_before"]
            )
        else:
            result["delta_fertility_entropy"] = None

        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 6. SWITCH-POINT ATTENTION COLLAPSE (SPAC)  ← Novel score from the paper
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_switch_point_attention_collapse(self) -> Dict[str, Any]:
        """
        SPAC(l) = H̄_att(SP) − H̄_att(non-SP)

        Here we approximate attention entropy via embedding cosine-similarity
        row entropy (without a full transformer forward pass).

        For each sentence:
          1. Find switch-point positions (lang(i) ≠ lang(i-1)).
          2. Compute token-level 'attention entropy' as the normalized cosine
             similarity row entropy against other tokens.
          3. SPAC = mean entropy at switch positions − mean entropy elsewhere.

        A strongly negative SPAC means the model attends diffusely (collapses)
        at language boundaries.
        """
        try:
            from _base import HINGLISH_LEXICON, ENGLISH_VOCAB
        except ImportError:
            HINGLISH_LEXICON = frozenset()
            ENGLISH_VOCAB = frozenset()

        def _lang(tok: str) -> str:
            t = tok.lower()
            if _contains_devanagari(t):
                return "HI"
            if t in HINGLISH_LEXICON:
                return "HI"
            if t in ENGLISH_VOCAB:
                return "EN"
            return "UNK"

        def _switch_points(toks: List[str]) -> List[bool]:
            langs = [_lang(t) for t in toks]
            sp = [False] * len(langs)
            for i in range(1, len(langs)):
                if langs[i] != langs[i - 1] and langs[i] != "UNK" and langs[i - 1] != "UNK":
                    sp[i] = True
            return sp

        # We need embeddings at token level. We use sentence-level embeddings
        # as a proxy: embed each token independently.
        try:
            proc_texts = self.df_proc[self.proc_col].astype(str).tolist()
        except Exception as e:
            return {"error": str(e)}

        sp_entropies: List[float] = []
        non_sp_entropies: List[float] = []
        total_switch_points = 0
        total_sentences_with_switch = 0

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
                # Cosine-similarity matrix (T×T).
                sim = pairwise.cosine_similarity(tok_embs)
                # Row entropy: how uniformly each token attends to others.
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
            "total_switch_points": total_switch_points,
            "sentences_with_switch_points": total_sentences_with_switch,
            "mean_entropy_at_switch_points": float(np.mean(sp_entropies)) if sp_entropies else None,
            "mean_entropy_at_non_switch_points": float(np.mean(non_sp_entropies)) if non_sp_entropies else None,
        }

        if sp_entropies and non_sp_entropies:
            spac = float(np.mean(sp_entropies)) - float(np.mean(non_sp_entropies))
            result["spac_score"] = spac
            result["interpretation"] = (
                "negative — switch-point attention collapse detected"
                if spac < 0 else
                "positive — no collapse; switch points attend more broadly"
            )
        else:
            result["spac_score"] = None
            result["interpretation"] = "Insufficient switch-point data."

        return result

    # ═════════════════════════════════════════════════════════════════════════
    # ALL
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_all(
        self,
        progress_callback: Optional[callable] = None,
        include_spac: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full Multilingual Preprocessing Profile.

        ``include_spac`` is False by default because SPAC requires per-token
        embeddings and is expensive on long datasets.  Enable it explicitly
        when you have GPU resources.

        Returns a JSON-serializable dict with keys:
          information_theory, spectral_analysis, embedding_stability,
          cross_lingual, fertility_entropy[, switch_point_attention_collapse]
        """
        def _cb(p: int, msg: str) -> None:
            if progress_callback:
                progress_callback(p, msg)

        out: Dict[str, Any] = {}

        _cb(5, "Computing information theory (ΔI, entropy)")
        out["information_theory"] = self.evaluate_information_theory()

        _cb(25, "Computing fertility entropy (H_F)")
        out["fertility_entropy"] = self.evaluate_fertility_entropy()

        _cb(40, "Computing spectral analysis (κ(E))")
        out["spectral_analysis"] = self.evaluate_spectral_analysis()

        _cb(65, "Computing embedding stability")
        out["embedding_stability"] = self.evaluate_embedding_stability()

        _cb(85, "Computing cross-lingual representation distance (CRD)")
        out["cross_lingual"] = self.evaluate_cross_lingual_consistency()

        if include_spac:
            _cb(92, "Computing switch-point attention collapse (SPAC)")
            out["switch_point_attention_collapse"] = (
                self.evaluate_switch_point_attention_collapse()
            )

        _cb(100, "Done")
        return out
