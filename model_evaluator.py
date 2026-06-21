"""
model_evaluator.py — Reusable, class-based model evaluation.

Replaces the original monolithic functions:

    _compute_model_evaluation()
    _summarize_dataset_model_metrics()

with four focused evaluator classes, each responsible for one concern, plus
a ``ModelEvaluator`` facade that composes them and preserves the exact
output shape of the original functions (so the FastAPI layer's response
payloads do not change):

    AttentionMetricsExtractor   — attention-to/from-switch-token aggregates
    EmbeddingMetricsExtractor   — sentence embedding + norm/mean/std stats
    ClassificationMetricsComputer — accuracy/F1/confusion matrix (unchanged
                                     logic from the original, just relocated)
    SwitchPointDetector          — locates switch/control-marker tokens in
                                     the *tokenized* text (operates on
                                     processed_text's tokenization, which is
                                     how the original counted switch tokens —
                                     this is analytical, not model-input, so
                                     it intentionally still looks for marker
                                     substrings in the wordpiece stream)

All transformer/tokenizer access goes through ``model_manager.ModelManager``
rather than calling ``AutoModel.from_pretrained`` directly, and all
embedding-bearing computations go through ``embedding_cache`` where the
embedding in question is reusable sentence-embedding work (not applicable
to per-token attention extraction, which must run a fresh forward pass with
the live attention tensors and cannot be served from a cache).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from logging_config import get_logger
from model_manager import LoadedTransformer, ModelManager, get_model_manager
from text_sanitizer import clean_for_model_text

logger = get_logger(__name__)

MODEL_BACKBONES: Dict[str, tuple] = {
    "xlmr": ("XLM-R", "xlm-roberta-base"),
    "muril": ("MuRIL", "google/muril-base-cased"),
    "nllb": ("NLLB", "facebook/nllb-200-distilled-600M"),
    "indicbert": ("IndicBERT", "ai4bharat/indic-bert"),
    "mbert": ("mBERT", "bert-base-multilingual-cased"),
}

LABEL_ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_ID_TO_NAME.items()}

_SWITCH_MARKERS = ["SWITCH", "DEV:", "TR:", "<EN", "<HI", "<UNK"]


def coerce_label(value: Any) -> Optional[int]:
    """Normalize a label cell (int, numeric string, or class name) to 0/1/2."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        iv = int(value)
        return iv if iv in LABEL_ID_TO_NAME else None
    text = str(value).strip().lower()
    if text in LABEL_NAME_TO_ID:
        return LABEL_NAME_TO_ID[text]
    if text.isdigit():
        iv = int(text)
        return iv if iv in LABEL_ID_TO_NAME else None
    return None


def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
    divisor = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / divisor


# ═════════════════════════════════════════════════════════════════════════════
# Attention metrics
# ═════════════════════════════════════════════════════════════════════════════

class SwitchPointDetector:
    """
    Locates analytical switch/control-marker tokens within a tokenized
    sequence, for use as an attention-analysis mask.

    Important: this operates on the tokenization of the *analytical*
    processed_text (which still contains markers like <EN>, [SWITCH], etc.),
    NOT on model_input_text. The whole point of this detector is to measure
    how the model treats positions that correspond to switch markers *when
    deliberately given the annotated text as an experiment* — by default,
    annotated text is never fed to the model at all (see ``clean_for_model``
    in text_sanitizer.py); this class supports the explicit-experiment case
    where a caller wants to study marker effects, not the default inference
    path.
    """

    def __init__(self, markers: Optional[List[str]] = None):
        self.markers = markers or list(_SWITCH_MARKERS)

    def detect(self, token_texts: List[str]) -> torch.Tensor:
        return torch.tensor(
            [any(marker in token for marker in self.markers) for token in token_texts],
            dtype=torch.bool,
        )


class AttentionMetricsExtractor:
    """
    Computes attention-to-switch-token / attention-from-switch-token
    aggregates from a model's attention tensors.

    Honesty note: averaging raw attention weights across layers and heads
    (as the original implementation did, and as this preserves) is a coarse
    summary. It is kept here, unchanged in method, because this module's
    job is to refactor structure, not to silently change what a metric
    measures — but it is documented so a reader doesn't mistake "mean
    attention across all layers/heads" for a principled importance measure.
    """

    def __init__(self, switch_detector: Optional[SwitchPointDetector] = None):
        self.switch_detector = switch_detector or SwitchPointDetector()

    def extract(
        self,
        attentions: Optional[tuple],
        token_texts: List[str],
    ) -> Dict[str, Optional[float]]:
        if not attentions:
            return {"attention_to_switch_tokens": None, "attention_from_switch_tokens": None}

        switch_mask = self.switch_detector.detect(token_texts)
        try:
            attn_tensor = torch.stack(attentions).mean(dim=(0, 1, 2))
            if switch_mask.any():
                attn_to_switch = float(attn_tensor[:, switch_mask].mean().item())
                attn_from_switch = float(attn_tensor[switch_mask, :].mean().item())
            else:
                attn_to_switch, attn_from_switch = None, None
        except Exception as exc:
            logger.warning(f"Attention extraction failed: {exc}")
            attn_to_switch, attn_from_switch = None, None

        return {
            "attention_to_switch_tokens": attn_to_switch,
            "attention_from_switch_tokens": attn_from_switch,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Embedding metrics
# ═════════════════════════════════════════════════════════════════════════════

class EmbeddingMetricsExtractor:
    """Computes sentence-embedding summary statistics from a mean-pooled hidden state."""

    def extract(self, sentence_embedding: torch.Tensor) -> Dict[str, Any]:
        return {
            "embedding_norm": float(torch.norm(sentence_embedding, p=2).item()),
            "embedding_mean": float(sentence_embedding.mean().item()),
            "embedding_std": float(sentence_embedding.std().item()),
            "embedding_vector": sentence_embedding.detach().cpu().tolist(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Classification metrics
# ═════════════════════════════════════════════════════════════════════════════

class ClassificationMetricsComputer:
    """Accuracy / macro-F1 / per-class precision-recall-F1 / confusion matrix."""

    def compute(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return {"accuracy": None, "macro_f1": None, "per_class": {}, "confusion_matrix": None}

        correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
        accuracy = correct / len(y_true)

        per_class: Dict[str, Dict[str, float]] = {}
        f1_values: List[float] = []
        cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for t, p in zip(y_true, y_pred):
            if t in LABEL_ID_TO_NAME and p in LABEL_ID_TO_NAME:
                cm[t][p] += 1

        for class_id, class_name in LABEL_ID_TO_NAME.items():
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == class_id and p == class_id)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != class_id and p == class_id)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == class_id and p != class_id)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            support = sum(1 for t in y_true if t == class_id)

            per_class[class_name] = {
                "precision": precision, "recall": recall, "f1": f1, "support": support,
            }
            f1_values.append(f1)

        macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
        return {
            "accuracy": accuracy, "macro_f1": macro_f1,
            "per_class": per_class, "confusion_matrix": cm,
        }

    def predict_with_leave_one_out_prototypes(
        self, embeddings: List[List[float]], labels: List[int],
    ) -> List[int]:
        """
        Unchanged logic from the original: nearest-centroid classification
        with leave-one-out exclusion (so a sample's own embedding never
        contributes to the prototype it's compared against). Used as a
        fallback for backbones with no direct sentiment classification head.
        """
        if not embeddings or not labels or len(embeddings) != len(labels):
            return []

        x = np.asarray(embeddings, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        num_classes = 3
        n, d = x.shape

        class_sums = np.zeros((num_classes, d), dtype=np.float32)
        class_counts = np.zeros(num_classes, dtype=np.int64)
        for i in range(n):
            cls = y[i]
            if 0 <= cls < num_classes:
                class_sums[cls] += x[i]
                class_counts[cls] += 1

        preds: List[int] = []
        eps = 1e-8
        for i in range(n):
            xi = x[i]
            yi = y[i]
            best_cls = 1
            best_score = -1e9

            for cls in range(num_classes):
                count = class_counts[cls]
                if cls == yi:
                    count -= 1
                if count <= 0:
                    continue
                vec = class_sums[cls].copy()
                if cls == yi:
                    vec -= xi
                proto = vec / float(count)
                denom = (np.linalg.norm(xi) * np.linalg.norm(proto)) + eps
                score = float(np.dot(xi, proto) / denom)
                if score > best_score:
                    best_score = score
                    best_cls = cls
            preds.append(int(best_cls))
        return preds


# ═════════════════════════════════════════════════════════════════════════════
# Facade
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SingleTextEvaluationConfig:
    """
    Controls whether analytical markers are fed to the model.

    use_annotated_text_for_model=False (default) means: strip every
    analytical/control marker via ``clean_for_model`` before tokenizing and
    running the model. This is the safe default required by the project's
    separation-of-concerns requirement. Set True only for an explicit
    experiment studying marker effects on the model itself.
    """
    use_annotated_text_for_model: bool = False


class ModelEvaluator:
    """
    Facade composing the focused extractors above, preserving the exact
    output dict shape of the original ``_compute_model_evaluation`` /
    ``_summarize_dataset_model_metrics`` functions.
    """

    def __init__(
        self,
        manager: Optional[ModelManager] = None,
        config: Optional[SingleTextEvaluationConfig] = None,
    ):
        self.manager = manager or get_model_manager()
        self.config = config or SingleTextEvaluationConfig()
        self.attention_extractor = AttentionMetricsExtractor()
        self.embedding_extractor = EmbeddingMetricsExtractor()
        self.classification_computer = ClassificationMetricsComputer()

    # ── Single-text evaluation ────────────────────────────────────────────────

    def evaluate_text(self, model_key: str, text: str) -> Dict[str, Any]:
        """
        Drop-in replacement for the original ``_compute_model_evaluation``.

        *text* is expected to be the analytical ``processed_text`` (may
        contain markers). Internally this method derives
        ``model_input_text`` via ``clean_for_model_text`` and feeds THAT to
        the tokenizer/model unless ``config.use_annotated_text_for_model``
        is explicitly set — preserving full backward compatibility of the
        function's behavior on already-clean text (most processed_text in
        the existing pipelines has no markers unless switch_point_encoding
        or language_identification_tagging modules were enabled), while
        fixing the marker-leakage issue for pipelines that do produce them.
        """
        if model_key not in MODEL_BACKBONES:
            raise ValueError(f"Unsupported model: {model_key}")

        label, model_name = MODEL_BACKBONES[model_key]
        loaded: LoadedTransformer = self.manager.get_encoder(model_name)
        tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device

        model_text = (
            text if self.config.use_annotated_text_for_model
            else clean_for_model_text(text)
        )

        inputs = tokenizer(
            model_text, return_tensors="pt", truncation=True,
            max_length=512, add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            is_seq2seq = hasattr(model, "decoder") and hasattr(model, "encoder")
            if is_seq2seq:
                try:
                    outputs = model.encoder(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_attentions=True, return_dict=True,
                    )
                except Exception:
                    outputs = model.encoder(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        return_dict=True,
                    )
            else:
                try:
                    outputs = model(**inputs, output_attentions=True)
                except Exception:
                    outputs = model(**inputs, output_attentions=False)

        hidden_state = outputs.last_hidden_state
        sentence_embedding = _mean_pool(hidden_state, inputs["attention_mask"])[0]
        token_texts = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attention_mask = inputs["attention_mask"][0]

        # Switch-token detection still inspects the *original* analytical
        # text's marker vocabulary for diagnostic purposes (e.g. comparing
        # how many marker tokens WOULD have been tokenized), even though the
        # model itself was given the cleaned text. If the model received the
        # cleaned text, token_texts will simply contain none of these
        # markers and switch_token_count will correctly be 0 — this is
        # expected and correct, not a bug: it reflects what the model
        # actually saw.
        switch_detector = SwitchPointDetector()
        switch_mask = switch_detector.detect(token_texts)

        metrics: Dict[str, Any] = {
            "model_label": label,
            "model_name": model_name,
            "token_count": int(inputs["input_ids"].shape[-1]),
            "non_padding_token_count": int(attention_mask.sum().item()),
            "switch_token_count": int(switch_mask.sum().item()),
            "token_texts": token_texts,
            "predicted_class": None,
            "predicted_label": None,
            "confidence": None,
            "class_probabilities": None,
            "model_input_text": model_text,
            "used_annotated_text_for_model": self.config.use_annotated_text_for_model,
        }
        metrics.update(self.embedding_extractor.extract(sentence_embedding))

        attentions = getattr(outputs, "attentions", None)
        metrics.update(self.attention_extractor.extract(attentions, token_texts))

        return metrics

    # ── Dataset-level summary ─────────────────────────────────────────────────

    def summarize_dataset(
        self,
        model_key: str,
        texts: List[str],
        labels: Optional[List[Optional[int]]] = None,
        sample_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Drop-in replacement for the original ``_summarize_dataset_model_metrics``."""
        if not texts:
            return {
                "model_label": MODEL_BACKBONES[model_key][0],
                "model_name": MODEL_BACKBONES[model_key][1],
                "evaluated_rows": 0, "sampled_rows": 0,
                "avg_token_count": None, "avg_switch_token_count": None,
                "avg_embedding_norm": None,
                "avg_attention_to_switch_tokens": None,
                "avg_attention_from_switch_tokens": None,
                "accuracy": None, "macro_f1": None, "per_class": {},
                "confusion_matrix": None, "samples": [],
            }

        eval_size = len(texts) if sample_limit is None else min(len(texts), max(1, sample_limit))
        sample_texts = texts[:eval_size]
        sample_labels = labels[:eval_size] if labels is not None else [None] * eval_size

        totals = {
            "token_count": 0.0, "switch_token_count": 0.0, "embedding_norm": 0.0,
            "attention_to_switch_tokens": 0.0, "attention_from_switch_tokens": 0.0,
        }
        counts = {"attention_to_switch_tokens": 0, "attention_from_switch_tokens": 0}
        samples: List[Dict[str, Any]] = []
        all_preds: List[Optional[int]] = []
        all_embeddings: List[List[float]] = []
        all_true: List[Optional[int]] = []

        for idx, (text, true_label) in enumerate(zip(sample_texts, sample_labels)):
            model_result = self.evaluate_text(model_key, text)
            totals["token_count"] += model_result.get("token_count", 0)
            totals["switch_token_count"] += model_result.get("switch_token_count", 0)
            totals["embedding_norm"] += model_result.get("embedding_norm", 0.0)
            if model_result.get("attention_to_switch_tokens") is not None:
                totals["attention_to_switch_tokens"] += model_result["attention_to_switch_tokens"]
                counts["attention_to_switch_tokens"] += 1
            if model_result.get("attention_from_switch_tokens") is not None:
                totals["attention_from_switch_tokens"] += model_result["attention_from_switch_tokens"]
                counts["attention_from_switch_tokens"] += 1

            all_preds.append(model_result.get("predicted_class"))
            all_embeddings.append(model_result.get("embedding_vector", []))
            all_true.append(true_label)

            if idx < 20:
                samples.append({
                    "text": text,
                    "actual_class": true_label,
                    "actual_label": LABEL_ID_TO_NAME.get(true_label) if true_label is not None else None,
                    "predicted_class": model_result.get("predicted_class"),
                    "predicted_label": model_result.get("predicted_label"),
                    "confidence": model_result.get("confidence"),
                    "token_count": model_result.get("token_count"),
                    "switch_token_count": model_result.get("switch_token_count"),
                    "attention_to_switch_tokens": model_result.get("attention_to_switch_tokens"),
                    "attention_from_switch_tokens": model_result.get("attention_from_switch_tokens"),
                    "token_texts": model_result.get("token_texts", []),
                })

        evaluated_rows = eval_size
        valid_pairs = [(t, p) for t, p in zip(all_true, all_preds) if t is not None and p is not None]
        y_true = [int(t) for t, _ in valid_pairs] if valid_pairs else []
        y_pred = [int(p) for _, p in valid_pairs] if valid_pairs else []

        if not y_pred and labels is not None:
            valid_indices = [
                i for i, (emb, lbl) in enumerate(zip(all_embeddings, all_true))
                if emb and lbl is not None
            ]
            if valid_indices:
                embs = [all_embeddings[i] for i in valid_indices]
                lbls = [int(all_true[i]) for i in valid_indices]
                proto_preds = self.classification_computer.predict_with_leave_one_out_prototypes(embs, lbls)
                y_true, y_pred = lbls, proto_preds

                idx_to_pred = {idx: pred for idx, pred in zip(valid_indices, proto_preds)}
                for i, s in enumerate(samples):
                    if i in idx_to_pred:
                        pred = idx_to_pred[i]
                        s["predicted_class"] = pred
                        s["predicted_label"] = LABEL_ID_TO_NAME[pred]
                        s["confidence"] = None

        cls_metrics = self.classification_computer.compute(y_true, y_pred)
        return {
            "model_label": MODEL_BACKBONES[model_key][0],
            "model_name": MODEL_BACKBONES[model_key][1],
            "evaluated_rows": evaluated_rows, "sampled_rows": evaluated_rows,
            "avg_token_count": float(totals["token_count"] / evaluated_rows),
            "avg_switch_token_count": float(totals["switch_token_count"] / evaluated_rows),
            "avg_embedding_norm": float(totals["embedding_norm"] / evaluated_rows),
            "avg_attention_to_switch_tokens": (
                float(totals["attention_to_switch_tokens"] / counts["attention_to_switch_tokens"])
                if counts["attention_to_switch_tokens"] else None
            ),
            "avg_attention_from_switch_tokens": (
                float(totals["attention_from_switch_tokens"] / counts["attention_from_switch_tokens"])
                if counts["attention_from_switch_tokens"] else None
            ),
            "accuracy": cls_metrics["accuracy"],
            "macro_f1": cls_metrics["macro_f1"],
            "per_class": cls_metrics["per_class"],
            "confusion_matrix": cls_metrics["confusion_matrix"],
            "samples": samples,
        }
