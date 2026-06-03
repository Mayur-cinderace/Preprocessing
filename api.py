from __future__ import annotations

import json
import os
import threading
import time
import uuid
import tempfile
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import mlflow
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from combined_preprocessing_pipeline import build_pipeline
from analytical_evaluator import AnalyticalEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
JOBS_DIR = ROOT / "analysis_jobs"
JOBS_DIR.mkdir(exist_ok=True)
MLFLOW_DB = ROOT.parent / "mlruns.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.resolve().as_posix()}")
mlflow.set_experiment("hinglish_preprocessing_analyses")

app = FastAPI(title="Hinglish Preprocessing Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=str(ROOT)), name="static")

# In-memory job registry (mirrors JOBS_DIR files)
jobs: Dict[str, Dict[str, Any]] = {}


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    try:
        if isinstance(obj, dict):
            return {k: _convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, torch.Tensor):
            return _convert_numpy_types(obj.detach().cpu().numpy())
        return obj
    except Exception as e:
        # If conversion fails for a specific value, return string representation
        return str(obj)


MODEL_BACKBONES: dict[str, tuple[str, str]] = {
    "xlmr": ("XLM-R", "xlm-roberta-base"),
    "muril": ("MuRIL", "google/muril-base-cased"),
    "nllb": ("NLLB", "facebook/nllb-200-distilled-600M"),
    "indicbert": ("IndicBERT", "ai4bharat/indic-bert"),
    "mbert": ("mBERT", "bert-base-multilingual-cased"),
    "custom": ("Custom CodeSwitched Transformer", "local://artifacts/custom_transformer"),
}
MODEL_CACHE: dict[str, Dict[str, Any]] = {}


def _load_transformer_resources(model_name: str) -> Dict[str, Any]:
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    # Handle custom local model
    if model_name.startswith("local://"):
        raise ValueError(
            f"Custom local model '{model_name}' requires CodeSwitchedSentimentTransformer "
            "and CodeSwitchedTokenizer classes.  Place architecture.py and tokenizer.py "
            "alongside api.py and re-run."
        )

    # Handle HuggingFace models
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    MODEL_CACHE[model_name] = {
        "tokenizer": tokenizer,
        "model": model,
        "is_custom": False,
    }
    return MODEL_CACHE[model_name]


def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
    divisor = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / divisor


LABEL_ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAME_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


def _coerce_label(value: Any) -> Optional[int]:
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


def _compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {
            "accuracy": None,
            "macro_f1": None,
            "per_class": {},
            "confusion_matrix": None,
        }

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
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _predict_with_leave_one_out_prototypes(
    embeddings: List[List[float]],
    labels: List[int],
) -> List[int]:
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


def _compute_model_evaluation(model_key: str, text: str) -> Dict[str, Any]:
    if model_key not in MODEL_BACKBONES:
        raise ValueError(f"Unsupported model: {model_key}")

    label, model_name = MODEL_BACKBONES[model_key]
    resources = _load_transformer_resources(model_name)
    tokenizer = resources["tokenizer"]
    model = resources["model"]
    is_custom = resources.get("is_custom", False)

    # Handle custom model differently
    if is_custom:
        encoded = tokenizer.batch_encode(
            [text],
            add_special_tokens=True,
            return_switch_mask=True,
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        lang_ids = encoded.lang_ids
        switch_mask = encoded.switch_mask if hasattr(encoded, 'switch_mask') else None

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                lang_ids=lang_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        hidden_state = outputs.get("hidden")
        pooled = outputs.get("pooled")
        lang_logits = outputs.get("lang_logits")
        sentiment_logits = outputs.get("logits")
        sentence_embedding = pooled[0] if pooled is not None else hidden_state[0, :].mean(dim=0)

        # Token reconstruction from input_ids (approximate)
        token_texts = [f"<token_{i}>" for i in range(input_ids.shape[1])]

        # Compute sentiment prediction and confidence
        predicted_class = None
        confidence = None
        class_probabilities = None
        
        if sentiment_logits is not None:
            probs = torch.softmax(sentiment_logits, dim=-1)[0]  # (num_classes,)
            predicted_class = int(probs.argmax().item())
            confidence = float(probs.max().item())
            class_probabilities = {
                "negative": float(probs[0].item()),
                "neutral": float(probs[1].item()),
                "positive": float(probs[2].item()),
            }

        # Compute attention-like metrics from language logits
        attn_to_switch = None
        attn_from_switch = None
        
        if lang_logits is not None and switch_mask is not None:
            try:
                # Get language predictions for each token
                lang_probs = torch.softmax(lang_logits[0], dim=-1)  # (T, 3)
                
                # Switch tokens are those with high entropy across languages (uncertain language)
                entropy = -(lang_probs * torch.log(lang_probs + 1e-10)).sum(dim=-1)  # (T,)
                switch_token_pred = entropy > entropy.median()
                
                # Compute how much attention flows to/from predicted switch tokens
                if switch_token_pred.any():
                    attn_to_switch = float(entropy[switch_token_pred].mean().item())
                    attn_from_switch = float(entropy[switch_token_pred].mean().item())
            except Exception:
                pass

        metrics: Dict[str, Any] = {
            "model_label": label,
            "model_name": model_name,
            "token_count": int(input_ids.shape[-1]),
            "non_padding_token_count": int(attention_mask.sum().item()),
            "switch_token_count": int(switch_mask.sum().item()) if switch_mask is not None else 0,
            "embedding_norm": float(torch.norm(sentence_embedding, p=2).item()),
            "embedding_mean": float(sentence_embedding.mean().item()),
            "embedding_std": float(sentence_embedding.std().item()),
            "token_texts": token_texts,
            "attention_to_switch_tokens": attn_to_switch,
            "attention_from_switch_tokens": attn_from_switch,
            "predicted_class": predicted_class,
            "predicted_label": LABEL_ID_TO_NAME[predicted_class] if predicted_class is not None else None,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "embedding_vector": sentence_embedding.detach().cpu().tolist(),
        }
        return metrics

    # Original HuggingFace model flow
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    )

    with torch.no_grad():
        # Check if model is seq2seq (like NLLB, M2M100)
        # These models need special handling
        is_seq2seq = hasattr(model, 'decoder') and hasattr(model, 'encoder')
        
        if is_seq2seq:
            # For seq2seq models, use encoder-only outputs
            try:
                outputs = model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_attentions=True,
                    return_dict=True
                )
            except Exception:
                # Fallback: try without output_attentions
                outputs = model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True
                )
        else:
            # For standard encoder models
            try:
                outputs = model(**inputs, output_attentions=True)
            except Exception:
                # Fallback: try without output_attentions
                outputs = model(**inputs, output_attentions=False)

    hidden_state = outputs.last_hidden_state
    sentence_embedding = _mean_pool(hidden_state, inputs["attention_mask"])[0]
    token_texts = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attention_mask = inputs["attention_mask"][0]
    switch_markers = ["SWITCH", "DEV:", "TR:", "<EN", "<HI", "<UNK"]
    switch_mask = torch.tensor([
        any(marker in token for marker in switch_markers)
        for token in token_texts
    ], dtype=torch.bool)

    metrics: Dict[str, Any] = {
        "model_label": label,
        "model_name": model_name,
        "token_count": int(inputs["input_ids"].shape[-1]),
        "non_padding_token_count": int(attention_mask.sum().item()),
        "switch_token_count": int(switch_mask.sum().item()),
        "embedding_norm": float(torch.norm(sentence_embedding, p=2).item()),
        "embedding_mean": float(sentence_embedding.mean().item()),
        "embedding_std": float(sentence_embedding.std().item()),
        "token_texts": token_texts,
        "predicted_class": None,
        "predicted_label": None,
        "confidence": None,
        "class_probabilities": None,
        "embedding_vector": sentence_embedding.detach().cpu().tolist(),
    }

    try:
        attentions = outputs.attentions
        if attentions:
            attn_tensor = torch.stack(attentions).mean(dim=(0, 1, 2))
            if switch_mask.any():
                attn_to_switch = float(attn_tensor[:, switch_mask].mean().item())
                attn_from_switch = float(attn_tensor[switch_mask, :].mean().item())
            else:
                attn_to_switch = None
                attn_from_switch = None
            metrics["attention_to_switch_tokens"] = attn_to_switch
            metrics["attention_from_switch_tokens"] = attn_from_switch
    except Exception:
        metrics["attention_to_switch_tokens"] = None
        metrics["attention_from_switch_tokens"] = None

    return metrics


def _summarize_dataset_model_metrics(
    model_key: str,
    texts: List[str],
    labels: Optional[List[Optional[int]]] = None,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    if not texts:
        return {
            "model_label": MODEL_BACKBONES[model_key][0],
            "model_name": MODEL_BACKBONES[model_key][1],
            "evaluated_rows": 0,
            "sampled_rows": 0,
            "avg_token_count": None,
            "avg_switch_token_count": None,
            "avg_embedding_norm": None,
            "avg_attention_to_switch_tokens": None,
            "avg_attention_from_switch_tokens": None,
            "accuracy": None,
            "macro_f1": None,
            "per_class": {},
            "confusion_matrix": None,
            "samples": [],
        }

    if sample_limit is None:
        eval_size = len(texts)
    else:
        eval_size = min(len(texts), max(1, sample_limit))

    sample_texts = texts[:eval_size]
    sample_labels = labels[:eval_size] if labels is not None else [None] * eval_size
    totals: Dict[str, float] = {
        "token_count": 0.0,
        "switch_token_count": 0.0,
        "embedding_norm": 0.0,
        "attention_to_switch_tokens": 0.0,
        "attention_from_switch_tokens": 0.0,
    }
    counts: Dict[str, int] = {
        "attention_to_switch_tokens": 0,
        "attention_from_switch_tokens": 0,
    }
    samples: list[Dict[str, Any]] = []
    all_preds: List[Optional[int]] = []
    all_embeddings: List[List[float]] = []
    all_true: List[Optional[int]] = []

    for idx, (text, true_label) in enumerate(zip(sample_texts, sample_labels)):
        model_result = _compute_model_evaluation(model_key, text)
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
    if valid_pairs:
        y_true = [int(t) for t, _ in valid_pairs]
        y_pred = [int(p) for _, p in valid_pairs]
    else:
        y_true = []
        y_pred = []

    # For backbones without direct sentiment logits, derive predictions from embeddings + labels
    if not y_pred and labels is not None:
        valid_indices = [
            i for i, (emb, lbl) in enumerate(zip(all_embeddings, all_true))
            if emb and lbl is not None
        ]
        if valid_indices:
            embs = [all_embeddings[i] for i in valid_indices]
            lbls = [int(all_true[i]) for i in valid_indices]
            proto_preds = _predict_with_leave_one_out_prototypes(embs, lbls)
            y_true = lbls
            y_pred = proto_preds

            idx_to_pred = {idx: pred for idx, pred in zip(valid_indices, proto_preds)}
            for i, s in enumerate(samples):
                if i in idx_to_pred:
                    pred = idx_to_pred[i]
                    s["predicted_class"] = pred
                    s["predicted_label"] = LABEL_ID_TO_NAME[pred]
                    s["confidence"] = None

    cls_metrics = _compute_classification_metrics(y_true, y_pred)
    return {
        "model_label": MODEL_BACKBONES[model_key][0],
        "model_name": MODEL_BACKBONES[model_key][1],
        "evaluated_rows": evaluated_rows,
        "sampled_rows": evaluated_rows,
        "avg_token_count": float(totals["token_count"] / evaluated_rows),
        "avg_switch_token_count": float(totals["switch_token_count"] / evaluated_rows),
        "avg_embedding_norm": float(totals["embedding_norm"] / evaluated_rows),
        "avg_attention_to_switch_tokens": float(totals["attention_to_switch_tokens"] / counts["attention_to_switch_tokens"]) if counts["attention_to_switch_tokens"] else None,
        "avg_attention_from_switch_tokens": float(totals["attention_from_switch_tokens"] / counts["attention_from_switch_tokens"]) if counts["attention_from_switch_tokens"] else None,
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "per_class": cls_metrics["per_class"],
        "confusion_matrix": cls_metrics["confusion_matrix"],
        "samples": samples,
    }


def _run_model_inference_job(job_id: str, csv_path: Path, modules: list[str], models: list[str], batch_size: int):
    dataset_name = csv_path.stem
    upload_name = csv_path.name
    pipeline = None
    processed_df = None
    results: Dict[str, Any] = {}
    run_started = False
    try:
        logger.info(f"Starting model inference job {job_id} with modules: {modules}, models: {models}")
        mlflow.start_run(run_name=f"model_inference_{dataset_name}_{job_id}")
        run_started = True
        mlflow.set_tag("job_id", job_id)
        mlflow.set_tag("status", "running")
        mlflow.set_tag("retention", "local_file_store")
        _update_progress(job_id, 1, "Loading CSV")
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        logger.info(f"Loaded {len(df)} rows from {dataset_name}")

        required = ["text", "sentiment", "label"]
        for r in required:
            if r not in df.columns:
                raise RuntimeError(f"Missing required column: {r}")

        _update_progress(job_id, 5, "Building pipeline")
        kwargs = {k: True for k in modules}
        augmentation_modules = {"code_switch_augmentation"}
        effective_modules = [m for m in modules if m not in augmentation_modules]
        if len(effective_modules) < len(modules):
            excluded = [m for m in modules if m in augmentation_modules]
            logger.info(f"Filtered out augmentation modules for inference: {excluded}")
        kwargs = {k: True for k in effective_modules}
        pipeline = build_pipeline(**kwargs) if kwargs else None
        mlflow.log_param("active_modules", ",".join(pipeline.active_modules) if pipeline and pipeline.active_modules else "(none)")
        mlflow.log_dict({"pipeline_config": pipeline.get_pipeline_config()} if pipeline else {}, "metadata/pipeline_config.json")

        _update_progress(job_id, 20, "Preprocessing dataset")
        processed_df = pipeline.process_dataframe(df, text_col="text", output_col="processed_text") if pipeline else df.copy()
        logger.info(f"Preprocessing complete: {len(processed_df)} rows")

        processed_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        try:
            processed_df.to_csv(processed_tmp.name, index=False, encoding="utf-8")
            processed_tmp.close()
            mlflow.log_artifact(processed_tmp.name, artifact_path="processed")
        finally:
            try:
                Path(processed_tmp.name).unlink(missing_ok=True)
            except Exception:
                pass

        _update_progress(job_id, 40, "Computing model summaries")
        # Use processed_text if it exists, otherwise use original text
        text_col = "processed_text" if "processed_text" in processed_df.columns else "text"
        texts = processed_df[text_col].astype(str).tolist()
        raw_labels = processed_df["label"].tolist() if "label" in processed_df.columns else [None] * len(processed_df)
        if "sentiment" in processed_df.columns:
            sentiment_labels = processed_df["sentiment"].tolist()
            labels = [
                _coerce_label(lbl) if _coerce_label(lbl) is not None else _coerce_label(sent)
                for lbl, sent in zip(raw_labels, sentiment_labels)
            ]
        else:
            labels = [_coerce_label(lbl) for lbl in raw_labels]

        model_summaries: Dict[str, Any] = {}
        for model_key in models:
            try:
                model_summaries[model_key] = _summarize_dataset_model_metrics(
                    model_key=model_key,
                    texts=texts,
                    labels=labels,
                    sample_limit=None,  # full dataset, not 100-row cap
                )
            except Exception as e:
                logger.error(f"Failed to compute metrics for model {model_key}: {e}")
                # Return partial results for this model
                model_summaries[model_key] = {
                    "model_label": MODEL_BACKBONES.get(model_key, ("Unknown", "unknown"))[0],
                    "model_name": MODEL_BACKBONES.get(model_key, ("Unknown", "unknown"))[1],
                    "error": str(e),
                    "evaluated_rows": 0,
                    "sampled_rows": 0,
                }
        pipeline_metadata = {
            "requested_modules": modules,
            "active_modules": pipeline.active_modules if pipeline else [],
            "pipeline_config": pipeline.get_pipeline_config() if pipeline else {}
        }

        results = {
            "job_id": job_id,
            "dataset_name": dataset_name,

            "preprocessing": pipeline_metadata,

            "processed_rows": len(processed_df),
            "labelled_rows": int(sum(1 for l in labels if l is not None)),
            "models": model_summaries,
            "sample_processed_texts": processed_df[text_col].astype(str).tolist()[:5],
        }

        jobs[job_id]["status"] = "done"
        jobs[job_id]["results"] = results
        jobs[job_id]["dataset_name"] = dataset_name
        jobs[job_id]["uploaded_filename"] = upload_name
        jobs[job_id]["batch_size"] = batch_size
        jobs[job_id]["modules"] = modules
        jobs[job_id]["models"] = models
        jobs[job_id]["pipeline_config"] = pipeline.get_pipeline_config() if pipeline is not None else None
        if mlflow.active_run() is not None:
            jobs[job_id]["mlflow_run_id"] = mlflow.active_run().info.run_id
            jobs[job_id]["mlflow_experiment_id"] = mlflow.active_run().info.experiment_id

        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("uploaded_filename", upload_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("models", ",".join(models) if models else "(none)")
        mlflow.log_dict(results, "results/model_inference.json")
        flattened = _flatten_numeric_metrics("model_inference", results)
        for metric_name, metric_value in flattened.items():
            try:
                mlflow.log_metric(metric_name, metric_value)
            except Exception:
                pass

        _save_job(job_id, jobs[job_id])
    except Exception as e:
        logger.error(f"Model inference job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["dataset_name"] = dataset_name
        jobs[job_id]["uploaded_filename"] = upload_name
        jobs[job_id]["batch_size"] = batch_size
        jobs[job_id]["modules"] = modules
        jobs[job_id]["models"] = models
        jobs[job_id]["pipeline_config"] = pipeline.get_pipeline_config() if pipeline is not None else None
        if mlflow.active_run() is not None:
            jobs[job_id]["mlflow_run_id"] = mlflow.active_run().info.run_id
            jobs[job_id]["mlflow_experiment_id"] = mlflow.active_run().info.experiment_id
        _save_job(job_id, jobs[job_id])
    finally:
        if run_started and mlflow.active_run() is not None:
            mlflow.end_run()


def _save_job(job_id: str, payload: Dict[str, Any]):
    path = JOBS_DIR / f"{job_id}.json"
    # Convert numpy types before serializing to JSON
    payload = _convert_numpy_types(payload)
    try:
        json_str = json.dumps(payload, default=str)
        path.write_text(json_str, encoding="utf-8")
        logger.info(f"Job {job_id} saved successfully ({len(json_str)} bytes)")
    except Exception as e:
        logger.error(f"Failed to save job {job_id}: {e}", exc_info=True)
        # If serialization fails, save error info instead of leaving file empty
        error_payload = {
            "status": "error",
            "error": f"Failed to save job: {str(e)}",
            "job_id": job_id,
        }
        try:
            path.write_text(json.dumps(error_payload), encoding="utf-8")
        except Exception:
            # Last resort: create a minimal valid JSON file
            path.write_text('{"status":"error","error":"Job save failed"}', encoding="utf-8")


def _flatten_numeric_metrics(prefix: str, payload: Dict[str, Any]):
    metrics: Dict[str, float] = {}
    for key, value in payload.items():
        metric_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(_flatten_numeric_metrics(metric_name, value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[metric_name] = float(value)
    return metrics


def _log_mlflow_run(job_id: str, csv_path: Path, dataset_name: str, upload_name: str, modules: list[str], batch_size: int, pipeline=None, processed: pd.DataFrame | None = None, results: Dict[str, Any] | None = None, status: str = "done", error: str | None = None):
    if mlflow.active_run() is None:
        return

    payload = jobs.get(job_id, {}).copy()
    payload.setdefault("status", status)
    payload.setdefault("dataset_name", dataset_name)
    payload.setdefault("uploaded_filename", upload_name)
    payload.setdefault("modules", modules)
    payload.setdefault("batch_size", batch_size)
    if pipeline is not None:
        payload.setdefault("pipeline_config", pipeline.get_pipeline_config())

    mlflow.set_tag("job_id", job_id)
    mlflow.set_tag("dataset_name", dataset_name)
    mlflow.set_tag("uploaded_filename", upload_name)
    mlflow.set_tag("status", status)
    mlflow.set_tag("retention", "local_file_store")

    mlflow.log_param("dataset_name", dataset_name)
    mlflow.log_param("uploaded_filename", upload_name)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("modules_enabled", ",".join(modules) if modules else "(none)")
    mlflow.log_param("active_modules", ",".join(pipeline.active_modules) if pipeline and pipeline.active_modules else "(none)")

    mlflow.log_dict({
        "job_id": job_id,
        "dataset_name": dataset_name,
        "uploaded_filename": upload_name,
        "modules": modules,
        "batch_size": batch_size,
        "pipeline_config": pipeline.get_pipeline_config() if pipeline is not None else None,
    }, "metadata/request.json")
    mlflow.log_dict(payload, "metadata/job.json")
    mlflow.log_dict({"status": status, "error": error}, "metadata/status.json")

    mlflow.log_artifact(str(csv_path), artifact_path="inputs")

    if processed is not None:
        processed_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        try:
            processed.to_csv(processed_tmp.name, index=False, encoding="utf-8")
            processed_tmp.close()
            mlflow.log_artifact(processed_tmp.name, artifact_path="artifacts")
        finally:
            try:
                Path(processed_tmp.name).unlink(missing_ok=True)
            except Exception:
                pass

    if results is not None:
        mlflow.log_dict(results, "results/results.json")
        metrics = _flatten_numeric_metrics("results", results)
        for metric_name, metric_value in metrics.items():
            try:
                mlflow.log_metric(metric_name, metric_value)
            except Exception:
                pass


def _update_progress(job_id: str, percent: int, message: str):
    jobs.setdefault(job_id, {})
    jobs[job_id]["progress"] = percent
    jobs[job_id]["message"] = message
    _save_job(job_id, jobs[job_id])


def _run_analysis_job(job_id: str, csv_path: Path, modules: list[str], batch_size: int):
    dataset_name = csv_path.stem
    upload_name = csv_path.name
    pipeline = None
    processed = None
    results = None
    run_started = False
    try:
        logger.info(f"Starting analysis job {job_id} with modules: {modules}")
        mlflow.start_run(run_name=f"{dataset_name}_{job_id}")
        run_started = True
        mlflow.set_tag("job_id", job_id)
        mlflow.set_tag("status", "running")
        mlflow.set_tag("retention", "local_file_store")
        _update_progress(job_id, 1, "Loading CSV")
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        logger.info(f"Loaded {len(df)} rows from {dataset_name}")
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("uploaded_filename", upload_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("modules_enabled", ",".join(modules) if modules else "(none)")
        # Ensure columns
        required = ["text", "sentiment", "label"]
        for r in required:
            if r not in df.columns:
                raise RuntimeError(f"Missing required column: {r}")

        _update_progress(job_id, 5, "Building pipeline")
        kwargs = {k: True for k in modules}
        # Filter out augmentation modules - they expand rows and aren't for normal preprocessing
        augmentation_modules = {"code_switch_augmentation"}
        effective_modules = [m for m in modules if m not in augmentation_modules]
        if len(effective_modules) < len(modules):
            excluded = [m for m in modules if m in augmentation_modules]
            logger.info(f"Filtered out augmentation modules for analysis: {excluded}")
        kwargs = {k: True for k in effective_modules}
        pipeline = build_pipeline(**kwargs)
        logger.info(f"Pipeline built with active modules: {pipeline.active_modules}")
        mlflow.log_param("active_modules", ",".join(pipeline.active_modules) if pipeline.active_modules else "(none)")
        mlflow.log_dict({"pipeline_config": pipeline.get_pipeline_config()}, "metadata/pipeline_config.json")
        _update_progress(job_id, 10, "Running preprocessing over full dataset")
        processed = pipeline.process_dataframe(df, text_col="text", output_col="processed_text")
        logger.info(f"Preprocessing complete: {len(processed)} rows")
        mlflow.log_artifact(str(csv_path), artifact_path="inputs")

        _update_progress(job_id, 20, "Initializing evaluator")
        evaluator = AnalyticalEvaluator(df_original=df, df_processed=processed, text_col="text", processed_col="processed_text", batch_size=batch_size)

        def progress_cb(p, msg):
            _update_progress(job_id, int(p), msg)

        _update_progress(job_id, 25, "Starting analysis")
        include_spac = bool(jobs[job_id].get("include_spac", False))
        results = evaluator.evaluate_all(progress_callback=progress_cb, include_spac=include_spac)
        logger.info(f"Evaluation complete for job {job_id}")
        
        # Convert numpy types to ensure JSON serializability
        results = _convert_numpy_types(results)
        logger.info(f"Numpy types converted for job {job_id}")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["results"] = results
        jobs[job_id]["dataset_name"] = dataset_name
        jobs[job_id]["uploaded_filename"] = upload_name
        jobs[job_id]["batch_size"] = batch_size
        jobs[job_id]["modules"] = modules
        jobs[job_id]["pipeline_config"] = pipeline.get_pipeline_config() if pipeline is not None else None
        if mlflow.active_run() is not None:
            jobs[job_id]["mlflow_run_id"] = mlflow.active_run().info.run_id
            jobs[job_id]["mlflow_experiment_id"] = mlflow.active_run().info.experiment_id
        _log_mlflow_run(job_id, csv_path, dataset_name, upload_name, modules, batch_size, pipeline=pipeline, processed=processed, results=results, status="done")
        _save_job(job_id, jobs[job_id])
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["dataset_name"] = dataset_name
        jobs[job_id]["uploaded_filename"] = upload_name
        jobs[job_id]["batch_size"] = batch_size
        jobs[job_id]["modules"] = modules
        jobs[job_id]["pipeline_config"] = pipeline.get_pipeline_config() if pipeline is not None else None
        if mlflow.active_run() is not None:
            jobs[job_id]["mlflow_run_id"] = mlflow.active_run().info.run_id
            jobs[job_id]["mlflow_experiment_id"] = mlflow.active_run().info.experiment_id
        _log_mlflow_run(job_id, csv_path, dataset_name, upload_name, modules, batch_size, pipeline=pipeline, processed=processed, results=results, status="failed", error=str(e))
        _save_job(job_id, jobs[job_id])
    finally:
        if run_started and mlflow.active_run() is not None:
            mlflow.end_run()


@app.get("/")
async def root():
    """Redirect to the web UI"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/preprocessing_preview.html")


@app.get("/preprocessing_preview.html")
async def get_preprocessing_preview():
    """Serve the preprocessing preview HTML"""
    html_path = ROOT / "preprocessing_preview.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML file not found")
    from fastapi.responses import FileResponse
    return FileResponse(str(html_path), media_type="text/html")


@app.post("/run_analysis")
async def run_analysis(background_tasks: BackgroundTasks, file: UploadFile = File(...), modules: str = Form("[]"), batch_size: int = Form(32), include_spac: bool = Form(False)):
    """Upload CSV and start asynchronous analysis job.

    - `modules` is a JSON list of module keys to enable (e.g. ["phonetic_normalization"]).
    """
    job_id = uuid.uuid4().hex
    dest = JOBS_DIR / f"upload_{job_id}.csv"
    content = await file.read()
    dest.write_bytes(content)
    dataset_name = Path(file.filename or dest.name).stem

    try:
        modules_list = json.loads(modules)
    except Exception:
        modules_list = []

    jobs[job_id] = {"status": "running", "progress": 0, "message": "queued", "modules": modules_list, "dataset_name": dataset_name, "uploaded_filename": file.filename or dest.name, "batch_size": int(batch_size), "include_spac": bool(include_spac)}
    _save_job(job_id, jobs[job_id])
    background_tasks.add_task(_run_analysis_job, job_id, dest, modules_list, int(batch_size))
    return {"job_id": job_id}


@app.post("/run_model_inference")
async def run_model_inference(background_tasks: BackgroundTasks, file: UploadFile = File(...), modules: str = Form("[]"), models: str = Form("[]"), batch_size: int = Form(32)):
    job_id = uuid.uuid4().hex
    dest = JOBS_DIR / f"upload_{job_id}.csv"
    content = await file.read()
    dest.write_bytes(content)
    dataset_name = Path(file.filename or dest.name).stem

    try:
        modules_list = json.loads(modules)
    except Exception:
        modules_list = []

    try:
        models_list = json.loads(models)
    except Exception:
        models_list = []

    allowed_modules = {
        "phonetic_normalization",
        "language_aware_normalization",
        "script_unification",
        "transliteration",
        "balanced_tokenization",
        "language_identification_tagging",
        "switch_point_encoding",
        "code_switch_augmentation",
        "context_aware_subword_sampling",
    }
    modules_list = [m for m in modules_list if m in allowed_modules]
    models_list = [m for m in models_list if m in MODEL_BACKBONES]

    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "queued",
        "modules": modules_list,
        "models": models_list,
        "dataset_name": dataset_name,
        "uploaded_filename": file.filename or dest.name,
        "batch_size": int(batch_size),
    }
    _save_job(job_id, jobs[job_id])
    background_tasks.add_task(_run_model_inference_job, job_id, dest, modules_list, models_list, int(batch_size))
    return {"job_id": job_id}


@app.get("/analysis_results/{job_id}")
def analysis_results(job_id: str):
    path = JOBS_DIR / f"{job_id}.json"
    if path.exists():
        try:
            content = path.read_text(encoding="utf-8")
            if not content or not content.strip():
                # File exists but is empty; job is still processing
                return {"status": "processing", "job_id": job_id}
            return json.loads(content)
        except json.JSONDecodeError as e:
            # File is corrupted; return error status
            raise HTTPException(status_code=500, detail=f"Job result file corrupted: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading job results: {str(e)}")
    raise HTTPException(status_code=404, detail="Job not found")


@app.post("/evaluate_models")
async def evaluate_models(payload: Dict[str, Any]):
    text = str(payload.get("text", "")).strip()
    models = payload.get("models", list(MODEL_BACKBONES))
    modules = payload.get("modules", [])

    if not isinstance(models, list):
        raise HTTPException(status_code=400, detail="models must be a list of model keys")
    if not isinstance(modules, list):
        raise HTTPException(status_code=400, detail="modules must be a list of module keys")

    allowed_modules = {
        "phonetic_normalization",
        "language_aware_normalization",
        "script_unification",
        "transliteration",
        "balanced_tokenization",
        "language_identification_tagging",
        "switch_point_encoding",
        "code_switch_augmentation",
        "context_aware_subword_sampling",
    }
    modules = [m for m in modules if m in allowed_modules]
    models = [m for m in models if m in MODEL_BACKBONES]

    job_id = uuid.uuid4().hex
    payload_record = {
        "job_id": job_id,
        "text": text,
        "modules": modules,
        "models": models,
        "status": "running",
    }
    jobs[job_id] = payload_record.copy()
    _save_job(job_id, jobs[job_id])

    try:
        pipeline_kwargs = {m: True for m in modules}
        pipeline = build_pipeline(**pipeline_kwargs) if pipeline_kwargs else None
        processed_text = pipeline.process(text) if pipeline is not None else text

        results: Dict[str, Any] = {
            "job_id": job_id,
            "preprocessed_text": processed_text,
            "models": {},
        }

        with mlflow.start_run(run_name=f"model_eval_{job_id}"):
            mlflow.set_tag("job_id", job_id)
            mlflow.set_tag("status", "running")
            _log_mlflow_model_eval(job_id, payload_record, results)
            for model_key in models:
                model_result = _compute_model_evaluation(model_key, processed_text)
                model_result.pop("embedding_vector", None)
                results["models"][model_key] = _convert_numpy_types(model_result)

            payload_record["status"] = "done"
            payload_record["results"] = results
            payload_record["mlflow_run_id"] = mlflow.active_run().info.run_id
            payload_record["mlflow_experiment_id"] = mlflow.active_run().info.experiment_id
            _log_mlflow_model_eval(job_id, payload_record, results)

        jobs[job_id] = payload_record
        _save_job(job_id, jobs[job_id])
        return {"job_id": job_id, "results": results, "mlflow_run_id": jobs[job_id].get("mlflow_run_id")}
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        _save_job(job_id, jobs[job_id])
        logger.error(f"Model evaluation job {job_id} failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/available_modules")
def available_modules():
    # Mirror details from combined_preprocessing_pipeline can be added later
    keys = [
        "phonetic_normalization",
        "language_aware_normalization",
        "script_unification",
        "transliteration",
        "balanced_tokenization",
        "language_identification_tagging",
        "switch_point_encoding",
        "code_switch_augmentation",
        "context_aware_subword_sampling",
    ]
    return {"modules": keys}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
