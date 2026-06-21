
"""api.py — FastAPI entrypoint for the Hinglish preprocessing framework.

This file replaces the older monolithic API layer while preserving the public
HTTP surface used by the frontend and jobs workflow.

Order-of-import note
--------------------
`hf_cache_config` must be imported before any module that can transitively
import `transformers`, `sentence_transformers`, or `datasets`, because the HF
cache environment variables are read at import time.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from available_modules_route import get_available_modules_payload
import numpy as np
import pandas as pd
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import hf_cache_config  # noqa: F401  (must be first, side-effect import)

# Keep project root on path for direct execution and local development.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytical_evaluator import AnalyticalEvaluator
from cached_analytical_evaluator import CachedAnalyticalEvaluator
from combined_preprocessing_pipeline import build_pipeline
from job_manager import JobManager
from logging_config import get_logger
from model_evaluator import MODEL_BACKBONES, coerce_label
from model_manager import get_model_manager
from text_sanitizer import add_model_input_column, clean_for_model_text

logger = get_logger(__name__)

ROOT = Path(__file__).parent
JOBS_DIR = ROOT / "analysis_jobs"
JOBS_DIR.mkdir(exist_ok=True)

MLFLOW_DB = ROOT.parent / "mlruns.db"

app = FastAPI(title="Hinglish Preprocessing Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (ROOT / "preprocessing_preview.html").exists():
    app.mount("/static", StaticFiles(directory=str(ROOT)), name="static")

jobs: Dict[str, Dict[str, Any]] = {}

job_manager = JobManager(
    jobs_dir=JOBS_DIR,
    mlflow_db_path=MLFLOW_DB,
    experiment_name="hinglish_preprocessing_analyses",
)


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy/torch values into JSON-safe Python types."""
    try:
        if isinstance(obj, dict):
            return {k: _convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert_numpy_types(item) for item in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, torch.Tensor):
            return _convert_numpy_types(obj.detach().cpu().numpy())
        return obj
    except Exception:
        return str(obj)


def _sync_job_cache() -> None:
    """
    Keep the legacy in-memory `jobs` dict in sync with JobManager so existing
    handlers that read from `jobs` continue to work.
    """
    try:
        for job_id in list(jobs.keys()):
            stored = job_manager.get_job(job_id)
            if stored is not None:
                jobs[job_id] = stored
    except Exception:
        pass


def _save_job(job_id: str, payload: Dict[str, Any]) -> None:
    jobs[job_id] = _convert_numpy_types(payload)
    job_manager.create_job(job_id, jobs[job_id]) if job_manager.get_job(job_id) is None else job_manager.update_fields(job_id, **jobs[job_id])
    job_manager.save_job(job_id)


def _update_progress(job_id: str, percent: int, message: str) -> None:
    jobs.setdefault(job_id, {})
    jobs[job_id]["progress"] = percent
    jobs[job_id]["message"] = message
    job_manager.update_progress(job_id, percent, message)
    jobs[job_id] = job_manager.get_job(job_id) or jobs[job_id]


def _flatten_numeric_metrics(prefix: str, payload: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in payload.items():
        metric_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(_flatten_numeric_metrics(metric_name, value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[metric_name] = float(value)
    return metrics


def _sanitize_model_input_df(df: pd.DataFrame, processed_col: str = "processed_text") -> pd.DataFrame:
    """
    Add a model_input_text column, stripping analytical markers before any
    transformer forward pass.
    """
    if processed_col not in df.columns:
        return df.copy()
    return add_model_input_column(df, processed_col=processed_col, model_input_col="model_input_text")


def _load_csv(upload: UploadFile, dest: Path) -> pd.DataFrame:
    content = dest.read_bytes()
    dest.write_bytes(content)
    df = pd.read_csv(dest)
    return df.loc[:, ~df.columns.str.startswith("Unnamed:")]


def _ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required column(s): {missing}")


def _get_pipeline(modules: List[str]):
    kwargs = {k: True for k in modules}
    return build_pipeline(**kwargs) if kwargs else None


def _default_models() -> List[str]:
    return list(MODEL_BACKBONES.keys())


def _model_keys(models: Any) -> List[str]:
    if not isinstance(models, list):
        return _default_models()
    return [m for m in models if m in MODEL_BACKBONES]


def _module_keys(modules: Any) -> List[str]:
    allowed = {
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
    if not isinstance(modules, list):
        return []
    return [m for m in modules if m in allowed]


def _run_analysis_job(job_id: str, csv_path: Path, modules: list[str], batch_size: int):
    dataset_name = csv_path.stem
    upload_name = csv_path.name
    pipeline = None
    processed = None
    results = None
    run_started = False

    try:
        logger.info("Starting analysis job", extra={"job_id": job_id, "modules": modules})
        job_manager.start_run(run_name=f"{dataset_name}_{job_id}", job_id=job_id)
        run_started = True
        job_manager.update_fields(job_id, status="running", dataset_name=dataset_name, uploaded_filename=upload_name, batch_size=batch_size, modules=modules)
        _update_progress(job_id, 1, "Loading CSV")

        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        _ensure_required_columns(df, ["text", "sentiment", "label"])

        _update_progress(job_id, 5, "Building pipeline")
        augmentation_modules = {"code_switch_augmentation"}
        effective_modules = [m for m in modules if m not in augmentation_modules]
        if len(effective_modules) < len(modules):
            logger.info("Filtered out augmentation modules for analysis", extra={"excluded": [m for m in modules if m in augmentation_modules]})
        pipeline = _get_pipeline(effective_modules)
        job_manager.log_params_safe({
            "dataset_name": dataset_name,
            "uploaded_filename": upload_name,
            "batch_size": batch_size,
            "modules_enabled": ",".join(modules) if modules else "(none)",
            "active_modules": ",".join(pipeline.active_modules) if pipeline and pipeline.active_modules else "(none)",
        })
        if pipeline:
            job_manager.log_dict_safe({"pipeline_config": pipeline.get_pipeline_config()}, "metadata/pipeline_config.json")

        _update_progress(job_id, 10, "Running preprocessing over full dataset")
        processed = pipeline.process_dataframe(df, text_col="text", output_col="processed_text") if pipeline else df.copy()
        processed = _sanitize_model_input_df(processed, processed_col="processed_text")
        job_manager.log_artifact_safe(str(csv_path), artifact_path="inputs")

        _update_progress(job_id, 20, "Initializing evaluator")
        evaluator = CachedAnalyticalEvaluator(
            df_original=df,
            df_processed=processed,
            text_col="text",
            processed_col="processed_text",
            label_col="label",
            batch_size=batch_size,
        )

        def progress_cb(p: int, msg: str):
            _update_progress(job_id, int(p), msg)

        _update_progress(job_id, 25, "Starting analysis")
        include_spac = bool(jobs.get(job_id, {}).get("include_spac", False))
        results = evaluator.evaluate_all(progress_callback=progress_cb, include_spac=include_spac)
        results = _convert_numpy_types(results)

        payload = {
            "job_id": job_id,
            "dataset_name": dataset_name,
            "processed_rows": len(processed),
            "labelled_rows": int(sum(1 for _ in processed["label"] if _ is not None)) if "label" in processed.columns else 0,
            "preprocessing": {
                "requested_modules": modules,
                "active_modules": pipeline.active_modules if pipeline else [],
                "pipeline_config": pipeline.get_pipeline_config() if pipeline else {},
            },
            "results": results,
            "sample_processed_texts": processed["processed_text"].astype(str).tolist()[:5] if "processed_text" in processed.columns else [],
        }

        job_manager.mark_done(**payload)
        jobs[job_id] = job_manager.get_job(job_id) or payload
        _save_job(job_id, jobs[job_id])
        job_manager.log_dict_safe(payload, "results/analysis.json")
        if results is not None:
            for metric_name, metric_value in _flatten_numeric_metrics("analysis", results).items():
                try:
                    job_manager.log_metrics_from_results("analysis", results)
                    break
                except Exception:
                    pass

    except Exception as e:
        logger.error("Analysis job failed", extra={"job_id": job_id, "error": str(e)}, exc_info=True)
        job_manager.mark_failed(job_id, str(e), dataset_name=dataset_name, uploaded_filename=upload_name, batch_size=batch_size, modules=modules, pipeline_config=pipeline.get_pipeline_config() if pipeline else None)
        jobs[job_id] = job_manager.get_job(job_id) or {"status": "failed", "error": str(e)}
        _save_job(job_id, jobs[job_id])
    finally:
        if run_started:
            job_manager.end_run()


def _run_model_inference_job(job_id: str, csv_path: Path, modules: list[str], models: list[str], batch_size: int):
    dataset_name = csv_path.stem
    upload_name = csv_path.name
    pipeline = None
    processed_df = None
    results: Dict[str, Any] = {}
    run_started = False

    try:
        logger.info("Starting model inference job", extra={"job_id": job_id, "modules": modules, "models": models})
        job_manager.start_run(run_name=f"model_inference_{dataset_name}_{job_id}", job_id=job_id)
        run_started = True
        job_manager.update_fields(job_id, status="running", dataset_name=dataset_name, uploaded_filename=upload_name, batch_size=batch_size, modules=modules, models=models)

        _update_progress(job_id, 1, "Loading CSV")
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        _ensure_required_columns(df, ["text", "sentiment", "label"])

        _update_progress(job_id, 5, "Building pipeline")
        augmentation_modules = {"code_switch_augmentation"}
        effective_modules = [m for m in modules if m not in augmentation_modules]
        pipeline = _get_pipeline(effective_modules)
        job_manager.log_params_safe({
            "dataset_name": dataset_name,
            "uploaded_filename": upload_name,
            "batch_size": batch_size,
            "models": ",".join(models) if models else "(none)",
            "active_modules": ",".join(pipeline.active_modules) if pipeline and pipeline.active_modules else "(none)",
        })
        job_manager.log_dict_safe({"pipeline_config": pipeline.get_pipeline_config() if pipeline else {}}, "metadata/pipeline_config.json")

        _update_progress(job_id, 20, "Preprocessing dataset")
        processed_df = pipeline.process_dataframe(df, text_col="text", output_col="processed_text") if pipeline else df.copy()
        processed_df = _sanitize_model_input_df(processed_df, processed_col="processed_text")
        job_manager.log_artifact_safe(str(csv_path), artifact_path="inputs")

        _update_progress(job_id, 40, "Computing model summaries")
        text_col = "model_input_text" if "model_input_text" in processed_df.columns else ("processed_text" if "processed_text" in processed_df.columns else "text")
        texts = processed_df[text_col].astype(str).tolist()
        raw_labels = processed_df["label"].tolist() if "label" in processed_df.columns else [None] * len(processed_df)
        if "sentiment" in processed_df.columns:
            sentiment_labels = processed_df["sentiment"].tolist()
            labels = [
                coerce_label(lbl) if coerce_label(lbl) is not None else coerce_label(sent)
                for lbl, sent in zip(raw_labels, sentiment_labels)
            ]
        else:
            labels = [coerce_label(lbl) for lbl in raw_labels]

        from model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator()
        for model_key in models:
            try:
                model_summary = evaluator.summarize_dataset(
                    model_key=model_key,
                    texts=texts,
                    labels=labels,
                    sample_limit=1000,
                )
                results[model_key] = model_summary
            except Exception as e:
                logger.error("Failed model metrics", extra={"job_id": job_id, "model_key": model_key, "error": str(e)}, exc_info=True)
                results[model_key] = {
                    "model_label": MODEL_BACKBONES.get(model_key, ("Unknown", "unknown"))[0],
                    "model_name": MODEL_BACKBONES.get(model_key, ("Unknown", "unknown"))[1],
                    "error": str(e),
                    "evaluated_rows": 0,
                    "sampled_rows": 0,
                }

        payload = {
            "job_id": job_id,
            "dataset_name": dataset_name,
            "preprocessing": {
                "requested_modules": modules,
                "active_modules": pipeline.active_modules if pipeline else [],
                "pipeline_config": pipeline.get_pipeline_config() if pipeline else {},
            },
            "processed_rows": len(processed_df),
            "labelled_rows": int(sum(1 for l in labels if l is not None)),
            "models": results,
            "sample_processed_texts": processed_df[text_col].astype(str).tolist()[:5],
        }

        job_manager.mark_done(**payload)
        jobs[job_id] = job_manager.get_job(job_id) or payload
        _save_job(job_id, jobs[job_id])
        job_manager.log_dict_safe(payload, "results/model_inference.json")
        job_manager.log_metrics_from_results("model_inference", payload)

    except Exception as e:
        logger.error("Model inference job failed", extra={"job_id": job_id, "error": str(e)}, exc_info=True)
        job_manager.mark_failed(job_id, str(e), dataset_name=dataset_name, uploaded_filename=upload_name, batch_size=batch_size, modules=modules, models=models, pipeline_config=pipeline.get_pipeline_config() if pipeline else None)
        jobs[job_id] = job_manager.get_job(job_id) or {"status": "failed", "error": str(e)}
        _save_job(job_id, jobs[job_id])
    finally:
        if run_started:
            job_manager.end_run()


@app.on_event("startup")
def _startup() -> None:
    _sync_job_cache()
    logger.info("API startup complete", extra={"jobs_recovered": len(jobs), "hf_cache_dir": hf_cache_config.HF_CACHE_DIR})


@app.get("/")
async def root():
    return RedirectResponse(url="/preprocessing_preview.html")


@app.get("/preprocessing_preview.html")
async def get_preprocessing_preview():
    html_path = ROOT / "preprocessing_preview.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML file not found")
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "hf_cache_dir": hf_cache_config.HF_CACHE_DIR,
        "embedding_cache_dir": hf_cache_config.EMBEDDING_CACHE_DIR,
        "model_manager": get_model_manager().status(),
    }


@app.post("/run_analysis")
async def run_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    modules: str = Form("[]"),
    batch_size: int = Form(32),
    include_spac: bool = Form(False),
):
    job_id = uuid.uuid4().hex
    dest = JOBS_DIR / f"upload_{job_id}.csv"
    dest.write_bytes(await file.read())
    dataset_name = Path(file.filename or dest.name).stem

    try:
        modules_list = _module_keys(json.loads(modules))
    except Exception:
        modules_list = []

    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "queued",
        "modules": modules_list,
        "dataset_name": dataset_name,
        "uploaded_filename": file.filename or dest.name,
        "batch_size": int(batch_size),
        "include_spac": bool(include_spac),
    }
    _save_job(job_id, jobs[job_id])
    background_tasks.add_task(_run_analysis_job, job_id, dest, modules_list, int(batch_size))
    return {"job_id": job_id}


@app.post("/run_model_inference")
async def run_model_inference(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    modules: str = Form("[]"),
    models: str = Form("[]"),
    batch_size: int = Form(32),
):
    job_id = uuid.uuid4().hex
    dest = JOBS_DIR / f"upload_{job_id}.csv"
    dest.write_bytes(await file.read())
    dataset_name = Path(file.filename or dest.name).stem

    try:
        modules_list = _module_keys(json.loads(modules))
    except Exception:
        modules_list = []

    try:
        models_list = _model_keys(json.loads(models))
    except Exception:
        models_list = _default_models()

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
    job = job_manager.get_job_or_from_disk(job_id)
    if job is not None:
        return _convert_numpy_types(job)
    path = JOBS_DIR / f"{job_id}.json"
    if path.exists():
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return {"status": "processing", "job_id": job_id}
        return json.loads(content)
    raise HTTPException(status_code=404, detail="Job not found")


@app.post("/evaluate_models")
async def evaluate_models(payload: Dict[str, Any]):
    text = str(payload.get("text", "")).strip()
    models = _model_keys(payload.get("models", _default_models()))
    modules = _module_keys(payload.get("modules", []))
    if not isinstance(payload.get("models", _default_models()), list):
        raise HTTPException(status_code=400, detail="models must be a list of model keys")
    if not isinstance(payload.get("modules", []), list):
        raise HTTPException(status_code=400, detail="modules must be a list of module keys")

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
        model_input_text = clean_for_model_text(processed_text)

        results: Dict[str, Any] = {
            "job_id": job_id,
            "processed_text": processed_text,
            "model_input_text": model_input_text,
            "models": {},
        }

        from model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator()
        for model_key in models:
            model_result = evaluator.evaluate_text(model_key, model_input_text)
            model_result.pop("embedding_vector", None)
            results["models"][model_key] = _convert_numpy_types(model_result)

        payload_record["status"] = "done"
        payload_record["results"] = results
        jobs[job_id] = payload_record
        _save_job(job_id, jobs[job_id])
        return {"job_id": job_id, "results": results}
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        _save_job(job_id, jobs[job_id])
        logger.error("Model evaluation failed", extra={"job_id": job_id, "error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

class ProcessRequest(BaseModel):
    text: str
    modules: List[str] = []


@app.post("/process")
async def process_text(payload: ProcessRequest):
    modules = _module_keys(payload.modules)
    try:
        pipeline = _get_pipeline(modules)
        result = pipeline.process(payload.text) if pipeline else payload.text
        return {"ok": True, "result": result}
    except Exception as e:
        logger.error("Process failed", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available_modules")
def available_modules():
    return get_available_modules_payload()

@app.get("/pipeline_info")
def pipeline_info():
    return {
        "transformation_modules": sorted(list({
            "phonetic_normalization",
            "language_aware_normalization",
            "script_unification",
            "transliteration",
            "balanced_tokenization",
            "context_aware_subword_sampling",
        })),
        "analysis_modules": sorted(list({
            "language_identification_tagging",
            "switch_point_encoding",
        })),
        "augmentation_modules": sorted(list({
            "code_switch_augmentation",
        })),
        "hf_cache_dir": hf_cache_config.HF_CACHE_DIR,
        "embedding_cache_dir": hf_cache_config.EMBEDDING_CACHE_DIR,
    }

@app.get("/system_info")
async def system_info():
    return {
        "gpu_available": torch.cuda.is_available(),
        "cached_models": [],  # populate from get_model_manager().status() if it exposes this
        "hf_cache_dir": hf_cache_config.HF_CACHE_DIR,
    }

@app.get("/available_models")
async def available_models():
    from model_evaluator import MODEL_BACKBONES
    return {
        "models": [
            {"key": key, "label": label}
            for key, (label, _hf_name) in MODEL_BACKBONES.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
