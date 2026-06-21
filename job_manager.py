"""
job_manager.py — Centralized job lifecycle management.

Consolidates what was previously duplicated across ``_run_analysis_job``,
``_run_model_inference_job``, ``_save_job``, ``_update_progress``,
``_log_mlflow_run``, and ``_log_mlflow_model_eval`` into one
``JobManager`` responsible for:

  * job state (in-memory dict, mirrored to JSON files on disk — unchanged
    storage format from the original, so existing job files written by a
    prior version of the server remain readable),
  * progress updates,
  * MLflow run lifecycle (start/end, tags, params, dict/artifact logging,
    metric flattening),
  * recovery after restart: on startup, ``JobManager.load_from_disk()``
    rehydrates the in-memory registry from JOBS_DIR so job status lookups
    (``GET /analysis_results/{job_id}``) keep working for jobs that were
    written before a server restart — the original code had no such
    recovery step, so an in-memory-only lookup would 404 after every
    restart even though the JSON file was sitting right there on disk.

The public surface (`create_job`, `update_progress`, `mark_done`,
`mark_failed`, `get_job`, `save_job`) is intentionally close to the original
function names/semantics so the FastAPI route handlers barely change.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow

from logging_config import get_logger

logger = get_logger(__name__)


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy/torch types to native Python types for JSON
    serialization. Unchanged logic from the original implementation —
    relocated here since job persistence is what actually needs it.
    """
    import numpy as np

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
        else:
            try:
                import torch
                if isinstance(obj, torch.Tensor):
                    return _convert_numpy_types(obj.detach().cpu().numpy())
            except ImportError:
                pass
        return obj
    except Exception:
        return str(obj)


def _flatten_numeric_metrics(prefix: str, payload: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in payload.items():
        metric_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(_flatten_numeric_metrics(metric_name, value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[metric_name] = float(value)
    return metrics


class JobManager:
    """
    Process-wide job registry + persistence + MLflow integration.

    Thread-safe for the in-memory dict; MLflow's own client handles its own
    concurrency (one active run per managed context at a time, as in the
    original code — this class does not change MLflow's concurrency model,
    only consolidates the bookkeeping around it).
    """

    def __init__(self, jobs_dir: Path, mlflow_db_path: Path, experiment_name: str):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(exist_ok=True, parents=True)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.resolve().as_posix()}")
        mlflow.set_experiment(experiment_name)

        self.load_from_disk()

    # ── Restart recovery ──────────────────────────────────────────────────────

    def load_from_disk(self) -> int:
        """
        Rehydrate the in-memory job registry from JOBS_DIR on startup.

        Without this, ``GET /analysis_results/{job_id}`` for a job created
        before a server restart would still work (it reads the JSON file
        directly), but anything relying on the in-memory ``jobs`` dict —
        e.g. progress updates for a job that's mid-flight during a restart,
        or any future endpoint that lists jobs from memory — would silently
        lose state. This makes the in-memory registry an eventually-correct
        cache of disk state rather than the sole source of truth.
        """
        loaded = 0
        for path in self.jobs_dir.glob("*.json"):
            job_id = path.stem
            try:
                content = path.read_text(encoding="utf-8")
                if not content.strip():
                    continue
                data = json.loads(content)
                with self._lock:
                    self._jobs[job_id] = data
                loaded += 1
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Skipping unreadable job file on recovery: {path.name}: {exc}")
        if loaded:
            logger.info(f"Recovered {loaded} job(s) from disk on startup",
                        extra={"jobs_dir": str(self.jobs_dir)})
        return loaded

    # ── Job lifecycle ──────────────────────────────────────────────────────────

    def create_job(self, job_id: str, initial_state: Dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id] = dict(initial_state)
        self.save_job(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_job_or_from_disk(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up *job_id* in memory; if absent (e.g. process restarted after
        ``load_from_disk`` ran but a job file was written by another
        process/worker), fall back to reading the file directly. Mirrors
        the original endpoint's direct-file-read behavior exactly, so
        ``GET /analysis_results/{job_id}`` keeps working unchanged.
        """
        job = self.get_job(job_id)
        if job is not None:
            return job
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return {"status": "processing", "job_id": job_id}
        return json.loads(content)

    def update_progress(self, job_id: str, percent: int, message: str) -> None:
        with self._lock:
            self._jobs.setdefault(job_id, {})
            self._jobs[job_id]["progress"] = percent
            self._jobs[job_id]["message"] = message
        self.save_job(job_id)

    def update_fields(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs.setdefault(job_id, {})
            self._jobs[job_id].update(fields)

    def mark_done(self, job_id: str, **fields: Any) -> None:
        self.update_fields(job_id, status="done", **fields)
        self.save_job(job_id)

    def mark_failed(self, job_id: str, error: str, **fields: Any) -> None:
        self.update_fields(job_id, status="failed", error=error, **fields)
        self.save_job(job_id)

    def save_job(self, job_id: str) -> None:
        with self._lock:
            payload = dict(self._jobs.get(job_id, {}))
        path = self.jobs_dir / f"{job_id}.json"
        payload = _convert_numpy_types(payload)
        try:
            json_str = json.dumps(payload, default=str)
            path.write_text(json_str, encoding="utf-8")
            logger.debug(f"Job {job_id} saved ({len(json_str)} bytes)")
        except Exception as exc:
            logger.error(f"Failed to save job {job_id}: {exc}", exc_info=True)
            error_payload = {"status": "error", "error": f"Failed to save job: {exc}", "job_id": job_id}
            try:
                path.write_text(json.dumps(error_payload), encoding="utf-8")
            except Exception:
                path.write_text('{"status":"error","error":"Job save failed"}', encoding="utf-8")

    # ── MLflow integration ───────────────────────────────────────────────────

    def start_run(self, run_name: str, job_id: str) -> bool:
        """Start an MLflow run, returning True if one was actually started."""
        mlflow.start_run(run_name=run_name)
        mlflow.set_tag("job_id", job_id)
        mlflow.set_tag("status", "running")
        mlflow.set_tag("retention", "local_file_store")
        return True

    def end_run(self) -> None:
        if mlflow.active_run() is not None:
            mlflow.end_run()

    def attach_mlflow_ids(self, job_id: str) -> None:
        """Record the active run's IDs onto the job record, if a run is active."""
        if mlflow.active_run() is not None:
            self.update_fields(
                job_id,
                mlflow_run_id=mlflow.active_run().info.run_id,
                mlflow_experiment_id=mlflow.active_run().info.experiment_id,
            )

    def log_dict_safe(self, payload: Dict[str, Any], artifact_path: str) -> None:
        if mlflow.active_run() is None:
            return
        try:
            mlflow.log_dict(payload, artifact_path)
        except Exception as exc:
            logger.warning(f"mlflow.log_dict failed for {artifact_path}: {exc}")

    def log_metrics_from_results(self, prefix: str, results: Dict[str, Any]) -> None:
        if mlflow.active_run() is None:
            return
        flattened = _flatten_numeric_metrics(prefix, results)
        for metric_name, metric_value in flattened.items():
            try:
                mlflow.log_metric(metric_name, metric_value)
            except Exception:
                pass

    def log_params_safe(self, params: Dict[str, Any]) -> None:
        if mlflow.active_run() is None:
            return
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception:
                pass

    def log_artifact_safe(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if mlflow.active_run() is None:
            return
        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as exc:
            logger.warning(f"mlflow.log_artifact failed for {local_path}: {exc}")
