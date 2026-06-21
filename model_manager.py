"""
model_manager.py — Centralized model loading, caching, and device management.

Replaces scattered direct calls to:
    AutoTokenizer.from_pretrained(...)
    AutoModel.from_pretrained(...)
    AutoModelForSequenceClassification.from_pretrained(...)
    SentenceTransformer(...)

with a single ``ModelManager`` singleton that:

  * always loads from the unified HF cache directory (`cache_dir=` is passed
    explicitly on every call, in addition to the env vars set by
    ``hf_cache_config`` — some code paths in `transformers` and
    `sentence-transformers` only respect the explicit kwarg),
  * loads each (model_name, model_class) pair exactly once per process,
  * automatically places models on the best available device (CUDA if
    available, else CPU) and supports a future multi-GPU device map without
    callers needing to change anything,
  * exposes a memory-cleanup hook (``unload`` / ``unload_all``) so long-running
    research servers can evict rarely-used backbones,
  * logs every load with timing via ``logging_config.log_timing``.

IMPORTANT: import ``hf_cache_config`` before importing this module (or before
importing ``transformers`` / ``sentence_transformers`` from anywhere), so the
HF_HOME / TRANSFORMERS_CACHE env vars are set before those libraries read them.
"""
from __future__ import annotations

import gc
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import hf_cache_config  # noqa: F401  side-effect import; must run before transformers
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from logging_config import get_logger, log_timing

logger = get_logger(__name__)


# ── Device selection ──────────────────────────────────────────────────────────

def get_default_device() -> torch.device:
    """
    Select the best available device.

    Today this is CPU-or-single-CUDA-device. The function is the single
    chokepoint callers go through, so adding multi-GPU device-map support
    later (e.g. via `accelerate`) only requires changing this function, not
    every call site that currently does `.to(device)`.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class LoadedTransformer:
    """Bundle of a loaded tokenizer + model + the device it lives on."""
    tokenizer: Any
    model: Any
    device: torch.device
    model_name: str


@dataclass
class ManagerStats:
    """Lightweight load-count bookkeeping, exposed for diagnostics endpoints."""
    loads: int = 0
    cache_hits: int = 0
    unloads: int = 0
    loaded_models: Dict[str, str] = field(default_factory=dict)  # key -> device


class ModelManager:
    """
    Process-wide singleton responsible for every transformer / sentence
    embedding model used by the research server.

    Thread-safe: a lock guards the cache dict so concurrent FastAPI background
    tasks requesting the same model don't trigger duplicate loads.
    """

    _instance: Optional["ModelManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], Any] = {}
        self._lock = threading.Lock()
        self.device = get_default_device()
        self.stats = ManagerStats()
        logger.info(f"ModelManager initialized on device={self.device}")

    # ── Singleton accessor ────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Encoder / embedding model loading ─────────────────────────────────────

    def get_encoder(self, model_name: str) -> LoadedTransformer:
        """
        Load (or return cached) tokenizer+encoder model for *model_name*.

        Used for embedding extraction / attention analysis (AutoModel, not a
        classification head).
        """
        key = (model_name, "encoder")
        with self._lock:
            if key in self._cache:
                self.stats.cache_hits += 1
                logger.debug(f"Encoder cache hit: {model_name}")
                return self._cache[key]

            with log_timing(logger, "load_encoder", model_name=model_name):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_fast=True, cache_dir=hf_cache_config.HF_CACHE_DIR,
                )
                model = AutoModel.from_pretrained(
                    model_name, output_attentions=True,
                    cache_dir=hf_cache_config.HF_CACHE_DIR,
                )
                model.eval()
                model.to(self.device)

            loaded = LoadedTransformer(
                tokenizer=tokenizer, model=model, device=self.device,
                model_name=model_name,
            )
            self._cache[key] = loaded
            self.stats.loads += 1
            self.stats.loaded_models[f"{model_name}:encoder"] = str(self.device)
            return loaded

    def get_sequence_classifier(self, model_name: str) -> LoadedTransformer:
        """
        Load (or return cached) tokenizer + AutoModelForSequenceClassification
        for *model_name*. Separate from ``get_encoder`` because a
        classification head is a structurally different model object than
        the bare encoder, even for the same backbone name — caching them
        under the same key would silently return the wrong model type to
        whichever caller asked second.
        """
        key = (model_name, "sequence_classifier")
        with self._lock:
            if key in self._cache:
                self.stats.cache_hits += 1
                logger.debug(f"Sequence classifier cache hit: {model_name}")
                return self._cache[key]

            with log_timing(logger, "load_sequence_classifier", model_name=model_name):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_fast=True, cache_dir=hf_cache_config.HF_CACHE_DIR,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, cache_dir=hf_cache_config.HF_CACHE_DIR,
                )
                model.eval()
                model.to(self.device)

            loaded = LoadedTransformer(
                tokenizer=tokenizer, model=model, device=self.device,
                model_name=model_name,
            )
            self._cache[key] = loaded
            self.stats.loads += 1
            self.stats.loaded_models[f"{model_name}:sequence_classifier"] = str(self.device)
            return loaded

    def get_sentence_transformer(self, model_name: str):
        """
        Load (or return cached) a ``sentence_transformers.SentenceTransformer``.

        Imported lazily inside the method (rather than at module top-level)
        so that environments without `sentence-transformers` installed can
        still import and use ``ModelManager`` for plain encoder/classifier
        workloads (the original codebase had the same optional-dependency
        pattern; we preserve it here rather than making sentence-transformers
        a hard requirement of this module).
        """
        key = (model_name, "sentence_transformer")
        with self._lock:
            if key in self._cache:
                self.stats.cache_hits += 1
                logger.debug(f"SentenceTransformer cache hit: {model_name}")
                return self._cache[key]

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed.\n"
                    "Run:  pip install sentence-transformers torch"
                ) from exc

            with log_timing(logger, "load_sentence_transformer", model_name=model_name):
                st_model = SentenceTransformer(
                    model_name,
                    cache_folder=hf_cache_config.HF_CACHE_DIR,
                    device=str(self.device),
                )

            self._cache[key] = st_model
            self.stats.loads += 1
            self.stats.loaded_models[f"{model_name}:sentence_transformer"] = str(self.device)
            return st_model

    # ── Memory management ─────────────────────────────────────────────────────

    def unload(self, model_name: str, kind: Optional[str] = None) -> int:
        """
        Evict cached model(s) matching *model_name* (optionally restricted to
        *kind* in {"encoder", "sequence_classifier", "sentence_transformer"}).

        Returns the number of cache entries removed. Calls ``torch.cuda.
        empty_cache()`` and ``gc.collect()`` afterward so freed GPU/CPU memory
        is actually released rather than merely dereferenced.
        """
        removed = 0
        with self._lock:
            keys_to_remove = [
                k for k in self._cache
                if k[0] == model_name and (kind is None or k[1] == kind)
            ]
            for k in keys_to_remove:
                del self._cache[k]
                self.stats.loaded_models.pop(f"{k[0]}:{k[1]}", None)
                removed += 1
                self.stats.unloads += 1

        if removed:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded {removed} model(s) for {model_name}",
                        extra={"model_name": model_name, "kind": kind})
        return removed

    def unload_all(self) -> int:
        """Evict every cached model. Returns the number removed."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self.stats.loaded_models.clear()
            self.stats.unloads += count

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded all {count} cached model(s)")
        return count

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Snapshot for a diagnostics endpoint: device, loaded models, stats."""
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "loads": self.stats.loads,
            "cache_hits": self.stats.cache_hits,
            "unloads": self.stats.unloads,
            "loaded_models": dict(self.stats.loaded_models),
        }


def get_model_manager() -> ModelManager:
    """Module-level convenience accessor for the singleton."""
    return ModelManager.instance()
