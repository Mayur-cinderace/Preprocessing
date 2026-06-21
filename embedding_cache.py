"""
embedding_cache.py — Persistent text -> embedding cache.

Backed by SQLite (stdlib, no extra dependency) rather than `diskcache`, so
this module has zero new third-party requirements. Stored under
``hf_cache_config.EMBEDDING_CACHE_DIR`` (a subdirectory of the unified HF
cache root) so the whole HF-related cache tree lives in one place.

Why this matters
-----------------
``AnalyticalEvaluator`` independently re-embeds largely-overlapping text
across several metrics — embedding stability embeds both df_original and
df_processed, CRD embeds (deduplicated) tokens, spectral analysis embeds
df_processed again, and Proxy-SPRC embeds individual tokens per sentence.
Without a shared cache, the same exact string can be sent through a
SentenceTransformer forward pass three or four times in a single
``evaluate_all`` call. This cache makes embedding lookups idempotent across
metrics *and* across job runs — re-running an analysis on the same dataset
with the same model after a server restart will hit the cache instead of
recomputing every embedding from scratch.

Cache key = sha256(model_name + "\x00" + text). Including the model name in
the key is required correctness, not an optimization — the same text
embedded by two different backbones produces unrelated vectors, and without
the model name in the key a cache hit would silently return the wrong
model's embedding.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import hf_cache_config
from logging_config import CacheHitMissCounter, get_logger

logger = get_logger(__name__)


def _cache_key(model_name: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _pack_vector(vec: np.ndarray) -> bytes:
    """Serialize a float32 vector to bytes (dimension-agnostic, no pickle)."""
    arr = np.asarray(vec, dtype=np.float32)
    return struct.pack(f"<{arr.size}f", *arr.tolist())


def _unpack_vector(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)


class EmbeddingCache:
    """
    Thread-safe SQLite-backed cache mapping (model_name, text) -> embedding.

    One instance is intended to be shared process-wide (see
    ``get_embedding_cache``), mirroring the ``ModelManager`` singleton pattern
    so callers don't need to wire a cache object through every function
    signature.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path(hf_cache_config.EMBEDDING_CACHE_DIR) / "embeddings.sqlite3")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counter = CacheHitMissCounter("embedding", logger=logger)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False because FastAPI background tasks may call
        # in from a different thread than __init__; the instance-level lock
        # serializes actual access, which is what matters for sqlite safety.
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key   TEXT PRIMARY KEY,
                    model_name  TEXT NOT NULL,
                    vector      BLOB NOT NULL,
                    created_at  REAL NOT NULL DEFAULT (strftime('%s','now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name)"
            )
            conn.commit()

    # ── Single-item API ────────────────────────────────────────────────────────

    def get(self, model_name: str, text: str) -> Optional[np.ndarray]:
        key = _cache_key(model_name, text)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT vector FROM embeddings WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            self._counter.miss()
            return None
        self._counter.hit()
        return _unpack_vector(row[0])

    def put(self, model_name: str, text: str, vector: np.ndarray) -> None:
        key = _cache_key(model_name, text)
        blob = _pack_vector(vector)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (cache_key, model_name, vector) "
                "VALUES (?, ?, ?)",
                (key, model_name, blob),
            )
            conn.commit()

    # ── Batch API ──────────────────────────────────────────────────────────────

    def get_many(self, model_name: str, texts: Sequence[str]) -> Dict[str, Optional[np.ndarray]]:
        """Return {text: vector_or_None} for every text in *texts*."""
        keys = {t: _cache_key(model_name, t) for t in texts}
        result: Dict[str, Optional[np.ndarray]] = {t: None for t in texts}
        if not keys:
            return result

        placeholders = ",".join("?" for _ in keys)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT cache_key, vector FROM embeddings WHERE cache_key IN ({placeholders})",
                list(keys.values()),
            ).fetchall()
        found = {k: _unpack_vector(v) for k, v in rows}

        for text, key in keys.items():
            if key in found:
                result[text] = found[key]
                self._counter.hit()
            else:
                self._counter.miss()
        return result

    def put_many(self, model_name: str, text_vector_pairs: Sequence[tuple]) -> None:
        rows = [
            (_cache_key(model_name, text), model_name, _pack_vector(vec))
            for text, vec in text_vector_pairs
        ]
        if not rows:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (cache_key, model_name, vector) "
                "VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()

    def encode_with_cache(
        self,
        model_name: str,
        texts: Sequence[str],
        encode_fn,
    ) -> np.ndarray:
        """
        Return embeddings for *texts*, computing only the cache misses via
        *encode_fn* (a callable taking a list[str] and returning an
        (n, dim) ndarray — typically ``SentenceTransformer.encode``).

        This is the main entry point ``AnalyticalEvaluator`` and
        ``model_evaluator`` should use instead of calling the encoder
        directly, so every embedding-consuming metric benefits from the
        cache without each metric needing its own caching logic.
        """
        cached = self.get_many(model_name, texts)
        missing = [t for t in texts if cached[t] is None]

        if missing:
            fresh = encode_fn(missing)
            fresh = np.asarray(fresh)
            self.put_many(model_name, list(zip(missing, fresh)))
            for t, v in zip(missing, fresh):
                cached[t] = v

        dim = next((v.shape[0] for v in cached.values() if v is not None), None)
        if dim is None:
            return np.empty((0, 0))

        return np.vstack([
            cached[t] if cached[t] is not None else np.zeros(dim, dtype=np.float32)
            for t in texts
        ])

    # ── Maintenance ────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, object]:
        with self._lock, self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            by_model = conn.execute(
                "SELECT model_name, COUNT(*) FROM embeddings GROUP BY model_name"
            ).fetchall()
        return {
            "total_rows": total,
            "rows_by_model": dict(by_model),
            "hit_rate": self._counter.hit_rate,
            "hits": self._counter.hits,
            "misses": self._counter.misses,
            "db_path": self.db_path,
        }

    def clear(self, model_name: Optional[str] = None) -> int:
        with self._lock, self._connect() as conn:
            if model_name:
                cur = conn.execute("DELETE FROM embeddings WHERE model_name = ?", (model_name,))
            else:
                cur = conn.execute("DELETE FROM embeddings")
            conn.commit()
            return cur.rowcount


_singleton: Optional[EmbeddingCache] = None
_singleton_lock = threading.Lock()


def get_embedding_cache() -> EmbeddingCache:
    """Process-wide singleton accessor, mirroring ``ModelManager.instance()``."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = EmbeddingCache()
    return _singleton
