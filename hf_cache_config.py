"""
hf_cache_config.py — Unified HuggingFace cache configuration.

CRITICAL: This module must be imported FIRST, before any other module that
transitively imports `transformers`, `sentence_transformers`, or `datasets`.
Setting these environment variables after those libraries have already been
imported has no effect, because they read the cache location at import time.

Usage (at the very top of main.py, before any other project import):

    import hf_cache_config  # noqa: F401  (side-effect import, must be first)

Every other module in this project should obtain the cache directory via
`hf_cache_config.HF_CACHE_DIR` rather than hardcoding the path, so the
location only needs to change in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Single source of truth for the cache location ────────────────────────────
#
# Overridable via the HF_CACHE_DIR environment variable so the same codebase
# can run in CI / Linux / Docker without code changes — only the original
# Windows path is hardcoded as the default, per the request.
HF_CACHE_DIR: str = os.environ.get("HF_CACHE_DIR", "D:/HF_CACHE")

# Set HF/transformers env vars BEFORE any transformers/sentence-transformers
# import happens anywhere in the process. This is the only correct place to
# do it — scattering `cache_dir=` kwargs everywhere is necessary too (some
# libraries ignore the env vars for certain sub-caches), but the env vars are
# the first line of defense and must be set first.
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", HF_CACHE_DIR)
# Hub download cache (used by huggingface_hub directly, e.g. by some
# AutoModel paths that bypass the legacy TRANSFORMERS_CACHE variable).
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)

# Embedding cache (sqlite-backed text->embedding cache) lives in a
# subdirectory of the same root so everything HF-related is under one tree.
EMBEDDING_CACHE_DIR: str = str(Path(HF_CACHE_DIR) / "embedding_cache")

try:
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(EMBEDDING_CACHE_DIR).mkdir(parents=True, exist_ok=True)
except OSError:
    # On some CI/sandbox systems the literal D:/ path won't exist and isn't
    # creatable. We do not raise here — model loading downstream will surface
    # a clear error if the directory genuinely can't be used, but we don't
    # want an import-time crash in environments where this module is merely
    # being imported for inspection (e.g. unit tests that mock the manager).
    pass
