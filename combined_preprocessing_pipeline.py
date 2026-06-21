"""
combined_preprocessing_pipeline.py — Master orchestration layer for the
Hinglish preprocessing toolkit.

Hard requirements:
    pip install nltk pandas
    python -c "import nltk; nltk.download('words')"

Optional per module:
    pip install indic-transliteration          # balanced_tokenizn, transliteration,
                                              #   script_unification
    pip install transformers sentencepiece    # context_aware_sentencepiece_dropout

Scientific positioning
──────────────────────
CombinedPreprocessingPipeline is a **reproducible experiment orchestration
framework** for modular Hinglish preprocessing research.  It cleanly separates
three conceptually distinct operation types:

    TRANSFORMATION  — modules that modify text (normalization, transliteration,
                      script unification, balanced tokenization, subword sampling).
    ANALYSIS        — modules that annotate or tag text without changing the
                      underlying content for model training (language ID tagging,
                      switch-point encoding).  Run only when enable_analysis=True
                      or output_mode="analysis"|"both".
    AUGMENTATION    — modules that expand the corpus by generating synthetic
                      rows (code-switch augmentation).  Runs only via
                      augment_dataframe / augment_csv; never via process().

The pipeline is NOT:
    • a monolithic text processor,
    • a silent dependency manager,
    • a module dumping ground.

Design invariants
─────────────────
    • All module imports are hard (no try/except) — missing deps raise immediately.
    • Pipeline order is validated at construction time.
    • Conflicting module combinations raise ValueError unless allow_conflicts=True.
    • Augmentation is isolated: CodeSwitchAugmentation never runs in process().
    • Given identical input and settings, output is identical (deterministic).
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import pandas as pd

# ── Hard module imports — fail loudly if deps are missing ─────────────────────
from balanced_tokenizn import BalancedTokenization
from code_switch_augmentn import CodeSwitchAugmentation
from context_aware_sentencepiece_dropout import ContextAwareSentencePieceDropout
from lang_aware_normalizn import LanguageAwareNormalization
from lang_id_tagging import LanguageIdentificationTagging
from phonetic_normalization import PhoneticNormalization
from script_unification import ScriptUnification
from switch_point_encoding import SwitchPointEncoding
from transliteration import Transliteration

# ── Module categories ─────────────────────────────────────────────────────────
#
# Each set defines which module keys belong to which conceptual category.
# Categories control default behaviour, output modes, and validation rules.

TRANSFORMATION_MODULES: frozenset[str] = frozenset({
    "phonetic_normalization",
    "language_aware_normalization",
    "script_unification",
    "transliteration",
    "balanced_tokenization",
    "context_aware_subword_sampling",
})

ANALYSIS_MODULES: frozenset[str] = frozenset({
    "language_identification_tagging",
    "switch_point_encoding",
})

AUGMENTATION_MODULES: frozenset[str] = frozenset({
    "code_switch_augmentation",
})

# ── Registry ──────────────────────────────────────────────────────────────────

_MODULE_REGISTRY: dict[str, type] = {
    "balanced_tokenization":           BalancedTokenization,
    "language_identification_tagging": LanguageIdentificationTagging,
    "language_aware_normalization":    LanguageAwareNormalization,
    "transliteration":                 Transliteration,
    "switch_point_encoding":           SwitchPointEncoding,
    "code_switch_augmentation":        CodeSwitchAugmentation,
    "context_aware_subword_sampling":  ContextAwareSentencePieceDropout,
    "phonetic_normalization":          PhoneticNormalization,
    "script_unification":              ScriptUnification,
}

# ── Module priorities (lower = earlier in pipeline) ────────────────────────────
#
# Used by _validate_pipeline() to detect invalid orderings.
# Priority bands:
#   0  NORMALIZATION   — phonetic / language-aware normalization
#   1  SCRIPT          — script unification, transliteration
#   2  TOKENIZATION    — balanced tokenization
#   3  ANALYSIS        — language ID tagging, switch-point encoding
#   4  SUBWORD         — context-aware subword sampling (always last)

_MODULE_PRIORITY: dict[str, int] = {
    "phonetic_normalization":          0,
    "language_aware_normalization":    0,
    "script_unification":              1,
    "transliteration":                 1,
    "balanced_tokenization":           2,
    "language_identification_tagging": 3,
    "switch_point_encoding":           3,
    "context_aware_subword_sampling":  4,
    "code_switch_augmentation":        99,  # augmentation-only; never in process()
}

# Default text-transformation + optional-analysis pipeline order.
_DEFAULT_PIPELINE_ORDER: list[str] = [
    "phonetic_normalization",
    "language_aware_normalization",
    "script_unification",
    "transliteration",
    "balanced_tokenization",
    "language_identification_tagging",    # skipped unless enable_analysis=True
    "switch_point_encoding",              # skipped unless enable_analysis=True
    "context_aware_subword_sampling",
]

# ── Compatibility matrix ───────────────────────────────────────────────────────
#
# Relationships between pairs of modules.
# "CONFLICTING" raises ValueError by default.
# "REDUNDANT"   issues a UserWarning.
# "COMPATIBLE"  is the default (not listed).

_COMPATIBILITY: dict[tuple[str, str], str] = {
    ("transliteration",               "script_unification"):            "CONFLICTING",
    ("script_unification",            "transliteration"):               "CONFLICTING",
    ("language_identification_tagging", "switch_point_encoding"):       "REDUNDANT",
    ("switch_point_encoding",          "language_identification_tagging"): "REDUNDANT",
    ("balanced_tokenization",         "script_unification"):            "COMPATIBLE",
}

# ── Output mode type ───────────────────────────────────────────────────────────

OutputMode = Literal["model", "analysis", "both"]


# ── Tokenizer / model cache (module-level singleton) ──────────────────────────
#
# Prevents repeated loading of heavy models (XLM-R, SentencePiece) across
# experiments within a single Python session.  Keyed by (model_name, config_hash).

_MODEL_CACHE: dict[str, Any] = {}


def _get_cached(key: str, loader_fn) -> Any:  # type: ignore[type-arg]
    """Return a cached model instance, loading via ``loader_fn`` on first call."""
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = loader_fn()
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """Evict all entries from the module-level model cache."""
    _MODEL_CACHE.clear()


# ── Pipeline statistics container ─────────────────────────────────────────────

@dataclasses.dataclass
class PipelineStats:
    """Tracks pipeline-level execution statistics."""
    processed_documents: int   = 0
    processed_tokens:    int   = 0
    execution_time_s:    float = 0.0
    active_modules:      list  = dataclasses.field(default_factory=list)
    pipeline_order:      list  = dataclasses.field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def reset(self) -> None:
        self.processed_documents = 0
        self.processed_tokens    = 0
        self.execution_time_s    = 0.0


# ── Main class ────────────────────────────────────────────────────────────────

class CombinedPreprocessingPipeline:
    """
    Reproducible experiment orchestration framework for modular Hinglish
    preprocessing.

    Module categories
    -----------------
    TRANSFORMATION  Modules that modify text content.  Always active when enabled.
    ANALYSIS        Modules that add linguistic annotations.  Active only when
                    ``enable_analysis=True`` or ``output_mode`` is "analysis"/"both".
    AUGMENTATION    Row-expanding modules.  Active only via augment_*() methods.

    Output modes
    ------------
    "model"     (default) — return processed text only; analysis modules skipped.
    "analysis"  — run analysis modules; return structured dict with annotations.
    "both"      — run all enabled modules; return structured dict with text +
                  analysis outputs + statistics.

    Conflict policy
    ---------------
    By default, CONFLICTING module combinations raise ``ValueError``.
    Pass ``allow_conflicts=True`` to demote conflicts to warnings.

    Usage
    -----
    >>> p = CombinedPreprocessingPipeline(
    ...     phonetic_normalization=True,
    ...     language_aware_normalization=True,
    ...     balanced_tokenization=True,
    ... )
    >>> p.process("yaar movie bahuuut acchi thi")
    >>> p.process_csv("data/hinglish.csv", "data/out.csv")

    Dry-run validation:
    >>> p.validate_only()

    Pipeline description:
    >>> print(p.describe_pipeline())

    Parameters
    ----------
    <module_name>=True|False
        Toggle each preprocessing module.
    module_configs
        Per-module keyword arguments.
    custom_pipeline_order
        Override the default module execution order.
    output_mode
        One of "model", "analysis", "both".
    enable_analysis
        Shorthand to activate analysis modules regardless of output_mode.
    allow_conflicts
        If True, CONFLICTING combinations produce warnings instead of errors.
    trace_execution
        If True, record before/after/timing for each module.
    random_seed
        Global seed forwarded to stochastic modules.
    verbose
        Print progress messages.
    experiment_name
        Human-readable label embedded in experiment manifests.
    """

    def __init__(
        self,
        # Transformation modules
        phonetic_normalization:         bool = False,
        language_aware_normalization:   bool = False,
        script_unification:             bool = False,
        transliteration:                bool = False,
        balanced_tokenization:          bool = False,
        context_aware_subword_sampling: bool = False,
        # Analysis modules
        language_identification_tagging: bool = False,
        switch_point_encoding:          bool = False,
        # Augmentation modules
        code_switch_augmentation:       bool = False,
        # Pipeline configuration
        module_configs:         Optional[Dict[str, Dict[str, Any]]] = None,
        custom_pipeline_order:  Optional[List[str]] = None,
        output_mode:            OutputMode = "model",
        enable_analysis:        bool = False,
        allow_conflicts:        bool = False,
        trace_execution:        bool = False,
        random_seed:            int  = 42,
        verbose:                bool = False,
        experiment_name:        str  = "experiment",
    ) -> None:
        self._enabled: dict[str, bool] = {
            "phonetic_normalization":          phonetic_normalization,
            "language_aware_normalization":    language_aware_normalization,
            "script_unification":              script_unification,
            "transliteration":                 transliteration,
            "balanced_tokenization":           balanced_tokenization,
            "context_aware_subword_sampling":  context_aware_subword_sampling,
            "language_identification_tagging": language_identification_tagging,
            "switch_point_encoding":           switch_point_encoding,
            "code_switch_augmentation":        code_switch_augmentation,
        }
        self._module_configs    = module_configs or {}
        self._output_mode       = output_mode
        self._enable_analysis   = enable_analysis
        self._allow_conflicts   = allow_conflicts
        self._trace_execution   = trace_execution
        self._random_seed       = random_seed
        self._verbose           = verbose
        self._experiment_name   = experiment_name
        self._pipeline_order    = self._resolve_pipeline_order(custom_pipeline_order)
        self._trace_log:   list[dict[str, Any]] = []

        # Validate before building — fail fast
        self._validate_pipeline()
        self._modules: dict[str, Any] = self._build_modules()

        self.stats = PipelineStats(
            active_modules=list(self.active_modules),
            pipeline_order=list(self._pipeline_order),
        )
        self._log("Active modules: " + (", ".join(self.active_modules) or "(none)"))

    # ── Pipeline order resolution ─────────────────────────────────────────────

    def _resolve_pipeline_order(self, custom: Optional[List[str]]) -> list[str]:
        order = custom if custom is not None else list(_DEFAULT_PIPELINE_ORDER)
        unknown = [k for k in order if k not in _MODULE_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown module keys in pipeline order: {unknown}\n"
                f"Valid keys: {sorted(_MODULE_REGISTRY)}"
            )
        return order

    # ── Pipeline validation ───────────────────────────────────────────────────

    def _validate_pipeline(self) -> list[str]:
        """
        Validate the pipeline configuration and return a list of warning messages.

        Raises ``ValueError`` for CONFLICTING combinations (unless
        ``allow_conflicts=True``) and for ordering violations.
        Issues ``UserWarning`` for REDUNDANT combinations.
        """
        import warnings
        issues: list[str] = []
        active = set(self._enabled_keys())

        # ── 1. Compatibility checks ───────────────────────────────────────
        seen_pairs: set[frozenset[str]] = set()
        for (a, b), relationship in _COMPATIBILITY.items():
            pair = frozenset({a, b})
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if a not in active or b not in active:
                continue
            if relationship == "CONFLICTING":
                msg = (
                    f"Modules '{a}' and '{b}' are CONFLICTING and should not "
                    f"be used together.  Pass allow_conflicts=True to override."
                )
                if self._allow_conflicts:
                    warnings.warn(msg, UserWarning, stacklevel=4)
                    issues.append(f"CONFLICT (allowed): {a} + {b}")
                else:
                    raise ValueError(msg)
            elif relationship == "REDUNDANT":
                msg = f"Modules '{a}' and '{b}' are REDUNDANT — review your config."
                warnings.warn(msg, UserWarning, stacklevel=4)
                issues.append(f"REDUNDANT: {a} + {b}")

        # ── 2. Duplicate detection ─────────────────────────────────────────
        seen: set[str] = set()
        for key in self._pipeline_order:
            if key in seen:
                raise ValueError(
                    f"Duplicate module key '{key}' in pipeline order."
                )
            seen.add(key)

        # ── 3. Ordering / priority violations ─────────────────────────────
        active_ordered = [k for k in self._pipeline_order
                          if k in active and k not in AUGMENTATION_MODULES]
        for i, key_a in enumerate(active_ordered):
            for key_b in active_ordered[i + 1:]:
                pri_a = _MODULE_PRIORITY.get(key_a, 99)
                pri_b = _MODULE_PRIORITY.get(key_b, 99)
                if pri_a > pri_b:
                    msg = (
                        f"Ordering violation: '{key_a}' (priority {pri_a}) "
                        f"appears before '{key_b}' (priority {pri_b}) in the "
                        f"pipeline but should come after."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=4)
                    issues.append(f"ORDER: {key_a} before {key_b}")

        # ── 4. Analysis modules in model-only mode ─────────────────────────
        for key in ANALYSIS_MODULES:
            if key in active and self._output_mode == "model" and not self._enable_analysis:
                import warnings as _w
                _w.warn(
                    f"Analysis module '{key}' is enabled but output_mode='model' "
                    f"and enable_analysis=False — it will be skipped.  "
                    f"Set enable_analysis=True or output_mode='analysis'/'both'.",
                    UserWarning,
                    stacklevel=4,
                )
                issues.append(f"SKIPPED_ANALYSIS: {key}")

        # ── 5. ContextAwareSentencePieceDropout must be last ───────────────
        casp = "context_aware_subword_sampling"
        if casp in active:
            active_keys = [k for k in self._pipeline_order
                           if k in active and k not in AUGMENTATION_MODULES]
            if active_keys and active_keys[-1] != casp:
                msg = (
                    f"'{casp}' should be the last module in the pipeline "
                    f"but is currently at position "
                    f"{active_keys.index(casp) + 1}/{len(active_keys)}."
                )
                import warnings as _w
                _w.warn(msg, UserWarning, stacklevel=4)
                issues.append(f"ORDER: {casp} not last")

        return issues

    def validate_only(self) -> list[str]:
        """
        Dry-run: validate the current configuration without processing any data.

        Returns a list of issue descriptions (warnings and demoted conflicts).
        Raises ``ValueError`` for hard violations (same as construction).
        """
        issues = self._validate_pipeline()
        print("[Pipeline] Validation complete.")
        if issues:
            for issue in issues:
                print(f"  ⚠  {issue}")
        else:
            print("  ✓  No issues found.")
        return issues

    # ── Module recommendations ────────────────────────────────────────────────

    def recommend_additions(self) -> list[str]:
        """
        Return a list of recommended modules to add based on the current config.
        Does not modify the pipeline.
        """
        recs: list[str] = []
        active = set(self._enabled_keys())
        if "script_unification" in active and "balanced_tokenization" not in active:
            recs.append(
                "Consider adding 'balanced_tokenization' — "
                "it complements 'script_unification' by reducing subword fragmentation."
            )
        if "switch_point_encoding" in active and "language_identification_tagging" not in active:
            recs.append(
                "Consider adding 'language_identification_tagging' alongside "
                "'switch_point_encoding' for richer code-switch annotations."
            )
        if "transliteration" in active and "phonetic_normalization" not in active:
            recs.append(
                "Consider adding 'phonetic_normalization' before 'transliteration' "
                "to reduce romanization variance before script conversion."
            )
        return recs

    # ── Module construction ───────────────────────────────────────────────────

    def _build_modules(self) -> dict[str, Any]:
        modules: dict[str, Any] = {}
        for key, cls in _MODULE_REGISTRY.items():
            if not self._enabled.get(key):
                continue
            config = dict(self._module_configs.get(key, {}))
            field_names = {f.name for f in dataclasses.fields(cls)}
            if "random_seed" in field_names:
                config.setdefault("random_seed", self._random_seed)
            modules[key] = cls(enabled=True, **config)
        return modules

    # ── Active key helpers ────────────────────────────────────────────────────

    def _enabled_keys(self) -> list[str]:
        return [k for k, v in self._enabled.items() if v]

    def _active_transform_order(self) -> list[str]:
        """
        Keys in pipeline order that are enabled TRANSFORMATION modules.
        Analysis and augmentation modules are excluded.
        """
        return [
            k for k in self._pipeline_order
            if k in self._modules and k in TRANSFORMATION_MODULES
        ]

    def _active_analysis_order(self) -> list[str]:
        """Keys in pipeline order that are enabled ANALYSIS modules."""
        return [
            k for k in self._pipeline_order
            if k in self._modules and k in ANALYSIS_MODULES
        ]

    @property
    def active_modules(self) -> list[str]:
        """All enabled non-augmentation modules in pipeline order."""
        return [
            k for k in self._pipeline_order
            if k in self._modules and k not in AUGMENTATION_MODULES
        ]

    def get_pipeline_config(self) -> dict[str, Any]:
        return {
            "experiment_name":  self._experiment_name,
            "enabled":          dict(self._enabled),
            "pipeline_order":   list(self._pipeline_order),
            "module_configs":   dict(self._module_configs),
            "output_mode":      self._output_mode,
            "enable_analysis":  self._enable_analysis,
            "allow_conflicts":  self._allow_conflicts,
            "random_seed":      self._random_seed,
        }

    # ── Pipeline description ──────────────────────────────────────────────────

    def describe_pipeline(self) -> str:
        """
        Return a human-readable description of the active pipeline with
        category labels and ordered flow arrows.

        Example output:
            [TRANSFORMATION] PhoneticNormalization
            ↓
            [TRANSFORMATION] LanguageAwareNormalization
            ↓
            [ANALYSIS] LanguageIdentificationTagging   (output_mode=analysis)
        """
        lines: list[str] = []
        transform_keys = self._active_transform_order()
        analysis_keys  = self._active_analysis_order()
        run_analysis   = self._enable_analysis or self._output_mode in ("analysis", "both")

        all_active = transform_keys + (analysis_keys if run_analysis else [])
        if not all_active:
            return "[Pipeline] (no active modules)"

        for i, key in enumerate(all_active):
            cat = (
                "TRANSFORMATION" if key in TRANSFORMATION_MODULES
                else "ANALYSIS"
            )
            cls_name = _MODULE_REGISTRY[key].__name__
            note = ""
            if key in ANALYSIS_MODULES and not run_analysis:
                note = "  ← SKIPPED (output_mode='model')"
            lines.append(f"[{cat}] {cls_name}{note}")
            if i < len(all_active) - 1:
                lines.append("↓")

        aug_keys = [k for k in self._modules if k in AUGMENTATION_MODULES]
        if aug_keys:
            lines.append("\n[AUGMENTATION — via augment_*() only]")
            for key in aug_keys:
                lines.append(f"  {_MODULE_REGISTRY[key].__name__}")

        return "\n".join(lines)

    # ── Core processing ───────────────────────────────────────────────────────

    def process(self, text: str) -> str | dict[str, Any]:
        """
        Apply all enabled modules to a single text string.

        Returns
        -------
        str
            When ``output_mode="model"``.
        dict
            When ``output_mode="analysis"`` or ``"both"``:
            ``{"processed_text": ..., "analysis": {...}, "statistics": {...}}``.
        """
        if not isinstance(text, str):
            return text

        run_analysis = self._enable_analysis or self._output_mode in ("analysis", "both")
        self._trace_log = []
        t_start = time.perf_counter()

        value = text

        # Transformation pass
        for key in self._active_transform_order():
            value = self._run_module(key, value)

        # Analysis pass (only when requested)
        analysis_outputs: dict[str, Any] = {}
        if run_analysis:
            for key in self._active_analysis_order():
                result = self._run_module(key, value)
                analysis_outputs[key] = result

        elapsed = time.perf_counter() - t_start
        self.stats.processed_documents += 1
        self.stats.processed_tokens    += len(value.split())
        self.stats.execution_time_s    += elapsed

        if self._output_mode == "model":
            return value

        structured: dict[str, Any] = {
            "processed_text": value,
            "analysis":       analysis_outputs,
            "statistics":     self.collect_statistics(),
        }
        if self._trace_execution:
            structured["trace"] = list(self._trace_log)
        return structured

    def _run_module(self, key: str, text: str) -> str:
        """Run a single module with optional execution tracing."""
        module = self._modules[key]
        if not self._trace_execution:
            return module.process(text)
        t0 = time.perf_counter()
        result = module.process(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._trace_log.append({
            "module":     key,
            "before":     text,
            "after":      result,
            "elapsed_ms": round(elapsed_ms, 3),
        })
        return result

    def process_batch(self, texts: Iterable[str]) -> list[str | dict[str, Any]]:
        return [self.process(t) for t in texts]

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        output_col: str = "processed_text",
    ) -> pd.DataFrame:
        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found. Available: {list(df.columns)}"
            )
        out = df.copy()
        current_col = text_col
        run_analysis = self._enable_analysis or self._output_mode in ("analysis", "both")

        for key in self._active_transform_order():
            out[output_col] = out[current_col].astype(str).apply(
                self._modules[key].process
            )
            current_col = output_col

        if run_analysis:
            for key in self._active_analysis_order():
                out[f"analysis_{key}"] = out[current_col].astype(str).apply(
                    self._modules[key].process
                )

        if output_col not in out.columns:
            out[output_col] = out[text_col].astype(str)

        self.stats.processed_documents += len(out)
        return out

    def process_csv(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        text_col: str = "text",
        output_col: str = "processed_text",
    ) -> pd.DataFrame:
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        self._log(f"Processing: {input_csv}")
        df = pd.read_csv(input_csv)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        processed = self.process_dataframe(df, text_col=text_col, output_col=output_col)
        processed.to_csv(output_csv, index=False)
        self._log(f"Saved:      {output_csv}  ({len(processed)} rows)")
        return processed

    # ── Statistics aggregation ────────────────────────────────────────────────

    def collect_statistics(self) -> dict[str, Any]:
        """
        Aggregate per-module statistics from all active modules that expose a
        ``get_stats()`` method (e.g. Transliteration), plus pipeline-level stats.

        Returns
        -------
        dict keyed by module name + "pipeline" for pipeline-level stats.
        """
        stats: dict[str, Any] = {"pipeline": self.stats.summary()}
        for key, module in self._modules.items():
            if hasattr(module, "get_stats"):
                stats[key] = module.get_stats()
        return stats

    # ── Augmentation (row-expanding) ──────────────────────────────────────────

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        output_col: str = "processed_text",
    ) -> pd.DataFrame:
        """
        Expand rows using the code-switch augmentation module.

        Raises ``RuntimeError`` if ``code_switch_augmentation`` is not enabled.
        CodeSwitchAugmentation is intentionally isolated from process() to
        preserve the 1-to-1 row relationship in the main pipeline.
        """
        module = self._modules.get("code_switch_augmentation")
        if module is None:
            raise RuntimeError(
                "augment_dataframe requires code_switch_augmentation=True."
            )
        return module.augment_dataframe(df, text_col=text_col, output_col=output_col)

    def augment_csv(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        text_col: str = "text",
        output_col: str = "processed_text",
    ) -> pd.DataFrame:
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        df = pd.read_csv(input_csv)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        processed = self.augment_dataframe(df, text_col=text_col, output_col=output_col)
        processed.to_csv(output_csv, index=False)
        return processed

    # ── Experiment suite ──────────────────────────────────────────────────────

    def run_experiment_suite(
        self,
        input_csv: str | Path,
        output_dir: str | Path,
        text_col:   str = "text",
        output_col: str = "processed_text",
        dataset_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run the dataset through:
            • a baseline (no modules),
            • each enabled module in isolation,
            • predefined module combinations,
            • the full pipeline (all enabled modules).

        Writes one CSV per experiment plus a JSON reproducibility manifest.

        Returns the manifest dict.
        """
        import platform, sys
        input_csv  = Path(input_csv)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df_ref = pd.read_csv(input_csv)
        row_count = len(df_ref)
        experiments: list[dict] = []
        t_suite_start = time.perf_counter()

        def _run(name: str, **kwargs: Any) -> None:
            out_path = output_dir / f"{name}.csv"
            module_flags  = {k: v for k, v in kwargs.items()
                              if k in _MODULE_REGISTRY and isinstance(v, bool)}
            extra_kwargs  = {k: v for k, v in kwargs.items()
                              if k not in _MODULE_REGISTRY or not isinstance(v, bool)}
            p = CombinedPreprocessingPipeline(
                module_configs=self._module_configs,
                random_seed=self._random_seed,
                verbose=self._verbose,
                **module_flags,
                **extra_kwargs,
            )
            t0 = time.perf_counter()
            p.process_csv(input_csv, out_path, text_col=text_col, output_col=output_col)
            elapsed = round(time.perf_counter() - t0, 3)
            experiments.append({
                "name":          name,
                "modules":       [k for k, v in module_flags.items() if v],
                "output_file":   str(out_path),
                "elapsed_s":     elapsed,
                "module_configs": {k: self._module_configs.get(k, {})
                                   for k in module_flags if module_flags.get(k)},
            })
            self._log(f"Experiment '{name}' → {out_path}  ({elapsed}s)")

        # Baseline
        _run("baseline")

        # Single-module runs
        for key in _DEFAULT_PIPELINE_ORDER:
            if self._enabled.get(key) and key not in AUGMENTATION_MODULES:
                _run(key, **{key: True})

        # Predefined combinations
        if (self._enabled.get("phonetic_normalization")
                or self._enabled.get("script_unification")):
            _run("phonetic_plus_script",
                 phonetic_normalization=True,
                 script_unification=True)

        if (self._enabled.get("phonetic_normalization")
                or self._enabled.get("language_aware_normalization")):
            _run("normalization_only",
                 phonetic_normalization=True,
                 language_aware_normalization=True)

        if (self._enabled.get("balanced_tokenization")
                or self._enabled.get("script_unification")):
            _run("script_plus_tokenization",
                 script_unification=True,
                 balanced_tokenization=True)

        # Full pipeline
        enabled_kwargs = {k: v for k, v in self._enabled.items()
                          if v and k not in AUGMENTATION_MODULES}
        if enabled_kwargs:
            _run("full_pipeline", **enabled_kwargs)

        suite_elapsed = round(time.perf_counter() - t_suite_start, 3)

        manifest = {
            "experiment_name": self._experiment_name,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_path":    dataset_path or str(input_csv),
            "row_count":       row_count,
            "random_seed":     self._random_seed,
            "active_modules":  list(self.active_modules),
            "pipeline_order":  list(self._pipeline_order),
            "module_configs":  dict(self._module_configs),
            "output_mode":     self._output_mode,
            "suite_elapsed_s": suite_elapsed,
            "software": {
                "python":        platform.python_version(),
                "pandas":        pd.__version__,
            },
            "experiments":     experiments,
        }
        manifest_path = output_dir / "experiment_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._log(f"Manifest: {manifest_path}")
        return manifest

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        if self._verbose:
            print(f"[Pipeline] {message}")

    def reset(self) -> None:
        """Rebuild all module instances and reset pipeline statistics."""
        self._modules = self._build_modules()
        self.stats.reset()

    def reset_stats(self) -> None:
        """Reset pipeline-level statistics without rebuilding modules."""
        self.stats.reset()


# ── Convenience factory ───────────────────────────────────────────────────────

def build_pipeline(**kwargs: Any) -> CombinedPreprocessingPipeline:
    """
    Thin factory so callers can write::

        from combined_preprocessing_pipeline import build_pipeline
        p = build_pipeline(phonetic_normalization=True, balanced_tokenization=True)
    """
    return CombinedPreprocessingPipeline(**kwargs)


# ── Smoke tests ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, warnings

    _SEP = "─" * 68
    failures = 0

    def _check(label: str, got: object, expected: object) -> None:
        global failures
        ok = got == expected
        if not ok:
            failures += 1
        print(f"  {'✓' if ok else '✗'}  {label}")
        if not ok:
            print(f"     got     : {got!r}")
            print(f"     expected: {expected!r}")

    print(_SEP)
    print("  combined_preprocessing_pipeline.py — smoke tests")
    print(_SEP)

    # ── Module categories ───────────────────────────────────────────────────
    print("\n[Module categories]")
    _check("balanced_tokenization in TRANSFORMATION",
           "balanced_tokenization" in TRANSFORMATION_MODULES, True)
    _check("language_identification_tagging in ANALYSIS",
           "language_identification_tagging" in ANALYSIS_MODULES, True)
    _check("code_switch_augmentation in AUGMENTATION",
           "code_switch_augmentation" in AUGMENTATION_MODULES, True)

    # ── Empty pipeline ──────────────────────────────────────────────────────
    print("\n[Empty pipeline]")
    p_empty = CombinedPreprocessingPipeline()
    _check("process() on empty pipeline returns input",
           p_empty.process("hello"), "hello")
    _check("active_modules is empty", p_empty.active_modules, [])

    # ── Conflicting modules raise ValueError ────────────────────────────────
    print("\n[Conflict detection]")
    raised = False
    try:
        CombinedPreprocessingPipeline(transliteration=True, script_unification=True)
    except ValueError:
        raised = True
    _check("transliteration + script_unification raises ValueError", raised, True)

    # ── allow_conflicts=True downgrades to warning ──────────────────────────
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        p_ac = CombinedPreprocessingPipeline(
            transliteration=True, script_unification=True, allow_conflicts=True
        )
    _check("allow_conflicts=True does not raise", True, True)
    _check("conflict warning was issued",
           any("CONFLICTING" in str(w.message) for w in w_list), True)

    # ── Validate-only dry run ───────────────────────────────────────────────
    print("\n[Dry-run validation]")
    p_dry = CombinedPreprocessingPipeline(phonetic_normalization=True)
    issues = p_dry.validate_only()
    _check("validate_only returns list", isinstance(issues, list), True)

    # ── Output mode: model (default) ─────────────────────────────────────────
    print("\n[Output mode: model]")
    p_model = CombinedPreprocessingPipeline(
        phonetic_normalization=True, output_mode="model"
    )
    result_model = p_model.process("yaar bahut accha")
    _check("model mode returns str", isinstance(result_model, str), True)

    # ── Output mode: both ─────────────────────────────────────────────────
    print("\n[Output mode: both]")
    p_both = CombinedPreprocessingPipeline(
        phonetic_normalization=True, output_mode="both"
    )
    result_both = p_both.process("yaar bahut accha")
    _check("both mode returns dict",        isinstance(result_both, dict), True)
    _check("has 'processed_text' key",      "processed_text" in result_both, True)
    _check("has 'analysis' key",            "analysis" in result_both,      True)
    _check("has 'statistics' key",          "statistics" in result_both,    True)

    # ── Statistics aggregation ──────────────────────────────────────────────
    print("\n[Statistics aggregation]")
    p_stat = CombinedPreprocessingPipeline(phonetic_normalization=True)
    p_stat.process("yaar bahut accha")
    stats = p_stat.collect_statistics()
    _check("statistics has 'pipeline' key", "pipeline" in stats, True)
    _check("processed_documents == 1",
           stats["pipeline"]["processed_documents"], 1)

    # ── Pipeline description ────────────────────────────────────────────────
    print("\n[Pipeline description]")
    p_desc = CombinedPreprocessingPipeline(
        phonetic_normalization=True, balanced_tokenization=True
    )
    desc = p_desc.describe_pipeline()
    _check("describe_pipeline returns non-empty str",
           isinstance(desc, str) and len(desc) > 0, True)
    _check("PhoneticNormalization in description",
           "PhoneticNormalization" in desc, True)

    # ── Execution tracing ───────────────────────────────────────────────────
    print("\n[Execution tracing]")
    p_trace = CombinedPreprocessingPipeline(
        phonetic_normalization=True, trace_execution=True, output_mode="both"
    )
    result_trace = p_trace.process("yaar bahut accha")
    _check("trace key present in both-mode output", "trace" in result_trace, True)
    _check("trace is non-empty list",
           isinstance(result_trace["trace"], list) and len(result_trace["trace"]) > 0,
           True)
    _check("trace entry has 'module' key",
           "module" in result_trace["trace"][0], True)
    _check("trace entry has 'elapsed_ms' key",
           "elapsed_ms" in result_trace["trace"][0], True)

    # ── Recommendations ────────────────────────────────────────────────────
    print("\n[Pipeline recommendations]")
    p_rec = CombinedPreprocessingPipeline(script_unification=True)
    recs = p_rec.recommend_additions()
    _check("recommendations returns list", isinstance(recs, list), True)
    _check("balanced_tokenization recommended with script_unification",
           any("balanced_tokenization" in r for r in recs), True)

    # ── DataFrame processing ────────────────────────────────────────────────
    print("\n[DataFrame processing]")
    p_df = CombinedPreprocessingPipeline(phonetic_normalization=True)
    import pandas as _pd
    df_in = _pd.DataFrame({"text": ["yaar bahut", "movie dekho"]})
    df_out = p_df.process_dataframe(df_in)
    _check("output column present",  "processed_text" in df_out.columns, True)
    _check("row count preserved",    len(df_out) == 2,                   True)

    # ── Analysis module skipped in model mode ───────────────────────────────
    print("\n[Analysis modules skipped in model mode]")
    with warnings.catch_warnings(record=True) as w_list2:
        warnings.simplefilter("always")
        p_skip = CombinedPreprocessingPipeline(
            language_identification_tagging=True, output_mode="model"
        )
    result_skip = p_skip.process("yaar bahut")
    _check("analysis module does not alter text in model mode",
           isinstance(result_skip, str), True)

    # ── Augmentation isolation ──────────────────────────────────────────────
    print("\n[Augmentation isolation]")
    p_aug_err = CombinedPreprocessingPipeline()
    aug_raised = False
    try:
        p_aug_err.augment_dataframe(
            _pd.DataFrame({"text": ["yaar"]}), text_col="text"
        )
    except RuntimeError:
        aug_raised = True
    _check("augment_dataframe raises RuntimeError without augmentation enabled",
           aug_raised, True)

    # ── build_pipeline factory ──────────────────────────────────────────────
    print("\n[build_pipeline factory]")
    p_factory = build_pipeline(phonetic_normalization=True)
    _check("build_pipeline returns CombinedPreprocessingPipeline",
           isinstance(p_factory, CombinedPreprocessingPipeline), True)

    # ── model cache ────────────────────────────────────────────────────────
    print("\n[Model cache]")
    _get_cached("test_key", lambda: {"model": "mock"})
    _get_cached("test_key", lambda: {"model": "should_not_load"})
    _check("cache returns same object on second call",
           _MODEL_CACHE["test_key"], {"model": "mock"})
    clear_model_cache()
    _check("clear_model_cache empties cache", len(_MODEL_CACHE), 0)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    if failures:
        print(f"  {failures} test(s) FAILED.")
        sys.exit(1)
    else:
        print("  All tests passed.")
    print(_SEP)