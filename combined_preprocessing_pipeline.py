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

Design:
    - Every module is imported directly (no try/except, no ``strict_imports``
      toggle, no silent disable-on-failure).  If a module's hard dependency is
      absent, the import raises immediately with a clear message.
    - ``code_switch_augmentation`` expands rows rather than transforming 1-to-1
      and is deliberately excluded from the main ``process`` / ``process_csv``
      pipeline.  Use ``augment_dataframe`` / ``augment_csv`` instead.
    - Pipeline order is validated at construction time.
    - Module configs use ``dataclasses.fields()`` for introspection — no
      fragile ``__code__.co_varnames`` hacks.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

# ── Registry ──────────────────────────────────────────────────────────────────

_MODULE_REGISTRY: dict[str, type] = {
    "balanced_tokenization":          BalancedTokenization,
    "language_identification_tagging": LanguageIdentificationTagging,
    "language_aware_normalization":   LanguageAwareNormalization,
    "transliteration":                Transliteration,
    "switch_point_encoding":          SwitchPointEncoding,
    "code_switch_augmentation":       CodeSwitchAugmentation,
    "context_aware_subword_sampling": ContextAwareSentencePieceDropout,
    "phonetic_normalization":         PhoneticNormalization,
    "script_unification":             ScriptUnification,
}

# Keys that participate in the ordered text-transformation pipeline.
# ``code_switch_augmentation`` is intentionally excluded — it expands rows.
_DEFAULT_PIPELINE_ORDER: list[str] = [
    "phonetic_normalization",
    "language_aware_normalization",
    "script_unification",
    "transliteration",
    "balanced_tokenization",
    "language_identification_tagging",
    "switch_point_encoding",
    "context_aware_subword_sampling",
]

# Combinations that are conceptually redundant; warn the caller.
_REDUNDANT_PAIRS: list[tuple[str, str]] = [
    ("transliteration",               "script_unification"),
    ("language_identification_tagging", "switch_point_encoding"),
]


class CombinedPreprocessingPipeline:
    """
    Research-grade orchestration layer for modular Hinglish preprocessing.

    Usage
    -----
    >>> pipeline = CombinedPreprocessingPipeline(
    ...     phonetic_normalization=True,
    ...     language_aware_normalization=True,
    ...     balanced_tokenization=True,
    ... )
    >>> pipeline.process("yaar movie bahuuut acchi thi")
    >>> pipeline.process_csv("data/hinglish.csv", "data/out.csv")

    For data augmentation (row-expanding):
    >>> pipeline = CombinedPreprocessingPipeline(code_switch_augmentation=True)
    >>> pipeline.augment_csv("data/hinglish.csv", "data/augmented.csv")

    Parameters
    ----------
    <module_name>=True|False
        Toggle each preprocessing module.
    module_configs
        Per-module keyword arguments, e.g.
        ``{"phonetic_normalization": {"normalize_phonetics": False}}``.
    custom_pipeline_order
        Override the default module execution order (list of module keys).
    random_seed
        Global seed forwarded to stochastic modules.
    verbose
        Print progress messages to stdout.
    """

    def __init__(
        self,
        balanced_tokenization:          bool = False,
        language_identification_tagging: bool = False,
        language_aware_normalization:   bool = False,
        transliteration:                bool = False,
        switch_point_encoding:          bool = False,
        code_switch_augmentation:       bool = False,
        context_aware_subword_sampling: bool = False,
        phonetic_normalization:         bool = False,
        script_unification:             bool = False,
        module_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        custom_pipeline_order: Optional[List[str]] = None,
        random_seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self._enabled: dict[str, bool] = {
            "balanced_tokenization":          balanced_tokenization,
            "language_identification_tagging": language_identification_tagging,
            "language_aware_normalization":   language_aware_normalization,
            "transliteration":                transliteration,
            "switch_point_encoding":          switch_point_encoding,
            "code_switch_augmentation":       code_switch_augmentation,
            "context_aware_subword_sampling": context_aware_subword_sampling,
            "phonetic_normalization":         phonetic_normalization,
            "script_unification":             script_unification,
        }
        self._module_configs  = module_configs or {}
        self._random_seed     = random_seed
        self._verbose         = verbose
        self._pipeline_order  = self._resolve_pipeline_order(custom_pipeline_order)

        self._warn_redundant_combinations()
        self._modules: dict[str, Any] = self._build_modules()
        self._log("Active modules: " + (", ".join(self.active_modules) or "(none)"))

    # ── Construction helpers ──────────────────────────────────────────────────

    def _resolve_pipeline_order(self, custom: Optional[List[str]]) -> list[str]:
        order = custom if custom is not None else list(_DEFAULT_PIPELINE_ORDER)
        unknown = [k for k in order if k not in _MODULE_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown module keys in pipeline order: {unknown}\n"
                f"Valid keys: {list(_MODULE_REGISTRY)}"
            )
        return order

    def _warn_redundant_combinations(self) -> None:
        import warnings
        for a, b in _REDUNDANT_PAIRS:
            if self._enabled.get(a) and self._enabled.get(b):
                warnings.warn(
                    f"Enabling both '{a}' and '{b}' is likely redundant; "
                    f"review your configuration.",
                    UserWarning,
                    stacklevel=3,
                )

    def _build_modules(self) -> dict[str, Any]:
        modules: dict[str, Any] = {}
        for key, cls in _MODULE_REGISTRY.items():
            if not self._enabled.get(key):
                continue
            config = dict(self._module_configs.get(key, {}))
            # Forward random_seed only to modules that declare it as a field.
            field_names = {f.name for f in dataclasses.fields(cls)}
            if "random_seed" in field_names:
                config.setdefault("random_seed", self._random_seed)
            modules[key] = cls(enabled=True, **config)
        return modules

    # ── Pipeline execution ────────────────────────────────────────────────────

    def _active_order(self) -> list[str]:
        """Keys in pipeline order that are enabled and instantiated."""
        return [k for k in self._pipeline_order if k in self._modules]

    @property
    def active_modules(self) -> list[str]:
        return self._active_order()

    def get_pipeline_config(self) -> dict[str, Any]:
        return {
            "enabled":        dict(self._enabled),
            "pipeline_order": list(self._pipeline_order),
            "module_configs": dict(self._module_configs),
            "random_seed":    self._random_seed,
        }

    def process(self, text: str) -> str:
        """Apply all enabled pipeline modules to a single text string."""
        if not isinstance(text, str):
            return text
        value = text
        for key in self._active_order():
            value = self._modules[key].process(value)
        return value

    def process_batch(self, texts: Iterable[str]) -> list[str]:
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
        for key in self._active_order():
            out[output_col] = out[current_col].astype(str).apply(
                self._modules[key].process
            )
            current_col = output_col
        if output_col not in out.columns:
            # No module ran — copy source column verbatim.
            out[output_col] = out[text_col].astype(str)
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
        text_col: str = "text",
        output_col: str = "processed_text",
    ) -> dict[str, Any]:
        """
        Run the dataset through every enabled module in isolation, a set of
        preset combinations, and the full pipeline.  Writes one CSV per
        experiment plus a JSON manifest.

        Returns the manifest dict.
        """
        input_csv  = Path(input_csv)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        experiments: list[dict] = []

        def _run(name: str, **kwargs: bool) -> None:
            out_path = output_dir / f"{name}.csv"
            p = CombinedPreprocessingPipeline(
                module_configs=self._module_configs,
                random_seed=self._random_seed,
                verbose=self._verbose,
                **kwargs,
            )
            p.process_csv(input_csv, out_path, text_col=text_col, output_col=output_col)
            experiments.append({"name": name, "modules": [k for k, v in kwargs.items() if v]})
            self._log(f"Experiment '{name}' → {out_path}")

        # Baseline (no modules).
        _run("baseline")

        # One module at a time.
        for key in _DEFAULT_PIPELINE_ORDER:
            if self._enabled.get(key):
                _run(key, **{key: True})

        # Preset combos.
        if self._enabled.get("phonetic_normalization") or self._enabled.get("script_unification"):
            _run(
                "phonetic_plus_script",
                phonetic_normalization=True,
                script_unification=True,
            )

        # Full pipeline (all enabled modules).
        enabled_kwargs = {k: v for k, v in self._enabled.items() if v}
        if enabled_kwargs:
            _run("full_pipeline", **enabled_kwargs)

        manifest = {
            "dataset":     str(input_csv),
            "schema":      [text_col, "sentiment", "label"],
            "experiments": experiments,
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
        """Rebuild all module instances (e.g. to reset RNG state)."""
        self._modules = self._build_modules()


# ── Convenience factory ───────────────────────────────────────────────────────

def build_pipeline(**kwargs: Any) -> CombinedPreprocessingPipeline:
    """
    Thin factory so callers can write::

        from combined_preprocessing_pipeline import build_pipeline
        p = build_pipeline(phonetic_normalization=True, balanced_tokenization=True)
    """
    return CombinedPreprocessingPipeline(**kwargs)
