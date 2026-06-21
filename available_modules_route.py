from typing import Any, Dict, List
 
# Mirrors combined_preprocessing_pipeline.TRANSFORMATION_MODULES /
# ANALYSIS_MODULES / AUGMENTATION_MODULES, expressed per-module for the
# frontend's category grouping (CATEGORY_ORDER = ['TRANSFORMATION','ANALYSIS','AUGMENTATION']).
_MODULE_METADATA: List[Dict[str, Any]] = [
    {
        "key": "phonetic_normalization",
        "label": "Phonetic Normalization",
        "tag": "PHON",
        "category": "TRANSFORMATION",
        "desc": "Normalizes phonetic/romanization variance in Hinglish spellings before script or tokenization steps run.",
        "conflicts": [],
    },
    {
        "key": "language_aware_normalization",
        "label": "Language Aware Normalization",
        "tag": "NORM",
        "category": "TRANSFORMATION",
        "desc": "Applies normalization rules conditioned on the detected language of each token span.",
        "conflicts": [],
    },
    {
        "key": "script_unification",
        "label": "Script Unification",
        "tag": "SCRIPT",
        "category": "TRANSFORMATION",
        "desc": "Unifies mixed-script text (Devanagari/Latin) into a single target script.",
        "conflicts": ["transliteration"],
    },
    {
        "key": "transliteration",
        "label": "Transliteration",
        "tag": "XLIT",
        "category": "TRANSFORMATION",
        "desc": "Transliterates text between scripts. Conflicts with Script Unification — both rewrite script identity and should not run together.",
        "conflicts": ["script_unification"],
    },
    {
        "key": "balanced_tokenization",
        "label": "Balanced Tokenization",
        "tag": "TOK",
        "category": "TRANSFORMATION",
        "desc": "Tokenizes mixed-language text with balanced subword fertility across languages.",
        "conflicts": [],
    },
    {
        "key": "context_aware_subword_sampling",
        "label": "Context Aware Subword Sampling",
        "tag": "SUBWORD",
        "category": "TRANSFORMATION",
        "desc": "SentencePiece dropout-based subword sampling, conditioned on surrounding context. Always runs last in the pipeline.",
        "conflicts": [],
    },
    {
        "key": "language_identification_tagging",
        "label": "Language Identification Tagging",
        "tag": "LANGID",
        "category": "ANALYSIS",
        "desc": "Annotates each token with its identified language. Only runs when analysis output is enabled; otherwise skipped.",
        "conflicts": [],
    },
    {
        "key": "switch_point_encoding",
        "label": "Switch Point Encoding",
        "tag": "SWITCH",
        "category": "ANALYSIS",
        "desc": "Marks code-switch boundaries between languages. Only runs when analysis output is enabled; otherwise skipped.",
        "conflicts": [],
    },
    {
        "key": "code_switch_augmentation",
        "label": "Code Switch Augmentation",
        "tag": "AUG",
        "category": "AUGMENTATION",
        "desc": "Expands the corpus with synthetic code-switched rows. Row-expanding — only runs via augment_dataframe/augment_csv, never via process().",
        "conflicts": [],
    },
]
 
 
def get_available_modules_payload() -> Dict[str, Any]:
    return {"modules": _MODULE_METADATA}