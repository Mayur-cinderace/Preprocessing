"""
_base.py — Shared base class and vocabulary constants for the Hinglish
preprocessing toolkit.

Hard requirements (no silent fallbacks):
    pip install nltk pandas
    python -c "import nltk; nltk.download('words')"

Every preprocessing module inherits ``HinglishBase`` and implements
``_process_token``.  All shared language-detection logic, vocabulary
constants, compiled regexes, and DataFrame/CSV helpers live here — no
duplication across module files.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import words as _nltk_corpus


# ── NLTK words corpus — hard requirement ──────────────────────────────────────

def _load_nltk_words() -> frozenset[str]:
    """Load the NLTK English words corpus.  Download it once if missing."""
    try:
        return frozenset(w.lower() for w in _nltk_corpus.words())
    except LookupError:
        nltk.download("words", quiet=False)
        try:
            return frozenset(w.lower() for w in _nltk_corpus.words())
        except LookupError as exc:
            raise LookupError(
                "NLTK 'words' corpus unavailable even after download.\n"
                "Run:  python -c \"import nltk; nltk.download('words')\""
            ) from exc


NLTK_WORDS: frozenset[str] = _load_nltk_words()


# ── Domain vocabulary constants ────────────────────────────────────────────────

SLANG_VOCAB: frozenset[str] = frozenset({
    "bro", "lol", "omg", "wtf", "idk", "pls", "thx", "insta",
    "dm", "ok", "okay", "yo", "bruh",
})
TECH_VOCAB: frozenset[str] = frozenset({
    "api", "json", "gpu", "cpu", "repo", "github", "login", "signup",
    "checkout",
})
SOCIAL_VOCAB: frozenset[str] = frozenset({
    "youtube", "whatsapp", "netflix", "instagram", "gmail",
})
INTERNET_VOCAB: frozenset[str] = frozenset({
    "wifi", "online", "offline", "app", "apps", "dm", "insta", "yt",
})
NLP_VOCAB: frozenset[str] = frozenset({
    "chatgpt", "openai", "leetcode", "kaggle", "huggingface",
    "transformers", "bert", "xlmr", "streamlit", "fastapi", "pytorch",
    "numpy", "pandas", "scikit", "colab", "jupyter", "keras",
    "aws", "gcp", "azure", "docker", "postgresql", "mongodb", "redis",
    "tensorflow", "scipy",
})

DOMAIN_VOCAB: frozenset[str] = (
    SLANG_VOCAB | TECH_VOCAB | SOCIAL_VOCAB | INTERNET_VOCAB | NLP_VOCAB
)

# Full English vocabulary (NLTK + domain terms)
ENGLISH_VOCAB: frozenset[str] = NLTK_WORDS | DOMAIN_VOCAB


# ── Canonical Hinglish lexicon (superset across all modules) ──────────────────

HINGLISH_LEXICON: frozenset[str] = frozenset({
    # Copula / auxiliaries
    "hai", "haan", "han", "nahi", "nahin", "na", "kya", "kyun", "kyu",
    "ho", "hun", "hu", "hoon", "hoga", "hogi", "hoge", "hota", "hoti", "hote",
    "tha", "thi",
    # Postpositions / particles
    "ka", "ki", "ke", "ko", "se", "mein", "me", "men", "par", "pe", "ab", "ya",
    "toh", "to", "bhi", "phir", "fir", "tab", "jab", "agar",
    # Pronouns
    "main", "mai", "hum", "ham", "tum", "aap",
    "mera", "meri", "mere", "tera", "teri", "tere",
    "unka", "unki", "unke", "iska", "iski", "iske",
    "wo", "woh", "ye", "yeh",
    # Conjunctions / discourse
    "aur", "ya",
    # Greetings / fillers
    "bhai", "yaar", "yar", "ji", "haanji", "theek", "thik",
    # Adjectives
    "acha", "accha", "achha", "bura", "sahi", "galat", "badhiya",
    "mast", "thoda", "thodi", "thora",
    # Quantifiers
    "bahut", "bohot", "bahot",
    # Aspect markers / verbs
    "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "unhonne", "inhonne", "hamne", "maine",
    "kiye", "kiya", "hua", "hui", "hue",
    "karna", "karta", "karti", "karte", "karen", "karke",
    "dekh", "dekho", "dekhna",
    "bol", "bolo", "sun", "suno",
    "chal", "chalo", "ja", "jao",
    "aao", "aana", "jana",
    "rakho", "rakha", "rakhna",
    "karo", "kar",
    # Common nouns / adverbs
    "samajh", "samjh", "ladki", "ghar", "khana",
    "padhai", "pyar", "sach", "jaldi",
})

# Romanisation patterns that signal a token is Hindi
_PHONETIC_CLUES: tuple[str, ...] = (
    "aa", "ai", "au", "bh", "dh", "kh", "ph", "sh", "th", "ch",
    "gh", "jh", "ky", "py", "ny", "ri", "ya",
)

# Compiled regexes shared by all modules
_TOKEN_RE:      re.Pattern[str] = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_DEVANAGARI_RE: re.Pattern[str] = re.compile(r"[\u0900-\u097F]")
_PUNCT_RE:      re.Pattern[str] = re.compile(r"[^\w\s]")


# ── Abstract base dataclass ────────────────────────────────────────────────────

@dataclass
class HinglishBase:
    """
    Abstract base for all Hinglish preprocessing components.

    Subclasses MUST implement ``_process_token``.
    Subclasses MAY override ``_setup`` for extra post-init logic.

    Language detection, vocab management, tokenisation, reconstruction, and
    DataFrame/CSV helpers are all handled here — never duplicated.
    """

    enabled:            bool          = True
    lowercase:          bool          = True
    preserve_punctuation: bool        = True
    english_min_len:    int           = 1
    # Replace wholesale for custom experiments; defaults to the canonical set.
    hinglish_lexicon:   frozenset[str] = field(
        default_factory=lambda: HINGLISH_LEXICON
    )

    def __post_init__(self) -> None:
        # Immutable after construction — no Optional wrapping, no None checks.
        self._english_vocab: frozenset[str] = ENGLISH_VOCAB
        self._setup()

    # ── Subclass hooks ────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Called at the end of ``__post_init__``.  Override for extra init."""

    def _process_token(self, token: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _process_token()"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, text: str) -> str:
        """Transform a single text string and return the result."""
        if not self.enabled or not isinstance(text, str):
            return text

        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()

        tokens = _TOKEN_RE.findall(text)
        processed: list[str] = []
        for tok in tokens:
            if _PUNCT_RE.fullmatch(tok):
                if self.preserve_punctuation:
                    processed.append(tok)
                continue
            processed.append(self._process_token(tok))
        return self._reconstruct(processed)

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
        out[output_col] = out[text_col].astype(str).apply(self.process)
        return out

    def process_csv(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        text_col: str = "text",
        output_col: str = "processed_text",
    ) -> pd.DataFrame:
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        df = pd.read_csv(input_csv)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        processed = self.process_dataframe(df, text_col=text_col, output_col=output_col)
        processed.to_csv(output_csv, index=False)
        return processed

    # ── Language detection ────────────────────────────────────────────────────

    def _detect_language(self, token: str) -> str:
        """Return ``'EN'``, ``'HI'``, or ``'UNK'``."""
        if self._is_english_word(token):
            return "EN"
        if self._is_hindi_token(token):
            return "HI"
        return "UNK"

    def _is_english_word(self, token: str) -> bool:
        return len(token) >= self.english_min_len and token in self._english_vocab

    def _is_hindi_token(self, token: str) -> bool:
        if not token:
            return False
        if self._contains_devanagari(token):
            return True
        if token in self.hinglish_lexicon:
            return True
        if token.isdigit() or self._is_english_word(token):
            return False
        return token.isalpha() and any(c in token for c in _PHONETIC_CLUES)

    def _contains_devanagari(self, token: str) -> bool:
        return bool(_DEVANAGARI_RE.search(token))

    # ── Reconstruction ────────────────────────────────────────────────────────

    def _reconstruct(self, tokens: list[str]) -> str:
        out: list[str] = []
        for tok in tokens:
            if not out:
                out.append(tok)
            elif _PUNCT_RE.fullmatch(tok):
                out[-1] += tok
            else:
                out.append(" " + tok)
        return "".join(out).strip()

    # ── Tokenisation helper (for subclasses that need raw tokens) ─────────────

    def _tokenize(self, text: str) -> list[str]:
        """NFKC-normalise and tokenise; applies ``lowercase`` setting."""
        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()
        return _TOKEN_RE.findall(text)
