"""
_base.py — Shared infrastructure layer for the Hinglish preprocessing toolkit.

Hard requirements (no silent fallbacks):
    pip install nltk pandas
    python -c "import nltk; nltk.download('words')"

What this module provides
─────────────────────────
  • Tokenization infrastructure  — a hierarchical regex tokenizer that
    preserves URLs, e-mail addresses, hashtags, @mentions, contractions,
    punctuated numbers, ordinary words, and punctuation as atomic units.

  • Lightweight language identification  — four-way per-token classification:
        EN       English (lexicon-assisted, heuristic)
        HI_DEV   Devanagari Hindi
        HI_ROM   Romanized Hindi / Hinglish
        UNK      Unclassified

  • Vocabulary resources  — NLTK English words corpus + curated domain
    vocabularies (SLANG_VOCAB, TECH_VOCAB, SOCIAL_VOCAB, INTERNET_VOCAB,
    NLP_VOCAB) assembled into ENGLISH_VOCAB; and a HINGLISH_LEXICON that
    contains ONLY canonical Hinglish forms (variant resolution is the
    responsibility of downstream modules).

  • Reconstruction utilities  — loss-minimizing token → string reconstruction
    with correct punctuation spacing.

  • DataFrame / CSV helpers  — process_dataframe and process_csv wrappers
    used by all downstream preprocessing classes.

What this module intentionally does NOT provide
───────────────────────────────────────────────
  • Module-specific normalization policies (e.g. transliteration, slang
    expansion, spelling-variant canonicalization) — those belong in
    BalancedTokenization, LanguageAwareNormalization, and their siblings.
  • External language-identification models (fastText, transformers, etc.).
  • Automatic NLTK corpus downloads — if the corpus is missing the module
    raises a deterministic LookupError with explicit installation instructions.

Subclass contract
─────────────────
  Every preprocessing module must:
    • inherit HinglishBase,
    • implement _process_token(token: str) -> str,
    • optionally override _setup() for post-init configuration.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from nltk.corpus import words as _nltk_corpus


# ── NLTK words corpus — hard requirement, no silent download ──────────────────

def _load_nltk_words() -> frozenset[str]:
    """
    Load the NLTK English words corpus as a lowercase frozenset.

    Raises
    ------
    LookupError
        If the corpus has not been downloaded.  The error message provides
        explicit installation instructions; no automatic download is attempted.
    """
    try:
        return frozenset(w.lower() for w in _nltk_corpus.words())
    except LookupError as exc:
        raise LookupError(
            "NLTK 'words' corpus is not available.\n"
            "Install it once with:\n"
            "    python -c \"import nltk; nltk.download('words')\"\n"
            "Then restart your Python session."
        ) from exc


NLTK_WORDS: frozenset[str] = _load_nltk_words()


# ── Domain vocabulary constants ────────────────────────────────────────────────
#
# English detection is lexicon-assisted and heuristic, not exhaustive.
# These curated sets extend NLTK_WORDS with domain-specific terms that are
# commonly used in Hinglish social-media and tech corpora.

SLANG_VOCAB: frozenset[str] = frozenset({
    "bro", "lol", "omg", "wtf", "idk", "pls", "plz", "thx", "insta",
    "dm", "ok", "okay", "yo", "bruh", "rn", "ngl", "tbh", "imo",
    "smh", "irl", "asap", "btw", "brb",
})
TECH_VOCAB: frozenset[str] = frozenset({
    "api", "json", "gpu", "cpu", "repo", "github", "login", "signup",
    "checkout", "ui", "ux", "sdk", "cli", "backend", "frontend",
})
SOCIAL_VOCAB: frozenset[str] = frozenset({
    "youtube", "whatsapp", "netflix", "instagram", "gmail", "twitter",
    "facebook", "linkedin", "snapchat", "telegram",
})
INTERNET_VOCAB: frozenset[str] = frozenset({
    "wifi", "online", "offline", "app", "apps", "dm", "insta", "yt",
    "url", "http", "https",
})
NLP_VOCAB: frozenset[str] = frozenset({
    "chatgpt", "openai", "leetcode", "kaggle", "huggingface",
    "transformers", "bert", "xlmr", "streamlit", "fastapi", "pytorch",
    "numpy", "pandas", "scikit", "colab", "jupyter", "keras",
    "aws", "gcp", "azure", "docker", "postgresql", "mongodb", "redis",
    "tensorflow", "scipy", "mlflow", "airflow", "pyspark",
})

DOMAIN_VOCAB: frozenset[str] = (
    SLANG_VOCAB | TECH_VOCAB | SOCIAL_VOCAB | INTERNET_VOCAB | NLP_VOCAB
)

# Full English vocabulary (NLTK corpus + domain terms).
# English detection against this set is heuristic — the corpus is large but
# not exhaustive, and domain terms evolve; downstream modules should not treat
# an EN classification as a guarantee.
ENGLISH_VOCAB: frozenset[str] = NLTK_WORDS | DOMAIN_VOCAB


# ── Canonical Hinglish lexicon ────────────────────────────────────────────────
#
# DESIGN INVARIANT:  This set contains ONLY canonical romanized Hinglish forms.
# Spelling variants (han, bohot, bahot, yar, achha, accha, kyu, thik, …) are
# intentionally excluded.  Variant resolution is the sole responsibility of
# downstream preprocessing modules (e.g. BalancedTokenization,
# LanguageAwareNormalization); including variants here would blur that boundary.
#
# A token in HINGLISH_LEXICON is classified as HI_ROM (romanized Hindi).

HINGLISH_LEXICON: frozenset[str] = frozenset({
    # Copula / auxiliaries
    "hai", "hain", "haan", "nahi", "na",
    "ho", "hoon", "hoga", "hogi", "honge", "hota", "hoti", "hote",
    "tha", "thi", "the",
    # Postpositions / particles
    "ka", "ki", "ke", "ko", "se", "mein", "par", "pe",
    "ab", "ya", "toh", "bhi", "phir", "tab", "jab", "agar",
    "tak", "liye", "bina", "saath",
    # Pronouns
    "main", "hum", "tum", "aap",
    "mera", "meri", "mere",
    "tera", "teri", "tere",
    "unka", "unki", "unke",
    "iska", "iski", "iske",
    "uska", "uski", "uske",
    "inka", "inki", "inke",
    "aapka", "aapki", "aapke",
    "humara", "humari", "humare",
    "tumhara", "tumhari", "tumhare",
    "wo", "ye", "is", "us",
    "koi", "sab", "sabhi", "kuch", "khud",
    "apna", "apni", "apne",
    # Conjunctions / discourse
    "aur", "lekin", "magar", "kyunki", "isliye",
    "sirf", "bas", "hi",
    # Question words
    "kya", "kyun", "kaise", "kaun", "kab", "kahan",
    "kitna", "kitni", "kitne",
    # Greetings / address terms
    "bhai", "yaar", "ji", "haanji", "arre", "oye", "wah",
    # Adjectives / intensifiers
    "bahut", "thoda", "thodi", "thode", "bilkul", "ekdum",
    "zyada", "kam", "kafi", "itna", "itni", "itne", "utna",
    "bada", "badi", "bade", "chota", "choti", "chote",
    "naya", "nayi", "naye", "purana", "purani", "purane",
    "sundar", "pyara", "pyari", "zabardast", "badhiya",
    "sahi", "galat", "mast", "theek",
    "accha",   # canonical adjective m.; achha/acha → accha via downstream modules
    "bura", "buri", "bure",
    # Common nouns
    "ghar", "khana", "paani", "dost", "pyar", "sach",
    "kaam", "din", "raat", "subah", "shaam", "padhai",
    "zindagi", "duniya", "log", "baat", "cheez", "jagah",
    "waqt", "naam", "mushkil", "matlab", "farak",
    # Verbs (root / infinitive / common conjugations)
    "karna", "karta", "karti", "karte", "karo", "kar",
    "karen", "karke", "kiya", "kiye",
    "dekh", "dekho", "dekhna", "dekha", "dekhi", "dekhe",
    "bol", "bolo", "bola", "boli",
    "sun", "suno", "suna",
    "chal", "chalo", "chala", "chali",
    "ja", "jao", "jana", "gaya", "gayi", "gaye",
    "aao", "aana", "aaya", "aayi",
    "rakho", "rakha", "rakhna",
    "samajh", "samjha", "samjhi",
    "lena", "lo", "liya",
    "dena", "do", "diya",
    "batao", "batana", "bata",
    "sochna", "socho", "socha",
    # Progressive / perfective auxiliaries
    "raha", "rahi", "rahe",
    "hua", "hui", "hue",
    # Ergative constructions
    "maine", "hamne", "tumne", "aapne", "usne", "unhonne", "inhonne",
    # Temporal / adverbial
    "jaldi", "dhire", "pehle", "baad", "kabhi", "hamesha",
    "aksar", "abhi", "seedha",
    # Politeness
    "shukriya", "dhanyawad", "maafi",
})


# ── Phonetic scoring constants for HI_ROM heuristic ──────────────────────────
#
# Used by _compute_hinglish_score to estimate whether an unrecognised alphabetic
# token is likely romanized Hindi.  All signals are computed on the lowercased
# token.  The design is deliberately simple — no external models.

_PHONETIC_BIGRAMS: tuple[str, ...] = (
    "aa", "ai", "au", "bh", "dh", "kh", "ph", "sh", "th", "ch",
    "gh", "jh", "ky", "py", "ny", "ri", "ya", "wa",
)
_HINDI_SUFFIXES: tuple[str, ...] = (
    "na", "kar", "ke", "ki", "ka", "ko", "se", "ta", "ti", "te",
    "ya", "yi", "ye", "oo", "hai", "hun", "hoon",
)
# Score threshold above which an unrecognised token is classified HI_ROM.
_HI_ROM_THRESHOLD: float = 0.45


def _compute_hinglish_score(token_lower: str) -> float:
    """
    Return a [0, 1] confidence score for the hypothesis that ``token_lower``
    is a romanized Hindi token.

    Scoring design
    --------------
    To reduce false positives on English words that happen to contain Hindi-
    like bigrams (e.g. "shadow" has "sh"), a *phonetic bigram hit alone* is
    insufficient to exceed the 0.45 threshold.  Both a bigram hit AND a suffix
    hit are required for the score to comfortably clear the bar; the vowel
    ratio provides a small tie-breaking bonus.

    Signal weights:
      +0.30  if at least one phonetic bigram is present.
      +0.25  if the token ends with a common Hindi suffix.
      +0.10  if the vowel ratio is in the typical Hindi range (0.25–0.55).
      −0.20  if token length ≤ 2 (ambiguous / likely abbreviation).

    Examples:
      "gaya"   → sh=0, ya=+0.30, suffix "ya"=+0.25, ratio=+0.10 → 0.65 ✓ HI_ROM
      "shadow" → sh=+0.30, no suffix → 0.30 + 0.10 = 0.40        → UNK  ✓
      "theek"  → th=+0.30, no suffix → 0.30                       → UNK  (lexicon handles it)
    """
    score = 0.0

    has_bigram = any(b in token_lower for b in _PHONETIC_BIGRAMS)
    if has_bigram:
        score += 0.30

    has_suffix = any(token_lower.endswith(s) for s in _HINDI_SUFFIXES)
    if has_suffix:
        score += 0.25

    vowels = sum(1 for c in token_lower if c in "aeiou")
    if len(token_lower) > 0:
        ratio = vowels / len(token_lower)
        if 0.25 <= ratio <= 0.55:
            score += 0.10

    if len(token_lower) <= 2:
        score -= 0.20

    return max(0.0, min(score, 1.0))


# ── Hierarchical tokenizer ────────────────────────────────────────────────────
#
# The tokenizer processes each pattern class in priority order so that longer,
# more specific patterns (URLs, e-mails, hashtags) are matched before simpler
# word/punctuation patterns.  This ensures that social-media artefacts are
# always emitted as single atomic tokens.

_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"""
    (?x)
    # Priority 1 — URLs (http/https and bare www.)
    # Stop before sentence-ending punctuation so trailing . , ! ? ; : are
    # not swallowed (e.g. "https://openai.com." → "https://openai.com" + ".")
    https?://[^\s.,!?;:]+
    | www\.[^\s.,!?;:]+

    # Priority 2 — E-mail addresses (same trailing-punctuation guard)
    | [A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?=[.,!?;:\s]|$)

    # Priority 3 — Hashtags and @mentions
    # Restricted to word characters only — trailing punctuation is excluded.
    | \#[A-Za-z0-9_]+
    | @[A-Za-z0-9_]+

    # Priority 4 — Numbers with embedded punctuation (98.6%, 12:30, 1,000)
    | \d[\d,._:/\-]*\d
    | \d+[%]?

    # Priority 5 — Contractions (English: can't, I'm, it's …)
    | [A-Za-z]+(?:\'[A-Za-z]+)+

    # Priority 6 — Unicode words (covers Devanagari, Latin, etc.)
    | \w+

    # Priority 7 — Individual punctuation / emoji
    | [^\w\s]
    """,
    re.UNICODE | re.VERBOSE,
)

_PUNCT_RE:      re.Pattern[str] = re.compile(r"^[^\w\s]+$",   re.UNICODE)

# Punctuation attachment classes for near-lossless reconstruction.
# Left-attaching: no space before (attach to the token on the left).
_PUNCT_LEFT:  frozenset[str] = frozenset(".,!?;:%)]}")
# Right-attaching: no space after (attach to the token on the right).
_PUNCT_RIGHT: frozenset[str] = frozenset("([{")
# Symmetric (quotes/apostrophes): heuristically left-attach when closing,
# right-attach when opening.  We treat them as left-attaching by default,
# which is correct for the majority of Hinglish social-media sentences.
_PUNCT_SYM:   frozenset[str] = frozenset("\"'")
_DEVANAGARI_RE: re.Pattern[str] = re.compile(r"[\u0900-\u097F]")


# ── Language label type ───────────────────────────────────────────────────────

LangLabel = Literal["EN", "HI_DEV", "HI_ROM", "UNK"]


# ── Abstract base dataclass ────────────────────────────────────────────────────

@dataclass
class HinglishBase:
    """
    Abstract base for all Hinglish preprocessing components.

    Provides
    --------
    • Hierarchical tokenization that preserves URLs, e-mails, hashtags,
      @mentions, contractions, punctuated numbers, and emoji as atomic units.
    • Four-way language identification: EN | HI_DEV | HI_ROM | UNK.
    • Shared vocabulary resources: ENGLISH_VOCAB, HINGLISH_LEXICON.
    • Loss-minimizing reconstruction with correct punctuation spacing.
    • process_dataframe and process_csv helpers.

    Subclass contract
    -----------------
    • Implement ``_process_token(token: str) -> str``.
    • Optionally override ``_setup()`` for post-init configuration.

    Notes
    -----
    ``lowercase`` defaults to False so that meaningful casing (NASA, GPT,
    OpenAI, BERT) is preserved at the base layer.  Individual preprocessing
    modules may set ``lowercase=True`` if their pipeline requires it.

    English detection is heuristic and corpus-assisted, not exhaustive.
    A token not in ENGLISH_VOCAB does not conclusively indicate Hindi.
    """

    enabled:              bool           = True
    lowercase:            bool           = False   # changed from True — see docstring
    preserve_punctuation: bool           = True
    english_min_len:      int            = 1
    hi_rom_threshold:     float          = _HI_ROM_THRESHOLD
    # Downstream modules may supply a custom lexicon for HI_ROM detection.
    hinglish_lexicon:     frozenset[str] = field(
        default_factory=lambda: HINGLISH_LEXICON
    )

    def __post_init__(self) -> None:
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
        """Tokenize, transform each token via ``_process_token``, reconstruct."""
        if not self.enabled or not isinstance(text, str):
            return text

        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()

        tokens = _TOKEN_PATTERN.findall(text)
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
        """Apply ``process`` to every row of ``df[text_col]``."""
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
        """Read ``input_csv``, apply ``process_dataframe``, write ``output_csv``."""
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        df = pd.read_csv(input_csv)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        processed = self.process_dataframe(
            df, text_col=text_col, output_col=output_col
        )
        processed.to_csv(output_csv, index=False)
        return processed

    # ── Language identification (four-way) ────────────────────────────────────

    def _detect_language(self, token: str) -> LangLabel:
        """Classify a token as EN | HI_DEV | HI_ROM | UNK. See ``_detect_language_with_score``."""
        label, _ = self._detect_language_with_score(token)
        return label

    def _detect_language_with_score(self, token: str) -> tuple[LangLabel, float]:
        """
        Classify a token and return ``(label, confidence)`` where confidence
        is a float in [0, 1].

        Confidence semantics
        --------------------
        HI_DEV  → 1.0 always  (script-level, unambiguous)
        HI_ROM  → 1.0 if the token is in the canonical lexicon; otherwise
                   the raw heuristic score from ``_compute_hinglish_score``.
        EN      → 1.0 always  (vocabulary-level, treated as certain)
        UNK     → 0.0 always

        This lets downstream modules threshold on confidence:

            label, conf = base._detect_language_with_score(token)
            if label == "HI_ROM" and conf > 0.7:
                # high-confidence romanized Hindi — apply transliteration
                ...

        Decision order
        --------------
        1. Devanagari script → HI_DEV, 1.0.
        2. Canonical Hinglish lexicon hit → HI_ROM, 1.0.
        3. English vocabulary hit → EN, 1.0.
        4. Heuristic phonetic scoring for alphabetic tokens:
             score ≥ hi_rom_threshold → HI_ROM, score
             otherwise                → UNK, 0.0.
        """
        if not token:
            return "UNK", 0.0
        tok_lower = token.lower()

        if self._contains_devanagari(token):
            return "HI_DEV", 1.0

        if tok_lower in self.hinglish_lexicon:
            return "HI_ROM", 1.0

        if self._is_english_word(token):
            return "EN", 1.0

        if token.isalpha():
            score = _compute_hinglish_score(tok_lower)
            if score >= self.hi_rom_threshold:
                return "HI_ROM", score

        return "UNK", 0.0

    def _is_english_word(self, token: str) -> bool:
        """
        Return True if ``token`` is likely English.

        Detection is lexicon-assisted and heuristic: membership in
        ENGLISH_VOCAB (NLTK words + domain terms).  A False result does not
        conclusively indicate the token is Hindi.
        """
        return (
            len(token) >= self.english_min_len
            and token.lower() in self._english_vocab
        )

    def _is_hinglish_token(self, token: str) -> bool:
        """
        Return True if the token is classified as HI_DEV or HI_ROM.

        This is the preferred binary Hindi-detection helper for subclasses
        that do not need the full four-way label.

        Backward-compatibility alias: ``_is_hindi_token`` is preserved as an
        alias so existing subclasses do not require modification.
        """
        label = self._detect_language(token)
        return label in ("HI_DEV", "HI_ROM")

    # Backward-compatibility alias for subclasses that call _is_hindi_token
    _is_hindi_token = _is_hinglish_token

    def _contains_devanagari(self, token: str) -> bool:
        """Return True if ``token`` contains at least one Devanagari character."""
        return bool(_DEVANAGARI_RE.search(token))

    # ── Tokenization helper ───────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """
        NFKC-normalise and tokenise ``text`` using the hierarchical tokenizer.

        Applies the ``lowercase`` setting.  Returns raw token strings including
        punctuation; callers are responsible for filtering as needed.
        """
        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()
        return _TOKEN_PATTERN.findall(text)

    # ── Reconstruction ────────────────────────────────────────────────────────

    def _reconstruct(self, tokens: list[str]) -> str:
        """
        Reconstruct a token list into a string with near-lossless spacing.

        Punctuation attachment rules
        ----------------------------
        Left-attaching  (.,!?;:%)]})  — no space before; space after.
        Right-attaching (([{)          — space before first occurrence,
                                         no space after (next token joins).
        Symmetric       (\"')          — treated as left-attaching by default,
                                         which is correct for closing quotes
                                         in the vast majority of sentences.
        All other tokens               — separated by a single space.
        """
        out: list[str] = []
        # Track whether the next word token should suppress its leading space
        # because a right-attaching bracket was just emitted.
        _suppress_next_space = False

        for tok in tokens:
            if not _PUNCT_RE.fullmatch(tok):
                # Ordinary word / Devanagari / contraction / artefact
                if not out:
                    out.append(tok)
                elif _suppress_next_space:
                    out.append(tok)          # join directly after "([{"
                else:
                    out.append(" " + tok)
                _suppress_next_space = False
            else:
                # Punctuation token
                if not out:
                    # First token is punctuation (e.g. leading quote)
                    out.append(tok)
                    if tok in _PUNCT_RIGHT:
                        _suppress_next_space = True
                elif tok in _PUNCT_LEFT or tok in _PUNCT_SYM:
                    out[-1] += tok           # attach to preceding token
                    _suppress_next_space = False
                elif tok in _PUNCT_RIGHT:
                    out.append(" " + tok)    # space before "([{"
                    _suppress_next_space = True
                else:
                    out[-1] += tok           # unknown punct → left-attach
                    _suppress_next_space = False

        return "".join(out).strip()


# ── Smoke tests ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _SEP = "─" * 70

    # Minimal concrete subclass for testing
    @dataclass
    class _PassThrough(HinglishBase):
        def _process_token(self, token: str) -> str:
            return token

    pt = _PassThrough(lowercase=False, preserve_punctuation=True)
    failures = 0

    def _check(label: str, got: object, expected: object) -> None:
        global failures
        status = "✓" if got == expected else "✗"
        if got != expected:
            failures += 1
        print(f"  {status}  {label}")
        if got != expected:
            print(f"     got     : {got!r}")
            print(f"     expected: {expected!r}")

    print(_SEP)
    print("  _base.py — smoke tests")
    print(_SEP)

    # ── Language detection ────────────────────────────────────────────────────
    print("\n[Language detection]")
    _check("English word 'movie'",        pt._detect_language("movie"),            "EN")
    _check("English word 'office'",       pt._detect_language("office"),           "EN")
    _check("English slang 'lol'",         pt._detect_language("lol"),              "EN")
    _check("Devanagari 'है'",             pt._detect_language("है"),               "HI_DEV")
    _check("Devanagari 'बहुत'",          pt._detect_language("बहुत"),             "HI_DEV")
    _check("Canonical HI_ROM 'bahut'",    pt._detect_language("bahut"),            "HI_ROM")
    _check("Canonical HI_ROM 'yaar'",     pt._detect_language("yaar"),             "HI_ROM")
    _check("Canonical HI_ROM 'mein'",     pt._detect_language("mein"),             "HI_ROM")
    _check("Heuristic HI_ROM 'gaya'",     pt._detect_language("gaya"),             "HI_ROM")
    _check("UNK short token 'xy'",        pt._detect_language("xy"),               "UNK")

    # ── Tokenizer — social-media artefacts as atomic tokens ───────────────────
    print("\n[Hierarchical tokenizer — atomic preservation]")
    toks = pt._tokenize("Check https://example.com and cinder@email.com #MondayMotivation @username")
    _check("URL preserved",      "https://example.com" in toks, True)
    _check("email preserved",    "cinder@email.com"    in toks, True)
    _check("hashtag preserved",  "#MondayMotivation"   in toks, True)
    _check("mention preserved",  "@username"           in toks, True)

    toks2 = pt._tokenize("score 98.6% can't")
    _check("punctuated number preserved", "98.6%" in toks2,  True)
    _check("contraction preserved",       "can't"  in toks2, True)

    # Trailing punctuation must NOT be swallowed into URL / hashtag / mention
    toks3 = pt._tokenize("Visit https://openai.com.")
    _check("URL stops before trailing '.'",     "https://openai.com" in toks3, True)
    _check("trailing '.' is separate token",    "."                  in toks3, True)

    toks4 = pt._tokenize("@user, thanks!")
    _check("@mention stops before comma",   "@user" in toks4,  True)
    _check("comma is separate token",       ","     in toks4,  True)
    _check("'thanks' is separate token",    "thanks" in toks4, True)

    toks5 = pt._tokenize("#MondayMotivation!!!")
    _check("#hashtag stops before '!'",     "#MondayMotivation" in toks5, True)
    _check("exclamations are separate",     "!" in toks5,               True)

    # ── Reconstruction fidelity ───────────────────────────────────────────────
    print("\n[Reconstruction fidelity]")
    cases = [
        ("yaar, kya hua bhai?",          "yaar, kya hua bhai?"),
        ("main office mein hoon.",        "main office mein hoon."),
        ("#MondayMotivation yaar sun",    "#MondayMotivation yaar sun"),
        ("@username check this out",      "@username check this out"),
        ("it's great, isn't it?",         "it's great, isn't it?"),
        ("price is 1,000 rupees only",    "price is 1,000 rupees only"),
    ]
    for text, expected in cases:
        got = pt.process(text)
        _check(repr(text), got, expected)

    # ── Mixed-language sentence ───────────────────────────────────────────────
    print("\n[Mixed-language sentence — pass-through processor]")
    mixed = "yaar this movie was bahut accha lol"
    got = pt.process(mixed)
    _check("mixed-language pass-through", got, mixed)

    # ── English detection helpers ─────────────────────────────────────────────
    print("\n[English detection]")
    _check("_is_english_word('movie')",   pt._is_english_word("movie"),   True)
    _check("_is_english_word('bahut')",   pt._is_english_word("bahut"),   False)
    _check("_is_english_word('ok')",      pt._is_english_word("ok"),      True)

    # ── Language detection — four-way ─────────────────────────────────────────
    print("\n[Language detection — four-way]")
    _check("English word 'movie'",          pt._detect_language("movie"),   "EN")
    _check("English word 'office'",         pt._detect_language("office"),  "EN")
    _check("English slang 'lol'",           pt._detect_language("lol"),     "EN")
    _check("Devanagari 'है'",               pt._detect_language("है"),      "HI_DEV")
    _check("Devanagari 'बहुत'",            pt._detect_language("बहुत"),    "HI_DEV")
    _check("Canonical HI_ROM 'bahut'",      pt._detect_language("bahut"),   "HI_ROM")
    _check("Canonical HI_ROM 'yaar'",       pt._detect_language("yaar"),    "HI_ROM")
    _check("Canonical HI_ROM 'mein'",       pt._detect_language("mein"),    "HI_ROM")
    _check("Canonical HI_ROM 'accha'",      pt._detect_language("accha"),   "HI_ROM")
    _check("Heuristic HI_ROM 'gaya'",       pt._detect_language("gaya"),    "HI_ROM")
    _check("UNK short token 'xy'",          pt._detect_language("xy"),      "UNK")
    # False-positive guard: "shadow" has "sh" bigram but no Hindi suffix
    _check("'shadow' is NOT HI_ROM (false-positive guard)",
           pt._detect_language("shadow"), "EN")   # 'shadow' is in NLTK_WORDS

    # ── _detect_language_with_score ───────────────────────────────────────────
    print("\n[Language detection — confidence scores]")
    lbl, conf = pt._detect_language_with_score("है")
    _check("HI_DEV confidence = 1.0",  (lbl, conf), ("HI_DEV", 1.0))

    lbl, conf = pt._detect_language_with_score("bahut")
    _check("HI_ROM lexicon confidence = 1.0", (lbl, conf), ("HI_ROM", 1.0))

    lbl, conf = pt._detect_language_with_score("movie")
    _check("EN confidence = 1.0", (lbl, conf), ("EN", 1.0))

    lbl, conf = pt._detect_language_with_score("gaya")
    _check("HI_ROM heuristic label", lbl, "HI_ROM")
    _check("HI_ROM heuristic confidence ∈ [0.45, 1.0]", 0.45 <= conf <= 1.0, True)

    lbl, conf = pt._detect_language_with_score("xy")
    _check("UNK confidence = 0.0", (lbl, conf), ("UNK", 0.0))

    # ── Hinglish detection helpers ────────────────────────────────────────────
    print("\n[Hinglish detection]")
    _check("_is_hinglish_token('yaar')",   pt._is_hinglish_token("yaar"),  True)
    _check("_is_hinglish_token('है')",     pt._is_hinglish_token("है"),    True)
    _check("_is_hinglish_token('movie')",  pt._is_hinglish_token("movie"), False)
    _check("_is_hindi_token alias works",  pt._is_hindi_token("yaar"),     True)

    # ── Canonical lexicon spot-checks ─────────────────────────────────────────
    print("\n[Canonical lexicon]")
    _check("'accha' in HINGLISH_LEXICON",   "accha"  in HINGLISH_LEXICON, True)
    _check("'acha' NOT in HINGLISH_LEXICON","acha"   in HINGLISH_LEXICON, False)
    _check("'samajh' in HINGLISH_LEXICON",  "samajh" in HINGLISH_LEXICON, True)
    _check("'samjha' in HINGLISH_LEXICON",  "samjha" in HINGLISH_LEXICON, True)
    _check("'samajha' NOT in HINGLISH_LEXICON", "samajha" in HINGLISH_LEXICON, False)
    _check("'bahut' in HINGLISH_LEXICON",   "bahut"  in HINGLISH_LEXICON, True)
    _check("'bohot' NOT in HINGLISH_LEXICON","bohot"  in HINGLISH_LEXICON, False)
    _check("'yaar' in HINGLISH_LEXICON",    "yaar"   in HINGLISH_LEXICON, True)
    _check("'yar' NOT in HINGLISH_LEXICON", "yar"    in HINGLISH_LEXICON, False)

    # ── Devanagari check ──────────────────────────────────────────────────────
    print("\n[Devanagari detection]")
    _check("contains_devanagari('है')",    pt._contains_devanagari("है"),    True)
    _check("contains_devanagari('yaar')",  pt._contains_devanagari("yaar"),  False)
    _check("contains_devanagari('movie')", pt._contains_devanagari("movie"), False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    if failures:
        print(f"  {failures} test(s) FAILED.")
        sys.exit(1)
    else:
        print("  All tests passed.")
    print(_SEP)