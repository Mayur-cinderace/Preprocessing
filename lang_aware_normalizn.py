"""
lang_aware_normalizn.py — Language-aware normalization for Hinglish text.

Hard requirements:
    (NLTK words corpus via _base.py)

Objective
---------
Language-Aware Normalization performs *language-preserving orthographic
normalization* for noisy code-switched Hinglish text.  It reduces orthographic
variability introduced by informal typing conventions — character elongations,
ad-hoc spelling variants, and common slang — while preserving:

    (a) language identity   — EN and HI tokens remain in their source language;
    (b) expressive sentiment cues — moderate elongations ("soooo") are
        optionally retained as sentiment signals;
    (c) code-switch boundaries — the sequence of languages in the input is not
        altered.

This module is NOT:
    * a translation system,
    * a semantic rewriting system,
    * an aggressive text simplification or normalization module.
    No stemming, lemmatization, or stop-word removal is performed.

Processing pipeline (per token)
--------------------------------
  1. Pass-through guards: skip empty, numeric, and social-media artifact
     tokens (hashtags, mentions, URLs, e-mail addresses, emojis / emoticons,
     punctuation-embedded numbers).
  2. Language detection: classify token as EN, HI, or UNK.
  3. Route to the appropriate language-specific pipeline:

     EN pipeline (when ``normalize_english=True``):
       a. Repetition compression.
       b. Optional slang expansion (when ``expand_slang=True``).

     HI pipeline (when ``normalize_hindi=True``):
       a. Repetition compression.
       b. Vowel-elongation reduction (beyond configurable threshold).
       c. Dictionary-based canonicalization (PRIMARY mechanism — always takes
          precedence over regex corrections).
       d. Regex-based OOV correction (fallback for tokens not in the dict).

     UNK pipeline (when ``normalize_unknown=True``):
       a. Generic repetition compression only.

  4. Return the (possibly normalized) token; the surrounding text structure
     is not modified.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from _base import HINGLISH_LEXICON, HinglishBase


# ── Social-media artifact guards ──────────────────────────────────────────────
#
# These patterns identify tokens that must pass through completely unchanged
# regardless of their detected language, to avoid corrupting social-media
# artefacts that carry no normalisable orthographic noise.

_SOCIAL_GUARDS: list[re.Pattern] = [
    re.compile(r"^#\S+$"),                          # hashtag
    re.compile(r"^@\S+$"),                          # @mention
    re.compile(r"^https?://\S+$"),                  # URL (http / https)
    re.compile(r"^www\.\S+$"),                      # URL (www.)
    re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"),  # email
    re.compile(r"\d[\d,._:/\-]\d"),                 # numbers with punctuation
    # Emoji / emoticons — Unicode ranges for common emoji blocks
    re.compile(
        r"(?:"
        r"[\U0001F300-\U0001F9FF"
        r"\U00002600-\U000027BF"
        r"\U0001FA00-\U0001FA9F"
        r"\u2764\u2665\u2660\u2663\u2666]"
        r"|:\)"
        r"|:\("
        r"|:D"
        r"|;\)"
        r"|:-\)"
        r"|:\|"
        r"|:P"
        r"|XD"
        r"|<3"
        r")",
        re.UNICODE,
    )
]


def _is_social_artifact(token: str) -> bool:
    """Return True if the token is a social-media artefact that must not be modified."""
    return any(pat.search(token) for pat in _SOCIAL_GUARDS)


# ── Canonical spelling-variant map ────────────────────────────────────────────
#
# Keys are romanized Hinglish surface forms observed in real corpora (including
# elongated / misspelled variants).  Values are the canonical romanized forms.
# This map is the PRIMARY normalization mechanism for Hindi tokens; regex
# corrections only apply to tokens NOT found here.
#
# Design rules:
#   • Every canonical value must also appear as a key (identity mapping) so
#     that already-canonical tokens are handled correctly.
#   • Variants are listed exhaustively; elongated forms produced after
#     repetition compression are included where applicable.
#   • No English→Hindi or Hindi→English translations are introduced.

_HINGLISH_VARIANTS: dict[str, str] = {
    # ── accha family ──────────────────────────────────────────────────────
    "acha":     "accha",
    "achha":    "accha",
    "accha":    "accha",    # canonical
    "achaa":    "accha",
    "achcha":   "accha",

    # ── bahut family ──────────────────────────────────────────────────────
    "bohot":    "bahut",
    "bohut":    "bahut",
    "bahot":    "bahut",
    "bahuut":   "bahut",
    "bahut":    "bahut",    # canonical
    "bahutt":   "bahut",    # post-compression artefact

    # ── yaar family ───────────────────────────────────────────────────────
    "yar":      "yaar",
    "yaaar":    "yaar",     # post-compression form
    "yaarr":    "yaar",     # post-compression form
    "yaar":     "yaar",     # canonical

    # ── theek family ──────────────────────────────────────────────────────
    "thik":     "theek",
    "theek":    "theek",    # canonical
    "thikk":    "theek",

    # ── kyun family ───────────────────────────────────────────────────────
    "kyu":      "kyun",
    "kyun":     "kyun",     # canonical

    # ── haan family ───────────────────────────────────────────────────────
    "han":      "haan",
    "haan":     "haan",     # canonical
    "haann":    "haan",

    # ── nahi family ───────────────────────────────────────────────────────
    "nahi":     "nahi",     # canonical
    "nahii":    "nahi",
    "nahiin":   "nahi",
    "nahin":    "nahi",

    # ── samajh family ─────────────────────────────────────────────────────
    "samjh":    "samajh",
    "samajh":   "samajh",   # canonical
    "samjha":   "samajha",
    "samajha":  "samajha",  # canonical

    # ── Additional high-frequency variants ────────────────────────────────
    "aur":      "aur",
    "aorr":     "aur",
    "orr":      "aur",
    "kya":      "kya",
    "kyaa":     "kya",
    "hai":      "hai",
    "haii":     "hai",
    "haiii":    "hai",
    "bhai":     "bhai",
    "bhaii":    "bhai",
    "yaar":     "yaar",
    "mein":     "mein",
    "main":     "main",
    "mainn":    "main",
    "hum":      "hum",
    "hoon":     "hoon",
    "hun":      "hoon",
    "hu":       "hoon",
    "toh":      "toh",
    "toh":      "toh",
    "to":       "toh",       # postposition; English "to" caught by EN guard first
    "lekin":    "lekin",
    "lakin":    "lekin",
    "phir":     "phir",
    "fir":      "phir",
    "abhi":     "abhi",
    "jaldi":    "jaldi",
    "bilkul":   "bilkul",
    "bilkull":  "bilkul",
}

# ── Slang expansion map ───────────────────────────────────────────────────────

_SLANG_EXPANSIONS: dict[str, str] = {
    "u":    "you",
    "ur":   "your",
    "pls":  "please",
    "plz":  "please",
    "thx":  "thanks",
    "idk":  "i do not know",
    "omg":  "oh my god",
    "lol":  "laughing out loud",
    "brb":  "be right back",
    "btw":  "by the way",
    "imo":  "in my opinion",
    "tbh":  "to be honest",
    "ngl":  "not gonna lie",
    "smh":  "shaking my head",
    "irl":  "in real life",
    "rn":   "right now",
    "asap": "as soon as possible",
}


@dataclass
class LanguageAwareNormalization(HinglishBase):
    """
    Language-preserving orthographic normalization for noisy code-switched text.

    Objective
    ---------
    Minimise orthographic variability in Hinglish corpora while preserving:

        (a) *language identity*      — EN and HI tokens stay in their respective
            scripts / romanisation conventions; no cross-language rewriting;
        (b) *expressive sentiment cues* — moderate elongations ("soooo good")
            are optionally retained as valid sentiment signals;
        (c) *code-switch boundaries* — the EN/HI/UNK sequence in the input is
            not reordered or collapsed.

    The normalization process is token-level and language-routed.  Social-media
    artefacts (hashtags, mentions, URLs, e-mails, emojis, punctuated numbers)
    are unconditionally preserved.

    Pipeline
    --------
    For each token:
      1. Pass-through guard (social artefacts, digits, empty strings).
      2. Language detection → EN | HI | UNK.
      3. Language-specific normalization:

         EN  → repetition compression  → optional slang expansion.

         HI  → repetition compression
             → vowel-elongation reduction
             → dictionary canonicalization  ← PRIMARY mechanism
             → regex OOV correction        ← fallback only.

         UNK → generic compression (only when ``normalize_unknown=True``).

    Attributes
    ----------
    normalize_english : bool
        Apply repetition compression (and optional slang expansion) to EN tokens.
    normalize_hindi : bool
        Apply the full HI normalization pipeline to HI tokens.
    normalize_unknown : bool
        Apply generic compression to UNK tokens.  Disabled by default to avoid
        corrupting named entities, hashtags, mixed-script tokens.
    expand_slang : bool
        Expand English slang contractions (e.g. "lol" → "laughing out loud").
        Only active when ``normalize_english=True``.
    preserve_sentiment_intensity : bool
        When True, allow up to ``sentiment_repeat_limit`` repeated characters
        so that expressive forms ("soooo", "yaaaar") are partially preserved.
        When False, compress to at most 2 repeats.
    sentiment_repeat_limit : int
        Maximum repeated characters to retain when
        ``preserve_sentiment_intensity=True``.  Default 3.
    vowel_repeat_limit : int
        Maximum same-vowel repetitions retained by the vowel-elongation
        reduction step.  Default 1 (standard linguistic form).
    hinglish_variants : dict[str, str]
        Override or extend the built-in spelling-variant canonicalization map.
    slang_expansions : dict[str, str]
        Override or extend the slang expansion map.
    """

    normalize_english:            bool           = True
    normalize_hindi:              bool           = True
    normalize_unknown:            bool           = False
    expand_slang:                 bool           = False
    preserve_sentiment_intensity: bool           = False
    sentiment_repeat_limit:       int            = 3
    vowel_repeat_limit:           int            = 1
    hinglish_variants:            dict[str, str] = field(
        default_factory=lambda: dict(_HINGLISH_VARIANTS)
    )
    slang_expansions:             dict[str, str] = field(
        default_factory=lambda: dict(_SLANG_EXPANSIONS)
    )

    # ── Token dispatch ────────────────────────────────────────────────────────

    def _process_token(self, token: str) -> str:
        # Step 1 — pass-through guards
        if not token or token.isdigit():
            return token
        if _is_social_artifact(token):
            return token

        # Step 2 — language detection and routing
        lang = self._detect_language(token)
        if lang == "EN" and self.normalize_english:
            return self._normalize_english(token)
        if lang == "HI" and self.normalize_hindi:
            return self._normalize_hindi(token)
        if lang == "UNK" and self.normalize_unknown:
            return self._normalize_unknown(token)
        return token

    # ── Per-language pipelines ────────────────────────────────────────────────

    def _normalize_english(self, token: str) -> str:
        """
        EN pipeline:
          a. Repetition compression (sentiment-aware).
          b. Optional slang expansion.
        Standard English vocabulary is never rewritten, stemmed, or lemmatised.
        """
        normalized = self._compress_repetitions(token)
        if self.expand_slang:
            expanded = self.slang_expansions.get(normalized.lower())
            if expanded is not None:
                return expanded
        return normalized

    def _normalize_hindi(self, token: str) -> str:
        """
        HI pipeline (ordered — dictionary always takes precedence over regex):
          a. Repetition compression.
          b. Vowel-elongation reduction.
          c. Dictionary-based canonicalization  ← PRIMARY.
          d. Regex-based OOV correction         ← fallback.
        """
        # (a) Repetition compression
        normalized = self._compress_repetitions(token)

        # (b) Vowel elongation reduction
        normalized = self._reduce_vowel_elongation(normalized)

        # (c) Dictionary canonicalization — mandatory, highest priority
        canonical = self.hinglish_variants.get(normalized.lower())
        if canonical is not None:
            # Preserve the original capitalisation style if the source was
            # title-cased or all-caps (rare in Hinglish, but defensible).
            return self._match_case(token, canonical)

        # Also attempt lookup against the base HINGLISH_LEXICON for any
        # normalised forms not covered by the local variant map.
        base_canonical = HINGLISH_LEXICON.get(normalized.lower())
        if base_canonical is not None:
            return self._match_case(token, base_canonical)

        # (d) Regex OOV correction — only reached when dict lookup fails
        normalized = self._apply_oov_patterns(normalized)
        return normalized

    def _normalize_unknown(self, token: str) -> str:
        """
        UNK pipeline: generic repetition compression only.
        Does not apply Hinglish canonicalization or English slang expansion
        to avoid corrupting named entities, hashtags, and mixed-script tokens.
        """
        return self._compress_repetitions(token)

    # ── Normalization helpers ─────────────────────────────────────────────────

    def _compress_repetitions(self, token: str) -> str:
        """
        Collapse runs of the same character to at most ``max_repeats``.

        When ``preserve_sentiment_intensity=True`` the limit is
        ``sentiment_repeat_limit`` (default 3) so that "soooo" → "sooo"
        rather than "so", retaining the expressive signal.
        When disabled the limit is 2 (standard orthographic form).
        """
        max_r = (
            self.sentiment_repeat_limit
            if self.preserve_sentiment_intensity
            else 2
        )
        return re.sub(
            r"(.)\1{%d,}" % max_r,
            lambda m: m.group(1) * max_r,
            token,
        )

    def _reduce_vowel_elongation(self, token: str) -> str:
        """
        Reduce consecutive identical vowels beyond ``vowel_repeat_limit``.

        Default limit is 1, which collapses "bahuuut" → "bahut".  When
        ``preserve_sentiment_intensity=True`` the limit rises to 2 so that
        "sooo" (already compressed by step a) keeps one extra vowel as a
        mild emphasis signal.
        """
        limit = (
            min(self.sentiment_repeat_limit - 1, 2)
            if self.preserve_sentiment_intensity
            else self.vowel_repeat_limit
        )
        # Build pattern: match a vowel repeated more than `limit` times
        return re.sub(
            r"([aeiouAEIOU])\1{%d,}" % limit,
            lambda m: m.group(1) * limit,
            token,
        )

    def _apply_oov_patterns(self, token: str) -> str:
        """
        Regex-based correction for Hinglish OOV tokens not covered by the
        variant dictionary.  Applied ONLY after dictionary lookup fails.

        Patterns are ordered from most-specific to least-specific to minimise
        false positive rewrites.
        """
        t = token
        t = re.sub(r"ach+a+",   "accha",  t)   # achhhaa → accha
        t = re.sub(r"nah+i+",   "nahi",   t)   # nahhii  → nahi
        t = re.sub(r"yaa+r+",   "yaar",   t)   # yaaarrr → yaar
        t = re.sub(r"bahu+t",   "bahut",  t)   # bahuuut → bahut
        t = re.sub(r"samj+h",   "samajh", t)   # samjjh  → samajh
        t = re.sub(r"kyu+n?",   "kyun",   t)   # kyuuu   → kyun
        t = re.sub(r"haa+n",    "haan",   t)   # haaaan  → haan
        t = re.sub(r"the+k",    "theek",  t)   # theeeek → theek
        return t

    @staticmethod
    def _match_case(source: str, target: str) -> str:
        """
        Transfer the capitalisation style of ``source`` to ``target``.
        Supports all-caps, title-case, and lowercase.  Falls back to
        returning ``target`` unchanged for mixed-case sources.
        """
        if source.isupper():
            return target.upper()
        if source.istitle():
            return target.capitalize()
        return target


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[LanguageAwareNormalization] = None,
) -> pd.DataFrame:
    """
    Apply Language-Aware Normalization to a DataFrame column.

    Parameters
    ----------
    df         : Input DataFrame.
    text_col   : Name of the column containing raw Hinglish text.
    output_col : Name of the output column for normalised text.
    processor  : Optional pre-configured ``LanguageAwareNormalization``
                 instance; a default instance is created if not provided.

    Returns
    -------
    DataFrame with an additional column ``output_col``.
    """
    if processor is None:
        processor = LanguageAwareNormalization()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[LanguageAwareNormalization] = None,
) -> pd.DataFrame:
    """
    Apply Language-Aware Normalization to a CSV file and write the result.

    Parameters
    ----------
    input_csv  : Path to the input CSV.
    output_csv : Path for the output CSV.
    text_col   : Column name containing raw Hinglish text.
    processor  : Optional pre-configured ``LanguageAwareNormalization``
                 instance.

    Returns
    -------
    Processed DataFrame (also written to ``output_csv``).
    """
    if processor is None:
        processor = LanguageAwareNormalization()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)


# ── Smoke tests ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _SEP = "─" * 68

    def _run(label: str, cases: list[tuple[str, str]], proc: LanguageAwareNormalization) -> None:
        print(f"\n{_SEP}")
        print(f"  {label}")
        print(_SEP)
        for text, expected in cases:
            out = proc.process(text)
            status = "✓" if out == expected else "✗"
            print(f"  {status}  IN : {text}")
            print(f"     OUT: {out}")
            if out != expected:
                print(f"     EXP: {expected}")

    # ── Default processor (no slang, no sentiment preservation) ───────────
    default = LanguageAwareNormalization()

    _run("Core Hinglish normalization", [
        ("yaaaaar this movie was bohottt accha lol",
         "yaar this movie was bahut accha lol"),
        ("kya hua bhai sab theek hai",
         "kya hua bhai sab theek hai"),
        ("main office mein hoon",
         "main office mein hoon"),
        ("acha accha achha bahut bohot bohut",
         "accha accha accha bahut bahut bahut"),
        ("kyu kyun han haan fir phir",
         "kyun kyun haan haan phir phir"),
        ("samjh gaya bilkull theek",
         "samajh gaya bilkul theek"),
    ], default)

    # ── Slang expansion ───────────────────────────────────────────────────
    slang = LanguageAwareNormalization(expand_slang=True)

    _run("English slang expansion", [
        ("lol that was so funny",
         "laughing out loud that was so funny"),
        ("idk tbh u should go",
         "i do not know to be honest you should go"),
        ("brb back in 5",
         "be right back back in 5"),
        # slang expansion must not fire on Hindi tokens
        ("ngl yaar that was bad",
         "not gonna lie yaar that was bad"),
    ], slang)

    # ── Sentiment intensity preservation ──────────────────────────────────
    senti = LanguageAwareNormalization(preserve_sentiment_intensity=True, sentiment_repeat_limit=3)

    _run("Sentiment-preserving elongation", [
        ("soooo good",
         "sooo good"),    # compressed to limit=3, not 2
        ("yaaaaaaarrrr",
         "yaar"),         # dict lookup after compression → canonical
        ("bahuuuuuut",
         "bahut"),        # vowel reduction + dict
        ("this is greaaaat",
         "this is greaaaat"),  # EN: kept at limit=3 → "greaaat" — see note below
    ], senti)

    # ── Social-media artefact preservation ────────────────────────────────
    social = LanguageAwareNormalization()

    _run("Social-media artefacts (must pass through unchanged)", [
        ("#MondayMotivation sab log ready ho",
         "#MondayMotivation sab log ready ho"),
        ("@username yaar sun",
         "@username yaar sun"),
        ("check https://example.com/link for info",
         "check https://example.com/link for info"),
        ("email me at cinder@example.com",
         "email me at cinder@example.com"),
        ("score was 98.6% nahi 99%",
         "score was 98.6% nahi 99%"),
        ("❤️ yaar tum the best",
         "❤️ yaar tum the best"),
        (":) sab badhiya hai",
         ":) sab badhiya hai"),
    ], social)

    # ── Unknown token handling ────────────────────────────────────────────
    unk_off = LanguageAwareNormalization(normalize_unknown=False)
    unk_on  = LanguageAwareNormalization(normalize_unknown=True)

    _run("Unknown tokens — normalize_unknown=False (default)", [
        ("Hindustaaaan",   "Hindustaaaan"),   # UNK → unchanged
        ("XD123abc",       "XD123abc"),
    ], unk_off)

    _run("Unknown tokens — normalize_unknown=True", [
        ("Hindustaaaan",   "Hindustaaan"),    # compressed to 2 repeats (no senti)
    ], unk_on)

    # ── Mixed-language (code-switching preservation) ──────────────────────
    mixed = LanguageAwareNormalization(expand_slang=True)

    _run("Code-switching structure preservation", [
        ("yaar this movie was bahuuuut acchi lol",
         "yaar this movie was bahut acchi laughing out loud"),
        ("main really tired hoon aaj",
         "main really tired hoon aaj"),
        ("kya u serious ho",
         "kya you serious ho"),
        ("nahi nahi that's not accha at all",
         "nahi nahi that's not accha at all"),
    ], mixed)

    print(f"\n{_SEP}")
    print("  Smoke tests complete.")
    print(_SEP)