"""
balanced_tokenizn.py — Balanced tokenization for code-switched Hinglish text.

Hard requirements:
    pip install indic-transliteration
    (+ NLTK words corpus via _base.py)

What it does:
    Balanced Tokenization performs *lexicon-guided canonicalization* to reduce
    subword fragmentation asymmetry in code-switched Hinglish text.  It does NOT
    translate Hindi tokens into English; the code-switching structure and
    semantic identity of the original text are fully preserved.

    Processing pipeline:
      1. Skip empty or purely numeric tokens — no transformation applied.
      2. Normalize elongated character repetitions (e.g. "bahuuuut" → "bahuut").
      3. Apply lightweight *phonetic canonicalization* of romanized spellings so
         that surface variants such as "acha/accha/achha" or "bahut/bohot/bahot"
         collapse to a single canonical romanized form before dictionary lookup.
         This step maximises dictionary coverage without altering meaning.
      4. Look up the canonical romanized form in the built-in Hinglish lexicon
         (extensible via ``map_path``).  The lexicon is the PRIMARY balancing
         mechanism; multiple romanized variants are mapped to a single
         Devanagari representation.
      5. Only for tokens *not* found in the lexicon:
           a. determine whether the token is Hindi-like;
           b. if yes, apply ITRANS → Devanagari transliteration as an OOV
              fallback — transliteration never overrides a lexicon mapping;
           c. otherwise leave the token unchanged.
      6. English words pass through unchanged, preserving the code-switching
         signal (e.g. "movie", "office", "bro" remain as-is).

    Example:
        Input : "yaar movie bahuuut acchi thi"
        Output: "यार movie बहुत अच्छी थी"
        ← Hindi tokens → Devanagari; English tokens preserved; NOT translated.

No silent fallbacks: missing ``indic-transliteration`` raises ``ImportError``
at import time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate

from _base import HINGLISH_LEXICON, HinglishBase


# ── Built-in romanized → Devanagari lexicon ───────────────────────────────────
#
# Each entry maps one *canonical* romanized form (and its common spelling
# variants, resolved by phonetic canonicalization below) to a single Devanagari
# representation.  The lexicon is the authoritative, mandatory balancing layer.
# Entries are grouped thematically for readability and maintainability.

_DEFAULT_HINGLISH_MAP: dict[str, str] = {
    # ── Affirmation / Negation ─────────────────────────────────────────────
    "haan":      "हाँ",   # variants: han, haa, ha → canonicalized to haan
    "nahi":      "नहीं",  # variants: nahin, nah, naheen
    "na":        "ना",

    # ── Question words ─────────────────────────────────────────────────────
    "kya":       "क्या",
    "kyun":      "क्यों",  # variants: kyu → canonicalized to kyun
    "kaise":     "कैसे",
    "kaun":      "कौन",
    "kab":       "कब",
    "kahan":     "कहाँ",   # variant: kaha
    "kitna":     "कितना",
    "kitni":     "कितनी",
    "kitne":     "कितने",

    # ── Postpositions / Case markers ───────────────────────────────────────
    "ka":        "का",
    "ki":        "की",
    "ke":        "के",
    "ko":        "को",
    "se":        "से",
    "mein":      "में",   # variants: me, men → canonicalized to mein
    "par":       "पर",
    "pe":        "पे",
    "tak":       "तक",
    "liye":      "लिए",
    "bina":      "बिना",
    "saath":     "साथ",   # variant: sath

    # ── Pronouns ───────────────────────────────────────────────────────────
    "main":      "मैं",   # variant: mai → canonicalized to main
    "hum":       "हम",    # variant: ham → canonicalized to hum
    "tum":       "तुम",
    "aap":       "आप",
    "wo":        "वो",    # variant: woh
    "ye":        "ये",    # variant: yeh → canonicalized to ye
    "is":        "इस",
    "us":        "उस",
    "koi":       "कोई",
    "sab":       "सब",
    "sabhi":     "सभी",
    "kuch":      "कुछ",   # variant: kucch
    "khud":      "खुद",
    "apna":      "अपना",
    "apni":      "अपनी",
    "apne":      "अपने",

    # ── Possessives ────────────────────────────────────────────────────────
    "mera":      "मेरा",
    "meri":      "मेरी",
    "mere":      "मेरे",
    "tera":      "तेरा",
    "teri":      "तेरी",
    "tere":      "तेरे",
    "aapka":     "आपका",
    "aapki":     "आपकी",
    "aapke":     "आपके",
    "humara":    "हमारा",
    "humari":    "हमारी",
    "humare":    "हमारे",
    "tumhara":   "तुम्हारा",
    "tumhari":   "तुम्हारी",
    "tumhare":   "तुम्हारे",
    "inka":      "इनका",
    "inki":      "इनकी",
    "inke":      "इनके",
    "iska":      "इसका",
    "iski":      "इसकी",
    "iske":      "इसके",
    "unka":      "उनका",
    "unki":      "उनकी",
    "unke":      "उनके",
    "uska":      "उसका",
    "uski":      "उसकी",
    "uske":      "उसके",

    # ── Conjunctions / Discourse markers ──────────────────────────────────
    "aur":       "और",
    "ya":        "या",
    "toh":       "तो",    # variant: to → canonicalized to toh
    "lekin":     "लेकिन", # variant: lakin
    "magar":     "मगर",
    "kyunki":    "क्योंकि",
    "isliye":    "इसलिए",
    "phir":      "फिर",   # variant: fir → canonicalized to phir
    "bhi":       "भी",
    "hi":        "ही",
    "sirf":      "सिर्फ",
    "bas":       "बस",

    # ── Temporal / Adverbial ───────────────────────────────────────────────
    "ab":        "अब",
    "tab":       "तब",
    "jab":       "जब",
    "agar":      "अगर",
    "kabhi":     "कभी",
    "hamesha":   "हमेशा",
    "aksar":     "अक्सर",
    "abhi":      "अभी",
    "pehle":     "पहले",  # variant: pahle
    "baad":      "बाद",
    "jaldi":     "जल्दी",
    "dhire":     "धीरे",
    "seedha":    "सीधा",  # variant: sidha

    # ── Copula / Auxiliaries ───────────────────────────────────────────────
    "hai":       "है",
    "hain":      "हैं",
    "tha":       "था",
    "thi":       "थी",
    "the":       "थे",
    "ho":        "हो",
    "hoon":      "हूं",   # variants: hun, hu → canonicalized to hoon
    "hoga":      "होगा",
    "hogi":      "होगी",
    "honge":     "होंगे",
    "hota":      "होता",
    "hoti":      "होती",
    "hote":      "होते",

    # ── Common verbs (root / imperative / infinitive) ──────────────────────
    "karna":     "करना",
    "karta":     "करता",
    "karti":     "करती",
    "karte":     "करते",
    "karo":      "करो",
    "kar":       "कर",
    "karen":     "करें",
    "karke":     "करके",
    "kiya":      "किया",
    "kiye":      "किए",
    "dekh":      "देख",
    "dekho":     "देखो",
    "dekhna":    "देखना",
    "dekha":     "देखा",
    "dekhi":     "देखी",
    "dekhe":     "देखे",
    "bol":       "बोल",
    "bolo":      "बोलो",
    "bola":      "बोला",
    "boli":      "बोली",
    "sun":       "सुन",
    "suno":      "सुनो",
    "suna":      "सुना",
    "chal":      "चल",
    "chalo":     "चलो",
    "chala":     "चला",
    "chali":     "चली",
    "ja":        "जा",
    "jao":       "जाओ",
    "jana":      "जाना",
    "gaya":      "गया",
    "gayi":      "गई",
    "gaye":      "गए",
    "aao":       "आओ",
    "aana":      "आना",
    "aaya":      "आया",
    "aayi":      "आई",
    "rakho":     "रखो",
    "rakha":     "रखा",
    "rakhna":    "रखना",
    "samajh":    "समझ",
    "samjha":    "समझा",
    "samjhi":    "समझी",
    "lena":      "लेना",
    "lo":        "लो",
    "liya":      "लिया",
    "dena":      "देना",
    "do":        "दो",
    "diya":      "दिया",
    "aana":      "आना",
    "jaana":     "जाना",
    "bata":      "बता",
    "batao":     "बताओ",
    "batana":    "बताना",
    "puchh":     "पूछ",
    "puchho":    "पूछो",
    "sochna":    "सोचना",
    "socho":     "सोचो",
    "socha":     "सोचा",
    "khelna":    "खेलना",
    "khelo":     "खेलो",
    "khana":     "खाना",   # also noun (food) — context preserved
    "khao":      "खाओ",
    "pina":      "पीना",
    "pio":       "पियो",

    # ── Progressive / Perfective auxiliaries ──────────────────────────────
    "raha":      "रहा",
    "rahi":      "रही",
    "rahe":      "रहे",
    "hua":       "हुआ",
    "hui":       "हुई",
    "hue":       "हुए",

    # ── Ergative constructions ─────────────────────────────────────────────
    "maine":     "मैंने",
    "hamne":     "हमने",
    "tumne":     "तुमने",
    "aapne":     "आपने",
    "usne":      "उसने",
    "unhonne":   "उन्होंने",
    "inhonne":   "इन्होंने",

    # ── Adjectives / Intensifiers ──────────────────────────────────────────
    "acha":      "अच्छा",  # variants: accha, achha → canonicalized to acha
    "achi":      "अच्छी",  # variants: acchi, achhi
    "ache":      "अच्छे",  # variants: acche, achhe
    "bura":      "बुरा",
    "buri":      "बुरी",
    "bure":      "बुरे",
    "sahi":      "सही",
    "galat":     "गलत",
    "badhiya":   "बढ़िया",
    "mast":      "मस्त",
    "zabardast": "ज़बरदस्त",
    "kamaal":    "कमाल",
    "bekar":     "बेकार",
    "sasta":     "सस्ता",
    "mehnga":    "महँगा",
    "bada":      "बड़ा",
    "badi":      "बड़ी",
    "bade":      "बड़े",
    "chota":     "छोटा",
    "choti":     "छोटी",
    "chote":     "छोटे",
    "naya":      "नया",
    "nayi":      "नई",
    "naye":      "नए",
    "purana":    "पुराना",
    "purani":    "पुरानी",
    "purane":    "पुराने",
    "bahut":     "बहुत",   # variants: bahot, bohot → canonicalized to bahut
    "thoda":     "थोड़ा",  # variant: thora
    "thodi":     "थोड़ी",
    "thode":     "थोड़े",
    "bilkul":    "बिल्कुल",
    "ekdum":     "एकदम",
    "zyada":     "ज़्यादा", # variant: jyada
    "kam":       "कम",
    "kafi":      "काफी",
    "aur":       "और",
    "itna":      "इतना",
    "itni":      "इतनी",
    "itne":      "इतने",
    "utna":      "उतना",
    "sundar":    "सुंदर",
    "pyara":     "प्यारा",
    "pyari":     "प्यारी",

    # ── Common nouns ───────────────────────────────────────────────────────
    "ghar":      "घर",
    "khana":     "खाना",
    "paani":     "पानी",  # variant: pani
    "dost":      "दोस्त",
    "pyar":      "प्यार",
    "sach":      "सच",
    "jhooth":    "झूठ",   # variant: jhut
    "kaam":      "काम",
    "din":       "दिन",
    "raat":      "रात",
    "subah":     "सुबह",
    "shaam":     "शाम",
    "padhai":    "पढ़ाई",
    "zindagi":   "ज़िंदगी",
    "duniya":    "दुनिया",
    "log":       "लोग",
    "baat":      "बात",
    "cheez":     "चीज़",
    "jagah":     "जगह",
    "waqt":      "वक्त",
    "baar":      "बार",
    "taraf":     "तरफ",
    "naam":      "नाम",
    "khayal":    "ख्याल",
    "sawaal":    "सवाल",
    "jawab":     "जवाब",
    "mushkil":   "मुश्किल",
    "matlab":    "मतलब",
    "farak":     "फ़र्क",
    "shauk":     "शौक",

    # ── Address terms / Interjections ──────────────────────────────────────
    "yaar":      "यार",   # variant: yar
    "bhai":      "भाई",
    "bhaiya":    "भैया",
    "didi":      "दीदी",
    "amma":      "अम्मा",
    "baba":      "बाबा",
    "ji":        "जी",
    "haanji":    "हाँजी",
    "arre":      "अरे",
    "oye":       "ओये",
    "aho":       "अहो",
    "wah":       "वाह",
    "shabash":   "शाबाश",
    "arey":      "अरे",

    # ── Politeness / Acknowledgement ──────────────────────────────────────
    "theek":     "ठीक",   # variant: thik → canonicalized to theek
    "shukriya":  "शुक्रिया",
    "dhanyawad": "धन्यवाद",
    "maafi":     "माफ़ी",
}


# ── Phonetic canonicalization rules ───────────────────────────────────────────
#
# Applied BEFORE dictionary lookup.  Each rule is a (pattern, replacement) pair
# operating on lowercase romanized tokens.  The goal is to collapse surface-
# level spelling variation so that the dictionary covers more real-corpus tokens
# without any meaning change.
#
# Rules are ordered from most-specific to least-specific.

_CANON_RULES: list[tuple[str, str]] = [
    # ── fir → phir (very common variant) ──────────────────────────────────
    (r"^fir$",              "phir"),

    # ── Trailing -n variants for affirmation / negation ────────────────────
    (r"^han$",              "haan"),
    (r"^haa$",              "haan"),
    (r"^ha$",               "haan"),        # isolated "ha" as affirmation
    (r"^nahin$",            "nahi"),
    (r"^nah$",              "nahi"),
    (r"^naheen$",           "nahi"),

    # ── Question word kyu(n) ───────────────────────────────────────────────
    (r"^kyu$",              "kyun"),

    # ── me / men / main / mai ─────────────────────────────────────────────
    (r"^men$",              "mein"),
    (r"^me$",               "mein"),        # postposition only; not pronoun
    (r"^mai$",              "main"),        # pronoun

    # ── woh → wo; yeh → ye ───────────────────────────────────────────────
    (r"^woh$",              "wo"),
    (r"^yeh$",              "ye"),

    # ── ham → hum ─────────────────────────────────────────────────────────
    (r"^ham$",              "hum"),

    # ── hun / hu → hoon ───────────────────────────────────────────────────
    (r"^hun$",              "hoon"),
    (r"^hu$",               "hoon"),

    # ── to → toh (postposition, not English "to") ─────────────────────────
    # Only apply when token is exactly "to" (handled carefully; English "to"
    # is already caught by _is_english_word before canonicalization matters,
    # but we keep this as a safety net for borderline cases)
    (r"^to$",               "toh"),

    # ── thik → theek ──────────────────────────────────────────────────────
    (r"^thik$",             "theek"),

    # ── kaha / kahan ──────────────────────────────────────────────────────
    (r"^kaha$",             "kahan"),

    # ── pahle → pehle ─────────────────────────────────────────────────────
    (r"^pahle$",            "pehle"),

    # ── sath → saath ──────────────────────────────────────────────────────
    (r"^sath$",             "saath"),

    # ── lakin → lekin ─────────────────────────────────────────────────────
    (r"^lakin$",            "lekin"),

    # ── sidha → seedha ───────────────────────────────────────────────────
    (r"^sidha$",            "seedha"),

    # ── pani → paani ──────────────────────────────────────────────────────
    (r"^pani$",             "paani"),

    # ── yar → yaar ────────────────────────────────────────────────────────
    (r"^yar$",              "yaar"),

    # ── samjh → samajh ────────────────────────────────────────────────────
    (r"^samjh$",            "samajh"),

    # ── kucch → kuch ──────────────────────────────────────────────────────
    (r"^kucch$",            "kuch"),

    # ── jyada → zyada ─────────────────────────────────────────────────────
    (r"^jyada$",            "zyada"),

    # ── jhut → jhooth ─────────────────────────────────────────────────────
    (r"^jhut$",             "jhooth"),

    # ── thora → thoda ─────────────────────────────────────────────────────
    (r"^thora$",            "thoda"),

    # ── accha / achha → acha (adjective masculine) ────────────────────────
    (r"^acch?h?a$",         "acha"),
    # ── acchi / achhi → achi (adjective feminine) ─────────────────────────
    (r"^acch?h?i$",         "achi"),
    # ── acche / achhe → ache (adjective plural/oblique) ───────────────────
    (r"^acch?h?e$",         "ache"),

    # ── bahot / bohot → bahut ─────────────────────────────────────────────
    (r"^ba[ho]ot$",         "bahut"),
    (r"^bohot$",            "bahut"),
]
# NOTE: Generic vowel-length rules (oo→u, ee→i) are intentionally absent.
# They produce unsafe substring substitutions that corrupt English tokens
# whose English detection may produce a false negative (e.g. "school" → "schul",
# "deep" → "dip").  All necessary romanized-variant coverage is handled by the
# explicit whole-word anchored rules above.

# Pre-compile for speed
_COMPILED_CANON: list[tuple[re.Pattern, str]] = [
    (re.compile(pat), repl) for pat, repl in _CANON_RULES
]


def _phonetic_canonicalize(token: str) -> str:
    """
    Apply lightweight phonetic canonicalization to a lowercase romanized token.

    Returns the canonical romanized form so that surface spelling variants
    (e.g. "accha", "achha", "acha") all resolve to the same dictionary key
    ("acha").  This step runs *before* dictionary lookup and transliteration;
    it does not alter meaning or introduce translations.

    All canonicalization rules are whole-word anchored (``^...$``) to prevent
    substring mutations on English or partially-romanized tokens.  The first
    matching rule wins; rules do not stack.
    """
    for pattern, replacement in _COMPILED_CANON:
        result = pattern.sub(replacement, token)
        if result != token:
            return result   # first matching rule wins; no further rules applied
    return token


@dataclass
class BalancedTokenization(HinglishBase):
    """
    Reduces tokenizer fragmentation asymmetry in code-switched Hinglish text
    via *lexicon-guided canonicalization*.

    Balanced Tokenization is NOT a translation system.  It maps romanized
    Hindi tokens to their Devanagari canonical representations so that a
    downstream subword tokenizer (e.g. mBERT, XLM-R) assigns them a single,
    consistent token rather than an unpredictable sequence of fragments.
    English tokens and the overall code-switching structure are preserved
    exactly as in the source.

    Processing pipeline (in order):
        1. Skip empty and purely numeric tokens.
        2. Normalize elongated repetitions (``normalize_repetitions=True``).
        3. Apply phonetic canonicalization to collapse romanized spelling
           variants to a single canonical form.
        4. Perform mandatory dictionary lookup.  The active lexicon is
           assembled from three layers (lowest → highest precedence):
             a. ``HINGLISH_LEXICON`` from ``_base`` (shared base vocabulary);
             b. ``_DEFAULT_HINGLISH_MAP`` defined in this module (thematically
                organized extensions, overrides base on conflict);
             c. user-supplied ``map_path`` entries (highest precedence).
           The lexicon is the PRIMARY balancing mechanism — transliteration
           never overrides a lexicon entry.
        5. For tokens not found in the lexicon, apply ITRANS → Devanagari
           transliteration as an OOV fallback (``transliterate_unknown=True``),
           but ONLY for Hindi-like tokens — never for English words.

    Attributes:
        normalize_repetitions: Collapse 3+ repeated characters to 2
                               (e.g. "bahuuuut" → "bahuut").
        transliterate_unknown: Apply ITRANS transliteration as an OOV
                               fallback for Hindi-like tokens absent from
                               the lexicon.  Transliteration never overrides
                               a lexicon mapping.
        map_path:              Optional path to a 2-column CSV (roman,devanagari)
                               or JSON dict that *extends* the built-in lexicon.
                               User-supplied entries take precedence.
    """

    normalize_repetitions: bool           = True
    transliterate_unknown: bool           = True
    map_path:              Optional[Path] = None

    def _setup(self) -> None:
        # Layer 1: base lexicon imported from _base (shared cross-module vocabulary).
        # Layer 2: _DEFAULT_HINGLISH_MAP (this module's extended, thematically
        #          organized entries) — overrides any conflicts with the base lexicon.
        # Layer 3: user-supplied map_path entries — highest precedence.
        self._hinglish_map: dict[str, str] = {
            **{str(k).strip().lower(): str(v).strip()
               for k, v in HINGLISH_LEXICON.items() if str(k).strip()},
            **_DEFAULT_HINGLISH_MAP,
        }
        if self.map_path is not None:
            # User-supplied entries override all built-in entries — intentional.
            self._hinglish_map.update(self._load_map(Path(self.map_path)))

    # ── External map loading ──────────────────────────────────────────────────

    def _load_map(self, path: Path) -> dict[str, str]:
        """
        Load an external romanized→Devanagari mapping from a JSON dict or a
        2-column CSV (roman, devanagari).  All keys are lowercased and stripped.
        """
        if not path.exists():
            raise FileNotFoundError(f"Hinglish map file not found: {path}")
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as fh:
                raw: dict = json.load(fh)
            if not isinstance(raw, dict):
                raise ValueError(
                    f"JSON map at '{path}' must be a flat {{roman: devanagari}} object."
                )
            return {
                str(k).strip().lower(): str(v).strip()
                for k, v in raw.items()
                if str(k).strip()
            }
        # CSV fallback
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError(
                f"Map CSV '{path}' must have at least two columns: roman, devanagari"
            )
        return {
            str(k).strip().lower(): str(v).strip()
            for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])
            if str(k).strip()
        }

    # ── Core token processing ─────────────────────────────────────────────────

    def _process_token(self, token: str) -> str:
        """
        Apply the Balanced Tokenization pipeline to a single token.

        The token is returned in Devanagari if it is a known Hinglish romanized
        form, or transliterated as an OOV Hindi-like token, or left unchanged
        if it is English or unrecognizable.  No translation into English is
        performed at any stage.
        """
        # ── Step 1: skip trivial tokens ────────────────────────────────────
        if not token or token.isdigit():
            return token

        # ── Step 2: normalize elongated repetitions ────────────────────────
        if self.normalize_repetitions:
            token = re.sub(r"(.)\1{2,}", r"\1\1", token)

        # ── Step 3: English words pass through unchanged ───────────────────
        #   (checked before canonicalization so we never alter English tokens)
        if self._is_english_word(token):
            return token

        # ── Step 4: phonetic canonicalization (romanized variants → key) ──
        canonical = _phonetic_canonicalize(token.lower())

        # ── Step 5: mandatory dictionary lookup (primary mechanism) ───────
        mapped = self._hinglish_map.get(canonical)
        if mapped is not None:
            return mapped
        # Also attempt lookup on the original (pre-canonicalization) form in
        # case the user-supplied lexicon lists a variant directly.
        mapped = self._hinglish_map.get(token.lower())
        if mapped is not None:
            return mapped

        # ── Step 6: OOV fallback — ITRANS transliteration ─────────────────
        #   Transliteration is never applied when a dictionary entry exists.
        if self.transliterate_unknown and self._is_hindi_token(token):
            return _transliterate(
                canonical, sanscript.ITRANS, sanscript.DEVANAGARI
            )

        return token


# ── Module-level convenience wrappers ─────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "processed_text",
    processor: Optional[BalancedTokenization] = None,
) -> pd.DataFrame:
    """
    Apply Balanced Tokenization to a DataFrame column.

    Parameters
    ----------
    df         : Input DataFrame.
    text_col   : Name of the column containing raw Hinglish text.
    output_col : Name of the output column for balanced tokens.
    processor  : Optional pre-configured ``BalancedTokenization`` instance;
                 a default instance is created if not provided.

    Returns
    -------
    DataFrame with an additional column ``output_col`` containing the
    lexicon-canonicalized text.
    """
    if processor is None:
        processor = BalancedTokenization()
    return processor.process_dataframe(df, text_col=text_col, output_col=output_col)


def process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_col: str = "text",
    processor: Optional[BalancedTokenization] = None,
) -> pd.DataFrame:
    """
    Apply Balanced Tokenization to a CSV file and write the result.

    Parameters
    ----------
    input_csv  : Path to the input CSV.
    output_csv : Path for the output CSV.
    text_col   : Column name containing raw Hinglish text.
    processor  : Optional pre-configured ``BalancedTokenization`` instance.

    Returns
    -------
    Processed DataFrame (also written to ``output_csv``).
    """
    if processor is None:
        processor = BalancedTokenization()
    return processor.process_csv(input_csv, output_csv, text_col=text_col)


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = BalancedTokenization(normalize_repetitions=True, transliterate_unknown=True)

    test_cases = [
        # Original examples — outputs must match the current implementation
        ("yaar movie bahuuut acchi thi",
         "यार movie बहुत अच्छी थी"),
        ("kya hua bhai, sab theek hai?",
         "क्या हुआ भाई, सब ठीक है?"),
        ("main office mein hun",
         "मैं office में हूं"),

        # Spelling-variant coverage (canonicalization paths)
        ("accha achha acha",
         "अच्छा अच्छा अच्छा"),
        ("bahut bahot bohot",
         "बहुत बहुत बहुत"),
        ("kyu kyun",
         "क्यों क्यों"),
        ("haan han",
         "हाँ हाँ"),
        ("fir phir",
         "फिर फिर"),
        ("thik theek",
         "ठीक ठीक"),
        ("mai main",
         "मैं मैं"),
        ("me mein men",
         "में में में"),

        # Elongation normalization
        ("yaaaaaar",
         "यार"),
        ("bhuuuuuut",
         "(transliterated — OOV)"),   # for illustration only

        # Code-switching preservation — English tokens unchanged
        ("yaar this movie was bahuuut acchi",
         "यार this movie was बहुत अच्छी"),
    ]

    print("=" * 64)
    print(" Balanced Tokenization — smoke-test")
    print("=" * 64)
    for text, expected in test_cases:
        out = p.process(text)
        print(f"  IN : {text}")
        print(f"  OUT: {out}")
        print(f"  EXP: {expected}")
        print()