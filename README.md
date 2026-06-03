# Hinglish Preprocessing Toolkit — Multilingual Preprocessing Profile (MLP)

Research-grade preprocessing and analysis toolkit for Hindi-English code-switched (Hinglish) text,
implementing the six-score **Multilingual Preprocessing Profile** from:

> *"Preprocessing is Not Language-Agnostic: An Information-Theoretic and Spectral Study of
> Code-Switched Text Preprocessing on Transformer Performance"*

---

## Multilingual Preprocessing Profile

### Standard scores
| Score | Symbol | What it measures |
|-------|--------|-----------------|
| Mutual Information Preservation | ΔI | Information discarded: I(X;Y) − I(f(X);Y) |
| Embedding Condition Number | κ(E) | Numerical stability of embedding matrix via SVD |
| Attention Spectral Entropy | S_att | Eigenvalue distribution uniformity of covariance |

### Novel multilingual scores
| Score | Symbol | What it measures |
|-------|--------|-----------------|
| Fertility Entropy | H_F | Tokeniser balance across EN/HI: H_F ∈ [0, log 2] |
| Switch-Point Attention Collapse | SPAC | H̄_att(SP) − H̄_att(non-SP) |
| Cross-lingual Representation Distance | CRD | L2 distance between EN/HI centroid embeddings |

---

## Quick Start

```bash
# Install deps
pip install -r requirements.txt
pip install sentence-transformers torch       # for embedding metrics
python -c "import nltk; nltk.download('words')"

# Start API
python api.py
# → http://127.0.0.1:8000

# Serve UI
python -m http.server 5500
# → Open http://127.0.0.1:5500/preprocessing_preview.html
```

---

## Preprocessing Variants (P0–P4)

| Variant | Modules | Targets |
|---------|---------|---------|
| **P0** (baseline) | none | reference for all scores |
| **P1** (fertility-balanced) | `balanced_tokenization` | H_F — equalises subword allocation |
| **P2** (switch-point normalization) | `language_aware_normalization`, `phonetic_normalization` | ΔI — script-specific rules |
| **P3** (transliteration) | `transliteration`, `script_unification` | CRD — native-script embeddings |
| **P4** (boundary tagging) | `language_identification_tagging`, `switch_point_encoding` | SPAC — explicit switch signals |
| **AUG** (augmentation) | `code_switch_augmentation` | row-expanding only, not in main pipeline |

---

## API Endpoints

### `POST /run_analysis`
Upload CSV + modules → async analysis job.

```bash
curl -X POST http://127.0.0.1:8000/run_analysis \
  -F "file=@hinglish.csv" \
  -F 'modules=["phonetic_normalization","balanced_tokenization"]' \
  -F "batch_size=32" \
  -F "include_spac=false"
```

Response: `{"job_id": "abc123..."}`

### `GET /analysis_results/{job_id}`
Poll for results.

```json
{
  "status": "done",
  "results": {
    "information_theory": { "entropy_before": ..., "entropy_delta": ..., "mutual_information_delta": ... },
    "fertility_entropy":  { "mean_fertility_entropy_before": ..., "delta_fertility_entropy": ... },
    "spectral_analysis":  { "top_k_eigenvalues": [...], "condition_number": ..., "spectral_entropy": ... },
    "embedding_stability":{ "mean_similarity": ..., "histogram_counts": [...] },
    "cross_lingual":      { "cross_lingual_representation_distance": ..., "cross_lingual_centroid_cosine": ... },
    "switch_point_attention_collapse": { "spac_score": ..., "interpretation": "..." }
  }
}
```

### `POST /run_model_inference`
Runs preprocessing + multilingual transformer evaluation over the full dataset.
Results logged to MLflow (`mlruns.db`).

### `GET /available_modules`
Lists all module keys.

---

## Module Reference

| Key | Class | Variant |
|-----|-------|---------|
| `phonetic_normalization` | `PhoneticNormalization` | P2 |
| `language_aware_normalization` | `LanguageAwareNormalization` | P2 |
| `script_unification` | `ScriptUnification` | P3 |
| `transliteration` | `Transliteration` | P3 |
| `balanced_tokenization` | `BalancedTokenization` | P1 |
| `language_identification_tagging` | `LanguageIdentificationTagging` | P4 |
| `switch_point_encoding` | `SwitchPointEncoding` | P4 |
| `code_switch_augmentation` | `CodeSwitchAugmentation` | AUG |
| `context_aware_subword_sampling` | `ContextAwareSentencePieceDropout` | SP |

---

## Supported Transformer Backbones

| Key | Model |
|-----|-------|
| `xlmr` | `xlm-roberta-base` |
| `muril` | `google/muril-base-cased` |
| `nllb` | `facebook/nllb-200-distilled-600M` |
| `indicbert` | `ai4bharat/indic-bert` |
| `mbert` | `bert-base-multilingual-cased` |

---

## Troubleshooting

**`sentence-transformers` not installed** → `pip install sentence-transformers torch`

**SPAC is slow** → Disable the SPAC toggle in the UI; it requires per-token embeddings.

**Chart.js CDN blocked** → Download `chart.umd.min.js` locally:
```bash
curl -o chart.umd.min.js https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js
```

**`mlflow` not installed** → `pip install mlflow`

**Missing columns** → CSV must have `text`, `sentiment`, `label`.

---

## Project Structure

```
hinglish_toolkit/
├── combined_preprocessing_pipeline.py   # Master orchestrator
├── analytical_evaluator.py              # MLP metrics (ΔI, H_F, κ(E), SPAC, CRD)
├── api.py                               # FastAPI backend + MLflow logging
├── server.py                            # Flask preview server (optional)
├── preprocessing_preview.html           # Redesigned interactive UI
├── hinglish.csv                         # Sample dataset
├── requirements.txt
├── _base.py                             # Shared base class + vocabulary
├── phonetic_normalization.py
├── lang_aware_normalizn.py
├── script_unification.py
├── transliteration.py
├── balanced_tokenizn.py
├── lang_id_tagging.py
├── switch_point_encoding.py
├── code_switch_augmentn.py
└── context_aware_sentencepiece_dropout.py
```
