# NYCU CSIC30169 Information Retrieval

Comprehensive coursework repository covering three modules:

1. HW1: Classic IR (custom BM25 & TF‑IDF implementations) for document retrieval.
2. HW2: Generative / Evidence Retrieval + Claim Verification (BERT fine-tuning + LLM prompting & ranking pipelines with BM25 / TF‑IDF / FAISS).
3. HW3: Multimodal Image Retrieval (VLM caption/object extraction + Sentence-BERT semantic search, and CLIP zero‑shot retrieval).

---
## Quick Start

```bash
git clone <this-repo>
cd InformationRetrieval
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

If using local LLM / VLM components install Ollama and pull models:
```bash
ollama pull vicuna:13b
ollama pull llava:34b
```

Apple Silicon (optional accelerated torch):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

---
## Repository Structure (excerpt)
```
HW1/
	BM25_train.py | BM25_test.py | TFIDF_train.py | TFIDF_test.py | utils.py | model/
HW2/
	Full_Preprocess.py | training.py | inference.py | prompt_gen.py | predict_llm.py | bm_25/ tf_idf/
HW3/
	hw3_313552001.ipynb (VLM + CLIP methods)
```

---
## HW1: Classic Information Retrieval
Custom from-scratch implementations (no external IR libs) focusing on scoring & evaluation.

### Core Components
| File | Purpose |
|------|---------|
| `model/bm25.py` | Minimal BM25 (configurable k1, b). |
| `model/tf_idf.py` | Vocabulary build + TF, IDF, TF‑IDF matrices. |
| `utils.py` | Lightweight HTML & URL stripping + normalization + cosine similarity. |

### Data
* `documents_data.csv` – Corpus (column: `Document_HTML`).
* `train_question.csv` / `test_question.csv` – Queries (column: `Question`).

### Training / Evaluation (Recall@3)
```bash
cd HW1
python BM25_train.py     # Prints top-3 doc indices per query + Recall@3
python TFIDF_train.py    # Same for TF‑IDF
```
Logic: For query i we consider a hit if doc i appears in top-3 (assumes parallel ordering of docs & questions).

### Generating Submission / Prediction CSV
```bash
python BM25_test.py      # -> output_BM25.csv (index, answer[space-separated top3])
python TFIDF_test.py     # -> output_tfidf.csv
```

### Example Output Row
```
index,answer
1,5 12 3
```

---
## HW2: Generative IR + Claim Verification
Pipeline: (1) Article Sentence Extraction → (2) Sentence Ranking (BM25 | TF‑IDF | FAISS Embeddings) → (3) Pack top‑k with claim → (4) Fine‑tune BERT (3‑class rating) → (5) Optional LLM prompting for explainable reasoning.

### 1. Preprocessing & Ranking
Script: `Full_Preprocess.py`

Supported ranking models:
* `bm25` – `rank_bm25.BM25Okapi`
* `tf-idf` – scikit-learn vector space similarity
* `faiss` – Dense sentence embeddings (`all-MiniLM-L6-v2`) + FAISS GPU index

Run (examples):
```bash
cd HW2
python Full_Preprocess.py --mode train --ranking_model bm25
python Full_Preprocess.py --mode valid --ranking_model bm25
python Full_Preprocess.py --mode test  --ranking_model bm25
```
Outputs: `train_dataset.json`, `valid_dataset.json`, `test_dataset.json` with fields:
```json
{
	"claim": "...",
	"id": 123,
	"top_10_sentences": ["...", "..."],
	"label": {"rating": 0}   // omitted for test mode
}
```

### 2. Model Fine‑Tuning (BERT)
Script: `training.py`
```bash
python training.py
```
Key settings:
* Base model: `bert-base-multilingual-uncased`
* Batch size: 16 | Epochs: 5 | LR: 2e-5 | Metric: accuracy | Padding/truncation: 512 tokens
* Saves best model to `./bm25_best_model`

### 3. Inference
```bash
python inference.py     # Reads valid (or test) json -> predictions.csv (id,rating)
```

### 4. Prompt Generation for LLM Evaluation
```bash
python prompt_gen.py    # Builds prompts from dataset -> output.json
```
Prompt format summarizes claim + top sentences and asks for 0/1/2 classification.

### 5. Local LLM Prediction (Ollama)
```bash
python predict_llm.py   # Uses vicuna:13b to answer prompts -> predicted_llm_valid.json
```
Ensure Ollama is running & model pulled.

### Extensibility Ideas
* Replace BERT with DeBERTa / Longformer for longer contexts.
* Fuse retrieval scores as features (score-weighted concatenation).
* Add calibration layer (temperature scaling) post training.

---
## HW3: Multimodal Image Retrieval
Two complementary methods implemented inside `hw3_313552001.ipynb`:
1. VLM Caption + Object Extraction (LLava via Ollama) → Consolidated textual description → Sentence-BERT embedding search.
2. CLIP Zero-Shot: Encode dialogue-derived query text + images → cosine similarity → top‑30.

### Running (Notebook)
Open the notebook and execute sequentially, or extract code segments for batch runs.

Artifacts produced:
* `test_vlm_caption.csv`, `test_vlm_caption_short.csv` – Object lists & concise captions.
* `caption_obj_descript_similarity_result.csv` – Top‑30 retrieval (VLM pipeline).
* `zeroshot_clip_results.csv` – Top‑30 retrieval (CLIP pipeline).

### Query Processing Highlights
* Dialogue parsing selects messages from user who shared a photo (context focus).
* Text normalized (lowercased, punctuation stripped, deduplicated tokens).
* Embeddings normalized L2 to enable cosine via dot product efficiency.

---
## Metrics
| Module | Metric | Notes |
|--------|--------|-------|
| HW1 | Recall@3 | Hit if doc index equals query index appears in top-3. |
| HW2 | Accuracy | Multi-class (0 / 1 / 2). |
| HW3 | (Implicit) Retrieval Quality | Top‑30 candidate ranking (no provided ground truth here). |

---
## Key Dependencies
See `requirements.txt` for full list: pandas, numpy, scikit-learn, torch, transformers, datasets, nltk, rank_bm25, sentence-transformers, faiss, tqdm, Pillow, ollama.

---
## Notes & Tips
* FAISS GPU path in `Full_Preprocess.py` assumes a visible GPU; switch to `faiss.IndexFlatL2` CPU if needed.
* Large local models (vicuna:13b, llava:34b) require significant RAM/VRAM; downscale if constrained.
* Re-run `nltk.download('punkt')` if tokenizer errors appear.
* Token length overflow → adjust `max_length` (currently 512) or adopt long-context transformer.

---
## Minimal Sanity Checks
```bash
# HW1 quick smoke
python HW1/BM25_train.py | head -n 15

# HW2 preprocessing dry run (first item only inside loop by design)
python HW2/Full_Preprocess.py --mode train --ranking_model bm25

# HW2 training (expect 5 epochs logs)
python HW2/training.py
```

---
## Future Improvements
* Add pytest unit tests for BM25 / TF‑IDF correctness.
* Introduce retrieval evaluation with standard IR metrics (MRR, MAP) using synthetic relevance labels.
* Batch + streaming caption generation with retry logic for robustness.
* Add Dockerfile for reproducible multi-platform setup.

---