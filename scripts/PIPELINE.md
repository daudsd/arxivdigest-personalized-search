# Evaluation Pipeline

## Baseline Evaluation

**Step 1 — Query Extraction** (`extract_trec_queries.py`)
- Downloads TREC OpenSearch 2016 dataset
- Parses CiteSeerX NDJSON → extracts `qid` + `qstr`
- Output: `data/trec_citeseerx_queries.csv` (969 queries)

**Step 2 — Candidate Retrieval** (`retrieve_candidates.py`)
- Runs each query against Elasticsearch (top-20 per query)
- Three modes:
  - `bm25`: `multi_match` on `title^2 + abstract` → `arxiv` index
  - `dense`: kNN on `abstract_vector` (`all-MiniLM-L6-v2`) → `arxiv_hybrid` index
  - `hybrid`: `bool/should` BM25 + kNN (0.5 boost each) → `arxiv_hybrid` index
- Output: `data/candidates_{bm25,dense,hybrid}.jsonl`

```bash
python3 scripts/retrieve_candidates.py --mode bm25
python3 scripts/retrieve_candidates.py --mode dense --index arxiv_hybrid
python3 scripts/retrieve_candidates.py --mode hybrid --index arxiv_hybrid
```

**Step 3 — LLM Judge** (`llm_judge.py`)
- For each (query, doc) pair: prompts LLM with query + title + abstract
- Scores 0–3 (not relevant → highly relevant), greedy decoding (temperature 0)
- Qrels saved to `data/<model_name>/qrels_{mode}.txt` (TREC qrels format)
- Supports `--resume` to continue interrupted runs

Backends:

```bash
# Ollama (University of Stavanger server)
python3 scripts/llm_judge.py --mode bm25 --backend openai \
  --model llama3.3:70b --api-base https://ollama.ux.uis.no --api-key <key> --resume

# Modal (OpenAI-compatible, GLM-5-FP8)
python3 scripts/llm_judge.py --mode bm25 --backend nim \
  --model zai-org/GLM-5-FP8 \
  --api-base https://api.us-west-2.modal.direct/v1 \
  --api-key <key> --resume

# NVIDIA NIM
python3 scripts/llm_judge.py --mode bm25 --backend nim \
  --model qwen/qwen3.5-122b-a10b \
  --api-key nvapi-... --resume

# Google Gemini Flash
python3 scripts/llm_judge.py --mode bm25 --backend gemini \
  --model gemini-flash-latest \
  --api-key AIza... --resume
```

`--delay` (default 1.0s) adds a per-request pause for rate-limited APIs.
429 errors trigger a 60s backoff automatically.

**Step 4 — Evaluation** (`evaluate.py`)
- Loads qrels + candidate runs, computes nDCG@10, MAP, MRR via `pytrec_eval`
- `--data-dir` points to the model-specific qrels folder

```bash
# Ollama llama3.3:70b results (stored in data/ root)
python3 scripts/evaluate.py --mode bm25 dense hybrid

# Model-specific results (e.g. after running Gemini judge)
python3 scripts/evaluate.py --mode bm25 dense hybrid \
  --data-dir scripts/data/qrels/gemini-flash
```

### Baseline Results (llama3.3:70b judge)

| System | nDCG@10 | MAP    | MRR    |
|--------|---------|--------|--------|
| BM25   | 0.8432  | 0.9176 | 0.9592 |
| Dense  | 0.8380  | 0.9163 | 0.9372 |
| Hybrid | 0.8362  | 0.9115 | 0.9529 |

---

## Personalization

**Step 1 — User Profile Construction** (`build_user_profiles.py`)
- Queries ArXivDigest DB `user_topics` table
- Filters: ≤4-word topics, ≥3 topics per user, cap at 20
- Output: `data/user_profiles.json` (671 users)

```bash
python3 scripts/build_user_profiles.py
```

**Step 2 — Personalized Reranking** (`retrieve_candidates_personalized.py`)
- Samples 10 users from `user_profiles.json` (sorted by profile size, every k=67th)
- Reads base candidates from `data/candidates_{base}.jsonl`
- For each (query, user) pair:
  - Fetches doc texts from ES
  - Computes TF-IDF cosine similarity between docs and user keywords (local IDF)
  - Combines: `α × norm_retrieval_score + (1-α) × cosine_sim` (default α=0.5)
  - Keeps top-10 reranked docs
- Output: `data/candidates_personalized_{base}.jsonl`

```bash
python3 scripts/retrieve_candidates_personalized.py --base bm25
python3 scripts/retrieve_candidates_personalized.py --base dense
python3 scripts/retrieve_candidates_personalized.py --base hybrid
```

**Step 3 — LLM Judge (personalized)** (`llm_judge_personalized.py`)
- For each (query, user, doc) triple: prompt includes user keyword profile
- Scores 0–3, keyed by `qid_uid`
- Output: `data/qrels_personalized_{base}.txt`
- Supports `--backend ollama` (default) or `--backend nim`

```bash
# Ollama
python3 scripts/llm_judge_personalized.py --base bm25 \
  --api-base https://ollama.ux.uis.no --api-key <key> \
  --model llama3.3:70b --top-k 10 --resume

# Modal / NIM
python3 scripts/llm_judge_personalized.py --base bm25 \
  --backend nim \
  --api-base https://api.us-west-2.modal.direct/v1 \
  --api-key <key> --model zai-org/GLM-5-FP8 --resume
```

**Step 4 — Evaluation**

```bash
python3 scripts/evaluate.py --mode personalized_bm25 personalized_dense personalized_hybrid
```

### Personalized Results (llama3.3:70b judge, α=0.5)

| System             | nDCG@10 | MAP    | MRR    |
|--------------------|---------|--------|--------|
| Personalized BM25  | 0.6102  | 0.5631 | 0.5878 |
| Personalized Hybrid| 0.5991  | 0.5503 | 0.5723 |
| Personalized Dense | 0.5760  | 0.5371 | 0.5525 |

---

## Alpha Sweep

Re-runs reranking at different α values against existing qrels. No ES or LLM calls needed after the doc text cache is built.

**Step 1 — Build doc text cache** (one-time, needs ES)

```bash
python3 scripts/alpha_sweep.py --build-cache --base bm25 dense hybrid \
  --es-url http://localhost:9200 --index arxiv
```

**Step 2 — Run sweep**

```bash
python3 scripts/alpha_sweep.py --base bm25 dense hybrid \
  --alphas 0.1 0.3 0.5 0.7 0.9
```

### Alpha Sweep Results

| α   | BM25 nDCG@10 | Dense nDCG@10 | Hybrid nDCG@10 |
|-----|-------------|---------------|----------------|
| 0.1 | **0.6491**  | **0.6016**    | **0.6365**     |
| 0.3 | 0.6201      | 0.5825        | 0.6085         |
| 0.5 | 0.6140      | 0.5786        | 0.6029         |
| 0.7 | 0.6119      | 0.5771        | 0.6007         |
| 0.9 | 0.6106      | 0.5763        | 0.5997         |

Lower α gives more weight to the user profile signal. α=0.1 is best across all base systems.

---

## 🚧 In Progress

### 1. Alpha Plot (0.0 → 1.0 in 0.1 steps)

**Status:** ✅ Done  
**Output:** `data/figures/alpha_sweep.pdf`

```bash
python3 scripts/alpha_sweep.py --base bm25 dense hybrid --plot
```

---

### 2. Query-Profile Likelihood Judge

**Status:** ✅ Done  
**Output:** `data/query_profile_scores.jsonl`

```bash
python3 scripts/score_query_profile.py \
  --api-base https://ollama.ux.uis.no \
  --api-key <key> --model llama3.3:70b --resume
```

**Results:**
- Total pairs scored: 9,610 (10 users × 961 queries)
- Score distribution: {0: 11352, 1: 7003, 2: 845}
- Score=2 (very likely): 845 pairs (4.4%)
- Per-user score=2 counts: min=6, max=247, avg=84.5

---

### 3. Filtered Evaluation on Likely Queries

**Status:** ✅ Done  

```bash
python3 scripts/evaluate.py \
  --mode personalized_bm25 personalized_dense personalized_hybrid \
  --filter-queries scripts/data/query_profile_scores.jsonl
```

**Results (score=2 subset, 426 pairs):**

| System             | nDCG@10 | MAP    | MRR    |
|--------------------|---------|--------|--------|
| Personalized Dense | **0.9680** | **0.9957** | 0.9968 |
| Personalized BM25  | 0.9626  | 0.9899 | **0.9988** |
| Personalized Hybrid| 0.9616  | 0.9924 | 0.9965 |

---

### 4. Candidate Pooling (up to 30 per query)

**Status:** ✅ Done  
**Script:** `pool_candidates.py`  
**Output:** `data/candidates/candidates_pooled.jsonl`

```bash
python3 scripts/pool_candidates.py
```

**Results:**
- Pool size: min=10, max=30, avg=23.2 per query
- 74.4% of docs retrieved by only 1 ranker
- 3.0% of docs retrieved by all 3 rankers
- Next: run LLM judge over pooled candidates for shared qrels

---

### 5. Dataset Section in Methods Chapter

**Status:** Not started  
**Plan:** New `\section{Dataset}` in Ch3 covering:
- User profiles: 671 total, 10 sampled, statistics (min/max/avg keywords per profile)
- Query subset: 969 total → filtered to "very likely" per user (stats after filtering)
- Candidate pooling: top-10 from each of 3 rankers, deduplicated, up to 30 per query
- Relevance judgments: LLM judge over pooled candidates
- Example table: sample (user, query, top-3 candidates) from the dataset

---

### 6. Second LLM Judge (Gemini Flash)

**Status:** Backend implemented, not yet run  
**Plan:**
- Run Gemini Flash on all 3 baseline modes
- Qrels saved to `data/qrels/gemini-flash-latest/`
- Compare nDCG@10/MAP/MRR between llama3.3:70b and Gemini in a table in Ch4
- Cost: ~$1.10 for 3 baseline systems

```bash
python3 scripts/llm_judge.py --mode bm25 --backend gemini \
  --model gemini-flash-latest --api-key AIza... --resume
python3 scripts/llm_judge.py --mode dense --backend gemini \
  --model gemini-flash-latest --api-key AIza... --resume
python3 scripts/llm_judge.py --mode hybrid --backend gemini \
  --model gemini-flash-latest --api-key AIza... --resume
```

---

### 7. Embedding Similarity for Personalization

**Status:** Not started  
**Plan:**
- Add `--sim-method tfidf|embedding` to `retrieve_candidates_personalized.py`
- For embedding mode: encode user keywords and doc texts with `all-MiniLM-L6-v2`, use cosine similarity on vectors
- Doc vectors already in ES (`abstract_vector`) — fetch directly
- Produces `candidates_personalized_{base}_embedding.jsonl`
- Requires new LLM judge run for the embedding-reranked candidates
- Compare TF-IDF vs embedding similarity in Ch4 as an ablation

---

## Data Files

```
data/
├── queries/
│   └── trec_citeseerx_queries.csv        # 969 TREC OpenSearch 2016 CiteSeerX queries
├── candidates/
│   ├── candidates_bm25.jsonl             # top-20 per query, BM25
│   ├── candidates_dense.jsonl            # top-20 per query, dense
│   ├── candidates_hybrid.jsonl           # top-20 per query, hybrid
│   ├── candidates_personalized_bm25.jsonl
│   ├── candidates_personalized_dense.jsonl
│   └── candidates_personalized_hybrid.jsonl
├── qrels/
│   └── llama3.3-70b/                     # one subfolder per judge model
│       ├── qrels_bm25.txt
│       ├── qrels_dense.txt
│       ├── qrels_hybrid.txt
│       ├── qrels_personalized_bm25.txt
│       ├── qrels_personalized_dense.txt
│       └── qrels_personalized_hybrid.txt
├── profiles/
│   └── user_profiles.json                # 671 user keyword profiles
├── cache/
│   ├── doc_texts_bm25.json               # cached doc texts for alpha sweep
│   ├── doc_texts_dense.json
│   └── doc_texts_hybrid.json
├── figures/
│   ├── score_distribution.pdf
│   └── score_distribution.png
├── logs/
│   └── judge_personalized.log
└── topics.csv                            # ArXivDigest seed topics (used by init_topic_list.py)
```
