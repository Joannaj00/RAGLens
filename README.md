# RAGLens

A RAG (Retrieval-Augmented Generation) evaluation and visualization platform.
RAGLens makes every stage of the RAG pipeline transparent so you can diagnose
failures, compare configurations, and make informed design decisions rather than
guessing from final outputs alone.

---

## Table of Contents

1. [What Problem Does This Solve?](#what-problem-does-this-solve)
2. [How RAG Works](#how-rag-works)
3. [Project Structure](#project-structure)
4. [Data Flow — End to End](#data-flow--end-to-end)
5. [Module Reference](#module-reference)
   - [raglens/ingest.py](#raglensingestpy)
   - [raglens/chunking.py](#raglenscchunkingpy)
   - [raglens/pipeline.py](#raglenspipelinepy)
   - [raglens/eval/retrieval_eval.py](#raglensevalretrieval_evalpy)
   - [raglens/eval/answer_eval.py](#raglensevalanswer_evalpy)
6. [Scripts Reference](#scripts-reference)
   - [scripts/make_docs_jsonl.py](#scriptsmake_docs_jsonlpy)
   - [scripts/generate_eval_set.py](#scriptsgenerate_eval_setpy)
   - [scripts/run_eval.py](#scriptsrun_evalpy)
7. [Web Dashboard](#web-dashboard)
   - [dashboard/api.py — FastAPI backend](#dashboardapipy--fastapi-backend)
   - [raglens-ui — Next.js frontend](#raglens-ui--nextjs-frontend)
8. [Metrics Explained](#metrics-explained)
9. [Running Everything From Scratch](#running-everything-from-scratch)
10. [Configuration Knobs](#configuration-knobs)
11. [Environment & Dependencies](#environment--dependencies)

---

## What Problem Does This Solve?

A standard RAG pipeline looks like this from the outside:

```
User question  →  [black box]  →  Final answer
```

When the answer is wrong you have no way to know *why*.
Was the correct document never retrieved? Was it retrieved but ranked too low?
Was the answer generated badly even though retrieval was good?

RAGLens opens the black box:

```
User question
  → Which chunks were retrieved?
  → Was the correct chunk in the top-k results?
  → At what rank did the correct chunk appear?
  → Did a different chunking strategy retrieve it higher?
  → Did the generated answer actually reflect what was retrieved?
```

Every one of these questions is answered with a number and a visualization.

---

## How RAG Works

RAG has two stages. Understanding them is essential to understanding the metrics.

**Stage 1 — Retrieval**

1. All documents are split into smaller pieces called *chunks*.
2. Each chunk is converted into a vector (a list of numbers) using an *embedding
   model*. Semantically similar text produces similar vectors.
3. These vectors are stored in a *vector store* (FAISS).
4. When a question arrives, the question is also embedded and the vector store
   finds the *k* chunks whose vectors are closest to the question vector.
   "Closest" is measured by L2 (Euclidean) distance — lower distance means
   more similar.

**Stage 2 — Generation**

5. The top-k retrieved chunks are concatenated into a *context* string.
6. A language model reads the question and the context and writes an answer.

The quality of the final answer depends heavily on Stage 1: if the correct
chunk is never retrieved, the LLM has no chance of answering correctly.
RAGLens measures both stages independently so you can tell them apart.

---

## Project Structure

```
RAGLens/
│
├── data/
│   ├── raw/
│   │   └── docs.jsonl              # 300 ArXiv abstracts (source corpus)
│   ├── eval_set.jsonl              # generated question/answer pairs for eval
│   └── eval_results/               # JSON output from run_eval.py
│       ├── fixed-500-ol50.json
│       ├── recursive-1000-ol150.json
│       ├── ...
│       └── summary.json            # compact table of all configs
│
├── raglens/                        # core Python package
│   ├── ingest.py                   # load JSONL → LangChain Documents
│   ├── chunking.py                 # fixed and recursive text splitters
│   ├── pipeline.py                 # RAGConfig + RAGPipeline class
│   └── eval/
│       ├── retrieval_eval.py       # Recall@k, Precision@k, MRR
│       └── answer_eval.py          # BERTScore + LLM-as-judge
│
├── scripts/
│   ├── make_docs_jsonl.py          # download ArXiv dataset → docs.jsonl
│   ├── generate_eval_set.py        # create eval_set.jsonl from docs
│   └── run_eval.py                 # evaluate all configs, write results
│
├── dashboard/
│   ├── api.py                      # FastAPI backend (serves results + live queries)
│   └── app.py                      # Streamlit dashboard (alternative)
│
├── raglens-ui/                     # Next.js + TypeScript web dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Overview page
│   │   │   ├── drill-down/         # Per-question chunk inspector
│   │   │   ├── compare/            # Side-by-side config comparison
│   │   │   └── inspector/          # Live query interface
│   │   ├── components/
│   │   │   ├── NavBar.tsx
│   │   │   └── ChunkCard.tsx
│   │   └── lib/
│   │       ├── api.ts              # API client (fetch wrappers)
│   │       └── types.ts            # TypeScript type definitions
│   └── package.json
│
├── .env                            # secrets (OPENAI-KEY=sk-...)
└── requirements.txt
```

---

## Data Flow — End to End

```
ArXiv dataset (HuggingFace)
        │
        ▼  scripts/make_docs_jsonl.py
data/raw/docs.jsonl   ←── 300 papers, one JSON object per line
        │
        ▼  scripts/generate_eval_set.py
data/eval_set.jsonl   ←── 50 questions with ground-truth doc IDs
        │
        ├──────────────────────────────────────┐
        ▼  raglens/ingest.py                   │
  LangChain Documents                          │
        │                                      │
        ▼  raglens/chunking.py                 │
  Text Chunks (fixed or recursive)             │
        │                                      │
        ▼  raglens/pipeline.py                 │
  FAISS Vector Index                           │
        │                                      │
        ▼  raglens/eval/retrieval_eval.py      │
  Recall@k / MRR / Precision@k per question   │
        │                                      │
        ▼  scripts/run_eval.py                 │
  data/eval_results/*.json  ◄──────────────────┘
        │
        ├─────────────────────────────────────────┐
        ▼  dashboard/api.py (FastAPI)             │
  GET /api/summary, /api/results/{config}        │
  POST /api/query (live retrieval)               │
        │                                         │
        ▼  raglens-ui (Next.js)                  │
  http://localhost:3000  ◄──────────────────────┘
```

---

## Module Reference

### raglens/ingest.py

**Purpose:** Reads the raw JSONL corpus file and converts each line into a
LangChain `Document` object.

A LangChain `Document` has two fields:
- `page_content` — the text (the abstract in this case)
- `metadata` — a dict of extra fields: `id`, `title`, `source`

**Key function:**

```python
load_jsonl(path: str) -> list[Document]
```

Reads every line of the file, parses the JSON, and returns a list of Documents.
The `id` field from the JSONL (e.g. `"0704.0001"`) is preserved in metadata
and is used later as the ground-truth identifier when checking whether the
correct document was retrieved.

---

### raglens/chunking.py

**Purpose:** Splits long documents into smaller chunks that fit inside the
embedding model's context window and produce focused, retrievable pieces.

Two strategies are provided:

**`chunk_docs_fixed(docs, chunk_size, chunk_overlap)`**

Splits text at hard character boundaries regardless of meaning. Every chunk
is exactly `chunk_size` characters (except possibly the last), with
`chunk_overlap` characters of overlap between consecutive chunks. Overlap
ensures a sentence that falls on a boundary is not cut in half and lost.

Example with `chunk_size=10, chunk_overlap=2`:
```
"Hello world foo bar"
 → "Hello worl"
 → "rld foo ba"
 → "bar"
```

**`chunk_docs_recursive(docs, chunk_size, chunk_overlap)`**

Tries to split on natural language boundaries first, falling back to smaller
units only when necessary. The priority order is:
1. Paragraph breaks (`\n\n`) — splits at blank lines first
2. Line breaks (`\n`)
3. Spaces (word boundaries)
4. Individual characters (last resort)

This preserves more semantic coherence than fixed splitting. A chunk is more
likely to contain a complete thought.

Both functions stamp each chunk with a `chunk_id` (sequential integer) in its
metadata so chunks can be traced back to their position in the corpus.

**Which should you use?**
Recursive is usually better for prose text like abstracts. Fixed is simpler
and sometimes faster to reason about. `run_eval.py` tests both so you can
compare their Recall@k directly.

---

### raglens/pipeline.py

**Purpose:** The central class that assembles all components into a runnable
pipeline.

**`RAGConfig` (dataclass)**

All tunable parameters in one place:

| Field | Default | Meaning |
|---|---|---|
| `chunking_strategy` | `"recursive"` | `"fixed"` or `"recursive"` |
| `chunk_size` | `1000` | Max characters per chunk |
| `chunk_overlap` | `150` | Characters of overlap between chunks |
| `embedding_model` | `"intfloat/e5-base-v2"` | HuggingFace model name |
| `top_k` | `5` | Number of chunks to retrieve per query |
| `config_name` | auto-generated | Label used in reports and charts |

`config_name` is auto-generated from the other fields if not supplied, e.g.
`"recursive-1000-ol150"`.

**`RAGPipeline` (class)**

```python
pipeline = RAGPipeline(config, data_path="data/raw/docs.jsonl").build()
```

`.build()` does four things in order:
1. Calls `load_jsonl` to load the corpus
2. Calls the chunker to split documents
3. Creates a `HuggingFaceEmbeddings` model (downloads on first use)
4. Calls `FAISS.from_documents` to embed every chunk and build the index

After building you can:

```python
# Retrieve only
chunks = pipeline.retrieve("What is attention?", k=5)

# Attach an LLM and get a generated answer
from langchain_openai import ChatOpenAI
pipeline.set_llm(ChatOpenAI(model="gpt-4o-mini"))
result = pipeline.run("What is attention?")
print(result.generated_answer)
print(result.retrieved_chunks)
```

**`RetrievedChunk` (dataclass)**

Each chunk returned by `retrieve()` carries:
- `doc_id` — the ArXiv paper ID
- `chunk_id` — position within the corpus
- `title` — paper title
- `l2_distance` — raw FAISS score (lower = more similar)
- `similarity` — converted to [0, 1] via `1 / (1 + l2_distance)` (higher = better)
- `content` — the actual text

**Why convert L2 distance to similarity?**

FAISS returns L2 (Euclidean) distances. A distance of 0 means identical vectors.
But humans expect "higher score = more relevant". The formula `1 / (1 + d)`
maps any non-negative distance to (0, 1] and inverts the direction, so 1.0 means
a perfect match and scores close to 0 mean very dissimilar.

---

### raglens/eval/retrieval_eval.py

**Purpose:** Given a set of questions with known-correct document IDs, run
retrieval and measure how well the vector store finds the right documents.

**`EvalExample` (dataclass)**

One row of the evaluation set:
```python
EvalExample(
    question="What is the main topic of 'Attention Is All You Need'?",
    reference_answer="We propose a new simple network architecture...",
    relevant_doc_ids=["1706.03762"],
)
```
`relevant_doc_ids` is the ground truth: these are the doc IDs that *should*
appear in the retrieval results.

**`evaluate_retrieval(eval_examples, vectorstore, ks, config_name)`**

The main evaluation loop:
1. For each `EvalExample`, queries the vectorstore with `max(ks)` results.
2. For every retrieved chunk, checks whether its `doc_id` is in the
   ground-truth set and marks it `is_relevant=True/False`.
3. Computes Recall@k, Precision@k, and Reciprocal Rank for each question.
4. Macro-averages all metrics across questions.
5. Returns a `RetrievalReport`.

**`RetrievalReport` (dataclass)**

```
config_name:          "recursive-1000-ol150"
macro_mrr:            0.643
macro_recall_at_k:    {1: 0.42, 3: 0.61, 5: 0.70, 10: 0.82}
macro_precision_at_k: {1: 0.42, 3: 0.20, 5: 0.14, 10: 0.08}
per_question:         [QuestionResult, ...]
```

---

### raglens/eval/answer_eval.py

**Purpose:** Measures whether the generated answer is semantically correct,
not just whether the right document was retrieved.

This matters because retrieval and generation can fail independently:
- Good retrieval + bad generation → still a wrong answer
- Bad retrieval + lucky generation → correct answer for the wrong reason

**BERTScore (`bertscore_eval`)**

Uses a pre-trained BERT model to compare the generated answer against the
reference answer. Instead of counting exact word matches (like BLEU), it
computes the cosine similarity between contextual token embeddings, which
captures paraphrases and synonyms.

Returns an F1 score between 0 and 1. ~0.85+ is generally considered good.

Requires: `pip install bert-score` (heavy dependency, installs PyTorch).

**LLM-as-judge (`llm_judge_single`)**

Sends a structured prompt to a LangChain chat model asking it to rate the
generated answer on a 1–5 scale and explain its reasoning. The LLM acts as
an automated human evaluator.

Scale:
- 1 = Completely wrong or irrelevant
- 2 = Mostly wrong, only minor correct elements
- 3 = Partially correct, missing key information
- 4 = Mostly correct with minor issues
- 5 = Fully correct and complete

The LLM returns `{"score": 4, "reasoning": "The answer correctly identifies..."}`.

Both methods are optional. If `bert-score` is not installed, BERTScore is
silently skipped. If no LLM is provided, judge scoring is skipped.

---

## Scripts Reference

### scripts/make_docs_jsonl.py

Downloads 300 papers from the `gfissore/arxiv-abstracts-2021` HuggingFace
dataset and saves them to `data/raw/docs.jsonl`. Each line is a JSON object:

```json
{
  "id": "0704.0001",
  "title": "Calculation of prompt diphoton production...",
  "abstract": "A fully differential calculation...",
  "source": "gfissore/arxiv-abstracts-2021"
}
```

Run once to set up the corpus. Only needed if `data/raw/docs.jsonl` is missing.

```bash
/opt/anaconda3/envs/RAGLens/bin/python scripts/make_docs_jsonl.py
```

---

### scripts/generate_eval_set.py

Creates `data/eval_set.jsonl` — the ground-truth file used for evaluation.

For each sampled paper it generates a question from the title using one of
five templates (e.g. "What is the main topic of '{title}'?", "What approach
does the paper '{title}' propose?"), and records the paper's ID as the
correct document to retrieve and its abstract as the reference answer.

```bash
/opt/anaconda3/envs/RAGLens/bin/python scripts/generate_eval_set.py
/opt/anaconda3/envs/RAGLens/bin/python scripts/generate_eval_set.py --n 100 --seed 99
```

Options:
- `--n` — number of examples to generate (default: 50)
- `--seed` — random seed for reproducibility (default: 42)
- `--input` — path to source JSONL (default: `data/raw/docs.jsonl`)
- `--output` — path to write eval set (default: `data/eval_set.jsonl`)

---

### scripts/run_eval.py

The main evaluation runner. Builds pipelines for four configurations, runs
`evaluate_retrieval` on each, and saves results.

```bash
/opt/anaconda3/envs/RAGLens/bin/python scripts/run_eval.py
```

The four built-in configurations compared:

| Config name | Strategy | Chunk size | Overlap |
|---|---|---|---|
| `fixed-500-ol50` | fixed | 500 | 50 |
| `fixed-1000-ol150` | fixed | 1000 | 150 |
| `recursive-500-ol50` | recursive | 500 | 50 |
| `recursive-1000-ol150` | recursive | 1000 | 150 |

For each config it prints a live progress line:
```
MRR=0.643  R@1=0.420  R@3=0.610  R@5=0.700  R@10=0.820  (12.3s)
```

And saves a JSON file to `data/eval_results/<config_name>.json` containing
the full per-question breakdown. A `summary.json` is also written with just
the aggregate numbers for the dashboard's comparison views.

**OpenAI key** is read automatically from `.env` (variable name `OPENAI-KEY`).
If present, `gpt-4o-mini` is used for answer generation and LLM judge scoring.
If absent, the script runs retrieval-only evaluation (no LLM calls, no cost).

---

## Web Dashboard

The dashboard is split into two processes that run simultaneously:

| Process | What it does | Default URL |
|---|---|---|
| **FastAPI backend** (`dashboard/api.py`) | Serves eval results as JSON and handles live retrieval queries | `http://localhost:8000` |
| **Next.js frontend** (`raglens-ui/`) | React app with four pages, interactive charts, live query UI | `http://localhost:3000` |

### dashboard/api.py — FastAPI backend

The backend exposes four endpoints that the frontend calls:

| Method | Path | What it returns |
|---|---|---|
| `GET` | `/api/summary` | Array of all config metrics from `summary.json` |
| `GET` | `/api/configs` | List of config names with saved result files |
| `GET` | `/api/results/{config}` | Full per-question breakdown for one config |
| `POST` | `/api/query` | Run live retrieval and return top-k chunks |

**`POST /api/query` request body:**
```json
{
  "question": "What is gamma-ray emission region?",
  "chunking": "recursive",
  "chunk_size": 1000,
  "chunk_overlap": 150,
  "top_k": 5
}
```

**Pipeline caching:** The first call to `/api/query` for a given
`(chunking, chunk_size, chunk_overlap)` combination builds the FAISS index,
which takes 30–60 seconds. Subsequent calls with the same parameters reuse
the cached pipeline and return results immediately.

**Start the backend:**
```bash
/opt/anaconda3/envs/RAGLens/bin/uvicorn dashboard.api:app --reload --port 8000
```

---

### raglens-ui — Next.js frontend

**Start the frontend:**
```bash
cd raglens-ui
npm run dev
# → http://localhost:3000
```

The app has four pages accessible from the top navigation bar:

---

#### Overview (`/`)

The landing page. Shows aggregate metrics for all evaluated configurations at a glance.

- **Stat cards** — Best MRR, Best Recall@5, number of configs tested
- **Recall@k line chart** — one line per config. The x-axis is k (1, 3, 5, 10),
  the y-axis is Recall@k. A line that is high at k=1 means the correct document
  is almost always the very top result.
- **MRR bar chart** — configs sorted by MRR descending. Instantly shows which
  chunking strategy produces the most consistently high rankings.
- **Full metrics table** — all configs × all metrics (MRR, R@1, R@3, R@5, R@10,
  build time, eval time) in one scrollable table.

---

#### Drill-down (`/drill-down`)

Lets you examine exactly what happened for each individual question.

1. **Select a configuration** from the dropdown.
2. **Browse the question list** on the left — each row shows the question text,
   its Reciprocal Rank (color-coded green/amber/red), and Recall@5.
3. **Click a question** to open the detail view on the right:
   - Headline metrics for that specific question (RR, Recall@1/3/5/10)
   - The ground-truth doc ID that should have been retrieved
   - Every retrieved chunk as a card, color-coded:
     - **Green border + ✓ Relevant** — this chunk came from the correct document
     - **Red border + ✗ Not relevant** — this chunk was retrieved instead of the correct one
   - Each card shows: rank, similarity score, paper title, doc ID, L2 distance, chunk ID, and full text

This is the most diagnostic page. When a question has low recall you can see
exactly which documents the pipeline retrieved instead of the right one, which
reveals what the embeddings confused.

---

#### Compare (`/compare`)

Side-by-side comparison of any two configurations.

1. **Select Config A and Config B** from the two dropdowns.
2. **Metric delta cards** — one card per metric (MRR, R@1, R@3, R@5, R@10).
   Each card shows Config B's value, a ▲ green arrow if B beats A or a ▼ red
   arrow if A beats B, and the absolute difference in percentage points.
3. **Overlapping Recall@k chart** — both configs plotted on the same axes so
   you can see exactly where they diverge as k increases.
4. **Per-question scatter plot** — each dot is one question. The x-axis is
   Config A's Reciprocal Rank for that question, the y-axis is Config B's.
   The diagonal line is where both configs perform equally. Points above the
   diagonal = Config B won on that question; points below = Config A won.
   The color of each dot shows who won (blue = A, green = B, gray = tied).
   A tight cluster above the diagonal means B is systematically better; a
   scattered cloud means performance depends heavily on the specific question.

---

#### Inspector (`/inspector`)

A live query interface. Type any question, adjust the pipeline parameters, and
see what gets retrieved in real time — no pre-computed eval data needed.

- **Sidebar controls:**
  - Chunking strategy (recursive / fixed)
  - Chunk size (slider, 200–2000 chars)
  - Chunk overlap (slider, 0–400 chars)
  - Top-k (slider, 1–15)
- **Sample questions** — click any to pre-fill the input
- **Query input** — type a question and press Enter or click Retrieve
- **Results:**
  - Similarity bar chart — one bar per rank, colored by score. Immediately
    shows how confidently the top result was retrieved vs. the others.
  - Chunk cards — one per retrieved result, showing title, doc ID, similarity
    score, L2 distance, and full text.

> **Note:** The first query for a new set of parameters (e.g. a new chunk size)
> triggers a full pipeline build on the backend (~30–60 seconds). Subsequent
> queries with the same parameters are instant because the index is cached.

---

## Metrics Explained

### Recall@k

> "Of all the documents that should have been retrieved, what fraction
> actually appeared in the top k results?"

Formula: `hits_in_top_k / total_relevant`

- Recall@1 = 0.4 means the correct doc was the top result for 40% of questions
- Recall@5 = 0.7 means retrieving 5 chunks finds the right one 70% of the time
- Higher is always better
- Recall@k increases monotonically as k grows (more chances to find the answer)

### Precision@k

> "Of the k chunks returned, what fraction were actually relevant?"

Formula: `relevant_in_top_k / k`

- Precision@1 = Recall@1 when there is exactly one relevant doc per question
- Precision decreases as k grows (more irrelevant chunks are pulled in)
- Useful for understanding how noisy retrieval is

### MRR (Mean Reciprocal Rank)

> "On average, at what rank does the correct document first appear?"

For each question: `Reciprocal Rank = 1 / rank_of_first_relevant_result`
- Rank 1 → RR = 1.0 (best)
- Rank 2 → RR = 0.5
- Rank 5 → RR = 0.2
- Not found → RR = 0.0

MRR is the average of RR across all questions. It heavily rewards finding the
correct document at rank 1 and penalizes pushing it down even slightly. A
system with MRR 0.5 finds the correct doc at rank 2 on average.

### BERTScore F1

> "How semantically similar is the generated answer to the reference answer?"

Scores are in [0, 1]. Unlike BLEU (exact word match), BERTScore uses
contextual embeddings so paraphrases score well. F1 ~0.85 is typically
considered good for English text.

### LLM Judge Score

> "Does the generated answer correctly answer the question?"

Score from 1 to 5 assigned by `gpt-4o-mini`. More flexible than BERTScore
because the LLM understands nuance, can detect hallucinations, and provides
a natural language explanation for its rating.

---

## Running Everything From Scratch

```bash
# 0. Activate the conda environment
conda activate RAGLens

# 1. Install Python dependencies
/opt/anaconda3/envs/RAGLens/bin/pip install -r requirements.txt

# 2. Add your OpenAI key to .env (skip if retrieval-only is fine)
echo 'OPENAI-KEY=sk-your-key-here' > .env

# 3. Download the corpus (skip if data/raw/docs.jsonl already exists)
/opt/anaconda3/envs/RAGLens/bin/python scripts/make_docs_jsonl.py

# 4. Generate evaluation questions
/opt/anaconda3/envs/RAGLens/bin/python scripts/generate_eval_set.py

# 5. Run evaluation across all four configs
/opt/anaconda3/envs/RAGLens/bin/python scripts/run_eval.py

# 6. Install frontend dependencies (one-time)
cd raglens-ui && npm install && cd ..

# 7. Start the FastAPI backend (Terminal 1)
/opt/anaconda3/envs/RAGLens/bin/uvicorn dashboard.api:app --reload --port 8000

# 8. Start the Next.js frontend (Terminal 2)
cd raglens-ui && npm run dev

# → Open http://localhost:3000
```

---

## Configuration Knobs

| Parameter | Where set | Effect |
|---|---|---|
| Chunking strategy | `RAGConfig` / Inspector sidebar | fixed vs recursive splitting |
| Chunk size | `RAGConfig` / Inspector sidebar | smaller = more chunks, more specific retrieval |
| Chunk overlap | `RAGConfig` / Inspector sidebar | more overlap = fewer boundary artifacts |
| Embedding model | `RAGConfig` | controls the semantic space |
| top_k | `RAGConfig` / Inspector sidebar | how many chunks to retrieve |
| eval set size `--n` | `generate_eval_set.py` | more questions = more reliable metrics |
| k values `--ks` | `run_eval.py` | which cutoffs to compute Recall@k for |

The general trade-off: smaller chunks improve retrieval precision (each chunk
is about one thing) but risk splitting a relevant answer across two chunks.
Larger chunks capture more context but may dilute the relevant signal in a
sea of unrelated text.

---

## Environment & Dependencies

### Python (`requirements.txt`)

| Package | Purpose |
|---|---|
| `langchain`, `langchain-core` | Document abstraction, pipeline primitives |
| `langchain-text-splitters` | Fixed and recursive text chunkers |
| `langchain-community` | FAISS vector store integration |
| `langchain-huggingface` | HuggingFace embedding model wrapper |
| `langchain-openai` | OpenAI chat model wrapper |
| `sentence-transformers` | Underlying HuggingFace embedding models |
| `faiss-cpu` | Fast approximate nearest-neighbour search |
| `datasets` | HuggingFace datasets (for downloading corpus) |
| `fastapi` | REST API framework for the dashboard backend |
| `uvicorn[standard]` | ASGI server that runs the FastAPI app |
| `streamlit` | Alternative dashboard (single-file, no JS required) |
| `plotly` | Charts for the Streamlit dashboard |
| `pandas` | Data manipulation for charts and tables |
| `bert-score` | BERTScore answer evaluation (optional) |
| `tqdm` | Progress bars |
| `pyyaml` | YAML config parsing |

### JavaScript (`raglens-ui/package.json`)

| Package | Purpose |
|---|---|
| `next` | React framework (App Router, server components) |
| `react`, `react-dom` | UI library |
| `recharts` | Composable charting library (LineChart, BarChart, ScatterChart) |
| `typescript` | Static typing |
| `tailwindcss` | Utility-first CSS framework |
