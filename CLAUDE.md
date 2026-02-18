# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python environment

All Python commands must use the conda environment's interpreter directly — the shell's `python` resolves to system Python 3.12 which is missing all project dependencies:

```bash
/opt/anaconda3/envs/RAGLens/bin/python <script>
/opt/anaconda3/envs/RAGLens/bin/pip install <package>
/opt/anaconda3/envs/RAGLens/bin/uvicorn dashboard.api:app --reload --port 8000
```

Python version: 3.11. No test suite exists yet.

## Running the system

**End-to-end pipeline (run in order on a fresh clone):**
```bash
# 1. Generate corpus (skip if data/raw/docs.jsonl exists)
/opt/anaconda3/envs/RAGLens/bin/python scripts/make_docs_jsonl.py

# 2. Generate eval questions
/opt/anaconda3/envs/RAGLens/bin/python scripts/generate_eval_set.py

# 3. Evaluate all configs → writes data/eval_results/*.json
/opt/anaconda3/envs/RAGLens/bin/python scripts/run_eval.py
```

**Web dashboard (two terminals):**
```bash
# Terminal 1 — FastAPI backend
/opt/anaconda3/envs/RAGLens/bin/uvicorn dashboard.api:app --reload --port 8000

# Terminal 2 — Next.js frontend
cd raglens-ui && npm run dev   # → http://localhost:3000
```

**Streamlit alternative (single terminal):**
```bash
streamlit run dashboard/app.py
```

## Architecture

The system has three layers that build on each other:

**1. Pipeline layer** (`raglens/`)
`ingest.py` loads JSONL → `chunking.py` splits into chunks → `pipeline.py` embeds with `intfloat/e5-base-v2` via HuggingFace and indexes with FAISS. `RAGConfig` is the single dataclass that parameterises all pipeline knobs. `RAGPipeline.build()` is expensive (30–60s); results are cached in `dashboard/api.py`'s `_pipeline_cache` dict keyed by `(chunking, chunk_size, chunk_overlap)`.

**2. Evaluation layer** (`raglens/eval/`)
`retrieval_eval.py` drives a batch eval loop: for each `EvalExample` (question + ground-truth doc IDs) it queries the vectorstore, marks each retrieved chunk `is_relevant`, and computes Recall@k / Precision@k / MRR. Results aggregate into a `RetrievalReport` and are serialised to `data/eval_results/<config_name>.json` by `scripts/run_eval.py`. `answer_eval.py` is independent — it evaluates generated answers via BERTScore and/or an LLM judge.

**3. Visualisation layer**
`dashboard/api.py` is a FastAPI app that serves `data/eval_results/` as JSON and exposes `POST /api/query` for live retrieval. `raglens-ui/` is a Next.js 13 app (App Router, TypeScript, Tailwind, Recharts) with four pages: Overview, Drill-down, Compare, Inspector. All API calls go through `raglens-ui/src/lib/api.ts`; types are in `src/lib/types.ts`.

## Key data contracts

**`data/raw/docs.jsonl`** — one JSON object per line: `{id, title, abstract, source}`. The `id` field is the ground-truth key used throughout evaluation.

**`data/eval_results/<config>.json`** — shape expected by both the Streamlit dashboard and the Next.js frontend:
```
{ config_name, macro_mrr, macro_recall_at_k, macro_precision_at_k,
  per_question: [{ question, relevant_doc_ids, reciprocal_rank,
                   recall_at_k, precision_at_k,
                   retrieved: [{ doc_id, chunk_id, title, l2_distance,
                                 similarity, content, is_relevant }] }] }
```

**FAISS similarity scores** — FAISS returns L2 distances (lower = more similar). All code converts to `similarity = 1 / (1 + l2_distance)` before displaying or returning to the frontend.

## Environment variables

Stored in `.env` at the project root. Variable name is `OPENAI-KEY` (hyphen, not underscore). Both `scripts/run_eval.py` and `dashboard/api.py` contain a `_load_env()` helper (no third-party dotenv dependency) that reads `.env` and re-exports `OPENAI-KEY` as `OPENAI_API_KEY` for LangChain/OpenAI SDK compatibility.

## Adding a new eval configuration

Add a `RAGConfig(...)` entry to the `CONFIGS` list in `scripts/run_eval.py`. Re-run the script; it writes a new JSON file to `data/eval_results/` which the dashboard picks up automatically on next load.
