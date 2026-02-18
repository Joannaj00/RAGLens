"""
RAGLens FastAPI backend.

Serves pre-computed evaluation results and handles live retrieval queries.

Endpoints:
  GET  /api/summary              → summary.json (all configs, aggregated metrics)
  GET  /api/configs              → list of config names with saved results
  GET  /api/results/{config}     → full per-question breakdown for one config
  POST /api/query                → live retrieval for an arbitrary question

Run:
    uvicorn dashboard.api:app --reload --port 8000
  or from the project root:
    /opt/anaconda3/envs/RAGLens/bin/uvicorn dashboard.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make raglens importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env (OPENAI-KEY → OPENAI_API_KEY)
def _load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_ROOT = Path(__file__).resolve().parent.parent
_load_env(_ROOT / ".env")
_raw_key = os.environ.get("OPENAI-KEY", "")
if _raw_key:
    os.environ["OPENAI_API_KEY"] = _raw_key

EVAL_DIR = _ROOT / "data" / "eval_results"
DATA_PATH = str(_ROOT / "data" / "raw" / "docs.jsonl")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RAGLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pipeline cache (built lazily on first /api/query call)
# ---------------------------------------------------------------------------

_pipeline_cache: dict = {}

def _get_pipeline(chunking: str, chunk_size: int, chunk_overlap: int):
    from raglens.pipeline import RAGConfig, RAGPipeline
    key = (chunking, chunk_size, chunk_overlap)
    if key not in _pipeline_cache:
        config = RAGConfig(
            chunking_strategy=chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        _pipeline_cache[key] = RAGPipeline(config, data_path=DATA_PATH).build()
    return _pipeline_cache[key]

# ---------------------------------------------------------------------------
# Evaluation result endpoints
# ---------------------------------------------------------------------------

@app.get("/api/summary")
def get_summary():
    path = EVAL_DIR / "summary.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="summary.json not found. Run: python scripts/run_eval.py"
        )
    with open(path) as f:
        return json.load(f)


@app.get("/api/configs")
def get_configs():
    if not EVAL_DIR.exists():
        return []
    return [
        p.stem for p in sorted(EVAL_DIR.glob("*.json"))
        if p.stem != "summary"
    ]


@app.get("/api/results/{config_name}")
def get_config_results(config_name: str):
    path = EVAL_DIR / f"{config_name}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results found for config '{config_name}'. Run scripts/run_eval.py first."
        )
    with open(path) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Live query endpoint
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    chunking: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5


@app.post("/api/query")
def run_query(req: QueryRequest):
    """Build (or reuse cached) pipeline and retrieve top-k chunks."""
    try:
        pipeline = _get_pipeline(req.chunking, req.chunk_size, req.chunk_overlap)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build pipeline: {e}")

    chunks = pipeline.retrieve(req.question, k=req.top_k)
    return {
        "question": req.question,
        "config": {
            "chunking": req.chunking,
            "chunk_size": req.chunk_size,
            "chunk_overlap": req.chunk_overlap,
            "top_k": req.top_k,
        },
        "chunks": [
            {
                "rank": i + 1,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "title": c.title,
                "similarity": round(c.similarity, 4),
                "l2_distance": round(c.l2_distance, 4),
                "content": c.content,
            }
            for i, c in enumerate(chunks)
        ],
    }
