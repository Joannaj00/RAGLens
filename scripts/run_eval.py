"""
Run retrieval evaluation across multiple RAG configurations.

For each config the script:
  1. Builds the pipeline (chunks docs, embeds, indexes with FAISS).
  2. Runs evaluate_retrieval() over the eval set.
  3. Saves a JSON result file to data/eval_results/<config_name>.json.

The OpenAI key is read from the .env file in the project root (OPENAI-KEY).
If present, the script also runs LLM answer generation and judge scoring.

Usage:
    python scripts/run_eval.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

# Add the project root to sys.path so raglens package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def _load_env(path: Path) -> None:
    """Minimal .env loader — no third-party deps required."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_load_env(_ENV_PATH)

# .env uses OPENAI-KEY; LangChain/OpenAI SDK expects OPENAI_API_KEY
_raw_key = os.environ.get("OPENAI-KEY", "")
if _raw_key:
    os.environ["OPENAI_API_KEY"] = _raw_key

from raglens.eval.retrieval_eval import EvalExample, evaluate_retrieval, RetrievalReport
from raglens.pipeline import RAGConfig, RAGPipeline


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _report_to_dict(report: RetrievalReport) -> dict:
    """Convert a RetrievalReport to a JSON-serialisable dict."""
    return {
        "config_name": report.config_name,
        "macro_mrr": report.macro_mrr,
        "macro_recall_at_k": {str(k): v for k, v in report.macro_recall_at_k.items()},
        "macro_precision_at_k": {str(k): v for k, v in report.macro_precision_at_k.items()},
        "per_question": [
            {
                "question": qr.question,
                "relevant_doc_ids": qr.relevant_doc_ids,
                "reciprocal_rank": qr.reciprocal_rank,
                "recall_at_k": {str(k): v for k, v in qr.recall_at_k.items()},
                "precision_at_k": {str(k): v for k, v in qr.precision_at_k.items()},
                "retrieved": [
                    {
                        "doc_id": c.doc_id,
                        "chunk_id": c.chunk_id,
                        "title": c.title,
                        "l2_distance": c.l2_distance,
                        "similarity": c.similarity,
                        "is_relevant": c.is_relevant,
                        "content": c.content,
                    }
                    for c in qr.retrieved
                ],
            }
            for qr in report.per_question
        ],
    }


# ---------------------------------------------------------------------------
# Evaluation configs to compare
# ---------------------------------------------------------------------------

CONFIGS = [
    RAGConfig(chunking_strategy="fixed",     chunk_size=500,  chunk_overlap=50),
    RAGConfig(chunking_strategy="fixed",     chunk_size=1000, chunk_overlap=150),
    RAGConfig(chunking_strategy="recursive", chunk_size=500,  chunk_overlap=50),
    RAGConfig(chunking_strategy="recursive", chunk_size=1000, chunk_overlap=150),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_eval_set(path: str) -> list[EvalExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(EvalExample(
                question=row["question"],
                reference_answer=row["reference_answer"],
                relevant_doc_ids=row["relevant_doc_ids"],
            ))
    return examples


def main():
    parser = argparse.ArgumentParser(description="Run RAGLens evaluation.")
    parser.add_argument(
        "--eval-set", default="data/eval_set.jsonl",
        help="Path to eval set JSONL (default: data/eval_set.jsonl)"
    )
    parser.add_argument(
        "--data", default="data/raw/docs.jsonl",
        help="Path to source docs JSONL (default: data/raw/docs.jsonl)"
    )
    parser.add_argument(
        "--out-dir", default="data/eval_results",
        help="Directory to write result JSON files (default: data/eval_results)"
    )
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[1, 3, 5, 10],
        help="Recall@k cutoffs (default: 1 3 5 10)"
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        print(f"Eval set not found at {eval_path}.")
        print("Run first: python scripts/generate_eval_set.py")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_examples = load_eval_set(str(eval_path))
    print(f"Loaded {len(eval_examples)} evaluation examples from {eval_path}\n")

    # Optional LLM setup — key comes from .env (OPENAI-KEY)
    llm = None
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("OpenAI key found — LLM answer generation enabled.\n")
    else:
        print("No OPENAI-KEY in .env — running retrieval-only evaluation.\n")

    summary_rows = []

    for config in CONFIGS:
        print(f"[{config.config_name}] Building pipeline...")
        t0 = time.time()
        pipeline = RAGPipeline(config, data_path=args.data).build()
        if llm:
            pipeline.set_llm(llm)
        build_time = time.time() - t0

        print(f"[{config.config_name}] Running retrieval eval ({len(eval_examples)} questions)...")
        t1 = time.time()
        report = evaluate_retrieval(
            eval_examples, pipeline.vectorstore, ks=args.ks,
            config_name=config.config_name,
        )
        eval_time = time.time() - t1

        out_path = out_dir / f"{config.config_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_report_to_dict(report), f, indent=2)

        row = {
            "config": config.config_name,
            "mrr": report.macro_mrr,
            **{f"recall@{k}": report.macro_recall_at_k[k] for k in args.ks},
            "build_s": round(build_time, 1),
            "eval_s": round(eval_time, 1),
        }
        summary_rows.append(row)

        print(
            f"  MRR={report.macro_mrr:.3f}  "
            + "  ".join(f"R@{k}={report.macro_recall_at_k[k]:.3f}" for k in args.ks)
            + f"  ({eval_time:.1f}s)"
        )
        print(f"  Saved -> {out_path}\n")

    # Print a compact summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Config':<30} {'MRR':>6}" + "".join(f"  R@{k:>2}" for k in args.ks)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        line = f"{row['config']:<30} {row['mrr']:>6.3f}"
        for k in args.ks:
            line += f"  {row[f'recall@{k}']:>4.3f}"
        print(line)

    # Save combined summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
