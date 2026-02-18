"""
Generate a synthetic evaluation set from data/raw/docs.jsonl.

Each example contains:
  - question: a natural-language question about the paper
  - reference_answer: the paper's abstract (used as ground truth for answer eval)
  - relevant_doc_ids: [doc_id]  (used as ground truth for retrieval eval)
  - relevant_titles: [title]

Three question templates are used per paper so the eval set covers
different phrasings and avoids relying on exact title matching.

Usage:
    python scripts/generate_eval_set.py [--n 50] [--seed 42]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


TEMPLATES = [
    # Topic / summary questions
    lambda title, abstract: f"What is the main topic of the paper titled '{title}'?",
    lambda title, abstract: f"Summarize the research on {title}.",
    # Method / approach questions  (use first sentence of abstract as a hint)
    lambda title, abstract: (
        f"What approach does the paper '{title}' propose?"
    ),
    # Finding / result questions
    lambda title, abstract: (
        f"What are the key findings or contributions of '{title}'?"
    ),
    # Domain / application questions
    lambda title, abstract: (
        f"In what domain or field is the work '{title}' situated?"
    ),
]


def make_question(doc: dict, rng: random.Random) -> dict:
    title = doc["title"].strip()
    abstract = doc["abstract"].strip()
    template = rng.choice(TEMPLATES)
    return {
        "question": template(title, abstract),
        "reference_answer": abstract,
        "relevant_doc_ids": [doc["id"]],
        "relevant_titles": [title],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic eval set.")
    parser.add_argument("--n", type=int, default=50, help="Number of examples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input", default="data/raw/docs.jsonl", help="Source JSONL path")
    parser.add_argument("--output", default="data/eval_set.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    docs = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    n = min(args.n, len(docs))
    sampled = rng.sample(docs, n)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in sampled:
            example = make_question(doc, rng)
            f.write(json.dumps(example) + "\n")

    print(f"Wrote {n} evaluation examples to {output_path}")
    print(f"Sample question: {sampled[0]['title']!r}")
    print(f"  -> {make_question(sampled[0], rng)['question']!r}")


if __name__ == "__main__":
    main()
