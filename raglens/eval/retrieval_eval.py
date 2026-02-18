"""
Retrieval evaluation metrics: Recall@k, MRR, Precision@k.

Scores are computed per-question then macro-averaged across the eval set.
FAISS returns L2 distances (lower = more similar); we store raw distances
and convert to a [0, 1] similarity for display: sim = 1 / (1 + distance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalExample:
    """A single evaluation question with ground-truth document IDs."""
    question: str
    reference_answer: str
    relevant_doc_ids: List[str]  # doc-level IDs that contain the answer


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: int | None
    title: str
    l2_distance: float        # raw FAISS L2 distance (lower = more similar)
    similarity: float         # 1 / (1 + distance) in [0, 1]
    content: str
    is_relevant: bool         # True if doc_id is in the ground-truth set


@dataclass
class QuestionResult:
    question: str
    relevant_doc_ids: List[str]
    retrieved: List[RetrievedChunk]
    recall_at_k: Dict[int, float]   # {k: recall}
    precision_at_k: Dict[int, float]
    reciprocal_rank: float


@dataclass
class RetrievalReport:
    config_name: str
    macro_recall_at_k: Dict[int, float]
    macro_precision_at_k: Dict[int, float]
    macro_mrr: float
    per_question: List[QuestionResult]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in set(relevant_ids))
    return min(hits, len(relevant_ids)) / len(relevant_ids)


def _precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of top-k retrieved results that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in set(relevant_ids))
    return hits / k


def _reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Reciprocal of the rank of the first relevant result (0 if none found)."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    eval_examples: List[EvalExample],
    vectorstore: FAISS,
    ks: List[int] = [1, 3, 5, 10],
    config_name: str = "default",
) -> RetrievalReport:
    """
    Run retrieval for every question in eval_examples and compute
    Recall@k, Precision@k, and MRR, then macro-average over all questions.

    Args:
        eval_examples: List of EvalExample with questions and ground-truth doc IDs.
        vectorstore: A built FAISS vectorstore to query against.
        ks: List of cutoffs to evaluate (e.g. [1, 3, 5, 10]).
        config_name: Label for this configuration (used in reports/charts).

    Returns:
        RetrievalReport with per-question results and macro-averaged metrics.
    """
    max_k = max(ks)
    per_question: List[QuestionResult] = []

    for ex in eval_examples:
        raw_results = vectorstore.similarity_search_with_score(ex.question, k=max_k)

        retrieved_chunks: List[RetrievedChunk] = []
        retrieved_ids: List[str] = []

        for doc, l2_dist in raw_results:
            doc_id = doc.metadata.get("id", "")
            retrieved_chunks.append(RetrievedChunk(
                doc_id=doc_id,
                chunk_id=doc.metadata.get("chunk_id"),
                title=doc.metadata.get("title", ""),
                l2_distance=float(l2_dist),
                similarity=1.0 / (1.0 + float(l2_dist)),
                content=doc.page_content,
                is_relevant=doc_id in set(ex.relevant_doc_ids),
            ))
            retrieved_ids.append(doc_id)

        recall_at_k = {k: _recall_at_k(retrieved_ids, ex.relevant_doc_ids, k) for k in ks}
        precision_at_k = {k: _precision_at_k(retrieved_ids, ex.relevant_doc_ids, k) for k in ks}
        rr = _reciprocal_rank(retrieved_ids, ex.relevant_doc_ids)

        per_question.append(QuestionResult(
            question=ex.question,
            relevant_doc_ids=ex.relevant_doc_ids,
            retrieved=retrieved_chunks,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            reciprocal_rank=rr,
        ))

    n = len(per_question)
    macro_recall_at_k = {
        k: sum(r.recall_at_k[k] for r in per_question) / n for k in ks
    }
    macro_precision_at_k = {
        k: sum(r.precision_at_k[k] for r in per_question) / n for k in ks
    }
    macro_mrr = sum(r.reciprocal_rank for r in per_question) / n

    return RetrievalReport(
        config_name=config_name,
        macro_recall_at_k=macro_recall_at_k,
        macro_precision_at_k=macro_precision_at_k,
        macro_mrr=macro_mrr,
        per_question=per_question,
    )
