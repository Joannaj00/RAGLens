"""
Answer quality evaluation: BERTScore and LLM-as-a-judge.

BERTScore measures semantic similarity between generated and reference answers
using contextual embeddings.  LLM-as-judge asks an LLM to score answer
correctness on a 1-5 scale with reasoning.

Both are optional: BERTScore requires `bert-score` to be installed;
LLM-as-judge requires a LangChain-compatible chat model to be passed in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AnswerScore:
    question: str
    generated_answer: str
    reference_answer: str
    bertscore_f1: Optional[float]   # None if bert-score not installed
    llm_score: Optional[int]        # 1-5, None if no LLM provided
    llm_reasoning: Optional[str]


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def bertscore_eval(predictions: List[str], references: List[str]) -> List[float]:
    """
    Compute BERTScore F1 for each (prediction, reference) pair.

    Requires: pip install bert-score
    Returns a list of float F1 scores in [0, 1].
    """
    try:
        from bert_score import score as _bert_score  # type: ignore
    except ImportError as e:
        raise ImportError(
            "bert-score is not installed. Run: pip install bert-score"
        ) from e

    _, _, F1 = _bert_score(predictions, references, lang="en", verbose=False)
    return F1.tolist()


# ---------------------------------------------------------------------------
# LLM-as-a-judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating the quality of a RAG system's generated answer.

Question: {question}

Reference Answer (ground truth):
{reference}

Generated Answer (to evaluate):
{generated}

Score the generated answer from 1 to 5 using these criteria:
  1 = Completely wrong or irrelevant
  2 = Mostly wrong, only minor correct elements
  3 = Partially correct, missing key information
  4 = Mostly correct with minor issues or omissions
  5 = Fully correct and complete

Respond with ONLY valid JSON, no markdown fences:
{{"score": <integer 1-5>, "reasoning": "<one or two sentences>"}}
"""


def llm_judge_single(
    question: str,
    generated: str,
    reference: str,
    llm,  # any LangChain chat model (e.g. ChatOpenAI)
) -> dict:
    """Ask an LLM to score one generated answer. Returns {score, reasoning}."""
    prompt = _JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        generated=generated,
    )
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
        return {"score": int(result["score"]), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": None, "reasoning": raw}


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_answers(
    questions: List[str],
    generated_answers: List[str],
    reference_answers: List[str],
    llm=None,
    use_bertscore: bool = True,
) -> List[AnswerScore]:
    """
    Evaluate a batch of generated answers against reference answers.

    Args:
        questions: The original questions.
        generated_answers: Answers produced by the RAG pipeline.
        reference_answers: Ground-truth / reference answers.
        llm: Optional LangChain chat model for LLM-as-judge scoring.
        use_bertscore: Whether to compute BERTScore (requires bert-score pkg).

    Returns:
        List of AnswerScore objects with per-question scores.
    """
    # BERTScore (optional)
    if use_bertscore:
        try:
            f1_scores = bertscore_eval(generated_answers, reference_answers)
        except ImportError:
            print("Warning: bert-score not installed; skipping BERTScore.")
            f1_scores = [None] * len(questions)
    else:
        f1_scores = [None] * len(questions)

    results: List[AnswerScore] = []
    for q, gen, ref, f1 in zip(questions, generated_answers, reference_answers, f1_scores):
        llm_score, llm_reasoning = None, None
        if llm is not None:
            judge = llm_judge_single(q, gen, ref, llm)
            llm_score = judge["score"]
            llm_reasoning = judge["reasoning"]

        results.append(AnswerScore(
            question=q,
            generated_answer=gen,
            reference_answer=ref,
            bertscore_f1=f1,
            llm_score=llm_score,
            llm_reasoning=llm_reasoning,
        ))

    return results
