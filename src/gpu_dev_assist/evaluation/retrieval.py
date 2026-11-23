"""Retrieval evaluation: Hit@5, Recall@5 and MRR@10 for each retriever."""

import json
from pathlib import Path

import pandas as pd

from gpu_dev_assist.retrieval.hybrid import build_retrievers
from gpu_dev_assist.ingestion.vector_store import PROJECT_ROOT

QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "results"

# Short document keys in questions.jsonl -> file names stored in chunk metadata
DOC_FILES = {
    "programming_guide": "oneapi_programming-guide_2025.1-771723-848694.pdf",
    "optimization_guide": "oneapi_optimization-guide-gpu_2025.0-771772-834474.pdf",
}


# ---------- metrics (pure functions, easy to test) ----------

def ranked_pages(results: list[dict]) -> list[tuple[str, int]]:
    """Ordered (source, page) list with duplicate pages removed, keeping the first occurrence."""
    seen, pages = set(), []
    for r in results:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            pages.append(key)
    return pages


def hit_at_k(ranked: list, relevant: set, k: int) -> float:
    return float(any(p in relevant for p in ranked[:k]))


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: list, relevant: set, k: int) -> float:
    for rank, p in enumerate(ranked[:k], start=1):
        if p in relevant:
            return 1.0 / rank
    return 0.0


# ---------- evaluation ----------

def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_pages(question: dict) -> set[tuple[str, int]]:
    return {
        (DOC_FILES[doc], page)
        for doc, pages in question["relevant_pages"].items()
        for page in pages
    }


def evaluate_retrievers(retrievers: dict, questions: list[dict], k: int = 5, depth: int = 10) -> pd.DataFrame:
    rows = []
    for name, retriever in retrievers.items():
        for q in questions:
            relevant = gold_pages(q)
            ranked = ranked_pages(retriever.query_retrieve(q["question"], top_k=depth))
            first_hit = next((i for i, p in enumerate(ranked, start=1) if p in relevant), None)
            rows.append({
                "retriever": name,
                "id": q["id"],
                "type": q["type"],
                "hit@5": hit_at_k(ranked, relevant, k),
                "recall@5": recall_at_k(ranked, relevant, k),
                "mrr@10": reciprocal_rank(ranked, relevant, depth),
                "first_relevant_rank": first_hit,
                "top_pages": [p for _, p in ranked[:k]],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    questions = [q for q in load_questions() if q["answerable"]]
    retrievers = build_retrievers()

    df = evaluate_retrievers(retrievers, questions)
    metrics = ["hit@5", "recall@5", "mrr@10"]

    print(f"\n{len(questions)} answerable questions\n")
    print(df.groupby("retriever")[metrics].mean().round(3))
    print("\nHit@5 by question type:")
    print(df.pivot_table(index="type", columns="retriever", values="hit@5", aggfunc="mean").round(2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "retrieval_per_question.csv", index=False)
    print(f"\nPer-question results saved to {RESULTS_DIR / 'retrieval_per_question.csv'}")