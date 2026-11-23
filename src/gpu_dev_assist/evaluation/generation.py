"""Generation evaluation: refusal accuracy, citation hit and faithfulness (hybrid retriever)."""

import json
import re

import pandas as pd

from gpu_dev_assist.evaluation.retrieval import RESULTS_DIR, gold_pages, load_questions
from gpu_dev_assist.generate import INSUFFICIENT_EVIDENCE, AnswerGenerator
from gpu_dev_assist.retrieval.hybrid import build_retrievers

JUDGE_PROMPT = """You check whether an answer is supported by its sources.

Split the answer into individual factual claims. For each claim, decide if the numbered sources support it.
Reply with JSON only: {"total_claims": <int>, "supported_claims": <int>, "unsupported": ["<claim>", ...]}"""


def judge_faithfulness(client, answer: str, retrieved: list[dict]) -> dict:
    sources = "\n\n".join(f"[{i}] {r['text']}" for i, r in enumerate(retrieved, start=1))
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": f"Sources:\n{sources}\n\nAnswer:\n{answer}"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    data = json.loads(re.search(r"\{.*\}", text, re.S).group())  # tolerate code fences
    data["faithfulness"] = data["supported_claims"] / max(data["total_claims"], 1)
    return data


def evaluate_generation(generator: AnswerGenerator, questions: list[dict]) -> pd.DataFrame:
    rows = []
    for q in questions:
        result = generator.answer(q["question"])
        refused = result["answer"].strip() == INSUFFICIENT_EVIDENCE
        row = {"id": q["id"], "type": q["type"], "answerable": q["answerable"], "refused": refused,
               "answer": result["answer"]}

        if q["answerable"] and not refused:
            gold = gold_pages(q)
            cited = {(s["source"], s["page"]) for s in result["sources"]}
            row["cited_any"] = bool(cited)
            row["citation_hit"] = bool(cited & gold)
            row.update(judge_faithfulness(generator.client, result["answer"], result["retrieved"]))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    generator = AnswerGenerator(build_retrievers()["hybrid"])
    gen_df = evaluate_generation(generator, load_questions())

    answerable = gen_df[gen_df["answerable"]]
    unanswerable = gen_df[~gen_df["answerable"]]
    answered = answerable[~answerable["refused"]]

    print("\nGeneration (hybrid retriever):")
    print(f"  Answered when answerable:  {len(answered)}/{len(answerable)}")
    print(f"  Refused when unanswerable: {unanswerable['refused'].sum()}/{len(unanswerable)}")
    print(f"  Citation hit (answered):   {answered['citation_hit'].mean():.2f}")
    print(f"  Mean faithfulness:         {answered['faithfulness'].mean():.2f}")
    gen_df.to_csv(RESULTS_DIR / "generation_per_question.csv", index=False)