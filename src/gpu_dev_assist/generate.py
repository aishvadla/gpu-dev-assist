"""Grounded answer generation with numbered source citations."""

import re

import anthropic
from dotenv import load_dotenv

from gpu_dev_assist.utils.logger import logger
from gpu_dev_assist.hybrid_retrieve import build_retrievers

load_dotenv()

MODEL = "claude-haiku-5-5"
INSUFFICIENT_EVIDENCE = "I don't have enough evidence in the documentation to answer this."

SYSTEM_PROMPT = f"""You answer questions about Intel oneAPI and GPU programming using only the numbered sources provided.

Rules:
- Every factual sentence must end with the number of the source that supports it, like [2].
- Use only the sources. Do not add outside knowledge.
- If the sources don't contain the answer, reply exactly: "{INSUFFICIENT_EVIDENCE}"
- Be concise and technical."""


def build_context(results: list[dict]) -> str:
    """Number each retrieved chunk so the model can cite it."""
    return "\n\n".join(
        f"[{i}] ({r['source']}, page {r['page']})\n{r['text']}"
        for i, r in enumerate(results, start=1)
    )


class AnswerGenerator:
    def __init__(self, retriever, client=None, model: str = MODEL):
        self.retriever = retriever
        self.client = client or anthropic.Anthropic()
        self.model = model

    def answer(self, question: str, top_k: int = 6) -> dict:
        results = self.retriever.query_retrieve(question, top_k=top_k)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Sources:\n{build_context(results)}\n\nQuestion: {question}",
            }],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        logger.info(
            f"Answer: {response.usage.input_tokens} in, {response.usage.output_tokens} out tokens"
        )

        # Map the [n] markers the model used back to their source and page
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
        sources = [
            {"id": i, "source": results[i - 1]["source"], "page": results[i - 1]["page"]}
            for i in cited
            if 1 <= i <= len(results)
        ]

        return {"question": question, "answer": text, "sources": sources, "retrieved": results}


if __name__ == "__main__":
    from gpu_dev_assist.chunking_embedding import ChunkingEmbeddingPipeline
    from gpu_dev_assist.vector_store import VectorStore

    retrievers = build_retrievers()
    generator = AnswerGenerator(retrievers["dense"])

    for q in [
        "What is the oneAPI programming model?",
        "When should I use shared local memory in a SYCL kernel?",
        "What is NVIDIA's CUDA warp size?",
    ]:
        result = generator.answer(q)
        print(f"\nQ: {q}\n{result['answer']}")
        for s in result["sources"]:
            print(f"  [{s['id']}] {s['source']}, p. {s['page']}")