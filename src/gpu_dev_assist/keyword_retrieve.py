"""BM25 keyword retrieval over the same chunks stored in the vector store."""

import re

import snowballstemmer
from rank_bm25 import BM25Okapi

from gpu_dev_assist.vector_store import VectorStore

_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stemmed so word forms match ("prefetching" -> "prefetch").

    Identifiers such as local_mem_size stay a single token, because underscores
    are part of the token pattern.
    """
    return _stemmer.stemWords(re.findall(r"[a-z0-9_]+", text.lower()))


class BM25Retriever:
    def __init__(self, vector_store: VectorStore):
        data = vector_store.collection.get(include=["documents", "metadatas"])
        self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])

    def query_retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "id": self.ids[i],
                "text": self.docs[i],
                "source": self.metas[i]["source_file"],
                "page": self.metas[i]["page"] + 1,
                "score": round(float(scores[i]), 3),
            }
            for i in top
        ]