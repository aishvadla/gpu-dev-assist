"""Hybrid retrieval: combine dense and BM25 rankings with reciprocal rank fusion."""

from gpu_dev_assist.ingestion.chunking_embedding import ChunkingEmbeddingPipeline
from gpu_dev_assist.ingestion.vector_store import VectorStore
from gpu_dev_assist.retrieval.dense import DenseRetriever
from gpu_dev_assist.retrieval.keyword import BM25Retriever


class HybridRetriever:
    def __init__(self, dense, keyword, k: int = 60, candidates: int = 20):
        self.dense = dense
        self.keyword = keyword
        self.k = k
        self.candidates = candidates

    def query_retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        ranked_lists = [
            self.dense.query_retrieve(query, top_k=self.candidates),
            self.keyword.query_retrieve(query, top_k=self.candidates),
        ]
        scores, items = {}, {}
        for results in ranked_lists:
            for rank, r in enumerate(results, start=1):
                scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (self.k + rank)
                items.setdefault(r["id"], r)

        best = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [{**items[i], "rrf_score": round(scores[i], 4)} for i in best]

def build_retrievers(vector_store=None, embed_pipe=None):
    """Build the dense, BM25 and hybrid retrievers sharing one store and one model."""
    vector_store = vector_store or VectorStore()
    embed_pipe = embed_pipe or ChunkingEmbeddingPipeline()

    dense = DenseRetriever(vector_store, embed_pipe)
    keyword = BM25Retriever(vector_store)
    hybrid = HybridRetriever(dense, keyword)
    return {"dense": dense, "bm25": keyword, "hybrid": hybrid}