from gpu_dev_assist.data_loader import load_all_documents
from gpu_dev_assist.chunking_embedding import ChunkingEmbeddingPipeline
from gpu_dev_assist.vector_store import VectorStore
from typing import Dict, Any, List
from gpu_dev_assist.utils.logger import logger

class DenseRetriever:
    def __init__(self, vector_store:VectorStore, embed_pipe: ChunkingEmbeddingPipeline):
        self.vector_store = vector_store
        self.embed_pipe = embed_pipe

    def query_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving context from the vector database...")
        # generate embedding for the query
        query_embedding = self.embed_pipe.generate_embeddings([query])[0]

        # query it with the client vectordb to get the top k chunks that match the query embedding
        # similarity search using cosine similarity
        results = self.vector_store.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

        logger.info(f"Completed context retrieval using query")
        return [
            {
                "id": chunk_id,
                "text": text,
                "source": meta["source_file"],
                "page": meta["page"] + 1,  # PyPDFLoader pages start at 0
                "distance": round(dist, 3),
            }
            for chunk_id, text, meta, dist in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]


if __name__ == "__main__":

    emb_pipe = ChunkingEmbeddingPipeline()
    vector_store = VectorStore()

    # Retieval using a query
    retrival_obj = DenseRetriever(vector_store, emb_pipe)
    query = 'what is oneapi programming model?'
    retieval_results = retrival_obj.query_retrieve(query, top_k=5)
    for r in retieval_results:
                print(r["page"], r["distance"], r["text"][:80].replace("\n", " "))