from gpu_dev_assist.data_loader import load_all_documents
from gpu_dev_assist.chunking_embedding import ChunkingEmbeddingPipeline
from gpu_dev_assist.vector_store import VectorStore
from typing import Dict, Any, List
from gpu_dev_assist.utils.logger import logger

class Retriever:
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
                "text": text,
                "source": meta["source_file"],
                "page": meta["page"] + 1,  # PyPDFLoader pages start at 0
                "distance": round(dist, 3),
            }
            for text, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]


if __name__ == "__main__":

    # loading all documents
    docs = load_all_documents("data")

    # Document chunking and converting them to embeddings
    emb_pipe = ChunkingEmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.generate_embeddings(chunks)

    # Storing the chunks and embeddings to a vector store database
    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)

    # Retieval using a query
    retrival_obj = Retriever(vector_store, emb_pipe)
    query = 'what is oneapi programming model?'
    retieval_results = retrival_obj.query_retrieve(query, top_k=5)
    print(retieval_results)