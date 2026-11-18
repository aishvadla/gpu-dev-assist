from gpu_dev_assist.utils.logger import logger
from gpu_dev_assist.data_loader import load_all_documents
from gpu_dev_assist.chunking_embedding import ChunkingEmbeddingPipeline
from gpu_dev_assist.vector_store import VectorStore
from gpu_dev_assist.retrieve import Retriever
from typing import Dict, Any, List

class DataIngestion:
    def __init__(self, data_dir: str = 'data/raw', chunk_size: int = 1000, chunk_overlap: int = 200, model_name: str = "all-MiniLM-L6-v2",
                 collection_name: str = 'pdf_documents', persist_directory: str= 'data/vector_store'):
        self.data_dir = data_dir
        self.embed_pipe = ChunkingEmbeddingPipeline(chunk_size, chunk_overlap, model_name)
        self.vector_store = VectorStore(collection_name, persist_directory)

    def initiate_data_ingestion(self):
        logger.info(f"Initiated data ingestion...")
        # Loading all the document files from the data directory
        docs = load_all_documents(self.data_dir)

        # Chunking and Embedding
        chunks = self.embed_pipe.chunk_documents(docs)
        embeddings = self.embed_pipe.generate_embeddings([chunk.page_content for chunk in chunks])

        # Store the documents and embeddings to vector database
        self.vector_store.add_documents(chunks, embeddings)
        logger.info(f"Completed data ingestion.")

if __name__ == "__main__":

    # Data ingestion
    dataingestion_obj = DataIngestion()
    dataingestion_obj.initiate_data_ingestion()

    # Retieval using a query
    retriever = Retriever(dataingestion_obj.vector_store, dataingestion_obj.embed_pipe)
    query = 'what is oneapi programming model?'
    retieval_results = retriever.query_retrieve(query, top_k=5)
    for r in retieval_results:
            print(r["page"], r["distance"], r["text"][:80].replace("\n", " "))