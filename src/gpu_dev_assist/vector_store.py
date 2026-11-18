import os
import chromadb
from typing import List, Any
import numpy as np

from gpu_dev_assist.data_loader import load_all_documents
from gpu_dev_assist.chunking_embedding import ChunkingEmbeddingPipeline

class VectorStore:
    def __init__(self, collection_name: str = 'pdf_documents', persist_directory: str= 'data/vector_store'):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description":"pdf document embeddings for RAG", "hnsw:space": "cosine"})
        print(f"vector store initialized")

    def add_documents(self, documents:List[Any], embeddings: np.ndarray):
        ids = []
        metadatas = []
        documents_text = []

        for i, document in enumerate(documents):
            metadata = document.metadata
            metadata['doc_index'] = i
            metadata['content_length'] = len(document.page_content)
            metadatas.append(metadata)
            ids.append(f"doc_{metadata['source_file']}-p{metadata['page']}-c{i}")

            documents_text.append(document.page_content)

        self.collection.add(ids=ids, metadatas=metadatas, documents=documents_text, embeddings=embeddings.tolist())

        print(f"Added the {len(documents)} chunks to vector store")
        print(f"number of chunks in the collection: {self.collection.count()}")


if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = ChunkingEmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.generate_embeddings(chunks)
    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)
    