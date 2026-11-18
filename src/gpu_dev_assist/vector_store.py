import os
import chromadb
from typing import List, Any
import numpy as np
from gpu_dev_assist.utils.logger import logger

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

    def reset(self):
        """Delete and recreate the collection, for a clean rebuild."""
        self.client.delete_collection(self.collection_name)
        self._initialize_vector_store()

    def add_documents(self, documents:List[Any], embeddings: np.ndarray):
        logger.info(f"Adding documents to the vector store...")
        ids = []
        metadatas = []
        documents_text = []

        for i, document in enumerate(documents):
            metadata = dict(document.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(document.page_content)
            metadatas.append(metadata)
            ids.append(f"doc_{metadata['source_file']}-p{metadata['page']}-c{i}")

            documents_text.append(document.page_content)

        self.collection.add(ids=ids, metadatas=metadatas, documents=documents_text, embeddings=embeddings.tolist())

        logger.info(f"Added the {len(documents)} chunks to vector store")
        logger.info(f"Number of chunks in the collection: {self.collection.count()}")