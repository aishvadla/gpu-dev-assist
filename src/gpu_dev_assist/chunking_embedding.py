from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from gpu_dev_assist.utils.logger import logger
from sentence_transformers import SentenceTransformer

from gpu_dev_assist.data_loader import load_all_documents

class ChunkingEmbeddingPipeline:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, model_name: str = "all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)

    def chunk_documents(self, docs: List[Any]) -> List[Any]:
        logger.info(f"Initiated document chunking...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"split {len(docs)} documents into {len(chunks)} chunks")

        # clean chunks
        clean_chunks = [c for c in chunks if not self.is_noise(c.page_content)]
        logger.info(f"kept {len(clean_chunks)} of {len(chunks)} chunks")
        logger.info(f"Finished document chunking")
        return clean_chunks

    def is_noise(self, text: str) -> bool:
        t = text.strip()
        if len(t) < 100:                       # title pages, stray headers
            return True
        if len(re.findall(r"\.{5,}", t)) >= 3:  # table-of-contents lines ("Overview.......5")
            return True
        return False

    def generate_embeddings(self, chunks):
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        if len(chunks) > 1:
            texts = [chunk.page_content for chunk in chunks]
        else:
            texts = chunks
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def get_embedding_model(self):
        return self.model
        

if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = ChunkingEmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.generate_embeddings(chunks)
    print("Example embedding:", embeddings[0] if len(embeddings) > 0 else None)

