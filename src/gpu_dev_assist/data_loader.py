import os
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from gpu_dev_assist.utils.logger import logger

### Read all the pdf's inside the directory
def load_all_documents(pdf_directory):
    """Process all PDF files in a directory"""
    
    logger.info(f"Loading the files into Documents...")
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


if __name__ == "__main__":
    docs = load_all_documents("data/raw")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)