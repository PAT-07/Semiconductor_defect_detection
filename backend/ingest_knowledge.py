import os
import re
import chromadb
from chromadb.config import Settings
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np

# Configuration
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "chroma_db", "Knowledge_base")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def chunk_text(text, chunk_size, overlap):
    """Dynamic chunking with sliding window."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def extract_pdf_data():
    """Extract text from PDFs in the knowledge base directory."""
    documents = []
    metadatas = []
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Directory not found: {KNOWLEDGE_BASE_DIR}")
        return documents, metadatas
        
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    full_text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + " "
                    
                    cleaned_text = clean_text(full_text)
                    chunks = chunk_text(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)
                    
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50: # Ignore very small chunks
                            documents.append(chunk)
                            metadatas.append({"filename": filename, "chunk_index": i})
                            
                print(f"Processed {filename}: extracted {len(chunks)} chunks.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    return documents, metadatas

def ingest_data():
    print("Extracting and chunking PDF data...")
    documents, metadatas = extract_pdf_data()
    
    if not documents:
        print("No documents extracted. Exiting.")
        return

    # Create IDs
    ids = [f"{meta['filename']}_chunk_{meta['chunk_index']}" for meta in metadatas]

    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating and L2 normalizing embeddings...")
    embeddings_raw = model.encode(documents)
    embeddings_l2 = normalize(embeddings_raw, norm='l2', axis=1).tolist()

    print("Initializing ChromaDB Persistent Client...")
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="semiconductor_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Clear existing data if any (for idempotency)
    existing_ids = collection.get()['ids']
    if existing_ids:
        print(f"Clearing {len(existing_ids)} old documents from collection...")
        collection.delete(ids=existing_ids)
    
    print(f"Adding {len(documents)} documents with explicit embeddings to the knowledge base...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_l2,
        ids=ids
    )
    
    print("Ingestion complete. ChromaDB is ready.")

if __name__ == "__main__":
    ingest_data()
