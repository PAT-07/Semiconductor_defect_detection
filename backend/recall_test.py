import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Setup
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="semiconductor_knowledge")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load models
print("Loading models for test...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Get all docs for BM25
all_data = collection.get()
documents = all_data['documents']
ids = all_data['ids']
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

def hybrid_search(query, k=10):
    # Vector Search
    user_emb = embedder.encode([query])
    user_emb_l2 = normalize(user_emb, norm='l2', axis=1).tolist()
    vector_results = collection.query(query_embeddings=user_emb_l2, n_results=k)
    vector_top_ids = vector_results['ids'][0] if vector_results['ids'] else []
    
    # BM25 Search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:k]
    bm25_top_ids = [ids[i] for i in top_bm25_idx]
    
    # RRF
    rrf_scores = {}
    for rank, doc_id in enumerate(vector_top_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
    for rank, doc_id in enumerate(bm25_top_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
        
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates_ids = [x[0] for x in sorted_rrf[:k]]
    
    # Cross-Encoder
    candidate_docs = [documents[ids.index(did)] for did in top_candidates_ids]
    cross_inp = [[query, doc] for doc in candidate_docs]
    cross_scores = cross_encoder.predict(cross_inp)
    top_2_idx = np.argsort(cross_scores)[::-1][:2]
    top_2_ids = [top_candidates_ids[i] for i in top_2_idx]
    
    return top_2_ids

def run_recall_test():
    # Pick 5 random indices
    test_indices = [50, 150, 250, 350, 450]
    hits = 0
    total = len(test_indices)
    
    print(f"\nStarting Synthetic Recall Test on {total} samples...")
    print("-" * 50)
    
    for idx in test_indices:
        target_id = ids[idx]
        target_text = documents[idx]
        
        # Generate question using Gemini
        prompt = f"Given this technical text, generate a short engineering question that is answered by this text. Only output the question text: '{target_text}'"
        response = gemini_client.models.generate_content(model='gemini-2.5-flash-lite', contents=prompt)
        question = response.text.strip()
        
        print(f"Fact from: {target_id[:30]}...")
        print(f"Generated Question: {question}")
        
        # Run search
        retrieved_ids = hybrid_search(question)
        
        if target_id in retrieved_ids:
            print("✅ HIT")
            hits += 1
        else:
            print("❌ MISS")
            
        print("-" * 30)
        
    recall_score = (hits / total) * 100
    print(f"\nFINAL RECALL @ 2: {recall_score}%")
    return recall_score

if __name__ == "__main__":
    run_recall_test()
