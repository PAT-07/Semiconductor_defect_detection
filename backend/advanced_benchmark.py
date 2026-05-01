import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import chromadb
from google import genai
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()

# Setup
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="semiconductor_knowledge")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load models
print("Loading models for benchmark...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Get all docs
all_data = collection.get()
documents = all_data['documents']
ids = all_data['ids']
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

def vanilla_search(query, k=10):
    user_emb = embedder.encode([query])
    user_emb_l2 = normalize(user_emb, norm='l2', axis=1).tolist()
    results = collection.query(query_embeddings=user_emb_l2, n_results=k)
    return results['ids'][0] if results['ids'] else []

def wdd_search(query, k=10):
    # Vector Search
    user_emb = embedder.encode([query])
    user_emb_l2 = normalize(user_emb, norm='l2', axis=1).tolist()
    v_ids = collection.query(query_embeddings=user_emb_l2, n_results=k)['ids'][0]
    # BM25 Search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:k]
    b_ids = [ids[i] for i in top_bm25_idx]
    # RRF
    rrf_scores = {}
    for r, did in enumerate(v_ids): rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + r)
    for r, did in enumerate(b_ids): rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + r)
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [x[0] for x in sorted_rrf[:k]]
    # Cross-Encoder
    candidate_docs = [documents[ids.index(did)] for did in top_candidates]
    scores = cross_encoder.predict([[query, doc] for doc in candidate_docs])
    final_idx = np.argsort(scores)[::-1]
    return [top_candidates[i] for i in final_idx]

def run_benchmark():
    indices = np.linspace(0, len(ids)-1, 10, dtype=int)
    results = {"vanilla": {"recall10": 0, "precision1": 0}, "wdd": {"recall10": 0, "precision1": 0}}
    
    print(f"Running Robust Benchmark on {len(indices)} samples...")
    
    for idx in indices:
        target_id = ids[idx]
        target_text = documents[idx]
        
        # Generate Question with Retry
        q = ""
        while not q:
            try:
                prompt = f"Generate a short technical question answered by: '{target_text}'"
                response = gemini_client.models.generate_content(model='gemini-2.5-flash-lite', contents=prompt)
                q = response.text.strip()
            except Exception as e:
                print("Rate limit reached, sleeping 10s...")
                time.sleep(10)
        
        # Test Vanilla
        v_res = vanilla_search(q, k=10)
        if target_id in v_res: results["vanilla"]["recall10"] += 1
        if v_res and target_id == v_res[0]: results["vanilla"]["precision1"] += 1
        
        # Test WDD
        w_res = wdd_search(q, k=10)
        if target_id in w_res: results["wdd"]["recall10"] += 1
        if w_res and target_id == w_res[0]: results["wdd"]["precision1"] += 1
        
        print(f"Sample {idx} complete.")
        time.sleep(2) # Prevent rate limit
        
    n = len(indices)
    print("\n" + "="*40)
    print("FINAL BENCHMARK RESULTS")
    print("="*40)
    print(f"STAGE          | VANILLA RAG | WDD ARCHITECTURE")
    print(f"Retrieval R@10 | {results['vanilla']['recall10']/n*100:.1f}%       | {results['wdd']['recall10']/n*100:.1f}%")
    print(f"Re-Ranking P@1 | {results['vanilla']['precision1']/n*100:.1f}%       | {results['wdd']['precision1']/n*100:.1f}%")
    
    v_f1 = (results['vanilla']['recall10'] + results['vanilla']['precision1']) / (2 * n)
    w_f1 = (results['wdd']['recall10'] + results['wdd']['precision1']) / (2 * n)
    print(f"Overall F1     | {v_f1:.2f}        | {w_f1:.2f}")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
