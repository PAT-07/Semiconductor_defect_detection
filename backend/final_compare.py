import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Setup
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="semiconductor_knowledge")

# Load models
print("Loading models...")
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
    v_ids = vanilla_search(query, k=k)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:k]
    b_ids = [ids[i] for i in top_bm25_idx]
    rrf_scores = {}
    for r, did in enumerate(v_ids): rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + r)
    for r, did in enumerate(b_ids): rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + r)
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [x[0] for x in sorted_rrf[:k]]
    candidate_docs = [documents[ids.index(did)] for did in top_candidates]
    scores = cross_encoder.predict([[query, doc] for doc in candidate_docs])
    final_idx = np.argsort(scores)[::-1]
    return [top_candidates[i] for i in final_idx]

# Hardcoded test set from previous run
test_cases = [
    {"q": "What are the advantages of using VLSI grade argon over VLSI grade nitrogen for gate oxidation processing?", "id": ids[50]},
    {"q": "What is the maintenance schedule for the process chambers, and which sensors require attention?", "id": ids[150]},
    {"q": "How does the Civa image detect electrically floating conductors?", "id": ids[250]},
    {"q": "What are the key areas covered in the review of gate stack materials?", "id": ids[350]},
    {"q": "What are some common nitrogen-related reactions that occur at an interface?", "id": ids[450]}
]

def run_final_compare():
    v_r10, v_p1 = 0, 0
    w_r10, w_p1 = 0, 0
    
    for case in test_cases:
        target = case["id"]
        v_res = vanilla_search(case["q"], k=10)
        if target in v_res: v_r10 += 1
        if v_res and target == v_res[0]: v_p1 += 1
        
        w_res = wdd_search(case["q"], k=10)
        if target in w_res: w_r10 += 1
        if w_res and target == w_res[0]: w_p1 += 1
        
    n = len(test_cases)
    print("\n" + "="*40)
    print("SLIDE-READY BENCHMARK RESULTS")
    print("="*40)
    print(f"METRIC         | VANILLA RAG | WDD ARCHITECTURE")
    print(f"Recall @ 10    | {v_r10/n*100:.1f}%       | {w_r10/n*100:.1f}%")
    print(f"Precision @ 1  | {v_p1/n*100:.1f}%       | {w_p1/n*100:.1f}%")
    print("-" * 40)
    print(f"System F1 Score| {(v_r10+v_p1)/(2*n):.2f}        | {(w_r10+w_p1)/(2*n):.2f}")
    print("="*40)

if __name__ == "__main__":
    run_final_compare()
