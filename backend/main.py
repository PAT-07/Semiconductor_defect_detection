import sys
import os
import pickle
import sqlite3
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import numpy as np
# Load environment variables from .env if it exists
load_dotenv()

# Add parent dir to path so we can import from middleware
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from middleware.material_predictor import predict_material_needs

app = FastAPI(title="Wafer Defect API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'middleware', 'wafer_control.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'middleware', 'material_model.pkl')

DEFECT_COLORS = {
    'Center': '#ef4444', 'Donut': '#f59e0b', 'Edge-Loc': '#10b981',
    'Edge-Ring': '#3b82f6', 'Loc': '#8b5cf6', 'Random': '#ec4899',
    'Scratch': '#06b6d4', 'Near-full': '#f97316', 'None': '#6b7280',
    'Undetected': '#374151',
}

# Globally load data so we don't block requests
print("Loading DB...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM wafer_logs", conn)
conn.close()

# Setup Vector DB and LLM
print("Connecting to ChromaDB...")
try:
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), 'chroma_db'))
    collection = chroma_client.get_collection(name="semiconductor_knowledge")
except Exception as e:
    print(f"Warning: Could not connect to ChromaDB collection. Ensure you run ingest_knowledge.py first. Error: {e}")
    collection = None

print("Initializing Gemini API...")
gemini_client = None
if os.getenv("GEMINI_API_KEY"):
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
else:
    print("Warning: GEMINI_API_KEY not found in environment.")

print("Loading SentenceTransformer & CrossEncoder...")
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    print(f"Warning: Could not load local ML models: {e}")
    embedder = None
    cross_encoder = None

print("Initializing BM25 index...")
bm25 = None
bm25_docs = []
bm25_ids = []
if collection:
    all_docs = collection.get()
    if all_docs and all_docs.get('documents'):
        bm25_docs = all_docs['documents']
        bm25_ids = all_docs['ids']
        tokenized_corpus = [doc.lower().split() for doc in bm25_docs]
        bm25 = BM25Okapi(tokenized_corpus)

df['scan_time'] = pd.to_datetime(df['scan_time'])
df['scan_date'] = df['scan_time'].dt.date

print("Loading ML model...")
model_pkg = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model_pkg = pickle.load(f)

@app.get("/api/kpi")
def get_kpis():
    total_scans = len(df)
    fail_df = df[df['status'] == 'FAIL']
    fail_count = len(fail_df)
    pass_count = len(df[df['status'] == 'PASS'])
    pass_rate = round((pass_count / total_scans) * 100, 1) if total_scans else 0
    scrap_count = len(df[df['action'] == 'ROUTE_TO_SCRAP'])
    avg_waste = round(fail_df['material_wasted_pct'].mean(), 2) if fail_count else 0
    avg_confidence = round(fail_df['confidence'].mean(), 2) if fail_count else 0
    
    return {
        "total_scans": total_scans,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "fail_count": fail_count,
        "fail_rate": round(100 - pass_rate, 1),
        "scrap_count": scrap_count,
        "avg_waste": avg_waste,
        "avg_confidence": avg_confidence
    }

@app.get("/api/charts/defects")
def get_defects():
    fail_df = df[df['status'] == 'FAIL']
    defect_counts = fail_df['defect_type'].value_counts().reset_index()
    defect_counts.columns = ['defect_type', 'count']
    
    gt_counts = fail_df['ground_truth'].value_counts().reset_index()
    gt_counts.columns = ['ground_truth', 'count']
    
    return {
        "predictions": defect_counts.to_dict(orient="records"),
        "ground_truth": gt_counts.head(15).to_dict(orient="records")
    }

@app.get("/api/charts/waste")
def get_waste():
    fail_df = df[df['status'] == 'FAIL']
    
    waste_by_type = fail_df.groupby('defect_type').agg(
        total_waste=('material_wasted_pct', lambda x: x.sum() / 100.0)
    ).reset_index().sort_values('total_waste', ascending=True)
    
    action_counts = df['action'].value_counts().reset_index()
    action_counts.columns = ['action', 'count']
    
    return {
        "waste_by_type": waste_by_type.to_dict(orient="records"),
        "actions": action_counts.to_dict(orient="records")
    }

@app.get("/api/charts/trends")
def get_trends():
    daily = df.groupby('scan_date').agg(
        scans=('id', 'count'),
        fails=('status', lambda x: (x == 'FAIL').sum()),
        waste=('material_wasted_pct', lambda x: x.sum() / 100.0)
    ).reset_index()
    daily['fail_rate'] = round((daily['fails'] / daily['scans']) * 100, 1)
    
    return {
        "dates": daily['scan_date'].astype(str).tolist(),
        "fail_rate": daily['fail_rate'].tolist(),
        "waste": daily['waste'].tolist()
    }

@app.get("/api/model/status")
def model_status():
    if not model_pkg:
        return {"loaded": False}
    
    m = model_pkg['metrics']
    imp = model_pkg['metrics']['importances']
    imp_df = pd.DataFrame({'feature': list(imp.keys()), 'importance': list(imp.values())})
    imp_df = imp_df.sort_values('importance', ascending=True).tail(10)
    
    return {
        "loaded": True,
        "metrics": {"r2": round(m['r2'], 4), "mae": round(m['mae'], 2)},
        "importance": imp_df.to_dict(orient="records")
    }

class PredictionRequest(BaseModel):
    scans: int
    fail_rate: float

@app.post("/api/predict")
def predict_waste(req: PredictionRequest):
    if not model_pkg:
        return {"error": "No model loaded"}
        
    fail_df = df[df['status'] == 'FAIL']
    dist = fail_df['defect_type'].value_counts(normalize=True).to_dict()
    
    pred = predict_material_needs(model_pkg['model'], model_pkg['feature_cols'], req.scans, req.fail_rate / 100.0, dist)
    pred['fail_rate'] = req.fail_rate
    return pred


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/api/chat")
def chat_with_bot(req: ChatRequest):
    if not gemini_client:
        return {"error": "Gemini API key not configured"}
        
    user_message = req.messages[-1].content if req.messages else ""
    
    # 1. Hybrid Search (Vector + BM25) & Re-ranking
    context_docs = ""
    if collection and user_message and embedder and cross_encoder and bm25:
        try:
            k = 10
            # 1A. Vector Search
            user_emb = embedder.encode([user_message])
            user_emb_l2 = normalize(user_emb, norm='l2', axis=1).tolist()
            vector_results = collection.query(query_embeddings=user_emb_l2, n_results=k)
            vector_top_ids = vector_results['ids'][0] if vector_results['ids'] else []
            
            # 1B. Keyword Search (BM25)
            tokenized_query = user_message.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:k]
            bm25_top_ids = [bm25_ids[i] for i in top_bm25_idx]
            
            # 1C. Reciprocal Rank Fusion
            rrf_scores = {}
            for rank, doc_id in enumerate(vector_top_ids):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
            for rank, doc_id in enumerate(bm25_top_ids):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
                
            sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            top_candidates_ids = [x[0] for x in sorted_rrf[:k]]
            
            # 1D. Cross-Encoder Re-ranking
            candidate_docs = []
            for doc_id in top_candidates_ids:
                if doc_id in bm25_ids:
                    idx = bm25_ids.index(doc_id)
                    candidate_docs.append(bm25_docs[idx])
                    
            if candidate_docs:
                cross_inp = [[user_message, doc] for doc in candidate_docs]
                cross_scores = cross_encoder.predict(cross_inp)
                top_2_idx = np.argsort(cross_scores)[::-1][:2]
                top_2_docs = [candidate_docs[i] for i in top_2_idx]
                context_docs = "\n\n---\n\n".join(top_2_docs)
                print(f"Hybrid RAG Success: Found {len(top_2_docs)} refined contexts.")
        except Exception as e:
            print(f"RAG Pipeline Error: {e}")
            
    # 2. Get Live Dashboard Context
    total_scans = len(df)
    fail_df = df[df['status'] == 'FAIL']
    fail_count = len(fail_df)
    pass_rate = round(((total_scans - fail_count) / total_scans) * 100, 1) if total_scans else 0
    top_defects = fail_df['defect_type'].value_counts().head(3).to_dict()
    
    live_kpis = f"""
    Current Dashboard State:
    - Total Wafers Scanned: {total_scans}
    - Current Pass Rate: {pass_rate}%
    - Total Defective Wafers: {fail_count}
    - Top Defect Types Right Now: {top_defects}
    """
    
    # 3. Construct System Prompt
    system_instruction = f"""
    You are the 'Gorilla Semiconductors Engineering Assistant', an expert semiconductor manufacturing assistant. 
    You help engineers understand dashboard data and troubleshoot wafer defects.
    Maintain a strictly professional, analytical, and authoritative engineering tone.
    
    Here is the LIVE DATA from the dashboard:
    {live_kpis}
    
    Here is retrieved technical context from our engineering database based on the user's query:
    {context_docs if context_docs else "No specific engineering docs retrieved."}
    
    Use the live data to answer questions about 'current status' or 'dashboard'. 
    Use the engineering docs to answer questions about 'why' a defect happens.
    """
    
    try:
        # Convert messages to format expected by google-genai
        contents = []
        for msg in req.messages:
            role = "user" if msg.role == "user" else "model"
            contents.append(
                genai.types.Content(role=role, parts=[genai.types.Part.from_text(text=msg.content)])
            )
            
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )
        return {"response": response.text}
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {"error": str(e)}

