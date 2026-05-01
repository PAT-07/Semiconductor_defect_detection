# Semiconductor Wafer Defect Detection: End-to-End AI & Antigravity RAG Pipeline

## Project Overview
This project is an enterprise-grade, end-to-end Applied AI pipeline designed for the semiconductor manufacturing industry. It integrates Computer Vision, Predictive Analytics, and a high-precision Retrieval-Augmented Generation (RAG) assistant to automate quality control, optimize material forecasting, and provide real-time scientific troubleshooting.

**Key Achievements:**
- **YOLOv8 Detection:** `0.962 mAP@50` (96.2% accuracy on unseen validation data).
- **Material Forecasting:** `R² = 0.9637` (Highly accurate waste prediction).
- **Antigravity RAG:** Hybrid Search + Cross-Encoder re-ranking for zero-hallucination engineering support.

## Business Value
In semiconductor fabrication, microscopic defects cost millions in scrapped materials and downtime. This project eliminates manual coordinate analysis, providing real-time visual detection and forecasting while bridging the gap between factory floor failures and scientific troubleshooting solutions.

---

## The Technical Pipeline

### Phase 1: Data Engineering & Computer Vision
- **Dataset Synthesis:** Parsed 25,000+ raw mathematical 2D arrays into high-contrast computer vision images.
- **Model Training:** Deployed a custom **YOLOv8 Nano** model to detect 8 specific manufacturing defect classes (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full).

### Phase 2: Predictive Middleware
- **Robotic Scanner Simulation:** Operates on a hybrid dataset of **823,953 wafers**, logging scan results into a centralized SQLite database.
- **Material Waste Predictor:** A Random Forest Regressor that forecasts material scrap percentages to optimize supply chain procurement.

### Phase 3: The Antigravity RAG Assistant (New)
The system now includes a state-of-the-art **Antigravity** vector search pipeline to assist fab engineers with root-cause analysis.
- **Dynamic Ingestion:** Automated PDF parsing of scientific papers with 500-char sliding-window chunking.
- **Hybrid Retrieval:** Dual-track search combining **Semantic Vector Search (ANN)** with **Keyword-based BM25** matching.
- **Reciprocal Rank Fusion (RRF):** Mathematical fusion of search results to ensure maximum retrieval recall.
- **Semantic Re-ranking:** A **Cross-Encoder** (ms-marco-MiniLM-L-6-v2) precision layer that analyzes the top candidates to eliminate hallucinations and find the absolute best technical context for the LLM.

---

## Performance Metrics

| Metric | Score | Note |
| :--- | :--- | :--- |
| **mAP50 (All Classes)** | **96.2%** | Overall YOLOv8 detection accuracy. |
| **Edge-Ring Precision** | **99.4%** | Near-flawless detection of Edge-Ring anomalies. |
| **Recall** | **93.1%** | Successfully located 93.1% of all physical defects. |
| **Predictive R² Score** | **0.9637** | Excellent correlation on material waste targets. |

---

## Tech Stack
- **Core:** Python 3.12
- **Computer Vision:** Ultralytics (YOLOv8), OpenCV
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Vector Search:** ChromaDB, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Ranking:** Rank-BM25, Cross-Encoders (`ms-marco-MiniLM-L-6-v2`)
- **LLM:** Google Gemini 2.5 Flash
- **Infrastructure:** FastAPI, Docker, Docker Compose

---

## Deployment (Docker)

This application is fully containerized with pre-cached AI models for instant startup.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Udayan2001/Semiconductor_defect_detection.git
    cd Semiconductor_defect_detection
    ```
2.  **Add API Key:**
    Create a `.env` file in the `backend/` directory:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```
3.  **Start the Application:**
    ```bash
    docker compose up --build
    ```
4.  **Access the Dashboard:**
    Open your browser and navigate to `http://localhost:5173`.

---
*Designed and engineered by Udayan Shashank Shukla.*