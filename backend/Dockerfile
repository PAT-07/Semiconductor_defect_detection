FROM python:3.12-slim

WORKDIR /app

# Install essential system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download ML models so the container starts instantly
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy backend and middleware code
COPY backend/ ./backend/
COPY middleware/ ./middleware/

EXPOSE 8000

# Run ingestion first to ensure vector DB is seeded, then start the server
CMD ["sh", "-c", "python backend/ingest_knowledge.py && uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
