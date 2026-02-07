# Stunting Prediction & Recommendation API
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY main.py .
COPY grab_monthly.py .
COPY src/ ./src/
 COPY data/ ./data/

# Use in-container paths for RAG (override via env at runtime if needed)
ENV RAG_DOCS_DIR=/app/data/guideline
ENV RAG_PERSIST_DIR=/app/data/chroma_guidelines
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
