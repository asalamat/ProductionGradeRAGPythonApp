#!/bin/bash
set -e
cd "$(dirname "$0")"

NODE="/Users/alisalamat/.nvm/versions/node/v24.12.0/bin/node"
NPX="/Users/alisalamat/.nvm/versions/node/v24.12.0/bin/npx"
PYTHON="/Users/alisalamat/miniconda3/bin/python"
UVICORN="/Users/alisalamat/miniconda3/bin/uvicorn"

echo "Starting RAG services..."

# Qdrant
docker start qdrant 2>/dev/null || \
  docker run -d --name qdrant --restart always \
    -p 6333:6333 \
    -v "$(pwd)/qdrant_storage":/qdrant/storage \
    qdrant/qdrant
echo "✓ Qdrant"

# Ollama
if ! pgrep -x ollama >/dev/null; then
  nohup ollama serve >/tmp/rag-ollama.log 2>&1 &
  sleep 2
fi
echo "✓ Ollama"

# FastAPI
if ! lsof -ti :8000 >/dev/null 2>&1; then
  nohup "$UVICORN" main:app --port 8000 >/tmp/rag-fastapi.log 2>&1 &
  sleep 3
fi
echo "✓ FastAPI"

# Inngest dev server
if ! lsof -ti :8288 >/dev/null 2>&1; then
  nohup "$NODE" "$NPX" inngest-cli@latest dev --port 8288 -u http://127.0.0.1:8000/api/inngest >/tmp/rag-inngest.log 2>&1 &
  sleep 5
fi
echo "✓ Inngest"

# Streamlit
if ! lsof -ti :8501 >/dev/null 2>&1; then
  nohup "$PYTHON" -m streamlit run streamlit_app.py >/tmp/rag-streamlit.log 2>&1 &
  sleep 3
fi
echo "✓ Streamlit"

echo ""
echo "All services running."
echo "  UI:      http://localhost:8501"
echo "  API:     http://localhost:8000"
echo "  Inngest: http://localhost:8288"
echo "  Qdrant:  http://localhost:6333/dashboard"
