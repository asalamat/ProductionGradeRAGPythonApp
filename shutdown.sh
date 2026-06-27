#!/bin/bash

echo "Shutting down RAG services..."

# Stop LaunchAgents
launchctl unload ~/Library/LaunchAgents/com.rag.fastapi.plist 2>/dev/null && echo "✓ FastAPI stopped"
launchctl unload ~/Library/LaunchAgents/com.rag.streamlit.plist 2>/dev/null && echo "✓ Streamlit stopped"
launchctl unload ~/Library/LaunchAgents/com.rag.inngest.plist 2>/dev/null && echo "✓ Inngest stopped"

# Stop Qdrant container (graceful — flushes writes to disk before stopping)
docker stop qdrant 2>/dev/null && echo "✓ Qdrant stopped"

# Kill any leftover processes on the ports
lsof -ti :8000 | xargs kill -9 2>/dev/null
lsof -ti :8501 | xargs kill -9 2>/dev/null
lsof -ti :8288 | xargs kill -9 2>/dev/null
lsof -ti :6333 | xargs kill -9 2>/dev/null

echo ""
echo "All services stopped. Memory freed."
echo "Run 'bash start.sh' to bring everything back up."
