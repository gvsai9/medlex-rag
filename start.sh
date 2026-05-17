#!/bin/bash
# MedLex RAG — Quick Start Script
set -e

echo "============================================"
echo "  MedLex RAG — Quick Start"
echo "============================================"

# Check Python
python3 --version || { echo "Python 3.9+ required"; exit 1; }

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -q -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm -q

# Check .env
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    echo ""
    echo "👉 Edit .env and add your GEMINI_API_KEY (free at https://ai.google.dev)"
    echo "   Then re-run this script."
    echo ""
    exit 1
fi

# Create data dirs
mkdir -p data/index data/raw data/processed

# Start backend
echo ""
echo "Starting FastAPI backend on http://localhost:8000 ..."
echo "API docs at http://localhost:8000/docs"
echo ""
uvicorn backend.api.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "Starting React frontend on http://localhost:3000 ..."
cd frontend
npm install -q
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================"
echo "  ✅ MedLex RAG running!"
echo ""
echo "  Frontend: http://localhost:3000"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "  Next: Ingest some FDA data:"
echo "  python scripts/ingest_fda.py --drugs ibuprofen aspirin --limit 5"
echo "============================================"

# Wait for both
wait $BACKEND_PID $FRONTEND_PID
