#!/bin/bash
# Start the FastAPI backend server

echo "Starting Document Q&A Application..."
echo ""
echo "🚀 Starting FastAPI Backend on http://localhost:8000"
echo "🌐 Frontend available at http://localhost:8000/index.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
