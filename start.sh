#!/bin/bash
# MedRAG — One-click startup script
# Run this from the medrag/ folder: bash start.sh

set -e

echo ""
echo "🏥 MedRAG — Clinical Intelligence Suite"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from https://python.org"
    exit 1
fi

echo "✅ Python: $(python3 --version)"

# Check .env
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Creating one..."
    cp .env.example .env 2>/dev/null || true
fi

# Check API key
if grep -q "your-key-here" .env; then
    echo ""
    echo "⚠️  Edit .env and add your Anthropic API key:"
    echo "   ANTHROPIC_API_KEY=sk-ant-your-key-here"
    echo ""
    echo "   Then run this script again."
    echo ""
    echo "   Or just leave it blank — you can enter the key in the UI."
    echo ""
fi

# Install dependencies
echo "📦 Installing dependencies (first run may take a few minutes)..."
pip install -r requirements.txt -q

echo ""
echo "🚀 Starting MedRAG backend..."
echo "   → Open http://localhost:8000 in your browser"
echo "   → Press Ctrl+C to stop"
echo ""

# Create data dirs
mkdir -p data/uploads data/chroma

# Start server
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
