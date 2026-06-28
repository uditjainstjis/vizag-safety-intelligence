#!/bin/bash
set -e
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  VIZAG SAFETY INTELLIGENCE — SETUP      ║"
echo "║  ET AI Hackathon 2026 · PS#1             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

cd ~/Desktop/vizag_safety

echo "[1/4] Installing dependencies..."
pip install -r requirements.txt -q

echo "[2/4] Generating Vizag incident simulation data..."
python3 generate_data.py

echo "[3/4] Training anomaly detection model..."
python3 models/train_anomaly.py

echo "[4/4] Building OISD regulatory knowledge base..."
python3 rag/ingest.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Run the dashboard:"
echo "  python3 app.py"
echo ""
echo "Then open: http://localhost:8001"
echo ""
