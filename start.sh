#!/bin/bash

echo "===================================================="
echo "  SWaT Anomaly Detection Dashboard - Startup Script"
echo "===================================================="

# Check for datasets
if [ ! -f "SWATDatasets/attack.csv" ]; then
    echo "[!] WARNING: SWATDatasets/attack.csv not found!"
    echo "    Please download the dataset from Hugging Face:"
    echo "    https://huggingface.co/datasets/minervar/SWaT-Curated"
    echo "    and place attack.csv, normal.csv, and merged.csv inside the SWATDatasets/ folder."
    echo "    The backend might crash if the files are missing!"
    echo "----------------------------------------------------"
    sleep 3
fi

# Set up virtual environment
if [ ! -d "venv" ]; then
    echo "[*] Creating Python virtual environment (venv)..."
    python3 -m venv venv
fi

echo "[*] Activating virtual environment..."
source venv/bin/activate

echo "[*] Installing requirements (this might take a moment if it's the first time)..."
pip install -r requirements.txt -q

echo "[*] Starting the FastAPI backend server..."
echo ""
echo ">>> PLEASE OPEN YOUR BROWSER AND DOUBLE-CLICK: frontend/index.html <<<"
echo ">>> (Or just drag frontend/index.html into a Chrome tab)            <<<"
echo ""

# Launch the backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
