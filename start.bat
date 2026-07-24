@echo off
echo ====================================================
echo   SWaT Anomaly Detection Dashboard - Startup Script
echo ====================================================

if not exist "SWATDatasets\attack.csv" (
    echo [!] WARNING: SWATDatasets\attack.csv not found!
    echo     Please download the dataset from Hugging Face:
    echo     https://huggingface.co/datasets/minervar/SWaT-Curated
    echo     and place attack.csv, normal.csv, and merged.csv inside the SWATDatasets\ folder.
    echo     The backend might crash if the files are missing!
    echo ----------------------------------------------------
    timeout /t 3
)

if not exist "venv\" (
    echo [*] Creating Python virtual environment (venv)...
    python -m venv venv
)

echo [*] Activating virtual environment...
call venv\Scripts\activate

echo [*] Installing requirements (this might take a moment if it's the first time)...
pip install -r requirements.txt -q

echo [*] Starting the FastAPI backend server...
echo.
echo ^>^>^> PLEASE OPEN YOUR BROWSER AND DOUBLE-CLICK: frontend\index.html ^<^<^<
echo ^>^>^> (Or just drag frontend\index.html into a Chrome tab)            ^<^<^<
echo.

cd backend
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
