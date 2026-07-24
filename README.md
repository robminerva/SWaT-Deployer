# SWaT Anomaly Detection Dashboard

Welcome to the Secure Water Treatment (SWaT) Anomaly Detection Dashboard! 
This project is an advanced, multi-model ensemble detection system designed to monitor and identify cyber-physical attacks on an industrial water treatment plant in real-time.

It features a fast, lightweight streaming architecture with a modern web dashboard and a highly accurate ensemble of 7 distinct behavioral models ranging from Machine Learning to Physical Invariants.

> **Note:** This project was built autonomously using [Google's Antigravity AI SDK](https://github.com/google/antigravity).

## 🚀 Quick Start (1-Click Install)

Running the dashboard locally on your machine is incredibly simple. We have provided startup scripts that automatically set up your Python environment, install dependencies, and launch the server.

### Prerequisites
1. **Python 3.9+** installed on your machine.
2. Download the required curated dataset files from Hugging Face:
   - Navigate to [minervar/SWaT-Curated](https://huggingface.co/datasets/minervar/SWaT-Curated)
   - Download `attack.csv`, `normal.csv`, and `merged.csv`.
   - Place them inside the `SWATDatasets/` folder in the root of this project. (Create the folder if it does not exist).

### Running on Linux / macOS
Open your terminal in the project directory and run:
```bash
bash start.sh
```

### Running on Windows
Double-click the `start.bat` file, or run it in Command Prompt/PowerShell:
```cmd
start.bat
```

Once the backend is running, the script will prompt you to open the `frontend/index.html` file in your web browser. The dashboard will automatically connect to the backend via WebSockets and begin streaming the attack data in real-time!

## 🧠 Architecture Overview

The system is separated into a lightweight streaming backend and a modern web dashboard.

* **Backend (`backend/main.py`)**: A high-performance Python FastAPI server. It streams the industrial control data row-by-row over a WebSocket to simulate real-time operations. As data flows, it evaluates each state against 7 distinct behavioral models simultaneously.
* **Frontend (`frontend/index.html`)**: A pure HTML/CSS/JS dashboard that receives the WebSocket stream. It renders a live scoreboard, an alert feed, and dynamic badges showing the operational status of every sensor and actuator in the plant.

## 🔬 The Ensemble Models
For a deep dive into the methodology, data curation, and the 7 behavioral models that make up the ensemble (Agent FSM, Process Mining, FIGS, ARF, XGBoost Regressors, SAB Bounds, Physical Mass Balance), please see [Methodology.md](Methodology.md).
