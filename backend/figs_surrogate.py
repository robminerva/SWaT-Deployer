import os
import json
import pickle
import pandas as pd
from river import stream
from sklearn.cluster import MiniBatchKMeans
try:
    from imodels import FIGSClassifier
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imodels"])
    from imodels import FIGSClassifier

WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"
MODELS_DIR = os.path.join(WORKING_DIR, "models")
DEDUP_RECORDS = os.path.join(WORKING_DIR, "deduplicated_states.parquet")
TOPOLOGY_FILE = os.path.join(WORKING_DIR, "topology.json")
STATUS_FILE = os.path.join(WORKING_DIR, "figs_status.json")

def update_status(status, processed=0, error=None, current_actuator="", progress=0, total=0, start_time=None):
    data = {
        "status": status,
        "processed": processed,
        "error": error,
        "current_actuator": current_actuator,
        "progress": progress,
        "total": total
    }
    if start_time:
        data["start_time"] = start_time
        
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f)

def run_figs(max_rules=100, archetypes=50):
    update_status("running", 0)
    
    if not os.path.exists(DEDUP_RECORDS):
        raise FileNotFoundError("deduplicated_states.parquet not found. Run Discretization pipeline first.")
        
    with open(TOPOLOGY_FILE, "r") as f:
        topology = json.load(f)

    continuous_sensors = [n["id"] for n in topology.get("nodes", []) if n["type"] == "sensor"]
    discrete_actuators = [n["id"] for n in topology.get("nodes", []) if n["type"] == "actuator"]
    
    print("Loading deduplicated states into memory...")
    df = pd.read_parquet(DEDUP_RECORDS)
    total_states = len(df)
    
    # Pre-process features into Numpy to avoid memory leaks
    X_full = df[continuous_sensors].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_numpy = X_full.values
    col_indices = {c: i for i, c in enumerate(continuous_sensors)}
    
    # We no longer train FIGS for sensor_bins because XGBoost handles continuous regressors
    all_targets = discrete_actuators
    total_targets = len(all_targets)
    
    print(f"Training FIGS on {total_states} unique states with Recompaction...")
    
    import time
    start_time = time.time()
    
    for i, target_col in enumerate(all_targets):
        print(f"Training FIGS for {target_col}...")
        
        model_path = os.path.join(MODELS_DIR, f"figs_model_{target_col}.pkl")
        if os.path.exists(model_path):
            pass # We want to overwrite the bloated models, do not skip!
                
        update_status("running", total_states, current_actuator=target_col, progress=i+1, total=total_targets, start_time=start_time)
        import numpy as np
        
        y = df[target_col].fillna("0").astype(str).values
        
        # Use all continuous sensors for actuator prediction
        X = X_numpy
        feature_names = continuous_sensors
            
        # If there's only one state, we skip or build a dummy
        if len(set(y)) <= 1:
            continue
            

        
        # ---------------------------------------------------------
        # Train highly granular FIGS on the full dataset (No Compression)
        # ---------------------------------------------------------
        # The user requested to train on 100% of the data without KMeans or random sampling,
        # provided the RAM can handle it with the reduced feature set (up to 16 features).
        model = FIGSClassifier(max_rules=16) # Constrained to prevent 4GB overfitting bloat
        y_np = np.array(y, dtype=str)
        model.fit(X.astype(float), y_np)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"figs_model_{target_col}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        # Extract rule string and manually replace X5 with real feature names
        rules_text = str(model)
        import re
        def replace_feature(match):
            idx = int(match.group(1))
            if idx < len(feature_names):
                return feature_names[idx]
            return match.group(0)
            
        rules_text = re.sub(r'X(\d+)', replace_feature, rules_text)
            
        text_path = os.path.join(MODELS_DIR, f"figs_rules_{target_col}.txt")
        with open(text_path, "w") as f:
            f.write(rules_text)
            
    update_status("completed", total_states, start_time=start_time)
    print("FIGS Extraction Complete.")

if __name__ == "__main__":
    try:
        import sys
        max_r = 100
        arch = 50
        if len(sys.argv) > 1:
            try: max_r = int(sys.argv[1])
            except ValueError: pass
        if len(sys.argv) > 2:
            try: arch = int(sys.argv[2])
            except ValueError: pass
        run_figs(max_rules=max_r, archetypes=arch)
    except Exception as e:
        update_status("error", error=str(e))
        raise e
