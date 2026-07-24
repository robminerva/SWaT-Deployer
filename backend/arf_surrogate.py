import os
import json
import random
import pickle
import pandas as pd
from river import stream
from river import forest
from river import metrics
from river.drift import binary
from sklearn.tree import DecisionTreeClassifier, export_text

WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"
DATA_DIR = os.path.join(WORKING_DIR, "SWATDatasets")
TOPOLOGY_FILE = os.path.join(WORKING_DIR, "topology.json")
NORMAL_CSV = os.path.join(DATA_DIR, "normal.csv")
MODELS_DIR = os.path.join(WORKING_DIR, "models")
STATUS_FILE = os.path.join(WORKING_DIR, "status.json")
EVENT_RECORDS = os.path.join(WORKING_DIR, "event_records.parquet")

os.makedirs(MODELS_DIR, exist_ok=True)

def update_status(status, processed=0, acc=0.0, kappa=0.0, error=None):
    data = {
        "status": status,
        "processed": processed,
        "acc": round(acc, 4),
        "kappa": round(kappa, 4),
        "error": error
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f)

def _train_logic(n_trees=16, split_criterion='hellinger', grace_period=30, split_confidence=1e-3):
    update_status("running", 0, 0, 0)
    if not os.path.exists(TOPOLOGY_FILE):
        raise FileNotFoundError("topology.json not found.")

    with open(TOPOLOGY_FILE, "r") as f:
        topology = json.load(f)

    continuous_sensors = [n["id"] for n in topology.get("nodes", []) if n["type"] == "sensor"]
    discrete_actuators = [n["id"] for n in topology.get("nodes", []) if n["type"] == "actuator"]

    sensor_bins = [f"{s}_BIN" for s in continuous_sensors]
    all_targets = discrete_actuators + sensor_bins

    print("Sensors:", continuous_sensors)
    print("Actuators:", discrete_actuators)
    print(f"Training ARFs to predict all {len(all_targets)} targets (actuators + sensor bins)...")

    # Initialize models and metrics per target
    models = {}
    for target in all_targets:
        models[target] = {
            "arf": forest.ARFClassifier(
                n_models=n_trees, 
                seed=42,
                grace_period=grace_period,
                delta=split_confidence,
                tau=0.05,
                split_criterion=split_criterion,
                leaf_prediction="nba",
                warning_detector=None,
                drift_detector=binary.DDM()
            ),
            "acc": metrics.Accuracy(),
            "kappa": metrics.CohenKappa(),
            "last_y": "0",
            "sample_records": []
        }

    print("Starting streaming ingestion...")
    
    chunksize = 100
    subsample_rate = 10
    rows_processed = 0
    surrogate_sample_size = 10000 
    sample_prob = 0.038
    
    kmeans_path = os.path.join(MODELS_DIR, "kmeans_sensors.pkl")
    if os.path.exists(kmeans_path):
        with open(kmeans_path, "rb") as f:
            kmeans_models = pickle.load(f)
    else:
        kmeans_models = {}
    
    sensor_bounds = {s: {"min": float('inf'), "max": float('-inf')} for s in continuous_sensors}

    for chunk in pd.read_csv(NORMAL_CSV, chunksize=chunksize * subsample_rate):
        chunk = chunk.iloc[::subsample_rate]
        chunk.columns = [col.strip() for col in chunk.columns]

        # Pre-process the features
        X_df = chunk[continuous_sensors].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Update bounds dynamically
        for sensor in continuous_sensors:
            s_min = float(X_df[sensor].min())
            s_max = float(X_df[sensor].max())
            if s_min < sensor_bounds[sensor]["min"]: sensor_bounds[sensor]["min"] = s_min
            if s_max > sensor_bounds[sensor]["max"]: sensor_bounds[sensor]["max"] = s_max

        # Add K-Means categorical symbols
        for sensor in continuous_sensors:
            if sensor in kmeans_models:
                # Add as string feature so River treats it as categorical
                X_df[f"{sensor}_BIN"] = kmeans_models[sensor].predict(X_df[[sensor]]).astype(str)

        # Convert chunk to dictionaries for River
        x_dicts = X_df.to_dict(orient="records")
        
        # Make a combined dict of target series for easy lookup
        y_targets = {act: chunk[act] for act in discrete_actuators}
        for sensor in continuous_sensors:
            if f"{sensor}_BIN" in X_df.columns:
                y_targets[f"{sensor}_BIN"] = X_df[f"{sensor}_BIN"]
                
        for target in all_targets:
            if target not in y_targets: continue
            y_series = y_targets[target]
            
            for idx, (x_dict, y_raw) in enumerate(zip(x_dicts, y_series)):
                if pd.isna(y_raw) or str(y_raw).strip() == "" or str(y_raw).strip() == "nan":
                    y_val = models[target]["last_y"]
                else:
                    # For bins it is already string, for actuators float->int->str
                    try:
                        y_val = str(int(float(y_raw))) if target in discrete_actuators else str(y_raw)
                    except ValueError:
                        y_val = str(y_raw)
                    models[target]["last_y"] = y_val
                    
                # For sensor prediction, exclude the sensor itself from the features in-place!
                base_sensor = None
                val = None
                if target in sensor_bins:
                    base_sensor = target.replace("_BIN", "")
                    val = x_dict.pop(base_sensor, None)
                    
                y_pred = models[target]["arf"].predict_one(x_dict)
                if y_pred is not None:
                    models[target]["acc"].update(y_val, y_pred)
                    models[target]["kappa"].update(y_val, y_pred)

                models[target]["arf"].learn_one(x_dict, y_val)
                
                # Keep a sample of records for the Decision Tree surrogate
                if len(models[target]["sample_records"]) < surrogate_sample_size:
                    if random.random() < sample_prob:
                        # Use a copy ONLY when we save it for the sample!
                        models[target]["sample_records"].append((x_dict.copy(), y_val))
                elif random.random() < (surrogate_sample_size / (rows_processed + idx + 1)):
                    replace_idx = random.randint(0, surrogate_sample_size - 1)
                    models[target]["sample_records"][replace_idx] = (x_dict.copy(), y_val)
                    
                # Restore the sensor value
                if base_sensor is not None and val is not None:
                    x_dict[base_sensor] = val
        
        # Multiply rows_processed by subsample_rate to reflect actual progress through the dataset
        rows_processed += len(chunk) * subsample_rate
        if (rows_processed / subsample_rate) % 100 == 0:
            avg_acc = sum([m["acc"].get() for m in models.values()]) / len(all_targets)
            avg_kappa = sum([m["kappa"].get() for m in models.values()]) / len(all_targets)
            print(f"Processed {rows_processed} rows (Subsampled 10x) | Avg Acc: {avg_acc:.4f} | Avg Kappa: {avg_kappa:.4f}")
            update_status("running", rows_processed, avg_acc, avg_kappa)

    print(f"Finished streaming {rows_processed} rows.")
    
    # Save sensor bounds
    bounds_path = os.path.join(MODELS_DIR, "sensor_bounds.json")
    with open(bounds_path, "w") as f:
        json.dump(sensor_bounds, f, indent=4)
    print(f"Saved global sensor bounds to {bounds_path}")

    print("Saving all ARF models and global surrogates...")
    for target, m in models.items():
        # Save ARF model
        arf_path = os.path.join(MODELS_DIR, f"arf_model_{target}.pkl")
        with open(arf_path, "wb") as f:
            pickle.dump(m["arf"], f)
            
        print(f"Generating Global Surrogate Model for {target}...")
        sample_x = [item[0] for item in m["sample_records"]]
        y_surrogate_preds = [item[1] for item in m["sample_records"]]
        
        if not sample_x:
            continue
            
        sample_df = pd.DataFrame(sample_x)
        # Increase depth slightly so it has enough complexity for complex valves
        surrogate_tree = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
        surrogate_tree.fit(sample_df, y_surrogate_preds)

        tree_text = export_text(surrogate_tree, feature_names=list(sample_df.columns))
        surrogate_path = os.path.join(MODELS_DIR, f"surrogate_tree_{target}.pkl")
        with open(surrogate_path, "wb") as f:
            pickle.dump(surrogate_tree, f)
            
        with open(os.path.join(MODELS_DIR, f"surrogate_tree_{target}.txt"), "w") as f:
            f.write(tree_text)
            
        with open(os.path.join(MODELS_DIR, f"arf_model_{target}.pkl"), "wb") as f:
            pickle.dump(m["arf"], f)
            
    print("Skipping discrete event extraction (handled by discretization pipeline).")
    
    final_acc = sum([m["acc"].get() for m in models.values()]) / len(all_targets)
    final_kappa = sum([m["kappa"].get() for m in models.values()]) / len(all_targets)
    update_status("completed", rows_processed, final_acc, final_kappa)
    print("Pipeline complete.")

def train_arf_and_surrogate(n_trees=16, split_criterion='hellinger', grace_period=30, split_confidence=1e-3):
    try:
        _train_logic(n_trees, split_criterion, grace_period, split_confidence)
    except Exception as e:
        update_status("error", error=str(e))
        raise e

if __name__ == "__main__":
    import sys
    n = 16
    sc = 'hellinger'
    gp = 30
    sco = 1e-3
    if len(sys.argv) > 1: n = int(sys.argv[1])
    if len(sys.argv) > 2: sc = sys.argv[2]
    if len(sys.argv) > 3: gp = int(sys.argv[3])
    if len(sys.argv) > 4: sco = float(sys.argv[4])
    train_arf_and_surrogate(n, sc, gp, sco)
