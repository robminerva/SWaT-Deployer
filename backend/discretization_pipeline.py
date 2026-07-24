import os
import json
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
import sys
from sklearn.cluster import MiniBatchKMeans
import pickle

WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"
DATA_DIR = os.path.join(WORKING_DIR, "SWATDatasets")
TOPOLOGY_FILE = os.path.join(WORKING_DIR, "topology.json")
NORMAL_CSV = os.path.join(DATA_DIR, "normal.csv")
MODELS_DIR = os.path.join(WORKING_DIR, "models")
EVENT_RECORDS = os.path.join(WORKING_DIR, "event_records.parquet")
DEDUP_RECORDS = os.path.join(WORKING_DIR, "deduplicated_states.parquet")
STATUS_FILE = os.path.join(WORKING_DIR, "discretize_status.json")

def update_status(status, processed=0, error=None):
    data = {
        "status": status,
        "processed": processed,
        "error": error
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f)

class StatusFileOut:
    def __init__(self, prefix=""):
        self.prefix = prefix
        
    def write(self, s):
        if "%" in s:
            parts = s.split("|")
            if len(parts) >= 2:
                percent_str = parts[1].strip()
                update_status("running", f"{self.prefix} {percent_str}")
                
    def flush(self):
        pass

def run_pipeline(k_clusters=5, n_init=10):
    update_status("running", 0)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(TOPOLOGY_FILE, "r") as f:
        topology = json.load(f)

    continuous_sensors = [n["id"] for n in topology.get("nodes", []) if n["type"] == "sensor"]
    discrete_actuators = [n["id"] for n in topology.get("nodes", []) if n["type"] == "actuator"]
    
    print(f"Discretization Pipeline Starting...")
    print(f"Sensors: {len(continuous_sensors)} | Actuators: {len(discrete_actuators)}")

    # 1. OUT-OF-CORE INGESTION
    df = dd.read_csv(NORMAL_CSV, dtype=str)
    df = df.rename(columns=lambda x: x.strip())
    
    all_cols = continuous_sensors + discrete_actuators
    df = df[all_cols]
    
    # 2. STATE PREPROCESSING
    for act in discrete_actuators:
        df[act] = df[act].replace({"nan": None, "": None})
        
    df[discrete_actuators] = df[discrete_actuators].map_partitions(lambda x: x.ffill().fillna("0"))
    
    for s in continuous_sensors:
        df[s] = dd.to_numeric(df[s], errors='coerce').fillna(0)
    
    # 3. K-MEANS DISCRETIZATION
    print(f"Training K-Means models (k={k_clusters}) on a 5% sample...")
    sample_df = df[continuous_sensors].sample(frac=0.05, random_state=42).compute()
    
    kmeans_models = {}
    for s in continuous_sensors:
        km = MiniBatchKMeans(n_clusters=k_clusters, random_state=42, n_init=n_init, batch_size=2048)
        km.fit(sample_df[[s]])
        kmeans_models[s] = km
        
    # Save the models for anomaly replay
    with open(os.path.join(MODELS_DIR, "kmeans_sensors.pkl"), "wb") as f:
        pickle.dump(kmeans_models, f)
        
    print("Applying K-Means to generate symbolic fields...")
    def apply_kmeans(part):
        for s in continuous_sensors:
            part[f"{s}_BIN"] = kmeans_models[s].predict(part[[s]]).astype(str)
        return part
        
    meta = df._meta.copy()
    for s in continuous_sensors:
        meta[f"{s}_BIN"] = pd.Series(dtype=str)
        
    df = df.map_partitions(apply_kmeans, meta=meta)
    
    # 4. SYMBOLIC EVENT EXTRACTION
    sensor_bins = [f"{s}_BIN" for s in continuous_sensors]
    symbolic_cols = discrete_actuators + sensor_bins
    
    df_sym = df[symbolic_cols]
    df_shifted = df_sym.shift(1)
    
    is_action = (df_sym != df_shifted).any(axis=1)
    df = df.assign(Is_Action=is_action)
    df['Record_Type'] = df['Is_Action'].where(df['Is_Action'], False)
    
    df_events = df[df['Is_Action'] == True]
    
    print("Computing Symbolic Event Records (Actuator + Sensor Bin transitions)...")
    with ProgressBar(out=StatusFileOut("Events: ")):
        events_pdf = df_events.compute()
        
    events_pdf['Record_Type'] = 'Action'
    events_pdf.drop(columns=['Is_Action'], inplace=True)
    events_pdf.to_parquet(EVENT_RECORDS)
    print(f"Saved {len(events_pdf)} symbolic event records to {EVENT_RECORDS}")
    
    # 5. DEDUPLICATION
    print("Performing Out-Of-Core Deduplication on Symbolic states...")
    df_dedup = df.drop_duplicates(subset=discrete_actuators + sensor_bins)
    
    with ProgressBar(out=StatusFileOut("Deduplication: ")):
        dedup_pdf = df_dedup.compute()
        
    dedup_pdf.drop(columns=['Is_Action', 'Record_Type'], inplace=True, errors='ignore')
    dedup_pdf.to_parquet(DEDUP_RECORDS)
    print(f"Saved {len(dedup_pdf)} deduplicated states to {DEDUP_RECORDS}")
    
    update_status("completed", len(dedup_pdf))

if __name__ == "__main__":
    try:
        k = 5
        n_init = 10
        if len(sys.argv) > 1:
            try:
                k = int(sys.argv[1])
            except ValueError:
                pass
        if len(sys.argv) > 2:
            try:
                n_init = int(sys.argv[2])
            except ValueError:
                pass
        run_pipeline(k_clusters=k, n_init=n_init)
    except Exception as e:
        update_status("error", error=str(e))
        raise e
