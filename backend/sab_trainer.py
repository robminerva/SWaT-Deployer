import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"
DATA_DIR = os.path.join(WORKING_DIR, "SWATDatasets")
MODELS_DIR = os.path.join(WORKING_DIR, "models")
NORMAL_CSV = os.path.join(DATA_DIR, "normal.csv")
SAB_MODELS_JSON = os.path.join(MODELS_DIR, "sab_models.json")
SENSOR_BOUNDS_JSON = os.path.join(MODELS_DIR, "sensor_bounds.json")

def load_global_bounds():
    with open(SENSOR_BOUNDS_JSON, "r") as f:
        return json.load(f)

def train_sab():
    print("Loading normal data...")
    df = pd.read_csv(NORMAL_CSV)
    df.columns = df.columns.str.strip()
    
    global_bounds = load_global_bounds()
    
    discrete_actuators = [
        "MV101", "P101", "P102", 
        "MV201", "P201", "P202", "P203", "P204", "P205", "P206",
        "MV301", "MV302", "MV303", "MV304", "P301", "P302",
        "P401", "P402", "P403", "P404", "UV401",
        "P501", "P502", 
        "P601", "P602", "P603"
    ]
    
    sensors = [c for c in df.columns if c not in discrete_actuators and c not in ["Timestamp", "Normal/Attack"]]
    lit_sensors = [s for s in sensors if "LIT" in s]
    other_sensors = [s for s in sensors if "FIT" in s or "PIT" in s or "AIT" in s or "DPIT" in s]
    
    # Cast actuators to int then string to create a clear state key
    for act in discrete_actuators:
        df[act] = df[act].astype(int).astype(str)
        
    print("Vectorizing string concatenation...")
    df['Situation'] = df[discrete_actuators[0]].astype(str)
    for act in discrete_actuators[1:]:
        df['Situation'] = df['Situation'] + "|" + df[act].astype(str)
        
    print("Grouping by Situation...")
    print("Grouping by Situation...")
    grouped = df.groupby('Situation')
    
    sab_model = {
        "frequent_states": {},
        "transient_states": {}
    }
    
    processed = 0
    total_groups = len(grouped)
    
    for situation, group in grouped:
        processed += 1
        count = len(group)
        is_frequent = count >= 1000
        
        state_model = {
            "count": count,
            "sensors": {},
            "lits": {}
        }
        
        # 1. Process LITs (Fluctuation)
        # We look at the differences (delta) between consecutive rows in the entire dataset, 
        # but wait, the group is not necessarily contiguous. 
        # It's better to sort the original DF by index, calculate diffs, and then group.
        pass # We will do LITs globally mapped to groups later
        
        # 2. Process FIT, PIT, AIT
        for s in other_sensors:
            vals = group[s].values
            s_min = float(np.min(vals))
            s_max = float(np.max(vals))
            # simple mode via histogram/rounding
            if len(vals) == 0: continue
            
            if is_frequent:
                # 1D clustering with Agglomerative Clustering (distance threshold = 10% of global range)
                glob_min = global_bounds[s]["min"]
                glob_max = global_bounds[s]["max"]
                threshold = max(0.1, (glob_max - glob_min) * 0.1) # 10% of global range
                
                if len(vals) > 5000:
                    # Subsample for speed if huge
                    sample_vals = np.random.choice(vals, 5000, replace=False)
                else:
                    sample_vals = vals
                    
                sample_vals = sample_vals.reshape(-1, 1)
                
                # If variance is zero (all same value), just 1 cluster
                if np.max(sample_vals) - np.min(sample_vals) < 1e-5:
                    clusters = [[float(np.min(sample_vals)), float(np.max(sample_vals))]]
                else:
                    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='complete')
                    labels = clustering.fit_predict(sample_vals)
                    
                    clusters = []
                    for lbl in np.unique(labels):
                        cluster_pts = sample_vals[labels == lbl]
                        clusters.append([float(np.min(cluster_pts)), float(np.max(cluster_pts))])
                
                state_model["sensors"][s] = {
                    "type": "clusters",
                    "clusters": clusters
                }
            else:
                # Transient
                # Find mode using a histogram
                hist, bin_edges = np.histogram(vals, bins='auto')
                max_bin = np.argmax(hist)
                mode = float((bin_edges[max_bin] + bin_edges[max_bin+1]) / 2)
                
                glob_min = global_bounds[s]["min"]
                glob_max = global_bounds[s]["max"]
                global_range = max(1.0, glob_max - glob_min)
                
                dist_max = s_max - mode
                dist_min = mode - s_min
                
                max_dist = max(dist_max, dist_min)
                flexibility = max_dist + (0.1 * global_range) # 10% of global range flexibility
                
                state_model["sensors"][s] = {
                    "type": "flexible",
                    "min": float(mode - flexibility),
                    "max": float(mode + flexibility),
                    "mode": mode,
                    "flexibility": flexibility
                }
                
        if is_frequent:
            sab_model["frequent_states"][situation] = state_model
        else:
            sab_model["transient_states"][situation] = state_model
            
        if processed % 50 == 0:
            print(f"Processed {processed}/{total_groups} situations...")

    # Now handle LITs globally to get correct deltas
    print("Calculating LIT fluctuations...")
    df_lit_diff = df[lit_sensors].diff().fillna(0)
    df_lit_diff['Situation'] = df['Situation']
    
    lit_grouped = df_lit_diff.groupby('Situation')
    for situation, group in lit_grouped:
        is_frequent = situation in sab_model["frequent_states"]
        target_dict = sab_model["frequent_states"] if is_frequent else sab_model["transient_states"]
        
        for lit in lit_sensors:
            vals = group[lit].values
            # 1st and 99th percentile of deltas
            if len(vals) == 0: continue
            min_delta = float(np.percentile(vals, 1))
            max_delta = float(np.percentile(vals, 99))
            
            # Allow some baseline fluctuation
            if max_delta - min_delta < 0.1:
                min_delta -= 0.5
                max_delta += 0.5
                
            target_dict[situation]["lits"][lit] = {
                "min_delta": min_delta,
                "max_delta": max_delta
            }

    with open(SAB_MODELS_JSON, "w") as f:
        json.dump(sab_model, f, indent=4)
        
    print(f"SAB models saved to {SAB_MODELS_JSON}")
    print(f"Frequent States: {len(sab_model['frequent_states'])}")
    print(f"Transient States: {len(sab_model['transient_states'])}")

if __name__ == "__main__":
    train_sab()
