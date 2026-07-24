import pandas as pd
import json
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pyarrow as pa
import pyarrow.parquet as pq

WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"
DATA_DIR = os.path.join(WORKING_DIR, "SWATDatasets")
TOPOLOGY_FILE = os.path.join(WORKING_DIR, "topology.json")
NORMAL_CSV = os.path.join(DATA_DIR, "normal.csv")

def run_discretization_pipeline():
    if not os.path.exists(TOPOLOGY_FILE):
        raise FileNotFoundError("topology.json not found. Run Phase 1 first.")
        
    with open(TOPOLOGY_FILE, "r") as f:
        topology = json.load(f)
        
    # Extract continuous sensors and discrete actuators
    continuous_sensors = []
    discrete_actuators = []
    
    for node in topology.get("nodes", []):
        if node["type"] == "sensor":
            continuous_sensors.append(node["id"])
        elif node["type"] == "actuator":
            discrete_actuators.append(node["id"])
            
    # Read the dataset
    print("Loading dataset...")
    df = pd.read_csv(NORMAL_CSV)
    df.columns = [col.strip() for col in df.columns]
    
    # Check if we have the needed columns
    missing_sensors = [c for c in continuous_sensors if c not in df.columns]
    missing_actuators = [c for c in discrete_actuators if c not in df.columns]
    
    if missing_sensors or missing_actuators:
        print(f"Missing sensors: {missing_sensors}")
        print(f"Missing actuators: {missing_actuators}")
    
    X = df[continuous_sensors].fillna(0)
    
    # We will train a simple decision tree for each actuator or a multi-output tree.
    # To keep it simple and extract thresholds per sensor, we can train one large tree
    # or separate trees. Let's train one tree per actuator to find split points.
    
    all_thresholds = {sensor: set() for sensor in continuous_sensors}
    
    print("Training Decision Trees...")
    for act in discrete_actuators:
        y = df[act].fillna(0)
        # Only train if there is more than 1 class
        if len(y.unique()) > 1:
            clf = DecisionTreeClassifier(max_depth=4, random_state=42)
            clf.fit(X, y)
            
            # Extract thresholds
            tree = clf.tree_
            for i in range(tree.node_count):
                if tree.children_left[i] != tree.children_right[i]: # not a leaf
                    feature_idx = tree.feature[i]
                    threshold = tree.threshold[i]
                    feature_name = continuous_sensors[feature_idx]
                    all_thresholds[feature_name].add(threshold)
                    
    # Discretize the data based on thresholds
    print("Discretizing continuous data...")
    df_discretized = df.copy()
    
    for sensor in continuous_sensors:
        thresholds = sorted(list(all_thresholds[sensor]))
        if len(thresholds) == 0:
            # no splits found, maybe binary
            pass
        else:
            # bins: [-inf, t1, t2, ..., inf]
            bins = [-np.inf] + thresholds + [np.inf]
            labels = [f"{sensor}_State_{i}" for i in range(len(bins)-1)]
            df_discretized[sensor] = pd.cut(df[sensor], bins=bins, labels=labels, right=False)
            df_discretized[sensor] = df_discretized[sensor].astype(str)
            
    # monitoring_records.parquet contains all data (discretized)
    print("Saving monitoring_records.parquet...")
    df_discretized.to_parquet(os.path.join(WORKING_DIR, "monitoring_records.parquet"))
    
    # event_records.parquet contains only rows where state changes (events)
    # Event definition: any change in any actuator or discretized sensor
    print("Extracting event records...")
    cols_to_check = continuous_sensors + discrete_actuators
    
    # Shift to find changes
    df_shifted = df_discretized[cols_to_check].shift(1)
    # A row is an event if any column is different from the previous row
    is_event = (df_discretized[cols_to_check] != df_shifted).any(axis=1)
    # First row is always an event
    is_event.iloc[0] = True
    
    df_events = df_discretized[is_event].copy()
    print(f"Total events found: {len(df_events)}")
    df_events.to_parquet(os.path.join(WORKING_DIR, "event_records.parquet"))
    
    print("Discretization Pipeline Completed.")
    return True

if __name__ == "__main__":
    run_discretization_pipeline()
