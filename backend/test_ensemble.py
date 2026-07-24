import pandas as pd
import json
import os
import sys

from validation_engine import ValidationEngine

def evaluate_ensemble():
    ground_truth_path = "models/ground_truth_events.json"
    dataset_path = "SWATDatasets/merged.csv"
    
    if not os.path.exists(ground_truth_path):
        print(f"Error: {ground_truth_path} not found.")
        sys.exit(1)
        
    with open(ground_truth_path, "r") as f:
        attacks = json.load(f)
        
    print(f"Loaded {len(attacks)} distinct attack events.")
    
    # Read just the header to get column names
    header_df = pd.read_csv(dataset_path, nrows=0)
    columns = header_df.columns
    
    # Load validation engine
    print("Loading Validation Engine...")
    engine = ValidationEngine()
    active_models = ["bounds", "pm"]
    
    # Let's test the first 2 attacks for speed
    test_attacks = attacks[:2]
    
    for attack in test_attacks:
        start_row = attack['start_row']
        end_row = attack['end_row']
        print(f"\n--- Evaluating Attack ID {attack['attack_id']} (Rows {start_row} to {end_row}) ---")
        
        # Read the attack window + 50 rows of stabilization
        df_attack = pd.read_csv(dataset_path, skiprows=range(1, start_row), nrows=(end_row - start_row + 50), names=columns, header=0)
        
        detected = False
        detection_delay = -1
        
        for i, row in df_attack.iterrows():
            actual_row_idx = start_row + i
            record_type, alarms = engine.evaluate_row(row, timestamp=str(actual_row_idx), active_models=active_models)
            
            if alarms and not detected:
                detected = True
                detection_delay = actual_row_idx - start_row
                print(f"✅ Attack {attack['attack_id']} Detected at Row {actual_row_idx} (Delay: {detection_delay} rows/sec)")
                for alarm in alarms:
                    print(f"   - {alarm['type']}: {alarm['message']}")
                break
                
        if not detected:
            print(f"❌ Attack {attack['attack_id']} was MISSED by the ensemble.")

if __name__ == "__main__":
    evaluate_ensemble()
