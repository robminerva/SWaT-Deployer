import pandas as pd
import json
import os

def extract_attack_windows(csv_filepath="SWATDatasets/merged.csv", output_filepath="models/ground_truth_events.json", label_column='Normal/Attack', attack_marker='Attack'):
    """
    Scans the SWaT dataset and groups contiguous attack rows into distinct events.
    """
    print(f"Loading dataset: {csv_filepath}...")
    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error loading {csv_filepath}: {e}")
        return

    # In SWaT, the label column is often 'Normal/Attack'
    if label_column not in df.columns:
        print(f"Column '{label_column}' not found. Available columns: {list(df.columns)}")
        # Try finding a column that looks like a label
        for col in df.columns:
            if 'label' in col.lower() or 'attack' in col.lower() or 'class' in col.lower():
                label_column = col
                print(f"Auto-selected label column: {label_column}")
                break
                
    # Standardize labels just in case there are trailing spaces in the CSV
    df[label_column] = df[label_column].astype(str).str.strip()
    
    # Create a boolean mask: True if it's an attack, False if Normal
    is_attack = (df[label_column] == attack_marker)
    
    # In our merged.csv, the Normal rows from the attack phase were deleted, 
    # meaning the attack rows are completely contiguous in the file.
    # To find the distinct attacks, we must look for jumps in the Timestamp!
    
    # Parse timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'].str.strip(), format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
    
    # Filter to only the attack rows
    attack_indices = df[df[label_column] == attack_marker].index.tolist()
    
    attack_starts = []
    attack_ends = []
    
    if attack_indices:
        current_start = attack_indices[0]
        for i in range(1, len(attack_indices)):
            curr_idx = attack_indices[i]
            prev_idx = attack_indices[i-1]
            
            # If the index jumps (due to Normal rows being filtered out)
            # OR if the timestamp jumps by more than 1 second (due to missing rows)
            time_diff = df.loc[curr_idx, 'Timestamp'] - df.loc[prev_idx, 'Timestamp']
            
            if (curr_idx - prev_idx > 1) or (time_diff > pd.Timedelta(seconds=1)):
                attack_ends.append(prev_idx)
                attack_starts.append(curr_idx)
                
        # Close the final attack
        attack_starts.insert(0, attack_indices[0])
        attack_ends.append(attack_indices[-1])

    print(f"\nTotal Distinct Attacks Found: {len(attack_starts)}\n")
    print("-" * 65)
    print(f"{'Attack #':<10} | {'Start Row':<12} | {'End Row':<12} | {'Duration (Sec/Rows)'}")
    print("-" * 65)
    
    attack_windows = []
    
    for i in range(len(attack_starts)):
        start_idx = attack_starts[i]
        end_idx = attack_ends[i]
        duration = end_idx - start_idx
        
        attack_windows.append({
            'attack_id': i + 1,
            'start_row': start_idx,
            'end_row': end_idx,
            'duration': duration
        })
        
        print(f"Attack {i+1:<3} | {start_idx:<12} | {end_idx:<12} | {duration} seconds")
        
    print("-" * 65)
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w") as f:
        json.dump(attack_windows, f, indent=4)
        
    print(f"Saved {len(attack_windows)} attack events to {output_filepath}")
    return attack_windows

if __name__ == "__main__":
    extract_attack_windows()
