import pandas as pd
import numpy as np
import os
import time

DATA_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem/SWATDatasets"
NORMAL_CSV = os.path.join(DATA_DIR, "normal.csv")
ATTACK_CSV = os.path.join(DATA_DIR, "attack.csv")
MERGED_CSV = os.path.join(DATA_DIR, "merged.csv")

def audit_dataset():
    print("--- SWaT Dataset Audit ---")
    print(f"Loading datasets from {DATA_DIR}...")
    
    t0 = time.time()
    df_normal = pd.read_csv(NORMAL_CSV)
    df_attack = pd.read_csv(ATTACK_CSV)
    df_merged = pd.read_csv(MERGED_CSV)
    print(f"Loaded in {time.time() - t0:.2f} seconds.")
    
    # Strip spaces from columns just to be safe
    df_normal.columns = df_normal.columns.str.strip()
    df_attack.columns = df_attack.columns.str.strip()
    df_merged.columns = df_merged.columns.str.strip()
    
    print("\n1. Schema & Length Verification")
    expected_len = len(df_normal) + len(df_attack)
    actual_len = len(df_merged)
    print(f"  Expected Rows (Normal + Attack): {len(df_normal)} + {len(df_attack)} = {expected_len}")
    print(f"  Actual Rows in merged.csv:       {actual_len}")
    if expected_len != actual_len:
        print("  [WARNING] Row count mismatch!")
        
    print("\n2. Time-Series Continuity Analysis")
    # Quick check for obvious timestamp inversions without full datetime parsing
    try:
        # Just grab the last 10 rows of normal and first 10 of attack from merged
        mid = len(df_normal)
        sample = df_merged['Timestamp'].iloc[mid-5 : mid+5].tolist()
        print("  Boundary Timestamps (Normal -> Attack):")
        for i, t in enumerate(sample):
            print(f"    {t}")
    except Exception as e:
        print(f"  [ERROR] Timestamp check failed: {e}")
        
    print("\n3. Boundary Value Validation (Truncation Check)")
    # Check the exact boundary where Normal ends and Attack begins
    # The last row of Normal should be index len(df_normal) - 1
    # The first row of Attack should be index len(df_normal) in merged.csv
    
    idx_normal_last = len(df_normal) - 1
    idx_attack_first = len(df_normal)
    
    print(f"  Checking Normal Dataset Last Row vs Merged Index {idx_normal_last}")
    try:
        val_norm_orig = df_normal.iloc[-1]['FIT101']
        val_norm_merge = df_merged.iloc[idx_normal_last]['FIT101']
        print(f"    Normal FIT101 Orig:   {val_norm_orig}")
        print(f"    Normal FIT101 Merged: {val_norm_merge}")
        if val_norm_orig != val_norm_merge:
            print("    [WARNING] Values do not match. Possible precision truncation.")
            
        print(f"  Checking Attack Dataset First Row vs Merged Index {idx_attack_first}")
        val_att_orig = df_attack.iloc[0]['FIT101']
        val_att_merge = df_merged.iloc[idx_attack_first]['FIT101']
        print(f"    Attack FIT101 Orig:   {val_att_orig}")
        print(f"    Attack FIT101 Merged: {val_att_merge}")
        if val_att_orig != val_att_merge:
            print("    [WARNING] Values do not match. Possible precision truncation.")
    except Exception as e:
        print(f"  [ERROR] Boundary check failed: {e}")

if __name__ == "__main__":
    audit_dataset()
