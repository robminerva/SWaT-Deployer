import pandas as pd
import sys

def compare_head():
    normal_file = "SWATDatasets/normal.csv"
    merged_file = "SWATDatasets/merged.csv"
    
    print("Loading first 1000 rows...")
    df_normal = pd.read_csv(normal_file, nrows=1000)
    df_merged = pd.read_csv(merged_file, nrows=1000)
    
    df_normal.columns = df_normal.columns.str.strip()
    df_merged.columns = df_merged.columns.str.strip()
    
    if len(df_normal) != len(df_merged):
        print("Length mismatch in first 1000 rows!")
        return
        
    differences = 0
    for col in df_normal.columns:
        if col not in df_merged.columns:
            print(f"Column {col} missing in merged!")
            continue
            
        # Compare values
        mask = df_normal[col] != df_merged[col]
        # Handle NaN equality
        mask = mask & ~(df_normal[col].isna() & df_merged[col].isna())
        
        diff_count = mask.sum()
        if diff_count > 0:
            print(f"Column '{col}' has {diff_count} differing rows in the first 1000.")
            differences += diff_count
            
            # Show first difference
            first_diff_idx = mask.idxmax()
            print(f"  Example at Row {first_diff_idx}:")
            print(f"    Normal: {df_normal.loc[first_diff_idx, col]} (Type: {type(df_normal.loc[first_diff_idx, col])})")
            print(f"    Merged: {df_merged.loc[first_diff_idx, col]} (Type: {type(df_merged.loc[first_diff_idx, col])})")
            
    if differences == 0:
        print("The first 1000 rows are perfectly identical between normal.csv and merged.csv!")
    else:
        print(f"Total differing cells: {differences}")

if __name__ == "__main__":
    compare_head()
