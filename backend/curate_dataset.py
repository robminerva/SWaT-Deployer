import os
import pandas as pd
import numpy as np

WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"
DATA_DIR = os.path.join(WORKING_DIR, "SWATDatasets")

def curate_datasets():
    datasets = ["normal.csv", "attack.csv", "merged.csv"]
    results = {}

    for ds in datasets:
        file_path = os.path.join(DATA_DIR, ds)
        orig_file_path = os.path.join(DATA_DIR, f"original_{ds}")

        if not os.path.exists(file_path):
            results[ds] = {"error": f"{ds} not found."}
            continue

        # Rename to original if not done yet
        if not os.path.exists(orig_file_path):
            os.rename(file_path, orig_file_path)
            read_path = orig_file_path
        else:
            # If original exists, it means we already curated once. We will read from the original
            # to avoid compounding curation on already curated data.
            read_path = orig_file_path

        df = pd.read_csv(read_path)
        metrics = {"original_rows": len(df)}

        # 1. Strip column names
        df.columns = df.columns.str.strip()

        # 2. Combine duplicate columns (if e.g. MV101 and  MV101 existed and both became MV101)
        cols = pd.Series(df.columns)
        duplicates = cols[cols.duplicated()].unique()
        for dup in duplicates:
            dup_cols = df.loc[:, df.columns == dup]
            df = df.drop(columns=[dup])
            # combine by taking first non-null
            df[dup] = dup_cols.bfill(axis=1).iloc[:, 0]
            
        metrics["duplicate_columns_merged"] = len(duplicates)

        # 3. Check for time gaps > 1s or < 0s
        time_col = next((c for c in df.columns if "Timestamp" in c), None)
        gaps_found = 0
        if time_col:
            try:
                # Need dayfirst=True for SWaT
                ts = pd.to_datetime(df[time_col].str.strip(), format='mixed', dayfirst=True, errors='coerce')
                diffs = ts.diff().dt.total_seconds()
                gap_mask = (diffs > 1.0) | (diffs < 0.0)
                gaps_found = int(gap_mask.sum())
                
                if gaps_found > 0:
                    gap_indices = df[gap_mask].index
                    # Create fractional indices for insertion
                    new_rows = []
                    for idx in gap_indices:
                        gap_row = df.iloc[idx-1].copy()
                        gap_row[time_col] = 'GAP'
                        new_rows.append(pd.DataFrame([gap_row], index=[idx - 0.5]))
                    
                    df = pd.concat([df] + new_rows).sort_index().reset_index(drop=True)
            except Exception as e:
                pass
        metrics["time_gaps_injected"] = gaps_found

        # 4. Fill missing values (ffill for sensors holding state)
        missing_before = int(df.isna().sum().sum())
        metrics["missing_values_before"] = missing_before
        
        if missing_before > 0:
            df.ffill(inplace=True)
            # if first row is NaN, bfill
            df.bfill(inplace=True)
            
        missing_after = int(df.isna().sum().sum())
        metrics["missing_values_after"] = missing_after

        # 5. Fix float cast issues caused by NaN interpolation
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if (df[col].dropna() % 1 == 0).all():
                    df[col] = df[col].astype(int)

        # Save perfectly clean dataset
        df.to_csv(file_path, index=False)
        results[ds] = metrics

    return results

if __name__ == "__main__":
    print(curate_datasets())
