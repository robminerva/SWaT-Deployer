import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import Ridge

def mine_invariants(csv_path="SWATDatasets/normal.csv", output_path="models/physical_invariants.json", delta_window=5):
    print(f"Mining physical invariants from {csv_path} with window={delta_window}...")
    df = pd.read_csv(csv_path)
    # Clean headers
    df.columns = df.columns.str.strip()
    
    invariants = {}
    
    # ---------------------------------------------------------
    # Category 1: Hydraulic & Volumetric (Delta Volume ~ Flow)
    # ---------------------------------------------------------
    lits = ["LIT101", "LIT301", "LIT401"]
    fits = ["FIT101", "FIT201", "FIT301", "FIT401", "FIT501", "FIT502", "FIT503", "FIT504", "FIT601"]
    
    print("Mining Category 1 (Hydraulic) Invariants...")
    for lit in lits:
        if lit not in df.columns: continue
        delta_col = df[lit].diff(periods=delta_window)
        X = pd.DataFrame()
        for fit in fits:
            if fit in df.columns:
                X[fit] = df[fit].rolling(window=delta_window).mean()
                
        valid_idx = delta_col.notna() & X.notna().all(axis=1)
        y_valid = delta_col[valid_idx]
        X_valid = X[valid_idx]
        
        if len(y_valid) < 100: continue
            
        model = Ridge(alpha=0.1)
        model.fit(X_valid, y_valid)
        preds = model.predict(X_valid)
        residuals = np.abs(y_valid - preds)
        epsilon = float(np.percentile(residuals, 99.9)) * 1.5
        coeffs = {fit: float(coef) for fit, coef in zip(X.columns, model.coef_) if abs(coef) > 1e-4}
        
        invariants[lit] = {
            "category": 1,
            "intercept": float(model.intercept_),
            "coefficients": coeffs,
            "epsilon": epsilon,
            "window": delta_window
        }
        
    # ---------------------------------------------------------
    # Category 2: Chemical & Water Quality (Delta AIT ~ Dosing)
    # ---------------------------------------------------------
    print("Mining Category 2 (Chemical) Invariants...")
    aits = ["AIT201", "AIT202", "AIT203", "AIT401", "AIT402", "AIT501", "AIT502", "AIT503", "AIT504"]
    pumps = ["P201", "P203", "P204", "P205", "P206", "P301", "P302", "P401", "P402", "P403", "P404", "P501", "P502", "P601", "P602"]
    
    for ait in aits:
        if ait not in df.columns: continue
        delta_col = df[ait].diff(periods=delta_window)
        X = pd.DataFrame()
        for p in pumps:
            if p in df.columns:
                X[p] = df[p].rolling(window=delta_window).mean()
        for fit in fits:
            if fit in df.columns:
                X[fit] = df[fit].rolling(window=delta_window).mean()
                
        valid_idx = delta_col.notna() & X.notna().all(axis=1)
        y_valid = delta_col[valid_idx]
        X_valid = X[valid_idx]
        
        if len(y_valid) < 100: continue
            
        model = Ridge(alpha=0.1)
        model.fit(X_valid, y_valid)
        preds = model.predict(X_valid)
        residuals = np.abs(y_valid - preds)
        epsilon = float(np.percentile(residuals, 99.9)) * 1.5
        coeffs = {col: float(coef) for col, coef in zip(X.columns, model.coef_) if abs(coef) > 1e-4}
        
        invariants[ait] = {
            "category": 2,
            "intercept": float(model.intercept_),
            "coefficients": coeffs,
            "epsilon": epsilon,
            "window": delta_window
        }

    # ---------------------------------------------------------
    # Category 3: Pressure & Membrane (Pressure ~ Flow^2 + Pumps)
    # ---------------------------------------------------------
    print("Mining Category 3 (Pressure) Invariants...")
    pits = ["DPIT301", "PIT501", "PIT502", "PIT503"]
    
    for pit in pits:
        if pit not in df.columns: continue
        y_col = df[pit] # Instantaneous pressure, not delta
        X = pd.DataFrame()
        
        for fit in fits:
            if fit in df.columns:
                X[f"{fit}^2"] = df[fit]**2
                
        for p in pumps:
            if p in df.columns:
                X[p] = df[p]
                
        valid_idx = y_col.notna() & X.notna().all(axis=1)
        y_valid = y_col[valid_idx]
        X_valid = X[valid_idx]
        
        if len(y_valid) < 100: continue
            
        model = Ridge(alpha=0.1)
        model.fit(X_valid, y_valid)
        preds = model.predict(X_valid)
        residuals = np.abs(y_valid - preds)
        epsilon = float(np.percentile(residuals, 99.9)) * 1.5
        coeffs = {col: float(coef) for col, coef in zip(X.columns, model.coef_) if abs(coef) > 1e-4}
        
        invariants[pit] = {
            "category": 3,
            "intercept": float(model.intercept_),
            "coefficients": coeffs,
            "epsilon": epsilon,
            "window": 1
        }
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(invariants, f, indent=4)
        
    print(f"Saved {len(invariants)} invariants to {output_path}")

if __name__ == "__main__":
    mine_invariants()
