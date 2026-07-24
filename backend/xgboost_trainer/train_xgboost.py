import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import joblib

def train_models():
    csv_path = "SWATDatasets/normal.csv"
    models_dir = "models/xgboost"
    graph_path = "models/causal_graph.json"
    
    print(f"Loading dataset {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Identify continuous targets
    targets = [col for col in df.columns if col.startswith(('FIT', 'LIT', 'AIT', 'PIT', 'DPIT')) and col != 'Normal/Attack' and col != 'Timestamp']
    all_features = [col for col in df.columns if col != 'Normal/Attack' and col != 'Timestamp']
    
    os.makedirs(models_dir, exist_ok=True)
    
    causal_graph = {}
    
    # Optional: Subsample to speed up training if memory is really constrained, 
    # but 4GB should be plenty for 450k rows in XGBoost.
    # We will use early stopping and a small number of trees to keep execution fast.
    
    print(f"Found {len(targets)} continuous targets to model.")
    
    for target in targets:
        print(f"\n--- Training XGBoost for {target} ---")
        features = [f for f in all_features if f != target]
        
        import gc
        
        # Subsample to avoid memory issues on 4GB limit
        subsample_idx = df.sample(n=min(50000, len(df)), random_state=42).index
        X = df.loc[subsample_idx, features]
        y = df.loc[subsample_idx, target]
        
        # We will use hist tree method which is highly optimized and memory efficient
        model = xgb.XGBRegressor(
            n_estimators=50, 
            max_depth=5, 
            learning_rate=0.1, 
            tree_method="hist",
            n_jobs=2,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Calculate training RMSE for dynamic thresholding
        preds = model.predict(X)
        rmse = np.sqrt(np.mean((y - preds)**2))
        
        # Save model
        model_path = os.path.join(models_dir, f"{target}_xgb.json")
        model.save_model(model_path)
        
        # Extract feature importances
        importances = model.feature_importances_
        # Filter zero importance features
        feature_scores = {feat: float(score) for feat, score in zip(features, importances) if score > 0}
        
        # Sort by importance descending
        feature_scores = dict(sorted(feature_scores.items(), key=lambda item: item[1], reverse=True))
        
        causal_graph[target] = {
            "rmse": float(rmse),
            "influencers": feature_scores
        }
        
        print(f"RMSE: {rmse:.4f}")
        print(f"Top 3 Influencers:")
        for k, v in list(feature_scores.items())[:3]:
            print(f"  {k}: {v:.4f}")
            
        del model, X, y, preds
        gc.collect()
            
    # Save the causal graph
    with open(graph_path, "w") as f:
        json.dump(causal_graph, f, indent=4)
        
    print(f"\nSuccessfully trained {len(targets)} models.")
    print(f"Causal graph saved to {graph_path}")

if __name__ == "__main__":
    train_models()
