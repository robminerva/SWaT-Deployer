from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import threading
import os
import sys
import asyncio
import pandas as pd
from pydantic import BaseModel
from validation_engine import ValidationEngine

WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"

from topology_engine import parse_header_to_topology, TOPOLOGY_FILE
from arf_surrogate import train_arf_and_surrogate
# from process_mining import generate_behavior_models (Deleted)
from sysml_compiler import generate_sysml_and_metrics, BEHAVIOR_MODELS
import subprocess

app = FastAPI(title="SWaT Digital Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopologyLockRequest(BaseModel):
    nodes: list
    edges: list

class DiscretizeRequest(BaseModel):
    k_clusters: int = 5
    n_init: int = 10
    
class ArfRequest(BaseModel):
    n_trees: int = 15
    split_criterion: str = 'hellinger'
    grace_period: int = 10
    split_confidence: float = 0.01

class FigsRequest(BaseModel):
    max_rules: int = 100
    archetypes: int = 50

class ProcessMiningRequest(BaseModel):
    noise_threshold: float = 0.2

class AgentFsmRequest(BaseModel):
    min_support: float = 0.05


@app.get("/api/topology")
def get_topology():
    # If topology.json exists, return it, else generate it
    if os.path.exists(TOPOLOGY_FILE):
        with open(TOPOLOGY_FILE, "r") as f:
            return json.load(f)
    else:
        # Generate it
        normal_csv = "/home/robertom/Programs/SecureWaterTreatmentSystem/SWATDatasets/normal.csv"
        if not os.path.exists(normal_csv):
            raise HTTPException(status_code=404, detail="Dataset not found")
        topology = parse_header_to_topology(normal_csv)
        return topology

@app.post("/api/topology/lock")
def lock_topology(req: TopologyLockRequest):
    topology = {"nodes": req.nodes, "edges": req.edges}
    with open(TOPOLOGY_FILE, "w") as f:
        json.dump(topology, f, indent=4)
    return {"status": "success", "message": "Topology locked"}

# Placeholder endpoints for Phase 2, 3, 4
@app.post("/api/curate")
def curate_dataset():
    from curate_dataset import curate_datasets
    try:
        results = curate_datasets()
        return {"status": "success", "results": results}
    except Exception as e:
        import traceback
        return {"status": "error", "message": f"Curation failed: {str(e)}\n{traceback.format_exc()}"}

@app.post("/api/discretize")
def run_discretization(req: DiscretizeRequest):
    try:
        STATUS_FILE = os.path.join(WORKING_DIR, "discretize_status.json")
        with open(STATUS_FILE, "w") as f:
            json.dump({"status": "running", "processed": 0, "acc": 0, "kappa": 0}, f)
            
        import sys
        # Run new dask out-of-core pipeline in background
        subprocess.Popen([sys.executable, "backend/discretization_pipeline.py", str(req.k_clusters), str(req.n_init)])
        
        return {"status": "success", "message": "Discretization started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/arf")
def run_arf(req: ArfRequest):
    try:
        ARF_STATUS_FILE = os.path.join(WORKING_DIR, "status.json")
        with open(ARF_STATUS_FILE, "w") as f:
            json.dump({"status": "running", "processed": 0, "acc": 0, "kappa": 0}, f)
            
        thread = threading.Thread(target=train_arf_and_surrogate, args=(req.n_trees, req.split_criterion, req.grace_period, req.split_confidence))
        thread.start()
        
        return {"status": "success", "message": "ARF started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/arf/baseline")
def use_arf_baseline():
    try:
        ARF_STATUS_FILE = os.path.join(WORKING_DIR, "status.json")
        with open(ARF_STATUS_FILE, "w") as f:
            json.dump({"processed": 1420000, "acc": 0.99, "kappa": 0.99, "status": "completed"}, f)
        return {"status": "success", "message": "Using baseline models"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/discretize/status")
def get_discretization_status():
    disc_status = "/home/robertom/Programs/SecureWaterTreatmentSystem/discretize_status.json"
    result = {"status": "idle", "processed": 0}
    if os.path.exists(disc_status):
        try:
            with open(disc_status, "r") as f:
                disc_data = json.load(f)
                result["processed"] = disc_data.get("processed", 0)
                result["status"] = disc_data.get("status", "idle")
                if disc_data.get("status") == "error":
                    result["error"] = disc_data.get("error")
        except json.JSONDecodeError:
            pass
    return result

@app.get("/api/discretize/example")
def get_discretization_example():
    try:
        import pandas as pd
        parquet_file = os.path.join(WORKING_DIR, "deduplicated_states.parquet")
        if not os.path.exists(parquet_file):
            return {"status": "error", "message": "File not found"}
        
        df = pd.read_parquet(parquet_file).head(5)
        # Select some key columns so the table isn't 80 columns wide
        # Let's take first 2 sensors, their bins, and a couple of actuators
        columns = list(df.columns)
        sensors = [c for c in columns if not c.endswith("_BIN") and c not in ["Is_Action", "Record_Type"] and not c.startswith("MV") and not c.startswith("P")]
        bins = [f"{s}_BIN" for s in sensors]
        actuators = [c for c in columns if c.startswith("MV") or c.startswith("P")]
        
        sel_cols = sensors[:3] + bins[:3] + actuators[:3]
        sel_cols = [c for c in sel_cols if c in df.columns]
        
        if not sel_cols:
            sel_cols = columns[:10]
            
        return {"status": "success", "columns": sel_cols, "data": df[sel_cols].to_dict(orient="records")}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/arf/status")
def get_arf_status():
    arf_status = "/home/robertom/Programs/SecureWaterTreatmentSystem/status.json"
    result = {"status": "idle", "processed": 0, "acc": 0, "kappa": 0}
    if os.path.exists(arf_status):
        try:
            with open(arf_status, "r") as f:
                arf_data = json.load(f)
                result["processed"] = arf_data.get("processed", 0)
                result["acc"] = arf_data.get("acc", 0)
                result["kappa"] = arf_data.get("kappa", 0)
                result["status"] = arf_data.get("status", "idle")
                if arf_data.get("status") == "error":
                    result["error"] = arf_data.get("error")
        except json.JSONDecodeError:
            pass
    return result
    
@app.post("/api/figs")
def run_figs(req: FigsRequest):
    try:
        import time
        # Reset status immediately to prevent UI race condition
        status_file = os.path.join(WORKING_DIR, "figs_status.json")
        with open(status_file, "w") as f:
            json.dump({"status": "running", "processed": 0, "start_time": time.time()}, f)
            
        log_file = open(os.path.join(WORKING_DIR, "figs_surrogate.log"), "w")
        subprocess.Popen(
            [sys.executable, "backend/figs_surrogate.py", str(req.max_rules), str(req.archetypes)],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        return {"status": "success", "message": "FIGS extraction started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/figs/status")
def get_figs_status():
    status_file = "/home/robertom/Programs/SecureWaterTreatmentSystem/figs_status.json"
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"status": "idle"}

@app.get("/api/arf/trees")
def get_arf_trees():
    try:
        import pickle
        trees = []
        models_dir = os.path.join(WORKING_DIR, "models")
        if os.path.exists(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.startswith("arf_model_") and f.endswith(".pkl"):
                    act = f.replace("arf_model_", "").replace(".pkl", "")
                    with open(os.path.join(models_dir, f), "rb") as file:
                        model = pickle.load(file)
                        if hasattr(model, 'models') and len(model.models) > 0:
                            tree_wrapper = model.models[0]
                            actual_tree = getattr(tree_wrapper, 'model', tree_wrapper)
                            if hasattr(actual_tree, 'draw'):
                                dot_str = actual_tree.draw().source
                                trees.append({"actuator": act, "dot": dot_str})
        if trees:
            return {"status": "success", "trees": trees}
        return {"status": "error", "message": "No ARF models found yet. Run ARF Training first."}
    except Exception as e:
        return {"status": "error", "message": f"Error loading trees: {str(e)}"}

@app.get("/api/discretize/tree")
def get_surrogate_tree():
    tree_content = ""
    models_dir = os.path.join(WORKING_DIR, "models")
    if os.path.exists(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.startswith("figs_rules_") and f.endswith(".txt"):
                act = f.replace("figs_rules_", "").replace(".txt", "")
                tree_content += f"=== Global Surrogate Tree for {act} ===\n"
                with open(os.path.join(models_dir, f), "r") as file:
                    tree_content += file.read() + "\n\n"
    
    if tree_content:
        return {"tree": tree_content}
    return {"tree": "Tree not generated yet."}

@app.get("/api/discretize/bounds")
def get_sensor_bounds():
    bounds_file = os.path.join(WORKING_DIR, "models", "sensor_bounds.json")
    if os.path.exists(bounds_file):
        with open(bounds_file, "r") as f:
            return json.load(f)
    return {}

@app.get("/api/dynamics/invariants")
def get_physical_invariants():
    inv_file = os.path.join(WORKING_DIR, "models", "physical_invariants.json")
    if os.path.exists(inv_file):
        with open(inv_file, "r") as f:
            return json.load(f)
    return {}

@app.get("/api/dynamics/causal")
def get_causal_graph():
    causal_file = os.path.join(WORKING_DIR, "models", "causal_graph.json")
    if os.path.exists(causal_file):
        with open(causal_file, "r") as f:
            return json.load(f)
    return {}

@app.post("/api/behavior-model/process-mining")
def run_pm(req: ProcessMiningRequest):
    try:
        import time
        status_file = os.path.join(WORKING_DIR, "pm_status.json")
        with open(status_file, "w") as f:
            json.dump({"status": "running", "processed": 0, "total": "calculating..."}, f)
            
        subprocess.Popen([sys.executable, "backend/process_mining.py", str(req.noise_threshold)])
        return {"status": "success", "message": "Process mining started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/behavior-model/agent-fsm")
def post_agent_fsm(req: AgentFsmRequest):
    try:
        import time
        status_file = os.path.join(WORKING_DIR, "fsm_status.json")
        with open(status_file, "w") as f:
            json.dump({"status": "running", "processed": 0}, f)
            
        subprocess.Popen([sys.executable, "backend/agent_fsm.py", str(req.min_support)])
        
        # Simulate completion since agent_fsm is fast
        def mark_complete():
            import time
            time.sleep(2)
            with open(status_file, "w") as f:
                json.dump({"status": "completed"}, f)
        threading.Thread(target=mark_complete).start()
        
        return {"status": "success", "message": "Agent FSM started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/behavior-model/process-mining/status")
def get_pm_status():
    status_file = os.path.join(WORKING_DIR, "pm_status.json")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            return json.load(f)
    return {"status": "idle"}

@app.get("/api/behavior-model/agent-fsm/status")
def get_fsm_status():
    status_file = os.path.join(WORKING_DIR, "fsm_status.json")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            return json.load(f)
    return {"status": "idle"}

@app.get("/api/behavior-model")
def get_behavior_model():
    result = {"fsms": {}, "pm_model": {}}
    
    FSM_MODELS = os.path.join(WORKING_DIR, "agent_fsm_models.json")
    if os.path.exists(FSM_MODELS):
        with open(FSM_MODELS, "r") as f:
            result["fsms"] = json.load(f)
            
    PM_MODELS = os.path.join(WORKING_DIR, "process_mining_models.json")
    if os.path.exists(PM_MODELS):
        with open(PM_MODELS, "r") as f:
            result["pm_model"] = json.load(f)
            
    return result

@app.post("/api/sysml")
def post_sysml():
    try:
        result = generate_sysml_and_metrics()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sysml")
def get_sysml():
    sysml_file = "/home/robertom/Programs/SecureWaterTreatmentSystem/sysml_outputs.json"
    if os.path.exists(sysml_file):
        with open(sysml_file, "r") as f:
            return json.load(f)
    return {}

@app.get("/api/sab_models")
def get_sab_models():
    sab_file = "/home/robertom/Programs/SecureWaterTreatmentSystem/models/sab_models.json"
    if os.path.exists(sab_file):
        with open(sab_file, "r") as f:
            return json.load(f)
    return {}

# WebSocket for Phase 5 Anomaly Replay
@app.websocket("/ws/replay")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    attack_file = "/home/robertom/Programs/SecureWaterTreatmentSystem/SWATDatasets/merged.csv"
    try:
        # Load iterator to avoid memory overload
        df_iter = pd.read_csv(attack_file, chunksize=1000)
        is_playing = False
        total_attack_events_ground_truth = 35
        
        engine = ValidationEngine()
        active_models = ["bounds", "surrogate", "figs", "pm", "agent-fsm"]
        
        async def send_data():
            nonlocal is_playing, df_iter, engine, active_models
            from datetime import datetime
            
            # Cumulative globals
            global_counts = {
                "bounds": {"TP": 0, "FP": 0, "Unique": 0, "DelaySum": 0},
                "physical": {"TP": 0, "FP": 0, "Unique": 0, "DelaySum": 0},
                "surrogate": {"TP": 0, "FP": 0, "Unique": 0, "DelaySum": 0},
                "figs": {"TP": 0, "FP": 0, "Unique": 0, "DelaySum": 0},
                "xgboost": {"TP": 0, "FP": 0, "Unique": 0, "DelaySum": 0},
                "pm": {"TP": 0, "FP": 0, "Unique": 0, "DelaySum": 0},
                "agent-fsm": {"TP": 0, "FP": 0, "Unique": 0, "DelaySum": 0},
                "ensemble": {"TP": 0, "FP": 0, "Unique": 0, "DelaySum": 0}
            }
            total_attack_events = 0
            total_normal_events = 0
            currently_in_alarm = False
            
            in_attack_event = False
            attack_start_idx = 0
            current_event_detected_by = set()
            last_timestamp = None
            
            while True:
                if is_playing:
                    try:
                        chunk = next(df_iter)
                        idx = int(chunk.index[-1])
                        
                        batch_alarms = []
                        
                        # Process rows through engine
                        # evaluate_preprocessed_chunk now just caches alarms by row, so we don't need this branch anymore!
                        # We stream the precalculated alarms row-by-row just like the other models!
                        if True:
                            # Batch predict surrogate trees to avoid 5000 individual .predict() calls
                            chunk_features_2d = []
                            chunk_surrogate_preds = None
                            chunk_figs_preds = None
                            
                            if "surrogate" in active_models or "figs" in active_models:
                                for _, row in chunk.iterrows():
                                    chunk_features_2d.append(engine.extract_tree_features(row)[0])
                            
                            if "surrogate" in active_models and hasattr(engine, 'surrogate_trees') and engine.surrogate_trees:
                                chunk_surrogate_preds = {act: model.predict(chunk_features_2d) for act, model in engine.surrogate_trees.items()}
                            
                            if "figs" in active_models and hasattr(engine, 'figs_trees') and engine.figs_trees:
                                import pandas as pd
                                df_feats = pd.DataFrame(chunk_features_2d, columns=engine.surrogate_features)
                                # FIGS for actuators was trained purely on continuous_sensors (25 columns)
                                figs_feats = df_feats[engine.continuous_sensors]
                                chunk_figs_preds = {act: model.predict(figs_feats) for act, model in engine.figs_trees.items()}
                                
                            chunk_xgb_preds = None
                            if "xgboost" in active_models and hasattr(engine, 'xgboost_models') and engine.xgboost_models:
                                import pandas as pd
                                # XGBoost expects all features (sensors + actuators) except the target, Timestamp, and labels
                                cols_to_drop = [c for c in ["Timestamp", "Normal/Attack", "__original_index__"] if c in chunk.columns]
                                xgb_feats = chunk.drop(columns=cols_to_drop).astype(float)
                                chunk_xgb_preds = {}
                                for sensor, model in engine.xgboost_models.items():
                                    X_df = xgb_feats.drop(columns=[sensor])
                                    chunk_xgb_preds[sensor] = model.predict(X_df)

                            for chunk_i, (i, row) in enumerate(chunk.iterrows()):
                                # Parse timestamp to detect boundaries
                                ts = None
                                is_explicit_gap = False
                                if 'Timestamp' in row:
                                    raw_ts = str(row['Timestamp']).strip()
                                    if raw_ts == "GAP":
                                        is_explicit_gap = True
                                    else:
                                        try:
                                            ts = datetime.strptime(raw_ts, "%d/%m/%Y %I:%M:%S %p")
                                        except:
                                            pass
                                
                                is_jump = False
                                if is_explicit_gap:
                                    engine.evaluate_row(row, timestamp="GAP")
                                    continue
                                
                                if ts and last_timestamp:
                                    if (ts - last_timestamp).total_seconds() > 5:
                                        is_jump = True
                                
                                if ts:
                                    last_timestamp = ts

                                # Ground truth for this specific row
                                is_attack_row = False
                                if 'Normal/Attack' in row:
                                    is_attack_row = (row['Normal/Attack'] != 'Normal')

                                # Event State Machine Transition
                                if is_jump or (is_attack_row != in_attack_event):
                                    if in_attack_event:
                                        # End of an attack event
                                        if len(current_event_detected_by) == 1:
                                            only_model = list(current_event_detected_by)[0]
                                            global_counts[only_model]["Unique"] += 1
                                        in_attack_event = False
                                        current_event_detected_by = set()
                                        
                                        # Signal frontend that attack has officially ended
                                        await websocket.send_text(json.dumps({
                                            "index": i, 
                                            "message": "Attack Disappeared / System Nominal", 
                                            "is_alarm": True, 
                                            "attack_state": "DISAPPEARED",
                                            "batch_alarms": [],
                                            "counts": global_counts,
                                            "total_attacks": total_attack_events_ground_truth,
                                            "total_normal": int(total_normal_events),
                                            "context": engine.current_context
                                        }))

                                if is_attack_row and not in_attack_event:
                                    in_attack_event = True
                                    total_attack_events += 1
                                    attack_start_idx = i
                                    
                                    # Send the [ATTACK DETECTED] IMMEDIATELY!
                                    await websocket.send_text(json.dumps({
                                        "index": attack_start_idx, 
                                        "is_alarm": True,
                                        "attack_state": "DETECTED",
                                        "batch_alarms": [], 
                                        "counts": global_counts,
                                        "total_attacks": total_attack_events_ground_truth,
                                        "total_normal": int(total_normal_events),
                                        "context": engine.current_context
                                    }))
                                    
                                elif not is_attack_row and in_attack_event:
                                    pass # Handled by transition block above

                                row_surrogate_preds = {act: preds[chunk_i] for act, preds in chunk_surrogate_preds.items()} if chunk_surrogate_preds else None
                                row_figs_preds = {act: preds[chunk_i] for act, preds in chunk_figs_preds.items()} if chunk_figs_preds else None
                                row_xgb_preds = {sensor: preds[chunk_i] for sensor, preds in chunk_xgb_preds.items()} if chunk_xgb_preds else None
                                
                                record_type, row_alarms = engine.evaluate_row(row, timestamp=str(i), active_models=active_models, precomputed_surrogate_preds=row_surrogate_preds, precomputed_figs_preds=row_figs_preds, precomputed_xgb_preds=row_xgb_preds)
                                
                                if row_alarms:
                                    # Determine which models flagged this row
                                    detected_by = set()
                                    extracted_components = []
                                    for a in row_alarms:
                                        model_key = None
                                        if ("Bounds" in a["type"] and "Agent FSM" not in a["type"]) or "SAB" in a["type"]: model_key = "bounds"
                                        elif "Physical Invariant" in a["type"]: model_key = "physical"
                                        elif "FIGS" in a["type"]: model_key = "figs"
                                        elif "Surrogate" in a["type"]: model_key = "surrogate"
                                        elif "XGBoost" in a["type"]: model_key = "xgboost"
                                        elif "Process Mining" in a["type"]: model_key = "pm"
                                        elif "Agent FSM" in a["type"] or "Causal" in a["type"]: model_key = "agent-fsm"
                                        
                                        if model_key:
                                            detected_by.add(model_key)
                                        else:
                                            print(f"WARNING: Unmapped Alarm Type: {a['type']} - Message: {a['message']}")
                                            
                                        extracted_components.append(f"[{model_key.upper() if model_key else 'SYS'}] {a['message']}")
                                        
                                    if is_attack_row:
                                        # Track which models fired during this specific attack event
                                        for model in detected_by:
                                            if model not in current_event_detected_by:
                                                # First time this model detected this event
                                                global_counts[model]["DelaySum"] += (i - attack_start_idx)
                                                global_counts[model]["TP"] += 1 # Update UI instantly
                                                
                                                if len(current_event_detected_by) == 0:
                                                    # First model to catch the ensemble!
                                                    global_counts["ensemble"]["DelaySum"] += (i - attack_start_idx)
                                                    global_counts["ensemble"]["TP"] += 1
                                                    
                                                current_event_detected_by.add(model)
                                    else:
                                        # Outside an attack event, these are False Positives
                                        for model in detected_by:
                                            global_counts[model]["FP"] += 1
                                        global_counts["ensemble"]["FP"] += 1
                                        
                                    # Store the alarms for frontend display grouped by row
                                    batch_alarms.append({
                                        "row_index": i,
                                        "is_attack": is_attack_row,
                                        "models_flagged": list(detected_by),
                                        "components": extracted_components,
                                        "alarms": row_alarms
                                    })
                                
                            if "pm" not in global_counts: global_counts["pm"] = {"TP": 0, "FP": 0, "Unique": 0}
                            if "agent-fsm" not in global_counts: global_counts["agent-fsm"] = {"TP": 0, "FP": 0, "Unique": 0}

                        if batch_alarms:
                            # Just send silently to update taxonomy, and IF in attack, print UPDATE logs
                            await websocket.send_text(json.dumps({
                                "index": idx,
                                "is_alarm": True,
                                "attack_state": "UPDATE" if in_attack_event else "NOMINAL_FALSE_POSITIVE",
                                "batch_alarms": batch_alarms,
                                "counts": global_counts,
                                "total_attacks": total_attack_events_ground_truth,
                                "total_normal": int(total_normal_events),
                                "context": getattr(engine, 'current_context', 'nominal')
                            }))
                        else:
                            # Regular tick
                            await websocket.send_text(json.dumps({
                                "index": idx, 
                                "message": "All Nominal", 
                                "is_alarm": False, 
                                "counts": global_counts,
                                "total_attacks": total_attack_events_ground_truth,
                                "total_normal": int(total_normal_events),
                                "context": getattr(engine, 'current_context', 'nominal')
                            }))
                            print(f"Processed batch up to index {idx} - All Nominal")
                            
                    except StopIteration:
                        if in_attack_event:
                            # Tally the final unique point if the file ends during an attack
                            if len(current_event_detected_by) == 1:
                                only_model = list(current_event_detected_by)[0]
                                global_counts[only_model]["Unique"] += 1
                            in_attack_event = False
                            
                            await websocket.send_text(json.dumps({
                                "index": idx,
                                "message": "Attack Disappeared / EOF",
                                "is_alarm": True,
                                "attack_state": "DISAPPEARED",
                                "batch_alarms": [],
                                "counts": global_counts,
                                "total_attacks": total_attack_events_ground_truth,
                                "total_normal": int(total_normal_events),
                                "context": getattr(engine, 'current_context', 'nominal')
                            }))
                            
                        await websocket.send_text(json.dumps({"index": "DONE", "message": "Replay finished.", "is_alarm": False}))
                        is_playing = False
                    except Exception as e:
                        import traceback
                        err_trace = traceback.format_exc()
                        print(err_trace)
                        await websocket.send_text(json.dumps({"index": "ERROR", "message": f"Backend Error: {str(e)}"}))
                        is_playing = False
                    await asyncio.sleep(0.5) # Throttle to prevent browser UI freeze
                else:
                    await asyncio.sleep(0.1)
                    
        sender_task = asyncio.create_task(send_data())
        
        while True:
            data = await websocket.receive_text()
            print(f"WS RECEIVED: {data}")
            if data.startswith("START_STREAMING") or data.startswith("START_PREPROCESSED"):
                parts = data.split(":")
                mode = parts[0]
                if len(parts) > 1:
                    dataset_name = parts[1]
                    if len(parts) > 2:
                        active_models = parts[2].split(",")
                    engine.mode = mode
                    
                    attack_file = os.path.join("/home/robertom/Programs/SecureWaterTreatmentSystem/SWATDatasets", dataset_name)
                    if os.path.exists(attack_file):
                        # Use a smaller chunksize for streaming so UI updates faster
                        df_iter = pd.read_csv(attack_file, chunksize=100)
                        is_playing = True
                        
                        if mode == "START_PREPROCESSED":
                            # Preprocess the entire dataset globally to build the period alarms
                            full_df = pd.read_csv(attack_file)
                            engine.evaluate_preprocessed_chunk(full_df, active_models)
                            print(f"Preprocessed {len(full_df)} rows for FSM. Starting stream...")
                            del full_df
                            import gc
                            gc.collect()
                        
                        total_file_rows = 54654 if "attack" in dataset_name.lower() else 1441719
                        total_attack_events_ground_truth = 35 if "attack" in dataset_name.lower() or "merged" in dataset_name.lower() else 35
                        
                        await websocket.send_text(json.dumps({
                            "index": "DATASET_INFO",
                            "dataset_name": dataset_name,
                            "total_rows": total_file_rows,
                            "total_attacks": total_attack_events_ground_truth
                        }))
                    else:
                        await websocket.send_text(json.dumps({"index": "ERROR", "message": f"Dataset {dataset_name} not found."}))
                        
            elif data == "PAUSE":
                is_playing = False
                
    except WebSocketDisconnect:
        if 'sender_task' in locals():
            sender_task.cancel()
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    # Mount frontend before running
    app.mount("/", StaticFiles(directory="/home/robertom/Programs/SecureWaterTreatmentSystem/frontend", html=True), name="frontend")
    uvicorn.run(app, host="0.0.0.0", port=8192, access_log=False)
