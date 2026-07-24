import json
import pandas as pd
import os
import time
from aalpy.learning_algs import run_Alergia
from aalpy.utils import save_automaton_to_file

WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"
NORMAL_CSV = os.path.join(WORKING_DIR, "SWATDatasets", "normal.csv")
TOPOLOGY_FILE = os.path.join(WORKING_DIR, "topology.json")
FSM_STATUS_FILE = os.path.join(WORKING_DIR, "fsm_status.json")
MODELS_DIR = os.path.join(WORKING_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

def generate_agent_fsm(min_support=0.05):
    if not os.path.exists(NORMAL_CSV):
        raise FileNotFoundError("normal.csv not found.")
    
    with open(TOPOLOGY_FILE, "r") as f:
        topology = json.load(f)
        
    print("Loading Symbolic Event Records for FSM Generation...")
    df = pd.read_csv(NORMAL_CSV)
    
    actuators = [node["id"] for node in topology.get("nodes", []) if node["type"] == "actuator"]
    sensors = [node["id"] for node in topology.get("nodes", []) if node["type"] == "sensor"]
    

    # Map nodes to stages
    stage_actuators = {}
    stage_sensors = {}
    
    # NEW CODE: Save sensor bounds
    sens_for_bounds = [col for col in df.columns if col.startswith('FIT') or col.startswith('LIT') or col.startswith('AIT') or col.startswith('DPIT') or col.startswith('PIT') or col.startswith('UV')]
    bounds = {}
    for s in sens_for_bounds:
        if s in df.columns:
            bounds[s] = {'min': float(df[s].min()), 'max': float(df[s].max())}
    with open(os.path.join(WORKING_DIR, 'sensor_bounds.json'), 'w') as bf:
        json.dump(bounds, bf, indent=4)

    
    for act in actuators:
        node_info = next((n for n in topology["nodes"] if n["id"] == act), None)
        if node_info and "stage" in node_info:
            stage_name = node_info["stage"].replace(" ", "_")
            if stage_name not in stage_actuators:
                stage_actuators[stage_name] = []
            stage_actuators[stage_name].append(act)
            
    for sens in sensors:
        node_info = next((n for n in topology["nodes"] if n["id"] == sens), None)
        if node_info and "stage" in node_info:
            stage_name = node_info["stage"].replace(" ", "_")
            if stage_name not in stage_sensors:
                stage_sensors[stage_name] = []
            stage_sensors[stage_name].append(sens)

    # Clean data types
    for col in df.columns:
        if col in actuators or col.endswith("_BIN"):
            df[col] = df[col].astype(str)
            
    print("Generating Probabilistic Automata (PDFA) with AALpy...")
    
    fsms = {}
    
    total_stages = sum(1 for acts in stage_actuators.values() if acts)
    processed_count = 0
    
    for stage, acts in stage_actuators.items():
        if not acts: continue
        print(f"Processing Stage {stage}...")
        
        sens = stage_sensors.get(stage, [])
        # Only use raw continuous sensors, not the _BIN ones, to calculate accurate derivatives
        
        if not acts:
            continue
            
        with open(FSM_STATUS_FILE, "w") as f:
            json.dump({
                "status": "running",
                "processed": processed_count,
                "total": total_stages,
                "stage": stage
            }, f)
        
        # 1. Identify periods where actuator states are entirely stable
        act_df = df[acts]
        changed = act_df.ne(act_df.shift(1)).any(axis=1)
        changed.iloc[0] = True
        # Cast to int to avoid pyarrow cumsum errors on bools
        period_ids = changed.astype(int).cumsum()
        
        # Calculate the maximum duration this stage ever sits in a single state
        max_period_len = int(df.groupby(period_ids).size().max())
        # Add a 5% margin to prevent micro-fluctuation false positives
        max_period_len = int(max_period_len * 1.05)
        
        # 2. Extract first value for actuators and first/last/min/max for sensors
        agg_funcs = {act: 'first' for act in acts}
        for s in sens:
            if s in df.columns:
                agg_funcs[s] = ['first', 'last', 'min', 'max']
                
        grouped = df.groupby(period_ids).agg(agg_funcs)
        
        # 3. Calculate sensor trends
        for s in sens:
            if s in df.columns:
                first_vals = grouped[(s, 'first')]
                last_vals = grouped[(s, 'last')]
                min_vals = grouped[(s, 'min')]
                max_vals = grouped[(s, 'max')]
                
                net_change = last_vals - first_vals
                spread = max_vals - min_vals
                
                # Dynamic threshold: 2% of the sensor's global physical range
                sensor_range = df[s].max() - df[s].min()
                eps = 0.02 * sensor_range if sensor_range > 0 else 0.001
                
                # 1. Determine Volatility
                is_fluctuating = spread > (2 * eps)
                
                # 2. Determine Trend
                is_increasing = net_change > eps
                is_decreasing = net_change < -eps
                is_stable = ~(is_increasing | is_decreasing)
                
                # 3. Final State Resolution (using pandas masks)
                trend = pd.Series("Stable", index=grouped.index)
                
                # pure trends
                trend.loc[is_increasing & ~is_fluctuating] = "Increasing"
                trend.loc[is_decreasing & ~is_fluctuating] = "Decreasing"
                
                # fluctuating combinations
                trend.loc[is_stable & is_fluctuating] = "Fluctuating"
                trend.loc[is_increasing & is_fluctuating] = "Increasing and Fluctuating"
                trend.loc[is_decreasing & is_fluctuating] = "Decreasing and Fluctuating"
                
                grouped[(s, 'trend')] = trend
                
        # 4. Construct the semantic state string for each period
        states = pd.Series("", index=grouped.index, dtype=str)
        for i, act in enumerate(acts):
            prefix = "|" if i > 0 else ""
            states += prefix + act + ":" + grouped[(act, 'first')].astype(int).astype(str)
        for s in sens:
            if s in df.columns:
                states += "|" + s + ":" + grouped[(s, 'trend')]
                
        # 5. Extract strictly unique consecutive states (since some sensor trends might map back to identical strings if actuators didn't actually change the physical macro-state sequence)
        final_changed = states != states.shift(1)
        final_changed.iloc[0] = True
        symbols = states[final_changed].tolist()
        
        # Split long trace into windows of length 15 to build PTA
        window_size = 15
        data = []
        for i in range(0, len(symbols), window_size):
            chunk = symbols[i:i+window_size]
            data.append(["START"] + chunk)
            
        # Run ALERGIA to learn Markov Chain / PDFA
        # eps=0.05 is the compatibility threshold
        mc = run_Alergia(data, automaton_type='mc', eps=min_support)
        print(f"Stage {stage} learned {len(mc.states)} states.")
        
        # Save visualization DOT format
        dot_path = os.path.join(MODELS_DIR, f"pdfa_{stage}.dot")
        save_automaton_to_file(mc, dot_path, file_type="dot")
        
        # Convert to JSON for frontend
        nodes_list = []
        edges_list = []
        
        for s in mc.states:
            s_id = str(s.state_id)
            lbl = str(s.output)
            # Simplify label if it's too long
            if lbl != "START":
                parts = lbl.split("|")
                lbl = "\\n".join(parts)
            nodes_list.append({"id": s_id, "label": lbl, "title": "PDFA State"})
            
            for t_tuple in s.transitions:
                if isinstance(t_tuple, tuple) and len(t_tuple) == 2:
                    target_state, prob = t_tuple
                else:
                    # Fallback just in case
                    target_state = t_tuple
                    prob = 1.0
                    
                target_id = str(target_state.state_id) if hasattr(target_state, 'state_id') else str(target_state)
                
                edges_list.append({
                    "source": s_id,
                    "target": target_id,
                    "weight": prob,
                    "label": f"{prob:.2f}"
                })
                
        fsms[stage] = {
            "type": "PDFA",
            "stage": stage,
            "max_period_len": max_period_len,
            "nodes": nodes_list,
            "edges": edges_list
        }
        
        processed_count += 1
        
    out_path = os.path.join(WORKING_DIR, "agent_fsm_models.json")
    with open(out_path, "w") as f:
        json.dump(fsms, f)
        
    with open(FSM_STATUS_FILE, "w") as f:
        json.dump({"status": "completed"}, f)
        
    print(f"Successfully generated Probabilistic Automata for {len(fsms)} stages.")

if __name__ == "__main__":
    try:
        import sys
        ms = 0.05
        if len(sys.argv) > 1:
            try: ms = float(sys.argv[1])
            except ValueError: pass
        generate_agent_fsm(min_support=ms)
    except Exception as e:
        with open(FSM_STATUS_FILE, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        raise e
