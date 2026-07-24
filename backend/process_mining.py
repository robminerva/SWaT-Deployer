import pandas as pd
import os
import pm4py
import json
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"
EVENT_RECORDS = os.path.join(WORKING_DIR, "event_records.parquet")
TOPOLOGY_FILE = os.path.join(WORKING_DIR, "topology.json")
PM_STATUS_FILE = os.path.join(WORKING_DIR, "pm_status.json")

def update_pm_status(status, processed=0, total=0, error=None):
    data = {
        "status": status,
        "processed": processed,
        "total": total,
        "error": error
    }
    with open(PM_STATUS_FILE, "w") as f:
        json.dump(data, f)

def generate_process_mining(noise_threshold=0.2):
    if not os.path.exists(EVENT_RECORDS):
        raise FileNotFoundError("event_records.parquet not found. Run Phase 2 first.")
    
    with open(TOPOLOGY_FILE, "r") as f:
        topology = json.load(f)
        
    df = pd.read_parquet(EVENT_RECORDS)
    total_records = len(df)
    update_pm_status("running", 0, total_records)
    
    actuators = [node["id"] for node in topology.get("nodes", []) if node["type"] == "actuator"]
    sensors = [node["id"] for node in topology.get("nodes", []) if node["type"] == "sensor"]
    df[actuators] = df[actuators].fillna(0).astype(float).astype(int).astype(str)
    for s in sensors:
        if s in df.columns:
            df[s] = pd.to_numeric(df[s], errors='coerce').fillna(0)
    
    # Extract Event Log
    events = []
    
    sensor_bins = [f"{s}_BIN" for s in sensors if f"{s}_BIN" in df.columns]
    
    # CRITICAL FIX: Only actuators should generate control-flow "events" in Process Mining.
    # Generating events for continuous sensor bin fluctuations causes PM4Py to process millions 
    # of microscopic events, leading to memory exhaustion and stalling. 
    # (The sensors will still be used later to annotate the Places between actuator events!)
    symbolic_cols = actuators
    
    # Using df.index as timestamp for sequencing
    for c in symbolic_cols:
        changed = df[c] != df[c].shift(1)
        changed.iloc[0] = False
        c_events = df.loc[changed].copy()
        
        # We need the timestamp and the action name
        event_df = pd.DataFrame({
            'time:timestamp': c_events.index,
            'concept:name': c + "=" + c_events[c].astype(str),
            'case:concept:name': "SWaT_Operation"
        })
        
        # Also carry over raw sensor values for later place annotation
        for s in sensors:
            if s in c_events.columns:
                event_df[s] = c_events[s].values
                
        events.append(event_df)
        
    event_log_df = pd.concat(events).sort_values('time:timestamp').reset_index(drop=True)
    print(f"Total events extracted: {len(event_log_df)}")
    
    # Convert arbitrary integer index into datetime so PM4Py accepts it
    event_log_df['time:timestamp'] = pd.to_datetime(event_log_df['time:timestamp'], unit='s', origin='unix')
    
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(event_log_df)
    
    print("Converting to PM4Py EventLog object...")
    log = log_converter.apply(event_log_df)
    print("Running Inductive Miner...")
    
    # Run Inductive Miner (Standard PM4Py Algorithm)
    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=noise_threshold)
    
    # Map Places to Sensor Ranges
    # A token is in a Place between the execution of its incoming transition and outgoing transition.
    # To approximate this without full trace replay, we will just sample the sensor values at the exact moment
    # the outgoing transitions from this place fire.
    
    places_list = []
    for p in net.places:
        outgoing_transitions = [arc.target.name for arc in p.out_arcs if arc.target.name]
        
        # If we have outgoing transitions, find when they fired in the log
        if outgoing_transitions:
            mask = event_log_df['concept:name'].isin(outgoing_transitions)
            firing_events = event_log_df[mask]
            
            bounds_str = []
            if not firing_events.empty:
                for s in sensors:
                    if s in firing_events.columns:
                        min_val = firing_events[s].min()
                        max_val = firing_events[s].max()
                        bounds_str.append(f"{s}: [{min_val:.2f}, {max_val:.2f}]")
            
            range_text = ", ".join(bounds_str) if bounds_str else "No outgoing events found"
        else:
            range_text = "Sink place (End of process)"
            
        places_list.append({
            "id": p.name,
            "sensor_ranges": range_text
        })
        
    transitions = [{"id": t.name, "label": t.label if t.label else ""} for t in net.transitions]
    arcs = [{"source": a.source.name, "target": a.target.name} for a in net.arcs]
    
    pm_model = {
        "places": places_list,
        "transitions": transitions,
        "arcs": arcs
    }
    
    output_file = os.path.join(WORKING_DIR, "process_mining_models.json")
    with open(output_file, "w") as f:
        json.dump(pm_model, f, indent=4)
        
    update_pm_status("completed", total_records, total_records)
    print(f"Process Mining Petri Net Generated and saved to {output_file}.")
    return pm_model

if __name__ == "__main__":
    try:
        import sys
        nt = 0.2
        if len(sys.argv) > 1:
            try: nt = float(sys.argv[1])
            except ValueError: pass
        generate_process_mining(noise_threshold=nt)
    except Exception as e:
        update_pm_status("error", error=str(e))
        raise e
