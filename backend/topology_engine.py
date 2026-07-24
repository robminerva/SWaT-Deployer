import pandas as pd
import json
import re
import os

DATA_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem/SWATDatasets"
WORKING_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem"
TOPOLOGY_FILE = os.path.join(WORKING_DIR, "topology.json")

def parse_header_to_topology(csv_path: str):
    # Read just the header
    df = pd.read_csv(csv_path, nrows=0)
    columns = [col.strip() for col in df.columns]
    
    topology = {
        "nodes": [],
        "edges": []
    }
    
    # 1. Create the Vessel (Stage) nodes
    for i in range(1, 7):
        topology["nodes"].append({
            "id": f"Stage {i}",
            "label": f"Vessel {i}",
            "type": "vessel",
            "stage": f"Stage {i}"
        })
        
    # 2. Connect Vessels sequentially
    for i in range(1, 6):
        topology["edges"].append({
            "from": f"Stage {i}",
            "to": f"Stage {i+1}",
            "label": "Flow",
            "type": "flow"
        })
    
    # regex to match standard SWaT tags: letters followed by digits
    pattern = re.compile(r"^([A-Za-z]+)(\d{3})$")
    
    for col in columns:
        match = pattern.match(col)
        if match:
            sensor_type_str = match.group(1).upper()
            number_str = match.group(2)
            stage_num = number_str[0] # first digit
            stage_id = f"Stage {stage_num}"
            
            # Determine if continuous sensor or discrete actuator
            is_actuator = sensor_type_str in ["MV", "P"]
            node_type = "actuator" if is_actuator else "sensor"
            
            topology["nodes"].append({
                "id": col,
                "label": col,
                "type": node_type,
                "sensor_type": sensor_type_str,
                "stage": stage_id
            })
            
            # Connect the sensor/actuator to its respective vessel
            topology["edges"].append({
                "from": stage_id,
                "to": col,
                "type": "association"
            })
            
    with open(TOPOLOGY_FILE, "w") as f:
        json.dump(topology, f, indent=4)
        
    return topology

if __name__ == "__main__":
    normal_csv = os.path.join(DATA_DIR, "normal.csv")
    if os.path.exists(normal_csv):
        print("Parsing topology from", normal_csv)
        parse_header_to_topology(normal_csv)
    else:
        print("Dataset not found at", normal_csv)
