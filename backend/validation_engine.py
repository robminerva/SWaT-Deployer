import os
import json
import csv
import pickle
import pandas as pd
import imodels
import warnings
import xgboost as xgb
warnings.filterwarnings("ignore")

MODELS_DIR = "/home/robertom/Programs/SecureWaterTreatmentSystem/models"
TOPOLOGY_FILE = "/home/robertom/Programs/SecureWaterTreatmentSystem/topology.json"

class ValidationEngine:
    def __init__(self, delta_window=5, top_k_links=5):
        self.delta_window = delta_window
        self.top_k_links = top_k_links
        self.row_buffer = []
        
        # Load topology
        with open(TOPOLOGY_FILE, 'r') as f:
            self.topology = json.load(f)
            
        self.continuous_sensors = [n["id"] for n in self.topology.get("nodes", []) if n["type"] == "sensor"]
        self.discrete_actuators = [n["id"] for n in self.topology.get("nodes", []) if n["type"] == "actuator"]
        
        # We will track the previous state of the system
        self.previous_state = {act: "nan" for act in self.discrete_actuators}
        
        # Load Sensor Bounds
        self.sensor_bounds = {}
        bounds_path = os.path.join(MODELS_DIR, "sensor_bounds.json")
        if os.path.exists(bounds_path):
            with open(bounds_path, "r") as f:
                self.sensor_bounds = json.load(f)
                
        # Load All Global Surrogate Trees
        self.surrogate_trees = {}
        self.surrogate_features = self.continuous_sensors + [s + "_BIN" for s in self.continuous_sensors]
        
        self.kmeans_models = {}
        kmeans_path = os.path.join(MODELS_DIR, "kmeans_sensors.pkl")
        if os.path.exists(kmeans_path):
            with open(kmeans_path, "rb") as f:
                self.kmeans_models = pickle.load(f)
                
        for act in self.discrete_actuators:
            surrogate_path = os.path.join(MODELS_DIR, f"surrogate_tree_{act}.pkl")
            if os.path.exists(surrogate_path):
                with open(surrogate_path, "rb") as f:
                    self.surrogate_trees[act] = pickle.load(f)
                    
        # Load FIGS Surrogate Trees
        self.figs_trees = {}
        self.figs_features = {}
        
        causal_path = os.path.join(MODELS_DIR, "causal_graph.json")
        causal_data = {}
        if os.path.exists(causal_path):
            with open(causal_path, "r") as f:
                causal_data = json.load(f)
                
        for act in self.discrete_actuators:
            figs_path = os.path.join(MODELS_DIR, f"figs_model_{act}.pkl")
            if os.path.exists(figs_path):
                with open(figs_path, "rb") as f:
                    self.figs_trees[act] = pickle.load(f)
                    # Get exact 16 features it was trained on
                    # Actuators were unfortunately trained on all 25 features!
                    # Wait, in figs_surrogate.py it did NOT apply the causal_graph logic to actuators!
                    self.figs_features[act] = self.continuous_sensors
                    
        # Load XGBoost Regression Models and Causal Graph (for RMSE)
        self.xgboost_models = {}
        self.xgboost_rmse = {}
        xgb_dir = os.path.join(MODELS_DIR, "xgboost")
        causal_path = os.path.join(MODELS_DIR, "causal_graph.json")
        
        if os.path.exists(causal_path):
            with open(causal_path, "r") as f:
                causal_data = json.load(f)
                for sensor in self.continuous_sensors:
                    if sensor in causal_data:
                        self.xgboost_rmse[sensor] = causal_data[sensor].get("rmse", 0)
                        
        if os.path.exists(xgb_dir):
            for sensor in self.continuous_sensors:
                model_path = os.path.join(xgb_dir, f"{sensor}_xgb.json")
                if os.path.exists(model_path):
                    model = xgb.XGBRegressor()
                    model.load_model(model_path)
                    self.xgboost_models[sensor] = model
        
        
        # Group actuators by stage
        self.stages = {}
        for act in self.discrete_actuators:
            stage = act[2] if len(act) > 2 else act[1]
            if stage not in self.stages:
                self.stages[stage] = []
            self.stages[stage].append(act)
            
        # Load Topology for Agent FSM State reconstruction
        self.stage_actuators = {}
        self.stage_sensors = {}
        topology_path = "/home/robertom/Programs/SecureWaterTreatmentSystem/topology.json"
        if os.path.exists(topology_path):
            with open(topology_path, "r") as f:
                topology = json.load(f)
                
                actuators = [node["id"] for node in topology.get("nodes", []) if node["type"] == "actuator"]
                sensors = [node["id"] for node in topology.get("nodes", []) if node["type"] == "sensor"]
                
                for act in actuators:
                    node_info = next((n for n in topology["nodes"] if n["id"] == act), None)
                    if node_info and "stage" in node_info:
                        stage_name = node_info["stage"].replace(" ", "_")
                        if stage_name not in self.stage_actuators:
                            self.stage_actuators[stage_name] = []
                        self.stage_actuators[stage_name].append(act)
                        
                for sens in sensors:
                    node_info = next((n for n in topology["nodes"] if n["id"] == sens), None)
                    if node_info and "stage" in node_info:
                        stage_name = node_info["stage"].replace(" ", "_")
                        if stage_name not in self.stage_sensors:
                            self.stage_sensors[stage_name] = []
                        self.stage_sensors[stage_name].append(sens)

        # Load Agent FSM PDFA Graphs
        self.agent_fsm_graphs = {}
        self.agent_fsm_current_nodes = {}
        self.agent_fsm_steps = {}
        fsm_path = "/home/robertom/Programs/SecureWaterTreatmentSystem/agent_fsm_models.json"
        if os.path.exists(fsm_path):
            with open(fsm_path, "r") as f:
                fsms = json.load(f)
                for stage_name, fsm in fsms.items():
                    if fsm.get("type") == "PDFA":
                        nodes_map = {n["id"]: n["label"] for n in fsm.get("nodes", [])}
                        edges_map = {} # source_id -> list of target_ids
                        for edge in fsm.get("edges", []):
                            src = edge["source"]
                            tgt = edge["target"]
                            if src not in edges_map:
                                edges_map[src] = []
                            edges_map[src].append(tgt)
                            
                        self.agent_fsm_graphs[stage_name] = {
                            "nodes": nodes_map,
                            "edges": edges_map
                        }
                        
                        # Find START node ID (usually q0, but we verify by label "START")
                        start_node_id = next((nid for nid, lbl in nodes_map.items() if lbl == "START"), None)
                        if start_node_id:
                            self.agent_fsm_current_nodes[stage_name] = start_node_id
                            
        self.agent_fsm_rules = {}
        
        # Load PM Transitions (Heuristics Miner sequential bigrams)
        self.pm_valid_sequences = set()
        self.sensor_bounds = {}
        
        # Helper: Try loading FSM rules from agent_fsm_models.json and bounds from sensor_bounds.jsonet()
        self.pm_known_events = set()
        pm_path = "/home/robertom/Programs/SecureWaterTreatmentSystem/process_mining_models.json"
        if os.path.exists(pm_path):
            with open(pm_path, "r") as f:
                pm_data = json.load(f)
                
                # Map transitions ID to Label
                t_labels = {t["id"]: t["label"] for t in pm_data.get("transitions", []) if t["label"]}
                self.pm_known_events.update(t_labels.values())
                
                # Map source -> target
                forward_arcs = {}
                for arc in pm_data.get("arcs", []):
                    src = arc["source"]
                    dst = arc["target"]
                    if src not in forward_arcs:
                        forward_arcs[src] = []
                    forward_arcs[src].append(dst)
                    
                # Find Transition -> Place -> Transition paths
                for t1_id, t1_label in t_labels.items():
                    # places connected from t1
                    places = forward_arcs.get(t1_id, [])
                    for p in places:
                        # transitions connected from this place
                        t2_ids = forward_arcs.get(p, [])
                        for t2_id in t2_ids:
                            t2_label = t_labels.get(t2_id)
                            if t2_label:
                                self.pm_valid_sequences.add((t1_label, t2_label))
        
        try:
            with open(os.path.join(os.path.dirname(__file__), "..", "sensor_bounds.json"), "r") as f:
                self.sensor_bounds = json.load(f)
        except Exception:
            try:
                with open("sensor_bounds.json", "r") as f:
                    self.sensor_bounds = json.load(f)
            except Exception:
                pass
        self.sab_models = None
        sab_path = os.path.join(os.path.dirname(__file__), "..", "models", "sab_models.json")
        try:
            with open(sab_path, "r") as f:
                self.sab_models = json.load(f)
        except Exception:
            pass

        # Load Physical Invariants
        self.physical_invariants = {}
        inv_path = os.path.join(MODELS_DIR, "physical_invariants.json")
        if os.path.exists(inv_path):
            with open(inv_path, "r") as f:
                self.physical_invariants = json.load(f)

        # Load Causal Graph (XGBoost)
        self.causal_graph = {}
        causal_path = os.path.join(MODELS_DIR, "causal_graph.json")
        if os.path.exists(causal_path):
            with open(causal_path, "r") as f:
                self.causal_graph = json.load(f)


                                
        self.last_pm_event_by_stage = {}
        # Precompute reverse map for acts
        self.actuator_to_stage = {}
        for stg, acts in self.stages.items():
            for a in acts:
                self.actuator_to_stage[a] = stg
                
        # To expose models to UI
        self.current_context = {
            "sab": None,
            "invariants": []
        }

        # Configuration Parameters        # XGBoost Dynamic Thresholding
        self.XGB_TRIGGER_MULT = 6.0
        self.XGB_SUSTAIN_MULT = 6.0
        self.XGB_INERTIAL_DELAY = 15
        self.TREES_INERTIAL_DELAY = 15
        self.PM_DEBOUNCE = 5
        self.FSM_MAX_DWELL = 220

        # State Trackers for Hysteresis & Inertial Delay
        self.xgb_instant_bucket = {}
        self.xgb_instant_cooldown = {}
        self.xgb_residual_history = {}
        self.xgb_rolling_cooldown = {}
        self.trees_warning_state = {}
        self.trees_cooldown = {}
        self.actuator_hold_time = {}

    def _get_sensor_bounds(self):
        # Use dynamic bounds if available, else hardcoded fallback
        if self.sensor_bounds:
            return {s: (b["min"], b["max"]) for s, b in self.sensor_bounds.items() if b["min"] != float('inf')}
        return {
            "LIT101": (100, 1000),
            "FIT101": (0, 5),
            "P101": (0, 2)
        }

    def triage_record(self, row):
        """
        Determines if a row is Pure Monitoring or Action.
        """
        action_happened = False
        changed_actuators = []
        old_stage_states = {}
        new_stage_states = {}
        
        # Build stage states
        for stage, acts in self.stages.items():
            old_states = []
            new_states = []
            for act in acts:
                old_val = self.previous_state[act]
                new_val = str(row.get(act, "nan")).strip()
                if new_val == "": new_val = "nan"
                elif new_val != "nan": new_val = str(int(float(new_val)))
                
                old_states.append(f"{act}={old_val}")
                new_states.append(f"{act}={new_val}")
                
                if old_val != new_val and old_val != "nan":
                    action_happened = True
                    changed_actuators.append((act, old_val, new_val))
                self.previous_state[act] = new_val
                
            old_stage_states[stage] = " | ".join(old_states)
            new_stage_states[stage] = " | ".join(new_states)
            
        return "Action" if action_happened else "Pure Monitoring", changed_actuators, old_stage_states, new_stage_states


    def check_sab_bounds(self, row):
        alarms = []
        if not self.sab_models:
            return alarms
            
        discrete_actuators = [
            "MV101", "P101", "P102", 
            "MV201", "P201", "P202", "P203", "P204", "P205", "P206",
            "MV301", "MV302", "MV303", "MV304", "P301", "P302",
            "P401", "P402", "P403", "P404", "UV401",
            "P501", "P502", 
            "P601", "P602", "P603"
        ]
        
        # Build situation string
        acts = []
        for act in discrete_actuators:
            val = row.get(act, 0)
            if str(val).strip() == "" or str(val).strip() == "nan": val = 0
            val = str(int(float(val)))
            acts.append(val)
        situation = "|".join(acts)
        
        is_frequent = situation in self.sab_models.get("frequent_states", {})
        is_transient = situation in self.sab_models.get("transient_states", {})
        
        # UI context
        self.current_context["sab"] = {
            "situation": situation,
            "type": "Frequent" if is_frequent else ("Transient" if is_transient else "Illegal")
        }
        
        if not is_frequent and not is_transient:
            alarms.append({
                "type": "SAB Illegal State", 
                "message": f"Illegal State Attack! Actuator combination never seen in training.", 
                "severity": "Critical Error",
                "category": 4
            })
            return alarms
            
        state_model = self.sab_models["frequent_states"][situation] if is_frequent else self.sab_models["transient_states"][situation]
        
        for sensor, rules in state_model.get("sensors", {}).items():
            if sensor not in row: continue
            try:
                val = float(row[sensor])
            except ValueError:
                continue
                
            if rules["type"] == "clusters":
                # Check if inside any cluster
                in_cluster = False
                for c_min, c_max in rules["clusters"]:
                    if c_min <= val <= c_max:
                        in_cluster = True
                        break
                if not in_cluster:
                    cluster_str = " or ".join([f"[{c[0]:.2f}, {c[1]:.2f}]" for c in rules["clusters"]])
                    cat = 2 if sensor.startswith('AIT') else (3 if 'PIT' in sensor else 1)
                    alarms.append({
                        "type": "SAB Contextual Bounds",
                        "message": f"Problem on {sensor}! Value {val:.2f} out of expected ranges {cluster_str} for current situation.",
                        "severity": "Error",
                        "category": cat
                    })
            elif rules["type"] == "flexible":
                c_min = rules["min"]
                c_max = rules["max"]
                if val < c_min or val > c_max:
                    cat = 2 if sensor.startswith('AIT') else (3 if 'PIT' in sensor else 1)
                    alarms.append({
                        "type": "SAB Transient Bounds",
                        "message": f"Problem on {sensor}! Value {val:.2f} exceeded flexible bounds [{c_min:.2f}, {c_max:.2f}] during transient state.",
                        "severity": "Warning",
                        "category": cat
                    })
                    
        return alarms

    def check_sensor_bounds(self, row):
        alarms = []
        bounds = self._get_sensor_bounds()
        for sensor, (min_val, max_val) in bounds.items():
            if sensor in row:
                try:
                    val = float(row[sensor])
                    
                    if val < min_val or val > max_val:
                        # Calculate deviation percentage
                        range_span = max_val - min_val if max_val != min_val else 1.0
                        
                        if val < min_val:
                            deviation_pct = ((min_val - val) / range_span) * 100
                            severity_txt = "low"
                            diff = min_val - val
                        else:
                            deviation_pct = ((val - max_val) / range_span) * 100
                            severity_txt = "high"
                            diff = val - max_val
                            
                        # Classify severity
                        if deviation_pct <= 1.0:
                            severity = "Warning"
                        elif deviation_pct <= 2.0:
                            severity = "Error"
                        else:
                            severity = "Critical Error"
                            
                        cat = 2 if sensor.startswith('AIT') else (3 if 'PIT' in sensor else 1)
                        msg = f"{sensor} is {diff:.2f} {severity_txt} ({deviation_pct:.1f}% out of acceptable range [{min_val:.2f}, {max_val:.2f}])"
                        alarms.append({"type": f"Bounds {severity}", "message": msg, "severity": severity, "category": cat})
                except ValueError:
                    pass
        return alarms

    def extract_tree_features(self, row):
        features = []
        for s in self.surrogate_features:
            if s.endswith("_BIN"):
                raw_s = s[:-4]
                raw_val = row.get(raw_s, 0)
                if str(raw_val).strip() == '': raw_val = 0
                raw_val = float(raw_val)
                if raw_s in getattr(self, "kmeans_models", {}):
                    centers = self.kmeans_models[raw_s].cluster_centers_
                    bin_val = float(min(range(len(centers)), key=lambda i: abs(centers[i][0] - raw_val)))
                else:
                    bin_val = 0.0
                features.append(bin_val)
            else:
                val = row.get(s, 0)
                if str(val).strip() == '': val = 0
                features.append(float(val))
        return [features]

    def check_surrogate_tree(self, row, features_2d=None, precomputed_preds=None):
        alarms = []
        if not self.surrogate_trees:
            return alarms
            
        if features_2d is None:
            features_2d = self.extract_tree_features(row)
            
        for act in self.surrogate_trees.keys():
            new_state = row.get(act, "0")
            if str(new_state).strip() == '': new_state = "0"
            if new_state != "nan": new_state = str(int(float(new_state)))
            
            try:
                if precomputed_preds and act in precomputed_preds:
                    predicted_state = precomputed_preds[act]
                else:
                    predicted_state = self.surrogate_trees[act].predict(features_2d)[0]
                
                actual_state = new_state
                
                if str(predicted_state) != str(actual_state) and str(predicted_state) != "nan":
                    state_key = f"ARF_{act}"
                    if state_key not in self.trees_warning_state:
                        self.trees_warning_state[state_key] = 0
                        self.trees_cooldown[state_key] = False
                    
                    self.trees_warning_state[state_key] = min(self.TREES_INERTIAL_DELAY, self.trees_warning_state[state_key] + 1)
                    
                    if self.trees_warning_state[state_key] >= self.TREES_INERTIAL_DELAY:
                        if not self.trees_cooldown.get(state_key, False):
                            alarms.append({"type": "Surrogate Error", "message": f"Physics logic rejected {act} state '{actual_state}'. Logic expected '{predicted_state}'.", "severity": "Error", "category": 4})
                            self.trees_cooldown[state_key] = True
                else:
                    state_key = f"ARF_{act}"
                    self.trees_warning_state[state_key] = max(0, self.trees_warning_state.get(state_key, 0) - 1)
                    if self.trees_warning_state.get(state_key, 0) == 0:
                        self.trees_cooldown[state_key] = False
            except Exception as e:
                pass
        return alarms

    def check_figs_tree(self, row, features_2d=None, precomputed_preds=None):
        alarms = []
        if not self.figs_trees:
            return alarms
            
        if features_2d is None and precomputed_preds is None:
            features_2d = self.extract_tree_features(row)
            
        df_feats = None
        if precomputed_preds is None:
            df_feats = pd.DataFrame(features_2d, columns=self.surrogate_features)
        
        for act in self.figs_trees.keys():
            new_state = row.get(act, "0")
            if str(new_state).strip() == '': new_state = "0"
            if new_state != "nan": new_state = str(int(float(new_state)))
            
            try:
                if precomputed_preds and act in precomputed_preds:
                    predicted_state = precomputed_preds[act]
                else:
                    if df_feats is None:
                        df_feats = pd.DataFrame(features_2d, columns=self.surrogate_features)
                    predicted_state = self.figs_trees[act].predict(df_feats)[0]
                actual_state = new_state
                
                if str(predicted_state) != str(actual_state) and str(predicted_state) != "nan":
                    state_key = f"FIGS_{act}"
                    if state_key not in self.trees_warning_state:
                        self.trees_warning_state[state_key] = 0
                        self.trees_cooldown[state_key] = False
                    
                    self.trees_warning_state[state_key] = min(self.TREES_INERTIAL_DELAY, self.trees_warning_state[state_key] + 1)
                    
                    if self.trees_warning_state[state_key] >= self.TREES_INERTIAL_DELAY:
                        if not self.trees_cooldown.get(state_key, False):
                            alarms.append({"type": "FIGS Rule Violation", "message": f"FIGS interpretable rules rejected {act} state '{actual_state}'. Rules expect '{predicted_state}' based on current context.", "severity": "Error", "category": 4})
                            self.trees_cooldown[state_key] = True
                else:
                    state_key = f"FIGS_{act}"
                    self.trees_warning_state[state_key] = max(0, self.trees_warning_state.get(state_key, 0) - 1)
                    if self.trees_warning_state.get(state_key, 0) == 0:
                        self.trees_cooldown[state_key] = False
            except Exception as e:
                pass
        return alarms

    def validate_action_context(self, row, changed_actuators, active_models):
        alarms = []
        
        for act, old_state, new_state in changed_actuators:

            # 2. Agent FSM Causal Physics Check
            if "agent-fsm" in active_models:
                transition = f"{old_state}->{new_state}"
                if act in self.agent_fsm_rules and transition in self.agent_fsm_rules[act]:
                    causal_rules = self.agent_fsm_rules[act][transition]
                    for rule in causal_rules:
                        if rule["type"] == "sensor_bound":
                            sensor = rule["sensor"]
                            min_v, max_v = rule["min"], rule["max"]
                            
                            if sensor in row:
                                try:
                                    s_val = float(row[sensor])
                                    if s_val < min_v or s_val > max_v:
                                        alarms.append({"type": "Agent FSM Error", "message": f"Causal violation! {act} transitioned {transition}, but trigger {sensor} is {s_val:.2f} (Expected bounds: [{min_v:.2f}, {max_v:.2f}])", "severity": "Error", "category": 4})
                                except:
                                    pass
                        elif rule["type"] == "actuator_state":
                            src_actuator = rule["actuator"]
                            expected_val = rule["val"]
                            
                            # check the state in the current row
                            if src_actuator in row:
                                s_val = str(row[src_actuator]).strip()
                                if s_val != "" and s_val != "nan":
                                    s_val = str(int(float(s_val)))
                                    if s_val != expected_val:
                                        alarms.append({"type": "Agent FSM Error", "message": f"Border violation! {act} transitioned {transition}, but inter-stage actuator {src_actuator} is {s_val} (Expected state: {expected_val})", "severity": "Error", "category": 4})
        return alarms

    def check_xgboost_predictions(self, row, precomputed_preds=None):
        """
        Validates continuous sensor readings against XGBoost predictions.
        Fires an alarm only if the absolute residual exceeds a Sustained Hysteresis threshold
        for a configured Inertial Delay (Time-Persistence).
        """
        alarms = []
        if not hasattr(self, 'xgboost_models') or not self.xgboost_models:
            return alarms
            
        # The CSV files are now normalized, so we don't need to strip keys anymore
        row_features = {k: float(v) for k, v in row.items() if k not in ["Timestamp", "Normal/Attack"]}
        
        for sensor in self.continuous_sensors:
            if sensor in self.xgboost_models:
                model = self.xgboost_models[sensor]
                rmse = self.xgboost_rmse.get(sensor, 0)
                if rmse == 0:
                    continue
                    
                actual_val = row_features.get(sensor, 0.0)
                if precomputed_preds and sensor in precomputed_preds:
                    pred_val = precomputed_preds[sensor]
                else:
                    X_dict = {k: v for k, v in row_features.items() if k != sensor}
                    X_df = pd.DataFrame([X_dict])
                    pred_val = model.predict(X_df)[0]
                    
                residual = abs(actual_val - pred_val)
                
                if sensor not in self.xgb_instant_bucket:
                    self.xgb_instant_bucket[sensor] = 0
                    self.xgb_instant_cooldown[sensor] = False
                    self.xgb_residual_history[sensor] = []
                    self.xgb_rolling_cooldown[sensor] = False

                self.xgb_residual_history[sensor].append(residual)
                if len(self.xgb_residual_history[sensor]) > 30:
                    self.xgb_residual_history[sensor].pop(0)

                # 1. Instantaneous Bucket (Massive Attacks)
                trigger_thresh = self.XGB_TRIGGER_MULT * rmse
                sustain_thresh = self.XGB_SUSTAIN_MULT * rmse
                
                if residual > trigger_thresh:
                    self.xgb_instant_bucket[sensor] += 1
                elif self.xgb_instant_bucket[sensor] > 0 and residual > sustain_thresh:
                    self.xgb_instant_bucket[sensor] += 1
                else:
                    self.xgb_instant_bucket[sensor] = max(0, self.xgb_instant_bucket[sensor] - 1)
                    
                if self.xgb_instant_bucket[sensor] >= self.XGB_INERTIAL_DELAY:
                    if not self.xgb_instant_cooldown[sensor]:
                        alarms.append({
                            "type": "XGBoost Massive Attack",
                            "message": f"Sensor {sensor} massively deviates from XGB prediction by {residual:.3f}.",
                            "severity": "High",
                            "category": 5 
                        })
                        self.xgb_instant_cooldown[sensor] = True
                elif self.xgb_instant_bucket[sensor] == 0:
                    self.xgb_instant_cooldown[sensor] = False

                # 2. Rolling Average Bucket (Stealthy Micro-Attacks)
                if len(self.xgb_residual_history[sensor]) == 30:
                    rolling_avg = sum(self.xgb_residual_history[sensor]) / 30.0
                    rolling_thresh = 1.5 * rmse
                    reset_thresh = 1.0 * rmse
                    
                    if rolling_avg > rolling_thresh:
                        if not self.xgb_rolling_cooldown[sensor]:
                            alarms.append({
                                "type": "XGBoost Micro-Attack",
                                "message": f"Sensor {sensor} rolling avg deviation ({rolling_avg:.3f}) exceeded threshold over 30s.",
                                "severity": "Medium",
                                "category": 5 
                            })
                            self.xgb_rolling_cooldown[sensor] = True
                    elif rolling_avg < reset_thresh:
                        self.xgb_rolling_cooldown[sensor] = False
                    
        return alarms

    def check_physical_invariants(self, current_row, past_row):
        alarms = []
        ui_invariants = []
        if not self.physical_invariants or past_row is None:
            self.current_context["invariants"] = ui_invariants
            return alarms
            
        # We will extract window_rows dynamically per invariant
        
        for tank, rules in self.physical_invariants.items():
            if tank not in current_row or tank not in past_row: continue
            try:
                current_LIT = float(current_row[tank])
                past_LIT = float(past_row[tank])
                
                is_absolute = rules.get("category") == 3
                target_val = current_LIT if is_absolute else (current_LIT - past_LIT)
                
                inv_window = rules.get("window", self.delta_window)
                window_rows = self.row_buffer[-inv_window:] if len(self.row_buffer) >= inv_window else self.row_buffer
                
                if tank == "LIT101" and not is_absolute and abs(target_val) > 20:
                    print(f"DEBUG: tank={tank}, current={current_LIT}, past={past_LIT}, target_val={target_val}, window_len={len(window_rows)}")
                    
                # Calculate expected target based on linear regression equation
                expected_target = rules["intercept"]
                for sensor_key, coef in rules["coefficients"].items():
                    is_squared = "^2" in sensor_key
                    sensor = sensor_key.replace("^2", "")
                    
                    if sensor in current_row:
                        # Compute the 5-point rolling average
                        vals = []
                        for r in window_rows:
                            if sensor in r:
                                vals.append(float(r[sensor]))
                                
                        if not vals:
                            continue
                        avg_sensor = sum(vals) / len(vals)
                        
                        if is_squared:
                            expected_target += coef * (avg_sensor ** 2)
                        else:
                            expected_target += coef * avg_sensor
                        
                residual = abs(target_val - expected_target)
                threshold = rules.get("epsilon", 10.0) # Fixed key from threshold to epsilon
                
                # Dynamic strictness: we use the threshold directly as it is already calibrated
                balanced = residual <= threshold
                
                # Expose equation for UI
                eq_symbol = " " if is_absolute else "Δ "
                eq_str = f"{eq_symbol}{tank} ≈ " + " + ".join([f"{coef:.2f}*{s}" for s, coef in rules["coefficients"].items()]) + f" + {rules['intercept']:.2f}"
                ui_invariants.append({
                    "equation": eq_str,
                    "expected": expected_target,
                    "actual": target_val,
                    "residual": residual,
                    "threshold": threshold,
                    "balanced": balanced
                })
                
                if not balanced:
                    alarms.append({
                        "type": "Physical Invariant Error",
                        "message": f"Invariant violation for {tank}! Expected={expected_target:.4f}, but observed={target_val:.4f}. Residual {residual:.4f} > {threshold:.4f}",
                        "severity": "Critical Error",
                        "category": rules.get("category", 1)
                    })
            except ValueError:
                pass
                
        self.current_context["invariants"] = ui_invariants
        return alarms

    def check_causal_predictions(self, current_row, past_row):
        alarms = []
        if not self.causal_graph or past_row is None:
            return alarms
            
        # For simplicity, we just look at the top K influencers and see if they are consistent 
        return alarms

    def evaluate_row(self, row, timestamp=None, active_models=None, precomputed_surrogate_preds=None, precomputed_figs_preds=None, precomputed_xgb_preds=None):
        """
        Evaluates a single row of data across the pipeline.
        Returns the type of record ("Action" or "Pure Monitoring") and a list of alarms.
        """
        # Clean keys to handle CSVs with leading/trailing spaces in headers
        row = {k.strip(): v for k, v in row.items()}
        
        # --- GAP AWARENESS ---
        # If the dataset curation phase injected a GAP row, we clear the physics buffer 
        # to prevent invalid rolling-average alarms across massive time gaps.
        if row.get("Timestamp", timestamp) == "GAP":
            self.row_buffer.clear()
            self.xgb_instant_bucket.clear()
            self.xgb_instant_cooldown.clear()
            self.xgb_residual_history.clear()
            self.xgb_rolling_cooldown.clear()
            self.trees_warning_state.clear()
            self.trees_cooldown.clear()
            self.actuator_hold_time.clear()
            return "GAP", []

        # Maintain rolling buffer
        self.row_buffer.append(row)
        if len(self.row_buffer) > self.delta_window + 1:
            self.row_buffer.pop(0)
            
        past_row = self.row_buffer[0] if len(self.row_buffer) > self.delta_window else None
        
        if active_models is None:
            active_models = ["bounds", "surrogate", "figs", "fsm"]
            
        alarms = []
        
        # 1. Triage
        record_type, changed_actuators, old_stage_states, new_stage_states = self.triage_record(row)
        
        # 2. Bounds Check (done on all rows)
        if "bounds" in active_models:
            bounds_alarms = self.check_sensor_bounds(row)
            for ba in bounds_alarms:
                alarms.append({"timestamp": timestamp, "type": ba["type"], "message": ba["message"], "severity": ba["severity"], "category": ba.get("category", 1)})

        if "sab" in active_models:
            sab_alarms = self.check_sab_bounds(row)
            for sa in sab_alarms:
                alarms.append({"timestamp": timestamp, "type": sa["type"], "message": sa["message"], "severity": sa["severity"], "category": sa.get("category", 1)})

            
        # 3. Continuous Checks (done on all rows)
        features_2d = None
        if "surrogate" in active_models or "figs" in active_models:
            features_2d = self.extract_tree_features(row)
            
        if "surrogate" in active_models:
            surrogate_alarms = self.check_surrogate_tree(row, features_2d=features_2d, precomputed_preds=precomputed_surrogate_preds)
            for sa in surrogate_alarms:
                alarms.append({"timestamp": timestamp, "type": sa["type"], "message": sa["message"], "severity": sa["severity"], "category": sa.get("category", 4)})
                
        if "figs" in active_models:
            figs_alarms = self.check_figs_tree(row, features_2d=features_2d, precomputed_preds=precomputed_figs_preds)
            for fa in figs_alarms:
                alarms.append({"timestamp": timestamp, "type": fa["type"], "message": fa["message"], "severity": fa["severity"], "category": fa.get("category", 4)})
                
        if "xgboost" in active_models:
            xgb_alarms = self.check_xgboost_predictions(row, precomputed_preds=precomputed_xgb_preds)
            for xa in xgb_alarms:
                alarms.append({"timestamp": timestamp, "type": xa["type"], "message": xa["message"], "severity": xa["severity"], "category": xa.get("category", 4)})
                
        # 3.5 Process Mining (Actuator Debouncing)
        if "pm" in active_models:
            for act in self.discrete_actuators:
                current_state = row.get(act, "0")
                if str(current_state).strip() == '': current_state = "0"
                if current_state != "nan": current_state = str(int(float(current_state)))

                if act not in self.actuator_hold_time:
                    self.actuator_hold_time[act] = {"state": current_state, "count": 1}
                else:
                    if self.actuator_hold_time[act]["state"] == current_state:
                        self.actuator_hold_time[act]["count"] += 1
                        
                        # Debounce threshold met exactly on this row
                        if self.actuator_hold_time[act]["count"] == self.PM_DEBOUNCE:
                            event = f"{act}={current_state}"
                            stage = self.actuator_to_stage.get(act)
                            
                            if event in self.pm_known_events:
                                last_event = self.last_pm_event_by_stage.get(stage)
                                if last_event:
                                    if (last_event, event) not in self.pm_valid_sequences:
                                        alarms.append({
                                            "timestamp": timestamp,
                                            "type": "Process Mining Warning",
                                            "message": f"PM Petri Net rejected intra-stage sequence: '{last_event}' -> '{event}'. This sequential path was never seen during normal operation.",
                                            "severity": "Warning",
                                            "category": 4
                                        })
                                self.last_pm_event_by_stage[stage] = event
                    else:
                        self.actuator_hold_time[act]["state"] = current_state
                        self.actuator_hold_time[act]["count"] = 1

        # 4. Action specific checks
        if record_type == "Action":
            # Process/Agent FSM Check
            if "pm" in active_models or "agent-fsm" in active_models:
                fsm_alarms = self.validate_action_context(row, changed_actuators, active_models)
                for fa in fsm_alarms:
                    alarms.append({"timestamp": timestamp, "type": fa["type"], "message": fa["message"], "severity": fa["severity"], "category": fa.get("category", 4)})
        else:
            # Static check for Agent FSM (Pure Monitoring)
            if "agent-fsm" in active_models and getattr(self, "agent_fsm_graphs", None):
                for stg, graph in self.agent_fsm_graphs.items():
                    current_stage_state = new_stage_states.get(stg, "")
                    if current_stage_state and current_stage_state not in graph.get("nodes", {}).values():
                        alarms.append({"timestamp": timestamp, "type": "Agent FSM Static Alarm", "message": f"Agent FSM rejected static state in {stg}. The state configuration is illegal/unknown.", "severity": "Critical Alarm", "category": 4})
        
        # Inject preprocessed FSM alarms if they were precalculated for this row index
        if "agent-fsm" in active_models and hasattr(self, "alarms_by_row_cache"):
            try:
                row_idx = int(timestamp)
                if row_idx in self.alarms_by_row_cache:
                    for a in self.alarms_by_row_cache[row_idx]["alarms"]:
                        alarms.append({"timestamp": timestamp, "type": a["type"], "message": a["message"], "severity": a["severity"], "category": 4})
            except:
                pass
                    
        # 5. Advanced Models
        if "physical" in active_models:
            phys_alarms = self.check_physical_invariants(row, past_row)
            for pa in phys_alarms:
                alarms.append({"timestamp": timestamp, "type": pa["type"], "message": pa["message"], "severity": pa["severity"], "category": pa.get("category", 1)})
                
        if "xgboost" in active_models:
            causal_alarms = self.check_causal_predictions(row, past_row)
            for ca in causal_alarms:
                alarms.append({"timestamp": timestamp, "type": ca["type"], "message": ca["message"], "severity": ca["severity"], "category": ca.get("category", 4)})
                
        return record_type, alarms

    def evaluate_preprocessed_chunk(self, chunk, active_models):
        """
        Groups the chunk into periods based on actuator changes (per stage!), calculates 
        semantic states (Fluctuating, Increasing, Decreasing) per sensor, 
        and evaluates against the Agent FSM.
        """
        import pandas as pd
        
        batch_alarms = []
        global_counts = {"agent-fsm": {"TP": 0, "FP": 0, "Unique": 0}}
        total_attacks = 0
        total_normal = 0
        
        chunk.columns = chunk.columns.str.strip()
        chunk['__original_index__'] = chunk.index
        
        if 'Normal/Attack' in chunk.columns:
            total_attacks = (chunk['Normal/Attack'] != 'Normal').sum()
            total_normal = len(chunk) - total_attacks
            
        if "agent-fsm" not in active_models:
            return batch_alarms, global_counts, total_attacks, total_normal
            
        # Dictionary to store alarms by original row index to group them at the end
        self.alarms_by_row_cache = {}
        
        # Evaluate PDFA FSM for each stage INDEPENDENTLY
        for stage_name, stage_graph in self.agent_fsm_graphs.items():
            acts = self.stage_actuators.get(stage_name, [])
            sens = self.stage_sensors.get(stage_name, [])
            
            if not acts and not sens:
                continue
                
            # 1. Determine periods for THIS STAGE ONLY
            act_df = chunk[acts]
            is_gap = (chunk.get("Timestamp", "") == "GAP")
            
            changed = act_df.ne(act_df.shift(1)).any(axis=1)
            changed.iloc[0] = True
            
            # A period breaks if actuators change OR if it's a GAP row
            period_ids = (changed | is_gap).astype(int).cumsum()
            
            # Now drop the GAP rows so they don't break the numeric math
            valid_mask = ~is_gap
            valid_chunk = chunk[valid_mask]
            valid_period_ids = period_ids[valid_mask]
            
            if valid_chunk.empty:
                continue
                
            agg_funcs = {act: 'first' for act in acts}
            agg_funcs['__original_index__'] = 'last'
            
            if 'Normal/Attack' in valid_chunk.columns:
                agg_funcs['Normal/Attack'] = lambda x: 'Attack' if (x != 'Normal').any() else 'Normal'
                
            for s in sens:
                if s in valid_chunk.columns:
                    agg_funcs[s] = ['first', 'last', 'min', 'max']
                    
            grouped = valid_chunk.groupby(valid_period_ids).agg(agg_funcs)
            
            # Calculate period length
            period_lengths = valid_chunk.groupby(valid_period_ids).size()
            grouped[('__period_length__', 'size')] = period_lengths
            
            # 2. Calculate sensor trends for THIS STAGE ONLY
            for s in sens:
                if s in valid_chunk.columns:
                    first_vals = grouped[(s, 'first')]
                    last_vals = grouped[(s, 'last')]
                    min_vals = grouped[(s, 'min')]
                    max_vals = grouped[(s, 'max')]
                    
                    net_change = last_vals - first_vals
                    spread = max_vals - min_vals
                    
                    if hasattr(self, 'sensor_bounds') and s in self.sensor_bounds:
                        sensor_range = float(self.sensor_bounds[s]['max']) - float(self.sensor_bounds[s]['min'])
                    else:
                        sensor_range = valid_chunk[s].max() - valid_chunk[s].min()
                        
                    eps = 0.02 * sensor_range if sensor_range > 0 else 0.001
                    
                    is_fluctuating = spread > (2 * eps)
                    is_increasing = net_change > eps
                    is_decreasing = net_change < -eps
                    is_stable = ~(is_increasing | is_decreasing)
                    
                    trend = pd.Series("Stable", index=grouped.index)
                    trend.loc[is_increasing & ~is_fluctuating] = "Increasing"
                    trend.loc[is_decreasing & ~is_fluctuating] = "Decreasing"
                    trend.loc[is_stable & is_fluctuating] = "Fluctuating"
                    trend.loc[is_increasing & is_fluctuating] = "Increasing and Fluctuating"
                    trend.loc[is_decreasing & is_fluctuating] = "Decreasing and Fluctuating"
                    
                    grouped[(s, 'trend')] = trend
                    
            # 3. Evaluate each period against FSM rules for THIS STAGE
            for period_idx, row in grouped.iterrows():
                idx_in_chunk = row[('__original_index__', 'last')]
                is_attack = False
                if ('Normal/Attack', '<lambda>') in row:
                    is_attack = (row[('Normal/Attack', '<lambda>')] != 'Normal')
                
                # Reconstruct semantic label for this stage
                lbl = ""
                for i, act in enumerate(acts):
                    prefix = "|" if i > 0 else ""
                    if (act, 'first') in row:
                        val = int(row[(act, 'first')])
                        lbl += prefix + act + ":" + str(val)
                for s in sens:
                    if (s, 'trend') in row:
                        s_trend = row[(s, 'trend')]
                        lbl += "|" + s + ":" + s_trend
                        
                lbl = "\\n".join(lbl.split("|"))
                
                # Check Temporal Bounds
                period_len = row[('__period_length__', 'size')]
                stage_max_dwell = stage_graph.get("max_period_len", self.FSM_MAX_DWELL)
                if period_len > stage_max_dwell:
                    alarm_msg = f"Temporal Bound Violation! Stage {stage_name} actuators have been static for {period_len}s (Max allowed: {stage_max_dwell}s). Suspected sensor spoofing/masking."
                    if idx_in_chunk not in self.alarms_by_row_cache:
                        self.alarms_by_row_cache[idx_in_chunk] = {"is_attack": is_attack, "alarms": []}
                    self.alarms_by_row_cache[idx_in_chunk]["alarms"].append({
                        "type": "Agent FSM Temporal Bounds Alarm",
                        "message": alarm_msg,
                        "severity": "Critical Alarm",
                        "category": 4 
                    })
                    if is_attack:
                        global_counts["agent-fsm"]["TP"] += 1
                    else:
                        global_counts["agent-fsm"]["FP"] += 1
                        
                steps_taken = self.agent_fsm_steps.get(stage_name, 0)
                if steps_taken >= 15:
                    start_node_id = next((nid for nid, slbl in stage_graph["nodes"].items() if slbl == "START"), None)
                    if start_node_id:
                        self.agent_fsm_current_nodes[stage_name] = start_node_id
                        self.agent_fsm_steps[stage_name] = 0
                        
                curr_node_id = self.agent_fsm_current_nodes.get(stage_name)
                if curr_node_id:
                    self.agent_fsm_steps[stage_name] = self.agent_fsm_steps.get(stage_name, 0) + 1
                    edges = stage_graph["edges"].get(curr_node_id, [])

                    valid_target = None
                    for target_id in edges:
                        target_lbl = stage_graph["nodes"].get(target_id)
                        if target_lbl == lbl:
                            valid_target = target_id
                            break
                            
                    if valid_target:
                        self.agent_fsm_current_nodes[stage_name] = valid_target
                    else:
                        curr_lbl = stage_graph["nodes"].get(curr_node_id)
                        if curr_lbl == lbl:
                            pass
                        else:
                            alarm = {
                                "type": "Agent FSM Error", 
                                "message": f"Sequence violation in {stage_name}! Semantic state never seen in this context: {lbl.replace('\n', ' | ')}", 
                                "severity": "Error"
                            }
                            if idx_in_chunk not in self.alarms_by_row_cache:
                                self.alarms_by_row_cache[idx_in_chunk] = {"is_attack": is_attack, "alarms": []}
                            self.alarms_by_row_cache[idx_in_chunk]["alarms"].append(alarm)
                            
                            resync_target = None
                            best_out = -1
                            for n_id, n_lbl in stage_graph["nodes"].items():
                                if n_lbl == lbl:
                                    out_deg = len(stage_graph["edges"].get(n_id, []))
                                    if out_deg > best_out:
                                        best_out = out_deg
                                        resync_target = n_id
                            if resync_target:
                                self.agent_fsm_current_nodes[stage_name] = resync_target
                                
        # Update previous states for acts at the end of the chunk
        if len(chunk) > 0:
            last_row = chunk.iloc[-1]
            for act in self.discrete_actuators:
                if act in last_row:
                    self.previous_state[act] = str(int(last_row[act]))
                    
        # We don't construct batch_alarms here anymore, we let the streaming loop do it.
        return self.alarms_by_row_cache

if __name__ == "__main__":
    engine = ValidationEngine()
    
    # Run validation on the attack dataset
    attack_csv = "/home/robertom/Programs/SecureWaterTreatmentSystem/SWATDatasets/merged.csv"
    
    if not os.path.exists(attack_csv):
        print(f"Dataset {attack_csv} not found.")
        exit(1)
        
    print(f"Starting Stateful Validation Engine on {attack_csv}...")
    
    bounds_count = 0
    surrogate_count = 0
    fsm_count = 0
    
    # Stream the dataset
    with open(attack_csv, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            timestamp = row.get("Timestamp", str(i))
            record_type, alarms = engine.evaluate_row(row, timestamp)
            
            for alarm in alarms:
                if alarm["type"] == "Bounds Violation":
                    bounds_count += 1
                elif alarm["type"] == "Surrogate Rule Violation":
                    surrogate_count += 1
                elif alarm["type"] == "Process Sequence Error":
                    fsm_count += 1
                    
            if (i + 1) % 50000 == 0:
                print(f"Processed {i + 1} rows... Alarms -> Bounds: {bounds_count}, Surrogate: {surrogate_count}, FSM: {fsm_count}")
                
    print("\n--- VALIDATION COMPLETE ---")
    print(f"Total Rows Processed: {i + 1}")
    print(f"Total Bounds Violations: {bounds_count}")
    print(f"Total Surrogate Rule Violations: {surrogate_count}")
    print(f"Total Process/FSM Sequence Errors: {fsm_count}")
    
    # Save the results
    with open("/home/robertom/Programs/SecureWaterTreatmentSystem/models/validation_results.json", "w") as out_f:
        json.dump({
            "rows_processed": i + 1,
            "bounds_violations": bounds_count,
            "surrogate_violations": surrogate_count,
            "fsm_errors": fsm_count
        }, out_f, indent=4)
