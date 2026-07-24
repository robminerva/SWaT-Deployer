// UI Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    btn.classList.add('active');
    const tabId = btn.getAttribute('data-tab');
    document.getElementById(tabId).classList.add('active');
  });
});

let topologyData = null;
let behaviorModels = null;
let pmModel = null;
let networkTopology = null;
let networkProcess = null;
let networkFSM = null;
let replayNetwork = null;

// Graph drawing helper using vis.js
function drawGraph(containerId, nodes, edges, options = {}) {
  const container = document.getElementById(containerId);
  const data = {
    nodes: new vis.DataSet(nodes),
    edges: new vis.DataSet(edges)
  };
  
  const defaultOptions = {
    nodes: {
      shape: 'dot',
      size: 16,
      font: { color: '#ffffff' },
      borderWidth: 2
    },
    edges: {
      width: 2,
      color: { color: 'rgba(255,255,255,0.2)', highlight: '#3b82f6' },
      arrows: { to: { enabled: true, scaleFactor: 0.5 } }
    },
    physics: {
      stabilization: false,
      barnesHut: { gravitationalConstant: -8000, springConstant: 0.04, springLength: 100 }
    },
    layout: {
      hierarchical: {
        enabled: false
      }
    }
  };
  
  return new vis.Network(container, data, { ...defaultOptions, ...options });
}

// Phase 1: Topology
document.getElementById('btn-load-topology').addEventListener('click', async () => {
  try {
    const res = await fetch('/api/topology');
    const data = await res.json();
    topologyData = data;
    
    const container = document.getElementById('topology-graph');
    container.innerHTML = ''; // Clear existing
    
    // Group nodes by stage
    const stages = {};
    for (let i = 1; i <= 6; i++) {
        stages[`Stage ${i}`] = { sensors: [], actuatorsIn: [], actuatorsOut: [] };
    }
    
    data.nodes.forEach(n => {
        if (n.type === 'vessel') return;
        const stageObj = stages[n.stage];
        if (!stageObj) return;
        
        if (n.type === 'sensor') {
            stageObj.sensors.push(n.id);
        } else if (n.type === 'actuator') {
            // Heuristic: MV = Inflow, P = Outflow
            if (n.id.startsWith('MV')) {
                stageObj.actuatorsIn.push(n.id);
            } else {
                stageObj.actuatorsOut.push(n.id);
            }
        }
    });
    
    // Generate DOM
    for (let i = 1; i <= 6; i++) {
        const stageName = `Stage ${i}`;
        const stageObj = stages[stageName];
        
        const vesselDiv = document.createElement('div');
        vesselDiv.className = 'vessel-box';
        
        const title = document.createElement('div');
        title.className = 'vessel-title';
        title.textContent = `Vessel ${i}`;
        vesselDiv.appendChild(title);
        
        // Sensors
        stageObj.sensors.forEach(s => {
            const badge = document.createElement('div');
            badge.className = 'sensor-badge';
            badge.textContent = s;
            badge.style.cursor = 'grab';
            const nData = data.nodes.find(n => n.id === s);
            if (nData && nData.ui_top) badge.style.top = nData.ui_top;
            if (nData && nData.ui_left) badge.style.left = nData.ui_left;
            if (nData && nData.ui_top) badge.style.position = 'absolute';
            vesselDiv.appendChild(badge);
        });
        
        // Inflow Actuators
        stageObj.actuatorsIn.forEach((a, idx) => {
            const badge = document.createElement('div');
            badge.className = 'actuator-badge actuator-in';
            badge.textContent = a;
            badge.style.cursor = 'grab';
            const nData = data.nodes.find(n => n.id === a);
            if (nData && nData.ui_top) {
                badge.style.top = nData.ui_top;
                badge.style.left = nData.ui_left;
            } else {
                badge.style.top = `${20 + (idx * 25)}%`;
            }
            vesselDiv.appendChild(badge);
        });
        
        // Outflow Actuators
        stageObj.actuatorsOut.forEach((a, idx) => {
            const badge = document.createElement('div');
            badge.className = 'actuator-badge actuator-out';
            badge.textContent = a;
            badge.style.cursor = 'grab';
            const nData = data.nodes.find(n => n.id === a);
            if (nData && nData.ui_top) {
                badge.style.top = nData.ui_top;
                badge.style.left = nData.ui_left;
            } else {
                badge.style.top = `${20 + (idx * 25)}%`;
            }
            vesselDiv.appendChild(badge);
        });
        
        // Connector to next stage
        if (i < 6) {
            const connector = document.createElement('div');
            connector.className = 'vessel-connector';
            vesselDiv.appendChild(connector);
        }
        
        container.appendChild(vesselDiv);
    }
  } catch (err) {
    alert("Error loading topology: " + err);
  }
});

// Drag and Drop Logic
let draggedElement = null;
let offsetX = 0;
let offsetY = 0;

document.addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('sensor-badge') || e.target.classList.contains('actuator-badge')) {
        draggedElement = e.target;
        
        if (draggedElement.style.position !== 'absolute') {
            draggedElement.style.position = 'absolute';
        }
        
        const rect = draggedElement.getBoundingClientRect();
        offsetX = e.clientX - rect.left;
        offsetY = e.clientY - rect.top;
        
        draggedElement.style.zIndex = 1000;
        draggedElement.style.cursor = 'grabbing';
    }
});

document.addEventListener('mousemove', (e) => {
    if (draggedElement) {
        const parentRect = draggedElement.parentElement.getBoundingClientRect();
        const newLeft = e.clientX - parentRect.left - offsetX;
        const newTop = e.clientY - parentRect.top - offsetY;
        
        draggedElement.style.left = `${newLeft}px`;
        draggedElement.style.top = `${newTop}px`;
    }
});

document.addEventListener('mouseup', () => {
    if (draggedElement) {
        const nodeId = draggedElement.textContent;
        if (topologyData && topologyData.nodes) {
            const node = topologyData.nodes.find(n => n.id === nodeId);
            if (node) {
                node.ui_top = draggedElement.style.top;
                node.ui_left = draggedElement.style.left;
            }
        }
        
        draggedElement.style.zIndex = 10;
        draggedElement.style.cursor = 'grab';
        draggedElement = null;
    }
});

document.getElementById('btn-lock-topology').addEventListener('click', async () => {
  if (!topologyData) return alert("Load topology first");
  try {
    const res = await fetch('/api/topology/lock', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(topologyData)
    });
    const result = await res.json();
    alert(result.message);
  } catch (err) {
    alert("Error locking topology: " + err);
  }
});

// Phase 2: Discretization
let pollInterval;

async function fetchSurrogateTree() {
    try {
        const res = await fetch('/api/discretize/tree');
        const data = await res.json();
        const treeDisplay = document.getElementById('surrogate-tree-display');
        treeDisplay.textContent = data.tree;
        treeDisplay.style.display = 'block';
        
        const resBounds = await fetch('/api/discretize/bounds');
        const dataBounds = await resBounds.json();
        const boundsDisplay = document.getElementById('sensor-bounds-display');
        if (Object.keys(dataBounds).length > 0) {
            boundsDisplay.textContent = JSON.stringify(dataBounds, null, 2);
        } else {
            boundsDisplay.textContent = "Bounds not extracted yet.";
        }
    } catch (e) {
        console.error("Failed to fetch tree or bounds", e);
    }
}

async function checkDiscretizeStatus(init = false) {
    try {
        const statusRes = await fetch('/api/discretize/status');
        const statusData = await statusRes.json();
        const logContainer = document.getElementById('discretize-log');
        
        if (statusData.status === "running") {
            logContainer.innerHTML = `Running Discretization... Processed ${statusData.processed} rows out of approx 1,300,000.`;
            return "running";
        } else if (statusData.status === "completed") {
            logContainer.innerHTML = `Discretization Completed Successfully!<br>Total Processed: ${statusData.processed}`;
            fetchDiscretizationExample();
            return "completed";
        } else if (statusData.status === "error") {
            logContainer.innerHTML = "Error: " + statusData.error;
            return "error";
        }
    } catch (e) {
        console.error("Polling error", e);
    }
    return "idle";
}

async function checkArfStatus(init = false) {
    try {
        const statusRes = await fetch('/api/arf/status');
        const statusData = await statusRes.json();
        const logContainer = document.getElementById('arf-log');
        
        if (statusData.status === "running") {
            logContainer.innerHTML = `Running ARF... Processed ${statusData.processed} rows out of approx 1,300,000.<br>Metrics -> Acc: ${statusData.acc} | Kappa: ${statusData.kappa}`;
            return "running";
        } else if (statusData.status === "completed") {
            logContainer.innerHTML = `ARF Training Completed Successfully!<br>Total Processed: ${statusData.processed}<br>Final Acc: ${statusData.acc} | Kappa: ${statusData.kappa}`;
            fetchSurrogateTree();
            return "completed";
        } else if (statusData.status === "error") {
            logContainer.innerHTML = "Error: " + statusData.error;
            return "error";
        }
    } catch (e) {
        console.error("Polling error", e);
    }
    return "idle";
}

let discretizePollInterval;
checkDiscretizeStatus(true).then(state => {
    if (state === "running") {
        discretizePollInterval = setInterval(async () => {
            const currentState = await checkDiscretizeStatus();
            if (currentState === "completed" || currentState === "error") {
                clearInterval(discretizePollInterval);
            }
        }, 1000);
    }
});

let arfPollInterval;
checkArfStatus(true).then(state => {
    if (state === "running") {
        arfPollInterval = setInterval(async () => {
            const currentState = await checkArfStatus();
            if (currentState === "completed" || currentState === "error") {
                clearInterval(arfPollInterval);
            }
        }, 1000);
    }
});

if (document.getElementById('surrogate-tree-display').textContent === "") {
    fetchSurrogateTree();
}

async function fetchDiscretizationExample() {
    try {
        const res = await fetch('/api/discretize/example');
        const data = await res.json();
        const container = document.getElementById('discretize-example-container');
        
        if (data.status === "success" && data.data && data.data.length > 0) {
            let html = `<table style="width: 100%; border-collapse: collapse; font-size: 0.85em; text-align: left;">`;
            html += `<thead><tr style="border-bottom: 1px solid #334155;">`;
            data.columns.forEach(col => {
                html += `<th style="padding: 8px;">${col}</th>`;
            });
            html += `</tr></thead><tbody>`;
            
            data.data.forEach(row => {
                html += `<tr style="border-bottom: 1px solid #1e293b;">`;
                data.columns.forEach(col => {
                    html += `<td style="padding: 8px;">${row[col]}</td>`;
                });
                html += `</tr>`;
            });
            
            html += `</tbody></table>`;
            container.innerHTML = html;
        } else if (data.status === "error") {
            container.innerHTML = `<p style="color: #ef4444;">Could not load example: ${data.message}</p>`;
        }
    } catch (e) {
        console.error("Failed to fetch discretization example", e);
    }
}

document.getElementById('btn-run-curate').addEventListener('click', async () => {
    document.getElementById('curate-log').innerHTML = "Curating datasets... This may take a minute.";
    document.getElementById('btn-run-curate').disabled = true;
    
    try {
        const res = await fetch("/api/curate", { method: "POST" });
        const data = await res.json();
        
        if (data.status === "success") {
            document.getElementById('curate-log').innerHTML = "Datasets curated successfully!";
            const metricsDisplay = document.getElementById('curate-metrics-display');
            metricsDisplay.style.display = "block";
            metricsDisplay.textContent = JSON.stringify(data.results, null, 2);
        } else {
            document.getElementById('curate-log').innerHTML = "Error: " + data.message;
        }
    } catch (e) {
        document.getElementById('curate-log').innerHTML = "Failed to run curation.";
    } finally {
        document.getElementById('btn-run-curate').disabled = false;
    }
});

document.getElementById('btn-run-discretize').addEventListener('click', async () => {
  const logContainer = document.getElementById('discretize-log');
  const kInput = document.getElementById('k-clusters-input');
  const nInitInput = document.getElementById('n-init-input');
  const kClusters = kInput ? parseInt(kInput.value, 10) : 5;
  const nInit = nInitInput ? parseInt(nInitInput.value, 10) : 10;
  logContainer.innerHTML = "Starting Discretization pipeline...";
  try {
    fetch('/api/discretize', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k_clusters: kClusters, n_init: nInit })
    });
    
    if (discretizePollInterval) clearInterval(discretizePollInterval);
    discretizePollInterval = setInterval(async () => {
        const state = await checkDiscretizeStatus();
        if (state === "completed" || state === "error") {
            clearInterval(discretizePollInterval);
        }
    }, 1000);
  } catch (err) {
    logContainer.innerHTML = "Error: " + err;
  }
});

document.getElementById('btn-run-arf').addEventListener('click', async () => {
  const logContainer = document.getElementById('arf-log');
  const nTreesInput = document.getElementById('n-trees-input');
  const splitCriterionInput = document.getElementById('split-criterion-input');
  const gracePeriodInput = document.getElementById('grace-period-input');
  const splitConfidenceInput = document.getElementById('split-confidence-input');
  
  const nTrees = nTreesInput ? parseInt(nTreesInput.value, 10) : 15;
  const splitCriterion = splitCriterionInput ? splitCriterionInput.value : 'hellinger';
  const gracePeriod = gracePeriodInput ? parseInt(gracePeriodInput.value, 10) : 10;
  const splitConfidence = splitConfidenceInput ? parseFloat(splitConfidenceInput.value) : 0.01;
  
  logContainer.innerHTML = "Starting ARF pipeline...";
  try {
    fetch('/api/arf', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_trees: nTrees, split_criterion: splitCriterion, grace_period: gracePeriod, split_confidence: splitConfidence })
    });
    
    if (arfPollInterval) clearInterval(arfPollInterval);
    arfPollInterval = setInterval(async () => {
        const state = await checkArfStatus();
        if (state === "completed" || state === "error") {
            clearInterval(arfPollInterval);
        }
    }, 1000);
    
  } catch (err) {
    logContainer.innerHTML = "Error: " + err;
  }
});

document.getElementById('btn-use-baseline').addEventListener('click', async () => {
    const logContainer = document.getElementById('arf-log');
    logContainer.innerHTML = "Skipping extraction: Using July 12 baseline models...";
    try {
        await fetch('/api/arf/baseline', { method: 'POST' });
        if (arfPollInterval) clearInterval(arfPollInterval);
        checkArfStatus();
    } catch (e) {
        logContainer.innerHTML = "Error: " + e;
    }
});

let figsPollInterval;
document.getElementById('btn-run-figs').addEventListener('click', async () => {
    const log = document.getElementById('figs-log');
    
    const maxRulesInput = document.getElementById('max-rules-input');
    const archetypesInput = document.getElementById('archetypes-input');
    const maxRules = maxRulesInput ? parseInt(maxRulesInput.value, 10) : 100;
    const archetypes = archetypesInput ? parseInt(archetypesInput.value, 10) : 50;
    
    log.innerHTML = "Starting FIGS pipeline...";
    try {
        fetch('/api/figs', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ max_rules: maxRules, archetypes: archetypes })
        });
        if (figsPollInterval) clearInterval(figsPollInterval);
        figsPollInterval = setInterval(async () => {
            const res = await fetch('/api/figs/status');
            const data = await res.json();
            if (data.status === "running") {
                const elapsedSecs = data.start_time ? Math.floor(Date.now()/1000 - data.start_time) : 0;
                const m = Math.floor(elapsedSecs / 60);
                const s = elapsedSecs % 60;
                const timeStr = `${m}m ${s}s`;
                
                if (data.current_actuator) {
                    log.innerHTML = `Extracting FIGS Rules... Training actuator ${data.current_actuator} (${data.progress} / ${data.total}). Elapsed Time: ${timeStr}`;
                } else {
                    log.innerHTML = `Extracting FIGS Rules...`;
                }
            } else if (data.status === "completed") {
                clearInterval(figsPollInterval);
                log.innerHTML = `FIGS Extraction Complete! Processed ${data.processed} unique states.`;
            } else if (data.status === "error") {
                clearInterval(figsPollInterval);
                log.innerHTML = `Error: ${data.error}`;
            }
        }, 1000);
    } catch (err) {
        log.innerHTML = "Error: " + err;
    }
});

document.getElementById('btn-fetch-arf-trees').addEventListener('click', async () => {
    const display = document.getElementById('arf-tree-display');
    const btn = document.getElementById('btn-fetch-arf-trees');
    btn.textContent = "Fetching...";
    display.style.display = 'block';
    display.innerHTML = "Loading trees...";
    try {
        const res = await fetch('/api/arf/trees');
        const data = await res.json();
        
        if (data.status === "success" && data.trees && data.trees.length > 0) {
            display.innerHTML = ""; // Clear
            
            data.trees.forEach((treeObj, idx) => {
                const header = document.createElement('h4');
                header.textContent = `Actuator: ${treeObj.actuator}`;
                header.style.color = "#94a3b8";
                header.style.marginTop = idx === 0 ? "0" : "20px";
                display.appendChild(header);
                
                const graphContainer = document.createElement('div');
                graphContainer.style.width = "100%";
                graphContainer.style.height = "300px";
                graphContainer.style.border = "1px solid #334155";
                graphContainer.style.borderRadius = "4px";
                graphContainer.style.marginBottom = "20px";
                graphContainer.style.backgroundColor = "#ffffff"; // Vis network usually looks better on white
                display.appendChild(graphContainer);
                
                try {
                    var parsedData = vis.parseDOTNetwork(treeObj.dot);
                    var dataSet = {
                        nodes: parsedData.nodes,
                        edges: parsedData.edges
                    };
                    var options = {
                        layout: {
                            hierarchical: {
                                direction: "UD",
                                sortMethod: "directed"
                            }
                        },
                        physics: false
                    };
                    new vis.Network(graphContainer, dataSet, options);
                } catch(err) {
                    graphContainer.textContent = "Error rendering Graphviz DOT: " + err;
                    graphContainer.style.color = "red";
                }
            });
            btn.textContent = "Fetch ARF Trees";
        } else {
            display.innerHTML = `<p style="color: red;">${data.message || 'No trees found.'}</p>`;
            btn.textContent = "Fetch ARF Trees";
        }
    } catch (e) {
        display.innerHTML = `<p style="color: red;">Error: ${e}</p>`;
        btn.textContent = "Fetch ARF Trees";
    }
});

// Phase 3: Behavior Models
let pmPollInterval;
let fsmPollInterval;

async function checkStatus(endpoint) {
    try {
        const res = await fetch(endpoint);
        const data = await res.json();
        return data;
    } catch (e) {
        console.error(e);
        return {status: "error", error: String(e)};
    }
}

document.getElementById('btn-run-process-mining').addEventListener('click', async () => {
  const log = document.getElementById('behavior-log');
  const noiseInput = document.getElementById('noise-threshold-input');
  const noiseThreshold = noiseInput ? parseFloat(noiseInput.value) : 0.2;
  
  log.innerHTML = "Starting PM4Py Inductive Miner...";
  try {
    fetch('/api/behavior-model/process-mining', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ noise_threshold: noiseThreshold })
    });
    
    if (pmPollInterval) clearInterval(pmPollInterval);
    pmPollInterval = setInterval(async () => {
        const data = await checkStatus('/api/behavior-model/process-mining/status');
        if (data.status === "running") {
            log.innerHTML = `Mining Process Model... Processed ${data.processed} out of ${data.total}`;
        } else if (data.status === "completed") {
            clearInterval(pmPollInterval);
            log.innerHTML = "Process Mining Complete! Loading PM4Py model...";
            await loadExistingBehaviorModels();
        } else if (data.status === "error") {
            clearInterval(pmPollInterval);
            log.innerHTML = "Error: " + data.error;
        }
    }, 1000);
  } catch (err) {
    log.innerHTML = "Error starting: " + err;
  }
});

document.getElementById('btn-run-agent-fsm').addEventListener('click', async () => {
    const log = document.getElementById('behavior-log');
    const minSupportInput = document.getElementById('min-support-input');
    const minSupport = minSupportInput ? parseFloat(minSupportInput.value) : 0.05;
    
    log.innerHTML = "Starting Causal Agent FSM Generation...";
    try {
        fetch('/api/behavior-model/agent-fsm', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ min_support: minSupport })
        });
        
        if (fsmPollInterval) clearInterval(fsmPollInterval);
        fsmPollInterval = setInterval(async () => {
            const data = await checkStatus('/api/behavior-model/agent-fsm/status');
            if (data.status === "running") {
                if (data.stage) {
                    log.innerHTML = `Generating Agent FSMs... Processing ${data.stage} (${data.processed + 1} of ${data.total})`;
                } else {
                    log.innerHTML = "Generating Agent FSMs with causal logic...";
                }
            } else if (data.status === "completed") {
                clearInterval(fsmPollInterval);
                log.innerHTML = "Agent FSM Generation Complete! Loading models...";
                await loadExistingBehaviorModels();
                // Select first stage automatically if empty
                if (document.getElementById('stage-select').value) {
                    renderFSMGraph(document.getElementById('stage-select').value);
                }
            } else if (data.status === "error") {
                clearInterval(fsmPollInterval);
                log.innerHTML = "Error: " + data.error;
            }
        }, 1000);
    } catch (err) {
        log.innerHTML = "Error starting: " + err;
    }
});

// Render PM4Py Graph automatically when models load
function renderPMGraph() {
    if (!pmModel || !pmModel.places) return;
    const nodes = [];
    const edges = pmModel.arcs.map(a => ({ from: a.source, to: a.target, color: '#3b82f6' }));
    
    pmModel.places.forEach(p => {
        nodes.push({ id: p.id, label: '', shape: 'circle', color: '#10b981', size: 10 });
    });
    
    pmModel.transitions.forEach(t => {
        nodes.push({ id: t.id, label: t.label, shape: 'box', color: '#60a5fa', font: { color: 'white' } });
    });
    
    if (nodes.length > 300) {
        document.getElementById('behavior-log').innerHTML = "Displayed mathematical formulation for Process Mining Petri Net (too dense).";
        document.getElementById('process-graph').innerHTML = `
          <div style="padding: 20px; font-family: monospace; overflow-y: auto; max-height: 100%; color: #34d399;">
            <h4 style="color: white; margin-top: 0;">Mathematical Formulation (PM4Py Inductive Miner Petri Net)</h4>
            <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 15px;">Graph rendering disabled because there are ${nodes.length} nodes (too dense to draw).</p>
            <div><strong>Places & Sensor Bounds (${pmModel.places.length}):</strong><br/> ${pmModel.places.map(p => p.id + " (Bounds: " + (p.sensor_ranges || "None") + ")").join('<br/><br/>')}</div>
            <br/>
            <div><strong>Transitions (${pmModel.transitions.length}):</strong><br/> ${pmModel.transitions.map(t => t.id + " => " + t.label).join('<br/>')}</div>
            <br/>
            <div><strong>Arcs (${pmModel.arcs.length}):</strong><br/> ${pmModel.arcs.map(a => a.source + " &rarr; " + a.target).join('<br/>')}</div>
          </div>
        `;
    } else {
        networkProcess = drawGraph('process-graph', nodes, edges, {
          layout: { hierarchical: { enabled: true, direction: "LR", sortMethod: "directed" } },
          physics: { hierarchicalRepulsion: { nodeDistance: 200, springLength: 200 } }
        });
    }
}

function populateStageDropdown() {
  const select = document.getElementById('stage-select');
  select.innerHTML = '<option value="">Select Stage...</option>';
  let firstStage = Object.keys(behaviorModels)[0];
  for (const stage in behaviorModels) {
    const opt = document.createElement('option');
    opt.value = stage;
    opt.textContent = stage.replace('_', ' ');
    select.appendChild(opt);
  }
  
  if (behaviorModels[firstStage]) {
    select.value = firstStage;
    renderFSMGraph(firstStage);
  }
}

function renderFSMGraph(stage) {
  const fsm = behaviorModels[stage];
  if (!fsm) return;
  
  const stageColors = {
    "Stage 1": "#ef4444",
    "Stage 2": "#f97316",
    "Stage 3": "#eab308",
    "Stage 4": "#22c55e",
    "Stage 5": "#3b82f6",
    "Stage 6": "#8b5cf6",
    "Unknown": "#64748b"
  };

  const nodes = fsm.nodes.map(n => {
    const isActuator = stage === "Global_System" ? n.type === 'actuator' : n.group === 'actuator';
    const nodeColor = stage === "Global_System" ? (stageColors[n.group] || stageColors["Unknown"]) : (isActuator ? '#4CAF50' : '#2196F3');

    return {
      id: n.id,
      label: n.id,
      color: nodeColor,
      font: { color: 'white' },
      shape: isActuator ? 'box' : 'ellipse'
    };
  });
  
  const edges = fsm.edges.map(e => ({
    from: e.source,
    to: e.target,
    label: e.label || "",
    arrows: 'to',
    font: { align: 'middle' }
  }));
  
  if (nodes.length > 200) {
      document.getElementById('behavior-log').innerHTML = `Displayed text formulation for Agent FSM for ${stage} (too dense).`;
      document.getElementById('fsm-graph').innerHTML = `
        <div style="padding: 20px; font-family: monospace; overflow-y: auto; max-height: 100%; color: #8b5cf6;">
          <h4 style="color: white; margin-top: 0;">FSM Formulation</h4>
          <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 15px;">Graph rendering disabled because there are ${nodes.length} states (too dense to draw).</p>
          <div><strong>States (${fsm.nodes.length}):</strong><br/> ${fsm.nodes.map(n => n.id).join('<br/><br/>')}</div>
          <br/>
          <div><strong>Transitions (${fsm.edges.length}):</strong><br/> ${fsm.edges.map(e => e.source + " &rarr; " + e.target + " (Weight: " + e.weight + (e.label ? " | " + e.label : "") + ")").join('<br/><br/>')}</div>
        </div>
      `;
  } else {
      document.getElementById('behavior-log').innerHTML = `Rendered Agent Causal Graph for ${stage}.`;
      networkFSM = drawGraph('fsm-graph', nodes, edges, {
        layout: { hierarchical: { enabled: true, direction: "LR", sortMethod: "directed" } },
        physics: { hierarchicalRepulsion: { nodeDistance: 250 } }
      });
  }
}

document.getElementById('stage-select').addEventListener('change', (e) => {
  const stage = e.target.value;
  if (!stage || !behaviorModels[stage]) return;
  // Automatically render the FSM graph when stage changes
  renderFSMGraph(stage);
});

async function loadExistingBehaviorModels() {
    try {
        const res = await fetch('/api/behavior-model');
        const data = await res.json();
        
        let loaded = false;
        if (data && Object.keys(data.fsms || {}).length > 0) {
            behaviorModels = data.fsms;
            populateStageDropdown();
            loaded = true;
        }
        
        if (data && data.pm_model && data.pm_model.places) {
            pmModel = data.pm_model;
            renderPMGraph();
            loaded = true;
        }
        
        if (loaded) {
            document.getElementById('behavior-log').innerHTML = "Existing Behavior Models loaded.";
        }
    } catch (e) {
        console.error("Failed to load behavior models on startup", e);
    }
}
loadExistingBehaviorModels();

// Phase 4: SysML Comparison
document.getElementById('btn-run-sysml').addEventListener('click', async () => {
  try {
    const res = await fetch('/api/sysml', { method: 'POST' });
    const result = await res.json();
    if (result.status === "success") {
      const data = result.data;
      
      // Update Table
      const tbody = document.querySelector('#metrics-table tbody');
      tbody.innerHTML = '';
      for (const [method, metrics] of Object.entries(data.metrics)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${method.replace('_', ' ')}</td>
          <td>${metrics.Precision}</td>
          <td>${metrics.Recall}</td>
          <td>${metrics['F1-Score']}</td>
          <td>${metrics.MTTD}</td>
          <td>${metrics.Model_Complexity_Index}</td>
        `;
        tbody.appendChild(tr);
      }
      
      
      // Update SysML Output
      let sysmlData = data.sysml;
      let updateSysmlView = async () => {
          let sel = document.getElementById('sysml-source-select').value;
          let textDisplayBox = document.getElementById('sysml-code-display');
          let visualDisplayBox = document.getElementById('sysml-visual-display');
          
          let selectedData = sysmlData[sel];
          let textData = selectedData && selectedData.text ? selectedData.text : (typeof selectedData === 'string' ? selectedData : "No Data Available");
          let mermaidData = selectedData && selectedData.mermaid ? selectedData.mermaid : "";
          let visData = selectedData && selectedData.vis_data ? selectedData.vis_data : null;
          
          if (textDisplayBox) {
              textDisplayBox.textContent = textData;
          }
          
          if (visualDisplayBox) {
              if (visData && visData.nodes.length > 0) {
                  visualDisplayBox.innerHTML = '<div id="sysml-vis-container" style="width: 100%; height: 600px;"></div>';
                  let container = document.getElementById('sysml-vis-container');
                  let data = {
                      nodes: new vis.DataSet(visData.nodes),
                      edges: new vis.DataSet(visData.edges)
                  };
                  let options = {
                      physics: { stabilization: false, barnesHut: { springLength: 100 } },
                      edges: { arrows: 'to', smooth: { type: 'continuous' } },
                      nodes: { shape: 'box', font: { size: 14 } }
                  };
                  new vis.Network(container, data, options);
              } else if (mermaidData && mermaidData.trim().length > 0) {
                  try {
                      visualDisplayBox.innerHTML = '<div class="mermaid-container"><div class="mermaid">' + mermaidData + '</div></div>';
                      // re-render using mermaid API
                      await mermaid.run({ nodes: visualDisplayBox.querySelectorAll('.mermaid') });
                  } catch (e) {
                      console.error("Mermaid Render Error:", e);
                      let errMsg = e.message ? e.message : JSON.stringify(e);
                      visualDisplayBox.innerHTML = '<div style="color: red; padding: 1rem;">Error rendering diagram:<br><pre>' + errMsg + '</pre></div>';
                  }
              } else {
                  visualDisplayBox.innerHTML = '<span style="color: #94a3b8;">No visual diagram available for this selection.</span>';
              }
          }
      };
      updateSysmlView();
      document.getElementById('sysml-source-select').addEventListener('change', updateSysmlView);
    }
  } catch (err) {
    alert("Error fetching SysML: " + err);
  }
});

// Phase 5: Anomaly Replay
document.getElementById('replay-dataset').addEventListener('change', (e) => {
    const warning = document.getElementById('dataset-warning');
    if (e.target.value === 'attack.csv') {
        warning.style.display = 'block';
    } else {
        warning.style.display = 'none';
    }
});

let ws = null;
let totalAttacksDetected = 0;

function resetScoreboard() {
    totalAttacksDetected = 0;
    const elCounter = document.getElementById('attack-counter');
    if (elCounter) elCounter.textContent = "0";
    
    const models = ['bounds', 'physical', 'surrogate', 'figs', 'xgboost', 'pm', 'agent-fsm', 'ensemble'];
    models.forEach(m => {
        const elTp = document.getElementById(`${m}-tp`);
        if (elTp) elTp.textContent = "0";
        const elFp = document.getElementById(`${m}-fp`);
        if (elFp) elFp.textContent = "0";
        const elAcc = document.getElementById(`${m}-acc`);
        if (elAcc) elAcc.textContent = "0.0%";
        const elDelay = document.getElementById(`${m}-delay`);
        if (elDelay) elDelay.textContent = "0";
        const elUniq = document.getElementById(`${m}-unique`);
        if (elUniq) elUniq.textContent = "0";
    });
}

function handleReplayWS(data) {
    const log = document.getElementById('replay-log');
    if (data.index === "DATASET_INFO") {
        const dsInfo = document.getElementById('dataset-info');
        if (dsInfo) dsInfo.style.display = 'block';
        const nameLbl = document.getElementById('dataset-name-label');
        if (nameLbl) nameLbl.textContent = data.dataset_name;
        
        const totalRowsSpan = document.getElementById('total-rows-scanned');
        if (totalRowsSpan) {
            totalRowsSpan.dataset.total = data.total_rows;
            totalRowsSpan.textContent = `Rows Scanned: 0 / ${data.total_rows.toLocaleString()}`;
        }
        
        const attackCounterTotalEl = document.getElementById('attack-counter-total');
        if (attackCounterTotalEl) {
            attackCounterTotalEl.textContent = data.total_attacks;
        }
        
        const replayDsName = document.getElementById('replay-dataset-name');
        if (replayDsName) replayDsName.textContent = data.dataset_name;
        
        log.innerHTML = `<div style="color: #60a5fa; font-weight: bold; margin-bottom: 10px;">Execution started for ${data.dataset_name}...</div>`;
        return;
    }
    if (data.index === "ERROR") {
        log.innerHTML += `<div style="color: #ef4444; font-weight: bold; margin-top: 10px; margin-bottom: 10px;">${data.message}</div>`;
        return;
    }
    if (data.index === "DONE") {
        log.innerHTML += `<div style="color: #10b981; font-weight: bold; margin-top: 10px; margin-bottom: 10px;">${data.message}</div>`;
        return;
    }

    if (data.counts) {
        const models = ['bounds', 'physical', 'surrogate', 'figs', 'xgboost', 'pm', 'agent-fsm', 'ensemble'];
        models.forEach(m => {
            const stats = data.counts[m] || {TP:0, FP:0, Unique:0, DelaySum:0};
            const prefix = m === 'bounds' ? 'sab' : m;
            
            const elTp = document.getElementById(`${prefix}-tp`);
            if (elTp) elTp.textContent = stats.TP;
            
            const elFp = document.getElementById(`${prefix}-fp`);
            if (elFp) elFp.textContent = stats.FP;
            
            const elUniq = document.getElementById(`${prefix}-unique`);
            if (elUniq) elUniq.textContent = stats.Unique;
            
            const elDelay = document.getElementById(`${prefix}-delay`);
            if (elDelay) {
                if (stats.TP > 0) {
                    const avgDelay = (stats.DelaySum / stats.TP).toFixed(0);
                    elDelay.textContent = avgDelay;
                } else {
                    elDelay.textContent = "0";
                }
            }
            
            const elAcc = document.getElementById(`${prefix}-acc`);
            if (elAcc) {
                if ((data.total_attacks || 0) > 0) {
                    const acc = ((stats.TP / data.total_attacks) * 100).toFixed(1);
                    elAcc.textContent = acc + "%";
                } else {
                    elAcc.textContent = "0.0%";
                }
            }
        });
    }
    if (data.total_attacks !== undefined) {
        document.getElementById('score-total-attacks').textContent = data.total_attacks;
    }
    if (data.total_normal !== undefined) {
        document.getElementById('score-total-normal').textContent = data.total_normal;
        
        const totalRowsSpan = document.getElementById('row-counter');
        if (totalRowsSpan && totalRowsSpan.dataset.total) {
            totalRowsSpan.textContent = `Rows Scanned: ${parseInt(data.index).toLocaleString()} / ${parseInt(totalRowsSpan.dataset.total).toLocaleString()}`;
        }
    }
    
    if (data.context) {
        if (data.context.sab) {
            const elTax = document.getElementById('sab-taxonomy');
            if (elTax) {
                let sabAlarm = null;
                if (data.batch_alarms) {
                    data.batch_alarms.forEach(ba => {
                        ba.alarms.forEach(a => {
                            if (a.type.includes("SAB") || a.type.includes("Bounds")) sabAlarm = a;
                        });
                    });
                }
                if (sabAlarm) {
                    if (elTax.innerHTML.includes("All Sensor Bounds Nominal")) {
                        elTax.innerHTML = "";
                    }
                    if (!elTax.innerHTML.includes(sabAlarm.message)) {
                        elTax.insertAdjacentHTML('afterbegin', `<div style="color: #ef4444; font-weight: bold; margin-bottom: 5px;">🚨 ${sabAlarm.message}</div>`);
                    }
                } else if (data.message === "All Nominal" || data.attack_state === "DISAPPEARED") {
                    elTax.innerHTML = `✅ All Sensor Bounds Nominal`;
                }
            }
        }
        if (data.context.invariants) {
            const invList = document.getElementById('invariants-display-list');
            if (invList) {
                if (data.context.invariants.length > 0) {
                    let broken = data.context.invariants.filter(inv => !inv.balanced);
                    if (broken.length === 0) {
                        invList.innerHTML = `<div style="color: #10b981; font-weight: bold; padding: 10px; background: rgba(16, 185, 129, 0.1); border-radius: 4px;">✅ All ${data.context.invariants.length} Invariants Nominal</div>`;
                    } else {
                        let h = `<div style="color: #ef4444; font-weight: bold; margin-bottom: 10px;">❌ ${broken.length} Physical Invariant(s) Violated:</div>`;
                        broken.forEach(inv => {
                            h += `<div style="margin-bottom: 8px; border-left: 3px solid #ef4444; padding-left: 10px; background: rgba(239, 68, 68, 0.1); padding: 5px 10px;">
                                <span style="color: #fca5a5; font-weight: bold;">${inv.equation}</span><br>
                                <span style="font-size: 0.95em; color: #cbd5e1;">Residual: <span style="color: #ef4444; font-weight: bold;">${inv.residual.toFixed(2)}</span> (Threshold: ${inv.threshold.toFixed(2)})</span>
                            </div>`;
                        });
                        invList.innerHTML = h;
                    }
                } else {
                    invList.innerHTML = "No active invariants evaluated.";
                }
            }
        }
    }
    
    let logHtml = "";
    
    if (data.attack_state === "DISAPPEARED") {
        logHtml += `<div style="margin-top: 15px; margin-bottom: 15px; padding: 10px; background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981;">
            <div style="font-weight: bold; font-size: 1.1rem; color: #10b981;">✅ [GROUND TRUTH: ATTACK FLOW TERMINATED]</div>
            <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 5px;">Row ${data.index}: Official dataset labels indicate return to nominal operation.</div>
        </div>`;
    } else if (data.is_alarm && data.attack_state === "DETECTED") {
        totalAttacksDetected++;
        const attackCounterEl = document.getElementById('attack-counter');
        if (attackCounterEl) attackCounterEl.textContent = totalAttacksDetected;

        logHtml += `<div style="margin-top: 15px; margin-bottom: 15px; padding: 10px; background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="font-weight: bold; font-size: 1.1rem; color: #ef4444;">🚨 [GROUND TRUTH: ATTACK FLOW STARTED]</div>
            <div style="font-size: 0.95rem; color: #f87171; margin-top: 5px;">Official dataset labels indicate attack sequence begins at Row ${data.index}</div>
        </div>`;
    } else if (data.is_alarm && data.attack_state === "UPDATE") {
        if (data.batch_alarms && data.batch_alarms.length > 0) {
            logHtml += `<div style="margin-top: 5px; margin-bottom: 15px; padding: 10px; background: rgba(0, 0, 0, 0.2); border-left: 2px solid #8b5cf6;">`;
            data.batch_alarms.forEach(row_data => {
                const modelsList = row_data.alarms ? [...new Set(row_data.alarms.map(a => a.type))].join(', ') : "Unknown Model";
                logHtml += `<div style="font-size: 0.85rem; color: #a8a29e; margin-bottom: 5px;">Alarms at Row <span style="color: #fff;">${row_data.row_index}</span> by: <span style="color: #8b5cf6; font-weight: bold;">${modelsList}</span></div>`;
                
                if (row_data.components && row_data.components.length > 0) {
                    row_data.components.forEach(comp => {
                        logHtml += `<div style="color: #cbd5e1; font-size: 0.95rem; margin-left: 10px; padding: 2px 0;">• ${comp}</div>`;
                    });
                } else if (row_data.alarms) {
                    row_data.alarms.forEach(alarm => {
                        logHtml += `<div style="color: #cbd5e1; font-size: 0.95rem; margin-left: 10px; padding: 2px 0;">• ${alarm.message}</div>`;
                    });
                }
            });
            logHtml += `</div>`;
        }
    }
    
    if (logHtml !== "") {
        log.insertAdjacentHTML('beforeend', logHtml);
        
        // Prevent infinite DOM growth and browser layout thrashing freeze
        while (log.children.length > 500) {
            log.removeChild(log.firstChild);
        }
        
        log.scrollTop = log.scrollHeight; // Auto-scroll to the newest message at the bottom
    }
}

document.getElementById('btn-run-streaming').addEventListener('click', () => {
    if (ws) ws.close();
    ws = new WebSocket(`ws://${window.location.host}/ws/replay`);
    
    ws.onopen = () => {
        document.getElementById('replay-log').innerHTML = "Connected. Starting Parallel Streaming Validation...";
        resetScoreboard();
        
        const dataset = document.getElementById('replay-dataset').value;
        const activeModels = [];
        if (document.getElementById('chk-bounds').checked) activeModels.push('sab');
        if (document.getElementById('chk-physical') && document.getElementById('chk-physical').checked) activeModels.push('physical');
        if (document.getElementById('chk-pm').checked) activeModels.push('pm');
        if (document.getElementById('chk-arf').checked) activeModels.push('surrogate');
        if (document.getElementById('chk-figs').checked) activeModels.push('figs');
        if (document.getElementById('chk-xgboost') && document.getElementById('chk-xgboost').checked) activeModels.push('xgboost');
        if (document.getElementById('chk-agent-fsm-stream') && document.getElementById('chk-agent-fsm-stream').checked) activeModels.push('agent-fsm');
        
        if (activeModels.length === 0) {
            alert("Please select at least one Streaming Method.");
            ws.close();
            return;
        }
        
        ws.send("START_STREAMING:" + dataset + ":" + activeModels.join(","));
    };
    
    ws.onmessage = (event) => {
        try {
            handleReplayWS(JSON.parse(event.data));
        } catch(e) {
            console.error(e);
        } finally {
            const log = document.getElementById('replay-log');
            while (log && log.children.length > 50) log.removeChild(log.lastChild);
        }
    };
    ws.onclose = () => {
        document.getElementById('replay-log').insertAdjacentHTML('afterbegin', "<div style='color: #fca5a5;'>Streaming Validation Completed / Closed.</div>");
    };
});

document.getElementById('btn-run-preprocessed').addEventListener('click', () => {
    if (ws) ws.close();
    ws = new WebSocket(`ws://${window.location.host}/ws/replay`);
    
    ws.onopen = () => {
        document.getElementById('replay-log').innerHTML = "Connected. Preprocessing dataset and running Agent FSM...";
        resetScoreboard();
        
        const dataset = document.getElementById('replay-dataset').value;
        const activeModels = [];
        if (document.getElementById('chk-agent-fsm').checked) activeModels.push('agent-fsm');
        
        if (activeModels.length === 0) {
            alert("Please select the Agent FSM to run preprocessed validation.");
            ws.close();
            return;
        }
        
        ws.send("START_PREPROCESSED:" + dataset + ":" + activeModels.join(","));
    };
    
    ws.onmessage = (event) => {
        try {
            handleReplayWS(JSON.parse(event.data));
        } catch(e) {
            console.error(e);
        } finally {
            const log = document.getElementById('replay-log');
            while (log && log.children.length > 50) log.removeChild(log.lastChild);
        }
    };
    ws.onclose = () => {
        document.getElementById('replay-log').insertAdjacentHTML('afterbegin', "<div style='color: #fca5a5;'>Preprocessed Validation Completed / Closed.</div>");
    };
});

// System Dynamics Tab
document.getElementById('btn-load-dynamics').addEventListener('click', async () => {
    const kInput = document.getElementById('top-k-links-input');
    const topK = kInput ? parseInt(kInput.value, 10) : 5;
    
    const invDisplay = document.getElementById('physical-invariants-display');
    const graphContainer = document.getElementById('causal-graph-display');
    
    invDisplay.innerHTML = "Loading...";
    graphContainer.innerHTML = "Loading...";
    
    try {
        const invRes = await fetch('/api/dynamics/invariants');
        const invData = await invRes.json();
        
        let invHtml = `<table class="sleek-table" style="width: 100%; font-size: 0.9em; text-align: left;">
            <thead><tr><th>Target Tank</th><th>Equation (Mass Balance)</th><th>Tolerance (ε)</th></tr></thead>
            <tbody>`;
        
        for (const [tank, data] of Object.entries(invData)) {
            if (!data.coefficients) continue;
            
            let eqTerms = [];
            for (const [sensor, coef] of Object.entries(data.coefficients)) {
                if (Math.abs(coef) > 0.01) {
                    eqTerms.push(`${coef.toFixed(2)}*${sensor}`);
                }
            }
            let eqStr = `Δ${tank} ≈ ` + eqTerms.join(" + ");
            if (data.intercept) {
                eqStr += ` + ${data.intercept.toFixed(2)}`;
            }
            eqStr = eqStr.replace("+ -", "- "); // clean up negative terms
            
            let eps = data.epsilon ? data.epsilon.toFixed(2) : "N/A";
            
            invHtml += `<tr><td style="font-weight:bold; color:#60a5fa;">${tank}</td><td style="font-family:monospace; white-space: normal; word-break: break-all;">${eqStr}</td><td style="color:#10b981;">±${eps}</td></tr>`;
        }
        invHtml += `</tbody></table>`;
        invDisplay.innerHTML = invHtml;
        
        const causalRes = await fetch('/api/dynamics/causal');
        const causalData = await causalRes.json();
        
        const nodes = [];
        const edges = [];
        const nodeSet = new Set();
        
        for (const [target, data] of Object.entries(causalData)) {
            if (!nodeSet.has(target)) {
                nodes.push({ id: target, label: target, shape: 'box', color: '#8b5cf6', font: {color: 'white'} });
                nodeSet.add(target);
            }
            
            if (data.influencers) {
                let sortedInf = Object.entries(data.influencers).sort((a,b) => b[1] - a[1]);
                let selectedInf = sortedInf.slice(0, topK);
                for (const [inf, score] of selectedInf) {
                    if (!nodeSet.has(inf)) {
                        nodes.push({ id: inf, label: inf, shape: 'ellipse', color: '#3b82f6', font: {color: 'white'} });
                        nodeSet.add(inf);
                    }
                    edges.push({
                        from: inf,
                        to: target,
                        label: score.toFixed(2),
                        font: {align: 'middle', size: 10, color: '#94a3b8'},
                        arrows: 'to',
                        color: {color: '#64748b'}
                    });
                }
            }
        }
        
        drawGraph('causal-graph-display', nodes, edges, {
            layout: { hierarchical: { enabled: true, direction: "UD", sortMethod: "directed" } },
            physics: { hierarchicalRepulsion: { nodeDistance: 150 } }
        });
        
    } catch (e) {
        invDisplay.innerHTML = "Error loading dynamics: " + e;
        graphContainer.innerHTML = "Error loading dynamics: " + e;
    }
});

// SAB Models Explorer Logic
let sabModelsData = null;

async function loadSABModels() {
    try {
        const res = await fetch('/api/sab_models');
        sabModelsData = await res.json();
        
        const select = document.getElementById('sab-situation-select');
        const stats = document.getElementById('sab-situation-stats');
        if (!select || !sabModelsData.frequent_states) return;
        
        select.innerHTML = '';
        
        // Convert to array and sort by count descending
        const states = Object.entries(sabModelsData.frequent_states)
            .map(([sit, data]) => ({ situation: sit, count: data.count, ...data }))
            .sort((a, b) => b.count - a.count);
            
        if (states.length === 0) {
            select.innerHTML = '<option value="">No SAB Models Found</option>';
            return;
        }
        
        states.forEach(state => {
            const opt = document.createElement('option');
            opt.value = state.situation;
            opt.textContent = `State: ${state.situation} (Occurrences: ${state.count})`;
            select.appendChild(opt);
        });
        
        // Trigger initial render
        select.addEventListener('change', renderSABSituation);
        renderSABSituation();
        
    } catch (e) {
        console.error("Failed to load SAB models", e);
        const select = document.getElementById('sab-situation-select');
        if (select) select.innerHTML = '<option value="">Error loading models</option>';
    }
}

function renderSABSituation() {
    const select = document.getElementById('sab-situation-select');
    const situation = select.value;
    if (!situation || !sabModelsData || !sabModelsData.frequent_states[situation]) return;
    
    const data = sabModelsData.frequent_states[situation];
    
    // Update Stats
    const stats = document.getElementById('sab-situation-stats');
    if (stats) {
        stats.textContent = `Total Training Rows Observed: ${data.count}`;
    }
    
    // Render Sensors (FIT, PIT, AIT)
    const tSensors = document.getElementById('sab-sensors-tbody');
    if (tSensors) {
        tSensors.innerHTML = '';
        if (data.sensors && Object.keys(data.sensors).length > 0) {
            for (const [sensor, bounds] of Object.entries(data.sensors)) {
                const tr = document.createElement('tr');
                let boundsHtml = '';
                if (bounds.type === 'clusters') {
                    boundsHtml = bounds.clusters.map(c => `[${c[0].toFixed(2)} - ${c[1].toFixed(2)}]`).join(', ');
                } else if (bounds.type === 'flexible') {
                    boundsHtml = `Mode: ${bounds.mode.toFixed(2)} &plusmn; ${bounds.flexibility.toFixed(2)} (Min: ${bounds.min.toFixed(2)}, Max: ${bounds.max.toFixed(2)})`;
                }
                
                tr.innerHTML = `
                    <td style="font-weight: bold; color: #60a5fa;">${sensor}</td>
                    <td>${bounds.type}</td>
                    <td style="color: #cbd5e1;">${boundsHtml}</td>
                `;
                tSensors.appendChild(tr);
            }
        } else {
            tSensors.innerHTML = '<tr><td colspan="3" style="text-align: center; color: #94a3b8;">No continuous sensors in this state</td></tr>';
        }
    }
    
    // Render LITs (Fluctuations)
    const tLits = document.getElementById('sab-lits-tbody');
    if (tLits) {
        tLits.innerHTML = '';
        if (data.lits && Object.keys(data.lits).length > 0) {
            for (const [lit, bounds] of Object.entries(data.lits)) {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight: bold; color: #34d399;">${lit}</td>
                    <td style="color: #cbd5e1;">${bounds.min_delta.toFixed(3)}</td>
                    <td style="color: #cbd5e1;">${bounds.max_delta.toFixed(3)}</td>
                `;
                tLits.appendChild(tr);
            }
        } else {
            tLits.innerHTML = '<tr><td colspan="3" style="text-align: center; color: #94a3b8;">No LITs found</td></tr>';
        }
    }
}

document.getElementById('btn-reload-sab')?.addEventListener('click', loadSABModels);

// Load models initially
loadSABModels();

