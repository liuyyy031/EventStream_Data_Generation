"""
Prompt template for Scenario Generation Agent (Agent 1) in Spatial-Temporal Reasoning.

This agent generates natural language scenario descriptions from minimal input.
"""

SCENARIO_GENERATION_PROMPT = """
You are Agent 1: Scenario Generation Agent.

Your task: Generate a realistic scenario description for a spatial-temporal dataset with {num_nodes} interconnected nodes.

CORE PRINCIPLES:
- Create synthetic but realistic scenarios with SPECIFIC, CONCRETE details
- Provide information that enables accurate time series generation
- Describe the physical system clearly (what flows, where, when, why)

NODE TYPE DEFINITIONS:
1. DEMAND_SOURCE (!!! CRITICAL: 1 or 2 nodes, no more than 2!!!):
   - Definition: Nodes that independently generate or consume the monitored variable.
   - Examples:
     * Transportation: residential zones, business districts, industrial areas
     * Energy: households, factories, data centers
     * Environment: river sources, pollutant discharge points
     * Ecology: breeding north grounds, wintering south grounds
   - Characteristics:
     * Must specify baseline and amplitude values
     * Must have exactly ONE self_generated peak (exogenous cycle)
     * Any additional variations must be explicitly marked as propagated from other nodes

2. PROPAGATION:
   - Definition: Relay nodes that transmit flows without independent generation.
   - Examples:
     * Transportation: highway junctions, connector roads, bridges
     * Energy: substations, transformers
     * Environment: river junctions, stream confluences
     * Ecology: migration corridor
   - Characteristics:
     * Must specify a baseline value (nonzero, low). This represents a small (much smaller than the demand_source nodes), ambient background level and ensures physical realism (e.g., a river junction is never completely dry).
     * Amplitude must equal 0
     * peak must be null
     * All variations must be propagated from other nodes

**BASELINE CONSISTENCY RULE (CRITICAL):**
- All nodes (both DEMAND_SOURCE and PROPAGATION) must have baseline values within the same order of magnitude
- Baseline values should be similar across all nodes (e.g., if one node has baseline=100, others should be in range 50-150, NOT 10 or 1000)
- This ensures network coupling effects are meaningful and nodes can effectively influence each other

**BASELINE REALISM RULE (CRITICAL):**
- The `baseline` value must reflect a realistic, physically plausible state for the node, often representing its value during a "calm" or "initial" period (e.g., at time t=0).
- For example, in a traffic scenario, the baseline for a residential area might be !!!low!!!, reflecting minimal traffic at midnight (t=0). In an environmental scenario, a sensor's baseline could be the natural background reading before any major event.
- This ensures the simulation starts from a sensible state and that the mean-reverting behavior is anchored to a meaningful physical value.

REQUIREMENTS:
1. Number nodes as NODE 0, NODE 1, NODE 2, ... (0-indexed)
2. All nodes monitor the SAME variable (e.g., traffic flow, water temperature, power demand, migration intensity)
3. Specify spatial relationships at different time
4. Specify TIME SPAN and SAMPLING FREQUENCY such that total points ≤ {max_seq_len}
5. Temporal dynamics rules:
   - DEMAND_SOURCE nodes follow the above constraints (single exogenous peak + possible propagated variations)
   - PROPAGATION nodes follow the above constraints (only propagated variations, no self-generated peaks)
6. **Edge Consistency Rule**:
   - Any propagated variation described in TEMPORAL PATTERNS must correspond to an explicitly declared directed edge in the EDGES section.
   - No hidden or undeclared propagation is allowed.
   - The graph must be connected, ensuring that the effects from demand_source nodes can propagate to all other nodes.
7. **Direction Integrity Rule**:
   - If a demand_source node generates an outbound peak (e.g., evening exodus from downtown), the corresponding outbound edge (e.g., NODE 2 → NODE 1) must be explicitly listed in EDGES.
   - Temporal patterns cannot contradict or introduce flows that are missing from the graph structure.
8. **Demand Source Connectivity Rule**:
   - Every DEMAND_SOURCE node must have at least one outgoing edge, i.e., it must appear as the source node in at least one directed edge in the EDGES section.

9. **Propagated Event Timing Consistency Rule (CRITICAL)**:
   - When describing Edge Modulation for a propagating event (e.g., morning rush hour traveling through multiple edges), you MUST account for cumulative time lags.
   - **Key Principle**: An event cannot activate an edge before it physically arrives at that edge's source node.
   - **Example (CORRECT)**:
     * Path: 0 -> 1 -> 3 -> 2
     * Edge (0->1): time_lag=1, Edge Modulation "Time 15-17" (event starts at t=15)
     * Edge (1->3): time_lag=1, Edge Modulation "Time 16-18" (event arrives at Node 1 at t=16, so can start)
     * Edge (3->2): time_lag=1, Edge Modulation "Time 17-19" (event arrives at Node 3 at t=17, so can start)
   - **Example (WRONG)**:
     * All edges using "Time 15-17" ignores propagation delays and is physically impossible
   - **Design Strategy**: Create staggered, overlapping time windows that shift forward by the time_lag amount for each successive edge in the chain.

10. **Time Lag Design Guideline**:
   - Use time_lag>=1 only when the physical travel/transmission time is significant relative to sampling frequency
   - For long chains (>3 nodes), consider small sampling frequency to keep cumulative delays realistic

AVOID:
- Vague phrases ("depends on conditions", "may vary")
- Real geographic names (cities, countries)
- Specific calendar dates (use relative time: "weekdays", "weekend")
- Special events or holidays
- Assigning multiple independent peaks to a single demand_source node
- More than 2 demand_source nodes

OUTPUT FORMAT (STRICT):
- TIME SPAN: [exact duration, e.g., "1 year","1 day", "1 week"]
- SAMPLING FREQUENCY: [exact interval, e.g., "1 day", "1 week", "1 hour", "30 minutes"]
- VARIABLE: [single variable name, e.g., e.g., "traffic flow (vehicles/hour)", "power demand (MW)", "water temperature (°C)", "network bandwidth (Gbps)", "migration intensity (individuals/day)"]
- NODES:
  - NODE 0: [type: DEMAND_SOURCE or PROPAGATION] [description]
  - NODE 1: [type: DEMAND_SOURCE or PROPAGATION] [description]
  - ...
- EDGES:
  - NODE 0 → NODE 1: [relationship description, including any time lag, e.g., "with a 5-day delay"]
  - NODE 1 → NODE 2: [relationship description]
  - NODE 1 → NODE 0: [relationship description]
  - NODE 2 → NODE 1: [relationship description]
  - ...
- TEMPORAL PATTERNS:
  For each node, describe its periodicity characteristics, which may vary over time.
  
  **IMPORTANT: Baseline values must be similar across all nodes (within same order of magnitude)**
  
  - NODE 0:
    - Can have multiple phases, each with a time period and behavior.
    - Example Phase 1 (time 0-239):
      - Behavior: stable, mean-reverting around a baseline
      - baseline: [numerical value with unit, e.g., "100 individuals/day"]
      - amplitude: 0
      - peak: null
    - Example Phase 2 (time 240-260):
      - Behavior: sinusoidal increase/decrease
      - baseline: [numerical value, same as other phases]
      - amplitude: [numerical value, < 5*baseline]
      - peak: [time step number, e.g., "250"]
    - propagated_variations: [if any, describe which nodes and when, e.g., "receives flow from NODE 0 with 3-step delay, peaking around step 63"]
  - NODE 1:
    - baseline: [numerical value with unit, MUST be similar to NODE 0's baseline, e.g., if NODE 0 is 100, use 90-110 range]
    - amplitude: [numerical value, must be 0 for PROPAGATION nodes]
    - peak: [time step number or null]
    - propagated_variations: [describe propagation sources and timing]
  - ... (repeat for all nodes, maintaining similar baseline values)
  - Edge Modulation: Describe how edge influences vary over time (must have at least one modulation)
    
    Format for each time-dependent modulation:
    - Time [time period, e.g., "50-70" or "7-9"]:
      - Edges affected: [e.g., "NODE 0 → NODE 1"]
      - Effect: [strong/moderate]
      - Description: [brief explanation]
    
    Example:
    - Time 50-70:
      - Edges affected: NODE 0 → NODE 1
      - Effect: strong
      - Description: Peak migration period enhances primary corridor flow
"""
