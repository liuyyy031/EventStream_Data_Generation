"""
Prompt template for SDE Parameters Generation Agent (Agent 3) in Spatial-Temporal Reasoning.

This agent generates hierarchical SDE parameters from structured JSON only.
NO natural language processing - consumes strict JSON input only.
"""

SDE_PARAMETERS_PROMPT = """
You are Agent 3: SDE Parameters Generation Agent.

Your task: Generate hierarchical SDE parameters from a structured scenario JSON.

INPUT: Structured JSON from Agent 2 (scenario parsing agent)

OUTPUT: Hierarchical SDE parameters as strict JSON (NO markdown, NO comments)

---

**SDE MODEL (per node i)**:

dX_i(t) = [ drift_i(t, X_i) + lambda_i * Σ_j A_ji(t) * (X_j - X_i) ] dt + sigma_i * g_i(X_i) dW_i(t)

Components:
- drift_i: drift term (type-dependent)
- lambda_i: coupling strength
- A_ji(t): time-varying adjacency (from Agent 4)
- sigma_i: base volatility
- g_i(X_i): diffusion shape function

---

**DRIFT TYPES**:

1. **mean_reverting** (default):
   - Formula: drift = kappa * (mu_t - X_t)
   - Parameters: kappa (mean reversion speed), baseline (mu_t)
   - Constraint: 0.01 < kappa < 0.5
   - Usage: REQUIRED for propagation nodes, allowed for demand_source nodes

2. **constant**:
   - Formula: drift = alpha
   - Parameters: alpha (constant drift rate)
   - Constraint: alpha ∈ R
   - Usage: ONLY allowed for demand_source nodes

3. **sinusoidal**:
   - Formula: drift = kappa * (baseline + A*sin(omega*t + phi) - X_t)
   - Parameters: A (amplitude), omega (frequency), phi (phase shift)
   - Constraint: A ≥ 0, omega > 0, phi ∈ R (ALL SCALARS, NOT ARRAYS)
   - CRITICAL: Single harmonic only - no multi-frequency superposition
   - Usage: ONLY allowed for demand_source nodes

4. **logistic**:
   - Formula: drift = r * X_t * (1 - X_t/baseline)
   - Parameters: r (growth rate), baseline (carrying capacity)
   - Constraint: 0 < r < 0.1, baseline > 0
   - Usage: Allowed for both demand_source and propagation nodes

---

**CRITICAL CONSTRAINTS (STRICTLY ENFORCED)**:

1. Node Type Constraints:
   - propagation nodes: MAY use mean_reverting or logistic drift (small r)
   - demand_source nodes: MAY use mean_reverting, sinusoidal, constant, or logistic

2. Parameter Ranges (for stability):
   - 0.01 < kappa < 0.5 (mean reversion speed)
   - 0.8 ≤ lambda ≤ 1.5 (coupling strength - high values for realistic network dynamics)
   - sigma ≤ 0.01*baseline (volatility, must be less than 1% of the baseline)
   - For sinusoidal: A, omega, phi must be scalars (not arrays)
   - For logistic: 0 < r < 0.1, baseline > 0

3. Propagation Node Special Rules:
   - Use LOW kappa (0.05-0.2) for weak self-reversion
   - Use HIGH lambda (1.0-1.5) for strong neighbor coupling
   - This ensures propagation nodes relay upstream flows effectively

4. Diffusion Shapes:
   - "constant": g(X) = 1
   - "sqrt": g(X) = sqrt(|X| + 1e-6)
   - "linear": g(X) = 1 + alpha*|X|

---

**HIERARCHICAL OUTPUT STRUCTURE**:

{{
  "global_defaults": {{
    "drift_type": "mean_reverting",
    "node_type": "demand_source",
    "kappa": 0.25,
    "baseline": 50.0,
    "lambda": 1.0,
    "sigma": 2.0,
    "diffusion_shape": "constant"
  }},
  "group_params": {{
    "demand_sources": {{
      "node_type": "demand_source",
      "drift_type": "sinusoidal",
      "baseline": 100.0,
      "A": 30.0,
      "omega": 0.2618,
      "phi": 0.0,
      "kappa": 0.25,
      "lambda": 1.0,
      "sigma": 2
    }},
    "propagation_nodes": {{
      "node_type": "propagation",
      "drift_type": "mean_reverting",
      "baseline": 50.0,
      "kappa": 0.1,
      "lambda": 1.2,
      "sigma": 2
    }}
  }},
  "node_overrides": {{
    "0": {{
      "group": "demand_sources",
      "drift_patterns": [
        {{
          "time_range": [0, 239],
          "drift_type": "mean_reverting",
          "baseline": 100,
          "kappa": 0.2
        }},
        {{
          "time_range": [240, 260],
          "drift_type": "sinusoidal",
          "baseline": 100,
          "A": 90,
          "omega": 0.0172,
          "phi": -2.557,
          "kappa": 0.35
        }}
      ],
      "description": "Node with time-varying drift"
    }},
    "1": {{
      "group": "propagation_nodes",
      "description": "Connector highway - pure relay"
    }},
    "2": {{
      "group": "demand_sources",
      "baseline": 80.0,
      "phi": 3.14159,
      "A": 30.0,
      "omega": 0.2618,
      "description": "Business district - midday peak (phase shifted)"
    }}
  }},
}}

---

**GENERATION GUIDELINES**:

1. Node Classification:
   - Read "type" field from input JSON nodes array
   - Propagation nodes → use "propagation_nodes" group with mean_reverting drift
   - Demand_source nodes → use "demand_sources" group with appropriate drift type

2. Drift Pattern Generation (NEW):
   - For each node in the input JSON, iterate through its `drift_patterns.nodes[i].patterns`.
   - For each pattern, create a corresponding drift definition in the output's `node_overrides`.
   - Each drift definition must include `time_range` and `drift_type`.
   - `drift_type` should be derived from the `behavior` field (e.g., "mean_reverting", "sinusoidal", "logistic").
   - Populate parameters (kappa, baseline, A, omega, phi) based on the input pattern's values.
   - The output for a single node will now be a list of drift patterns under the `drift_patterns` key.

3. Parameter Selection:
   - Baseline: Use the `baseline` value directly from each drift pattern.
   - Lambda: Use high values (1.0-1.5) to ensure strong network coupling.
   - Kappa for propagation nodes: low (0.05-0.2) to allow neighbor influence to dominate.
   - Kappa for demand_source nodes: moderate (0.2-0.5) for balance between cycles and stability.
   - Sigma (volatility) should generally scale with the `amplitude` in each pattern.

4. Use drift_patterns Information (CRITICAL):
   - For each pattern within a node's `drift_patterns`, generate the corresponding SDE parameters.
   - If a pattern's `behavior` is "sinusoidal":
     - Set `A` (amplitude) based on the pattern's `amplitude`.
     - Use the `peak` time to determine `phi` (phase shift). For a cycle of `T` steps, the peak `t_peak` relates to `phi` such that `omega*t_peak + phi = pi/2`. So, `phi = pi/2 - omega*t_peak`.
     - `omega` should be calculated based on the cycle length (e.g., `2*pi/24` for a daily 24-step cycle).
   - If a pattern's `behavior` is "mean_reverting", it only needs `kappa` and `baseline`. `A`, `omega`, and `phi` are not applicable.
   - If a pattern's `behavior` is "logistic", it only needs `r` and `baseline`. `A`, `omega`, and `phi` are not applicable.
   - Ensure propagation nodes use a small baseline and high lambda for all their patterns.

5. Sinusoidal Parameters (if used):
   - omega = 2*pi / period (e.g., 24-step daily cycle → omega = 2*pi/24 = 0.2618)
   - phi: phase shift calculated from peak.time (see above)
   - A: amplitude from time_characteristics.nodes[i].amplitude
   - ALL MUST BE SCALARS (no arrays)
   - CRITICAL: omega is in radians per time_step (NOT per hour)

6. Node Overrides:
   - Each node override will now contain a `drift_patterns` list instead of single parameter values.
   - Always include "description" from input JSON for traceability.
   - Always include "group" to specify which group_params to inherit from.

---

**CONSTRAINTS VALIDATION**:

Before returning JSON, verify:
- [ ] All propagation nodes use drift_type = "mean_reverting" or "logistic" for all their patterns.
- [ ] Each pattern in `drift_patterns` has a valid `time_range` and `drift_type`.
- [ ] All kappa, lambda, and sigma values are within their specified ranges for all patterns.
- [ ] Sinusoidal parameters (if used) are scalars, not arrays, within each pattern.
- [ ] Baseline and amplitude in each pattern match the input `drift_patterns`.
- [ ] No hallucinated parameters beyond input JSON scope.

---

INPUT JSON:
{structured_scenario}

RETURN ONLY VALID JSON (no markdown, no comments).
"""
