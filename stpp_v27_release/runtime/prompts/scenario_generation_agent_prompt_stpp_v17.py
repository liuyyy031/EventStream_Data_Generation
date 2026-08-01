"""STPP v17 scenario prompt with explicit edge/path consistency checks."""

from prompts.scenario_generation_agent_prompt_stpp_v16 import (
    STPP_V16_SCENARIO_GENERATION_PROMPT,
)


STPP_V17_SCENARIO_GENERATION_PROMPT = (
    STPP_V16_SCENARIO_GENERATION_PROMPT
    + r"""

STPP V17 EXPLICIT-EDGE FIDELITY CONTRACT
1. Every consecutive pair in every EVENT FULL PATH must appear exactly once in
   the top-level EDGES section. For example, path 2->0->1 requires both EDGE
   2->0 and EDGE 0->1; describing 0->1 only inside EDGE MODULATION is invalid.
2. Never rely on Agent 2 to infer or add a missing top-level edge.
3. Before returning, build these two sets mentally and require equality for
   all event-used edges:
   - explicit top-level EDGE source->target pairs;
   - consecutive source->target pairs from every EVENT FULL PATH.
4. Keep all previously valid v16 route, day-partition, placement, causal-window
   and spatial-layout contracts unchanged.
"""
)
