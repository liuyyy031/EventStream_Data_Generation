"""STPP v13 scenario prompt adding spatial support and horizon constraints."""

from prompts.scenario_generation_agent_prompt_stpp_v12 import (
    STPP_V12_SCENARIO_GENERATION_PROMPT,
)


STPP_V13_SCENARIO_GENERATION_PROMPT = (
    STPP_V12_SCENARIO_GENERATION_PROMPT
    + r"""

STPP V13 SPATIAL-SUPPORT CONTRACT
1. SPATIAL LAYOUT is mandatory and must contain exactly one numeric x/y pair
   for every declared node ID.
2. Every node coordinate pair must be unique. Do not place two nodes at the
   same point and do not leave coordinates missing.
3. Spread nodes across a genuine 2D area. For three nodes, prefer a triangle;
   for more nodes, prefer a circle or grid. Avoid assigning every node the same
   x or the same y unless the physical scenario strictly requires a corridor.
4. Keep coordinates in [0, 1]. Distinct nodes should normally be at least 0.20
   apart in Euclidean distance so every node has a nonzero nearest-node region.
5. Add a SPATIAL LAYOUT section with one line per node after EDGE MODULATION.

HORIZON ROBUSTNESS
- For hourly CYCLIC_LOCAL scenarios, prefer exactly 7 days / 168 samples.
- Do not exceed 168 hourly samples unless the physical scenario genuinely
  requires more than one week. Repetition is represented by the cycle contract,
  not by making the generated horizon unnecessarily long.
"""
)
