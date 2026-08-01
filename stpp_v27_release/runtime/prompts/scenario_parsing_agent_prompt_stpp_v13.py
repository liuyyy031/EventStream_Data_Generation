"""STPP v13 parser prompt preserving usable spatial support."""

from prompts.scenario_parsing_agent_prompt_stpp_v12 import (
    STPP_V12_SCENARIO_PARSING_PROMPT,
)


_SPATIAL_RULES = r"""
STPP V13 SPATIAL RULES
13. spatial_layout must contain every node ID exactly once with finite numeric
    x and y coordinates in [0,1].
14. Copy the scenario coordinates faithfully. Coordinates must be pairwise
    unique; never fill missing nodes by copying another node's position.
15. Do not collapse a triangle, circle, or grid into a single point or duplicate
    coordinate. Preserve enough separation for every node to own a nonzero
    nearest-node region.

"""

STPP_V13_SCENARIO_PARSING_PROMPT = STPP_V12_SCENARIO_PARSING_PROMPT.replace(
    "SCENARIO TO PARSE\n{scenario}",
    _SPATIAL_RULES + "SCENARIO TO PARSE\n{scenario}",
)
