"""STPP v13 Judge prompt including spatial identifiability checks."""

from prompts.judge_scenario_parsing_prompt_stpp_v12 import (
    STPP_V12_JUDGE_SCENARIO_PARSING_PROMPT,
)


_SPATIAL_CHECKS = r"""
SPATIAL SUPPORT CHECKS
1. spatial_layout contains every node exactly once and every x/y value is a
   finite number in [0,1].
2. All coordinate pairs are unique. Reject duplicate coordinates because
   nearest-node assignment would make the higher-ID node unreachable.
3. Reject a layout whose nodes are so close that a node has no meaningful
   nearest-node region. For ordinary synthetic layouts, pairwise separation
   should be at least 0.05 after normalization.
4. If JSON changes, omits, duplicates, or invents coordinates relative to the
   scenario, assign the error to Agent 2. If the scenario itself declares a
   duplicate/degenerate layout, assign the error to Agent 1.

"""

STPP_V13_JUDGE_SCENARIO_PARSING_PROMPT = (
    STPP_V12_JUDGE_SCENARIO_PARSING_PROMPT.replace(
        "Only genuine blocking problems belong in issues.",
        _SPATIAL_CHECKS + "Only genuine blocking problems belong in issues.",
    )
)
