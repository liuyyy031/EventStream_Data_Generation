"""STPP v17 Judge 1 prompt with explicit-edge evidence requirements."""

from prompts.judge_scenario_parsing_prompt_stpp_v16 import (
    STPP_V16_JUDGE_SCENARIO_PARSING_PROMPT,
)


_V17_CHECKS = r"""
STPP V17 EXPLICIT-EDGE CHECKS
1. Extract three edge sets separately: top-level EDGES, consecutive pairs in
   EVENT FULL PATH declarations, and JSON edges.
2. JSON edges must equal top-level EDGES exactly. A path pair missing from the
   top-level EDGES section is an Agent 1 inconsistency, even when Agent 2 added
   the logically implied edge.
3. Never approve when a path pair is absent from the explicit edge set or when
   Agent 2 added/deleted any explicit edge.
4. Cite the concrete source->target pair in every edge-related issue.

"""

STPP_V17_JUDGE_SCENARIO_PARSING_PROMPT = (
    STPP_V16_JUDGE_SCENARIO_PARSING_PROMPT.replace(
        "Only genuine blocking problems belong in issues.",
        _V17_CHECKS + "Only genuine blocking problems belong in issues.",
    )
)
