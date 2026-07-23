"""STPP v17 parser prompt forbidding inferred graph edges."""

from prompts.scenario_parsing_agent_prompt_stpp_v16 import (
    STPP_V16_SCENARIO_PARSING_PROMPT,
)


_V17_RULES = r"""
STPP V17 EDGE-FIDELITY RULES
33. The JSON edges array must be an exact transcription of the top-level EDGES
    section. Do not add an edge merely because EVENT FULL PATH, ARRIVAL PATH or
    EDGE MODULATION mentions it.
34. If a path uses a pair absent from the top-level EDGES section, preserve the
    top-level edge list exactly. The scenario is internally inconsistent and
    Judge 1 must attribute the problem to Agent 1.
35. Before returning JSON, compare the unordered set of (source,target) pairs
    in JSON edges with the explicit top-level EDGE lines. The sets must match.

"""

STPP_V17_SCENARIO_PARSING_PROMPT = STPP_V16_SCENARIO_PARSING_PROMPT.replace(
    "SCENARIO TO PARSE\n{scenario}",
    _V17_RULES + "SCENARIO TO PARSE\n{scenario}",
)
