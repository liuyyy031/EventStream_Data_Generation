"""STPP v16 parser prompt with deterministic variation placement invariant."""

from prompts.scenario_parsing_agent_prompt_stpp_v15 import (
    STPP_V15_SCENARIO_PARSING_PROMPT,
)


_V16_RULES = r"""
STPP V16 FAIL-CLOSED PARSING RULES
28. A propagated_variations entry belongs only to the drift node whose id is
    arrival_path[-1]. Never place an arrival at Node 1 under Node 0 merely
    because Node 0 is the event source.
29. Deduplicate propagated arrivals by (event_id, arrival_path[-1]). Emit each
    required receiving node exactly once. Do not copy EVT-002 path [2,0] into
    Node 2; it belongs only under Node 0.
30. For every schedule branch and stage k>0, preserve or calculate
    stage[k].time_period equal to stage[k-1].destination_arrival_period.
31. Before returning JSON, perform this checklist:
    - container node id == arrival_path[-1] for every propagated variation;
    - one entry per (event_id, receiving node);
    - every non-origin node on event_full_path has one entry;
    - downstream activation never precedes upstream arrival.
32. If the scenario itself violates the stage recurrence, preserve it exactly
    for fidelity and let the Judge identify Agent 1 as the error source. Do not
    silently invent corrected values while claiming faithful parsing.

"""

STPP_V16_SCENARIO_PARSING_PROMPT = STPP_V15_SCENARIO_PARSING_PROMPT.replace(
    "SCENARIO TO PARSE\n{scenario}",
    _V16_RULES + "SCENARIO TO PARSE\n{scenario}",
)
