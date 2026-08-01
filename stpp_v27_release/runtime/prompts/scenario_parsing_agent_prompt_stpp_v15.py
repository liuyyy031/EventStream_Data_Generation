"""STPP v15 parser prompt for exclusive route and day-branch preservation."""

from prompts.scenario_parsing_agent_prompt_stpp_v14 import (
    STPP_V14_SCENARIO_PARSING_PROMPT,
)


_V15_RULES = r"""
STPP V15 EXECUTABLE ROUTE PARSING RULES
23. Treat event_full_path/path as an exclusive executable route. Never append
    another graph edge merely because it exists in EDGES.
24. Preserve exactly one event_id family per demand_source. Its schedule_id
    branches must share the same path and collectively partition days 0..6.
25. For each (event_id, schedule_id), emit every path_stage from 0 through
    len(path)-2 exactly once.
26. Preserve stage TIME as the parent activation window and
    destination_arrival_period as the permitted child-arrival window.
27. Do not merge two schedule branches or reuse one schedule_id for different
    day sets. Do not invent an additional route for a demand source.

"""

STPP_V15_SCENARIO_PARSING_PROMPT = STPP_V14_SCENARIO_PARSING_PROMPT.replace(
    "SCENARIO TO PARSE\n{scenario}",
    _V15_RULES + "SCENARIO TO PARSE\n{scenario}",
)
