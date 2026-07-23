"""STPP v14 parser prompt for node-local paths and schedule IDs."""

from prompts.scenario_parsing_agent_prompt_stpp_v13 import (
    STPP_V13_SCENARIO_PARSING_PROMPT,
)


_V14_RULES = r"""
STPP V14 EVENT PARSING RULES
16. Every demand_source must retain at least one outgoing edge declared by the
    scenario. Do not remove return or outbound edges.
17. A propagated_variations entry must use these exact fields:
    - event_id;
    - time: arrival time at the node containing the entry;
    - source: original event node;
    - arrival_path: list from source ending at the node containing the entry;
    - event_full_path: complete event path;
    - cumulative_lag_to_node: sum of edge lags along arrival_path;
    - origin: propagated;
    - description.
18. Never put a later downstream node in arrival_path. For a NODE 1 arrival in
    0->1->2, arrival_path is [0,1], not [0,1,2].
19. A propagation node's zero-amplitude background pattern must use
    origin="ambient_baseline", not origin="self_generated".
20. Every adjacency modulation must preserve schedule_id. Workday and weekend
    branches use different schedule IDs even when event_id is shared.
21. Preserve edge_activation_window in time_period. Also emit
    destination_arrival_period by shifting time_period by the current edge lag.
22. All patterns in one schedule branch must have identical event_id,
    schedule_id, days, and path.

"""

STPP_V14_SCENARIO_PARSING_PROMPT = STPP_V13_SCENARIO_PARSING_PROMPT.replace(
    '"path": [2, 1, 0], "cumulative_lag": 2,',
    (
        '"arrival_path": [2, 1, 0], '
        '"event_full_path": [2, 1, 0],\n'
        '           "cumulative_lag_to_node": 2,'
    ),
).replace(
    '{"event_id": "morning_commute", "path": [0, 1, 2],\n'
    '       "path_stage": 0, "time_period": "6-8",',
    (
        '{"event_id": "morning_commute", '
        '"schedule_id": "morning_commute_weekday",\n'
        '       "path": [0, 1, 2], "path_stage": 0, '
        '"time_period": "6-8",\n'
        '       "destination_arrival_period": "7-9",'
    ),
).replace(
    "SCENARIO TO PARSE\n{scenario}",
    _V14_RULES + "SCENARIO TO PARSE\n{scenario}",
)
