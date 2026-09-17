"""STPP v14 Judge prompt for node-local lag and schedule-branch validation."""

from prompts.judge_scenario_parsing_prompt_stpp_v13 import (
    STPP_V13_JUDGE_SCENARIO_PARSING_PROMPT,
)


_V14_CHECKS = r"""
STPP V14 NODE-LOCAL ARRIVAL CHECKS
1. For each propagated_variations entry, arrival_path must begin at source and
   end at the node containing that entry. event_full_path may continue farther.
2. cumulative_lag_to_node is calculated only along arrival_path. Do not require
   an intermediate node to use the full event path's lag.
3. For 0->1->2 with lags 2 and 1, NODE 1 has cumulative_lag_to_node=2 and NODE 2
   has cumulative_lag_to_node=3. Requiring NODE 1 to use 3 is an error.
4. time must equal the source peak plus cumulative_lag_to_node in the same
   cyclic or absolute coordinate system.
5. A propagation node baseline has origin=ambient_baseline.

STPP V14 SCHEDULE CHECKS
1. Group modulation stages by (event_id, schedule_id), never by event_id alone.
2. Different DAYS or shifted workday/weekend schedules require different
   schedule_id values.
3. Within each schedule, downstream edge activation starts no earlier than the
   preceding stage start plus the preceding edge lag.
4. destination_arrival_period equals time_period shifted by the current edge
   lag. Do not confuse edge activation with destination arrival.
5. Every demand_source has at least one outgoing edge. If the scenario omits an
   outgoing edge, this is Agent 1's error; if Agent 2 drops it, this is Agent 2's
   error.

"""

STPP_V14_JUDGE_SCENARIO_PARSING_PROMPT = (
    STPP_V13_JUDGE_SCENARIO_PARSING_PROMPT.replace(
        "Only genuine blocking problems belong in issues.",
        _V14_CHECKS + "Only genuine blocking problems belong in issues.",
    )
)
