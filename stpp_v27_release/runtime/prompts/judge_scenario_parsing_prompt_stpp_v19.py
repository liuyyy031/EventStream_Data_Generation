"""STPP v19 Judge 1 prompt for branch-level schedule fidelity."""

from prompts.judge_scenario_parsing_prompt_stpp_v17 import (
    STPP_V17_JUDGE_SCENARIO_PARSING_PROMPT,
)


_V19_CHECKS = r"""
STPP V19 SCHEDULE-BRANCH CHECKS
These checks supersede earlier Judge checks that deduplicated only by event_id
and receiving_node.
1. Use (event_id, schedule_id, receiving_node), not merely
   (event_id, receiving_node), as the propagated-arrival uniqueness key.
2. Weekday/weekend arrivals with different schedule_id values are required
   separate records when their times differ; do not call them duplicates.
3. Every positive source pattern and propagated variation must carry the DAYS
   and SCHEDULE ID of its matching adjacency branch.
4. Validate each variation time against the matching stage's DESTINATION
   ARRIVAL WINDOW. Never validate all branches using the last source peak in a
   node's pattern list.
5. Reject missing/extra schedule records, mismatched days, path-prefix errors,
   wrong receiving-node placement, or arrival times outside branch windows.

"""

STPP_V19_JUDGE_SCENARIO_PARSING_PROMPT = (
    STPP_V17_JUDGE_SCENARIO_PARSING_PROMPT.replace(
        "Only genuine blocking problems belong in issues.",
        _V19_CHECKS + "Only genuine blocking problems belong in issues.",
    )
)
