"""STPP v15 Judge prompt for executable route exclusivity."""

from prompts.judge_scenario_parsing_prompt_stpp_v14 import (
    STPP_V14_JUDGE_SCENARIO_PARSING_PROMPT,
)


_V15_CHECKS = r"""
STPP V15 EXECUTABLE ROUTE CHECKS
1. EVENT FULL PATH is exclusive. A generated lineage may end at a path prefix,
   but no semantic event may continue along a non-next graph edge.
2. Each demand_source has exactly one EVENT ID family. All schedule branches
   for that source use the same full path.
3. Those branches partition days 0..6 exactly once. Missing or overlapping
   days make root-to-schedule assignment ambiguous and are blocking errors.
4. Each (event_id, schedule_id) contains every path stage exactly once.
5. Every stage activation and destination-arrival window is compatible with
   its edge lag. A graph cycle does not authorize extending a shorter event
   path.

"""

STPP_V15_JUDGE_SCENARIO_PARSING_PROMPT = (
    STPP_V14_JUDGE_SCENARIO_PARSING_PROMPT.replace(
        "Only genuine blocking problems belong in issues.",
        _V15_CHECKS + "Only genuine blocking problems belong in issues.",
    )
)
