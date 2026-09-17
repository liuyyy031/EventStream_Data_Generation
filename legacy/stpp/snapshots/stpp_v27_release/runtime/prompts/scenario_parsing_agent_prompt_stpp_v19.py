"""STPP v19 parser prompt for schedule-aware variation keys."""

from prompts.scenario_parsing_agent_prompt_stpp_v17 import (
    STPP_V17_SCENARIO_PARSING_PROMPT,
)


_V19_RULES = r"""
STPP V19 SCHEDULE-AWARE PARSING RULES
Rules 36-41 supersede v16 rules 28-31 wherever the older rules omit
schedule_id from the uniqueness key.
36. Every positive self-generated item in drift_patterns.nodes[].patterns must
    include schedule_id and days copied from its matching source branch.
37. Every propagated_variations item must include schedule_id and days. Emit
    exactly one item per (event_id, schedule_id, arrival_path[-1]).
38. Weekday and weekend arrivals are not duplicates when schedule_id differs.
    Keep their separate branch-specific arrival times.
39. Match a propagated variation to adjacency_modulation by event_id and
    schedule_id. Its time must be inside the destination_arrival_period of the
    stage whose target is arrival_path[-1].
40. Do not validate all branches against one arbitrary source peak. Each
    schedule branch uses its own self-generated pattern and days.
41. Final checklist:
    - self-generated pattern has schedule_id + days;
    - propagated variation has event_id + schedule_id + days;
    - days agree with the matching adjacency schedule;
    - one variation for every schedule and receiving node;
    - variation time is inside that stage's arrival window.

"""

STPP_V19_SCENARIO_PARSING_PROMPT = STPP_V17_SCENARIO_PARSING_PROMPT.replace(
    "SCENARIO TO PARSE\n{scenario}",
    _V19_RULES + "SCENARIO TO PARSE\n{scenario}",
)
