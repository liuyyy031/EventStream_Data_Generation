"""STPP v15 scenario prompt with exclusive executable event routes."""

from prompts.scenario_generation_agent_prompt_stpp_v14 import (
    STPP_V14_SCENARIO_GENERATION_PROMPT,
)


STPP_V15_SCENARIO_GENERATION_PROMPT = (
    STPP_V14_SCENARIO_GENERATION_PROMPT
    + r"""

STPP V15 EXECUTABLE-ROUTE CONTRACT
1. EVENT FULL PATH is exclusive, not merely descriptive. An event may stop at
   any prefix of that path, but it must never take an outgoing edge that is not
   the next edge of EVENT FULL PATH.
2. Define exactly one EVENT ID family for each DEMAND_SOURCE in v15. Its
   weekday/weekend or other day branches share one EVENT FULL PATH and differ
   only by SCHEDULE ID, DAYS, windows, and effects.
3. The SCHEDULE ID branches for one demand source must partition all DAYS
   0..6: every day occurs in exactly one branch. This lets every root event be
   assigned to one unambiguous semantic schedule.
4. Every path stage must be present in every schedule branch. Do not declare a
   full path 0->1->2 and omit the 1->2 stage from a weekend branch.
5. Actual propagation is allowed only while the parent event is inside that
   stage's TIME activation window; its child must land inside DESTINATION
   ARRIVAL WINDOW. Keep the windows wide enough for the declared integer lag.
6. Do not use a graph cycle to extend an event beyond its full path. If
   EVT-002 has FULL PATH 2->0, it stops at Node 0 even when edge 0->1 exists.
7. Every strong or moderate propagation edge described for an event must be a
   stage of that event's EVENT FULL PATH.
"""
)
