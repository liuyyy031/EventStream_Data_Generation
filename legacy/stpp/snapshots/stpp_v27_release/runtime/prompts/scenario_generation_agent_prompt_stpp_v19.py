"""STPP v19 scenario prompt for schedule-aware source and arrival branches."""

from prompts.scenario_generation_agent_prompt_stpp_v17 import (
    STPP_V17_SCENARIO_GENERATION_PROMPT,
)


STPP_V19_SCENARIO_GENERATION_PROMPT = (
    STPP_V17_SCENARIO_GENERATION_PROMPT
    + r"""

STPP V19 SCHEDULE-AWARE PATTERN CONTRACT
These v19 rules supersede earlier rules that used only
(event_id, receiving_node) as an arrival key.
1. A weekday and weekend branch may have different source peaks and propagated
   arrival times. They are distinct schedule records, not duplicates.
2. Every positive self-generated TEMPORAL PATTERN must state SCHEDULE ID and
   DAYS. Its SCHEDULE ID and DAYS must equal exactly one stage-0 branch in EDGE
   MODULATION for the same demand-source origin.
3. Under PROPAGATED ARRIVALS, emit one record for every
   (EVENT ID, SCHEDULE ID, RECEIVING NODE). Each record must state EVENT ID,
   SCHEDULE ID, DAYS, RECEIVING NODE, ARRIVAL PATH, EVENT FULL PATH,
   CUMULATIVE LAG TO NODE, and branch-specific ARRIVAL TIME TO NODE.
4. For one schedule branch, ARRIVAL TIME TO NODE must lie inside the
   DESTINATION ARRIVAL WINDOW of the path stage entering that receiving node.
5. Weekday/weekend records with the same EVENT ID and receiving node are valid
   only when their SCHEDULE IDs differ. Never merge branch-specific times and
   never omit SCHEDULE ID or DAYS.
6. Before returning, verify exactly one propagated-arrival record for every
   (event_id, schedule_id, non-origin node on EVENT FULL PATH).
"""
)
