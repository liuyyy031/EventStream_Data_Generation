"""STPP v16 scenario prompt with exact stage-window recurrence."""

from prompts.scenario_generation_agent_prompt_stpp_v15 import (
    STPP_V15_SCENARIO_GENERATION_PROMPT,
)


STPP_V16_SCENARIO_GENERATION_PROMPT = (
    STPP_V15_SCENARIO_GENERATION_PROMPT
    + r"""

STPP V16 EXACT CAUSAL-WINDOW CONTRACT
1. For consecutive path stages, the downstream stage TIME must equal the
   preceding stage DESTINATION ARRIVAL WINDOW. Do not repeat the origin window
   at every edge.
2. For path 0->1->2 with lags 2 and 1 and stage-0 TIME 6-10:
   - stage 0 TIME 6-10, DESTINATION ARRIVAL WINDOW 8-12;
   - stage 1 TIME 8-12, DESTINATION ARRIVAL WINDOW 9-13.
   Stage-1 TIME 6-10 is invalid because traffic has not reached Node 1 at 6.
3. Apply the same recurrence independently to every weekday/weekend branch.
4. Under PROPAGATED ARRIVALS, write exactly one entry for every receiving node
   on EVENT FULL PATH and no entry for the origin node. Make the receiving node
   explicit on the same line as ARRIVAL PATH.
5. Before returning, verify for every stage k>0:
   stage[k].TIME == stage[k-1].DESTINATION ARRIVAL WINDOW.
"""
)
