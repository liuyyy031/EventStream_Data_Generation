"""STPP v23 prompt for an unambiguous single-wave schedule contract."""

from prompts.scenario_generation_agent_prompt_stpp_v19 import (
    STPP_V19_SCENARIO_GENERATION_PROMPT,
)


STPP_V23_SCENARIO_GENERATION_PROMPT = (
    STPP_V19_SCENARIO_GENERATION_PROMPT
    + r"""

STPP V23 SINGLE-WAVE SCHEDULE CONTRACT
These rules clarify the existing one-pattern-per-demand-source contract. They
do not permit multiple daily waves inside one schedule branch.
1. One (EVENT ID, SCHEDULE ID) identifies exactly one source wave on its DAYS.
   For a path with N edges, emit exactly N EDGE MODULATION records: one record
   for every PATH STAGE 0 through N-1. Never repeat a PATH STAGE under the same
   SCHEDULE ID.
2. Each SCHEDULE ID has exactly one positive source peak and exactly one
   propagated-arrival time for each receiving node. Do not place morning and
   evening peaks, windows, or arrival times in the same schedule.
3. Because schedule branches for a demand source must partition DAYS 0..6,
   do not create two SCHEDULE IDs whose DAYS overlap. If a domain naturally
   has several daily waves, choose one dominant wave and omit the secondary
   wave from this scenario.
4. Write SAMPLING FREQUENCY as a canonical interval such as "1 hour" or
   "30 minutes". Do not write rate wording such as "1 sample per hour".
5. Before returning, group EDGE MODULATION records by
   (EVENT ID, SCHEDULE ID) and verify that the PATH STAGE list is exactly
   [0, ..., len(PATH)-2], without duplicates.
"""
)
