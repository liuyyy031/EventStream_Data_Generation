"""STPP v25 prompt for non-wrapping executable event routes."""

from prompts.scenario_generation_agent_prompt_stpp_v23 import (
    STPP_V23_SCENARIO_GENERATION_PROMPT,
)


STPP_V25_SCENARIO_GENERATION_PROMPT = (
    STPP_V23_SCENARIO_GENERATION_PROMPT
    + r"""

STPP V25 NON-WRAPPING ROUTE CONTRACT
1. In CYCLIC_LOCAL mode, an event route must finish inside the same local
   cycle. The final destination-arrival window must end at or before
   REPEAT PERIOD-1 (normally step 23).
2. Never write a wrapping or split-next-day range such as 22-0, 23-1,
   "22-23 and 0-0", or "next day". Choose an earlier source peak/window or
   smaller physically valid integer edge lags so every stage and arrival is a
   single non-wrapping range a-b with a <= b.
3. Stage k+1 TIME must equal stage k DESTINATION ARRIVAL WINDOW. For every
   stage, DESTINATION ARRIVAL WINDOW is TIME shifted forward by that stage's
   edge time_lag_steps, without crossing the cycle boundary.
4. Physical duration must agree with the sampling interval:
   physical_duration = time_lag_steps * SAMPLING FREQUENCY.
   For hourly sampling, 2 steps means 2 hours, never 30 minutes.
5. Before returning, compute the cumulative lag of every full path and verify
   that source_window_end + cumulative_lag <= REPEAT PERIOD-1.
"""
)
