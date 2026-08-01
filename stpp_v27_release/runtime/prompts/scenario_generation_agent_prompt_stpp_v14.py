"""STPP v14 scenario prompt with node-local arrival paths and schedule branches."""

from prompts.scenario_generation_agent_prompt_stpp_v13 import (
    STPP_V13_SCENARIO_GENERATION_PROMPT,
)


STPP_V14_SCENARIO_GENERATION_PROMPT = (
    STPP_V13_SCENARIO_GENERATION_PROMPT
    + r"""

STPP V14 EVENT-SEMANTICS CONTRACT
1. Every DEMAND_SOURCE must have at least one outgoing directed edge and a
   directed path by which its independent events can affect another node.
   A demand source that only receives events is invalid.
2. For every propagated event, state EVENT FULL PATH once. Then give one
   propagated-arrival entry for every receiving node on that path.
3. Each propagated-arrival entry must contain:
   - RECEIVING NODE;
   - ARRIVAL PATH, starting at the event origin and ending at that receiving
     node, not at a later node;
   - EVENT FULL PATH;
   - CUMULATIVE LAG TO NODE, equal to the edge-lag sum along ARRIVAL PATH;
   - ARRIVAL TIME TO NODE, equal to the origin peak plus that cumulative lag.
4. Example for 0->1->2 with lags 2 and 1 and origin peak 8:
   - NODE 1: ARRIVAL PATH 0->1, EVENT FULL PATH 0->1->2,
     CUMULATIVE LAG TO NODE 2, ARRIVAL TIME TO NODE 10;
   - NODE 2: ARRIVAL PATH 0->1->2, EVENT FULL PATH 0->1->2,
     CUMULATIVE LAG TO NODE 3, ARRIVAL TIME TO NODE 11.
5. A propagation node's zero-amplitude baseline pattern has
   ORIGIN: ambient_baseline, never self_generated.

SCHEDULE-BRANCH CONTRACT
1. Every edge-modulation entry must have a SCHEDULE ID in addition to EVENT ID.
2. Use a different SCHEDULE ID for different day groups or shifted schedules,
   for example EVT-001-weekday and EVT-001-weekend.
3. All stages of one schedule branch share EVENT ID, SCHEDULE ID, DAYS, and
   EVENT FULL PATH.
4. TIME on a stage is the EDGE ACTIVATION WINDOW at that edge's source node.
   It is not the destination arrival window.
5. Also state DESTINATION ARRIVAL WINDOW, calculated by shifting that stage's
   edge activation window by the current edge's own lag.
6. Do not describe an early edge window as an arrival window and do not justify
   an inconsistent window by saying it is merely centered on the peak.
"""
)
