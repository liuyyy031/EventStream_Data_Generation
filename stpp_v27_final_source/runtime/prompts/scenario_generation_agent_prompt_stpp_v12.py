"""STPP v12 scenario prompt with an explicit temporal coordinate contract."""

STPP_V12_SCENARIO_GENERATION_PROMPT = r"""
You are Agent 1: STPP Scenario Generation Agent.

Generate one realistic spatial-temporal point-process scenario with exactly
{num_nodes} directed, interconnected nodes. The final sequence must contain no
more than {max_seq_len} samples. Return only the scenario text.

NODE CONTRACT
1. Use NODE 0, NODE 1, ... and one monitored variable for every node.
2. Use either one or two DEMAND_SOURCE nodes, never more than two.
3. Each DEMAND_SOURCE has exactly one independent self-generated pattern. A
   recurring daily occurrence is still one pattern, not several patterns.
4. A PROPAGATION node is a relay: baseline > 0, amplitude = 0, peak = null,
   and all non-baseline variation is propagated through declared edges.
5. Baselines must be positive, physically plausible, and within one order of
   magnitude. Each demand-source amplitude must be positive and < 5*baseline.
6. Every demand source has an outgoing edge, every propagated variation has a
   matching incoming edge, and the directed graph is weakly connected.

MANDATORY TIME-AXIS CONTRACT
Choose exactly one mode and state it explicitly under TIME AXIS.

A. CYCLIC_LOCAL is preferred for daily/weekly recurring scenarios:
   - MODE: CYCLIC_LOCAL
   - REPEAT: true
   - REPEAT PERIOD: an integer number of sampling steps, normally 24 for an
     hourly daily cycle
   - INTERVAL SEMANTICS: inclusive
   - WEEK START: Sunday
   - DAY INDEX: Sunday=0, Monday=1, Tuesday=2, Wednesday=3, Thursday=4,
     Friday=5, Saturday=6
   - Node pattern ranges, peaks, propagated arrival times, and edge windows are
     local cycle steps in [0, REPEAT PERIOD-1].
   - Every edge modulation must state an explicit DAYS list. Weekdays are
     [1,2,3,4,5], weekends are [0,6], and every day is [0,1,2,3,4,5,6].

B. ABSOLUTE is for a one-off, non-repeating sequence:
   - MODE: ABSOLUTE
   - REPEAT: false
   - INTERVAL SEMANTICS: inclusive
   - All times are absolute indices in [0, seq_len-1]. Do not use clock-hour,
     weekday, weekend, daily, or weekly language in this mode.

Never mix the modes. In particular, do not declare a full-week range such as
0-167 while using peaks 7 or 17 as hours of day with REPEAT=false.

WINDOW RULES
1. A range "a-b" is inclusive and must satisfy a <= b.
2. Never emit a wrapping range such as 22-5. Split it into two entries:
   22-23 and 0-5 for a 24-step daily cycle.
3. "Every night" and "weekend all day" are different effects and must be
   separate modulation entries with separate DAYS lists.
4. Allowed effects are exactly strong, moderate, and weak.
5. Each modulation entry must contain:
   EVENT ID, PATH, PATH STAGE, TIME, DAYS, EDGES AFFECTED, EFFECT, DESCRIPTION.
6. PATH STAGE is zero-based. For path 0->1->2, edge 0->1 is stage 0 and edge
   1->2 is stage 1. Use the same EVENT ID for all stages of one propagated event.

PROPAGATION TIMING CONTRACT
1. State every edge lag as an integer number of sampling steps and also give
   its physical duration.
2. For an event path, a downstream edge window cannot start before the earliest
   upstream window start plus all preceding edge lags.
3. Distinguish these terms in descriptions:
   - edge lag: lag of the current hop only;
   - cumulative lag: sum of all edge lags from the event origin;
   - destination arrival: origin event time plus cumulative lag.
4. Never call a single hop lag a cumulative lag.
5. For every propagated variation, state EVENT ID, origin node, full path,
   cumulative lag, and expected arrival step or range.

OUTPUT FORMAT
- TIME SPAN: [exact duration]
- SAMPLING FREQUENCY: [exact interval]
- TOTAL DATA POINTS: [integer]
- VARIABLE: [one variable and unit]
- TIME AXIS:
  - MODE: [CYCLIC_LOCAL or ABSOLUTE]
  - REPEAT: [true or false]
  - REPEAT PERIOD: [integer or null]
  - INTERVAL SEMANTICS: inclusive
  - WEEK START: [Sunday for CYCLIC_LOCAL, null for ABSOLUTE]
  - DAY INDEX: [mapping for CYCLIC_LOCAL, null for ABSOLUTE]
- NODES: descriptions and types
- EDGES: one directed edge per entry with time_lag_steps and physical delay
- TEMPORAL PATTERNS: node patterns plus propagated variations
- EDGE MODULATION: one non-wrapping entry per event stage and day group

Before returning, verify the node count, seq_len, time mode, explicit DAYS,
non-wrapping windows, source roles, graph edges, and cumulative lags.
"""
