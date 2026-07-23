"""STPP v16 Judge prompt with evidence-based error attribution."""

from prompts.judge_scenario_parsing_prompt_stpp_v15 import (
    STPP_V15_JUDGE_SCENARIO_PARSING_PROMPT,
)


_V16_CHECKS = r"""
STPP V16 EVIDENCE AND ATTRIBUTION CHECKS
1. First make an explicit table of each propagated arrival:
   (event_id, scenario receiving node, JSON container node, arrival_path[-1]).
   A JSON placement is valid only when all receiving-node values agree.
2. Never claim an entry is missing if it is present. Never call a value wrong
   and correct in the same sentence. Re-read the JSON before adding an issue.
3. Deduplicate by (event_id, receiving node). An entry under the wrong node is
   both misplaced and invalid; do not count it as satisfying the correct node.
4. Independently calculate every causal stage recurrence. If Agent 1 states
   stage-0 TIME 6-10, lag 2, but stage-1 TIME 6-10, the scenario is wrong and
   error_source must be agent1 even when Agent 2 copied it faithfully.
5. Attribute agent2 only when JSON differs from a valid scenario statement.
   Attribute agent1 when the source scenario itself violates route, placement,
   day partition, or causal-window logic.
6. If both contain errors, report the earliest blocking source that must be
   corrected; do not approve and do not emit contradictory feedback.

"""

STPP_V16_JUDGE_SCENARIO_PARSING_PROMPT = (
    STPP_V15_JUDGE_SCENARIO_PARSING_PROMPT.replace(
        "Only genuine blocking problems belong in issues.",
        _V16_CHECKS + "Only genuine blocking problems belong in issues.",
    )
)
