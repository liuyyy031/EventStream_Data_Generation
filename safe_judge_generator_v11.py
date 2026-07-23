"""Grounded Judge contract for STPP v11 without changing earlier prompts."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import safe_judge_generator as _safe_module
from safe_judge_generator import SafeJudgeNetworkSDEGenerator


_GROUNDING = """
**GROUND-TRUTH CONSTRAINTS FOR STEP 2 (OVERRIDE INFERRED RULES):**
- A valid scenario may contain EITHER 1 OR 2 demand_source nodes. There is no
  fixed node ID or agent that must be the sole demand source.
- Each demand_source must have exactly one independent self-generated pattern;
  a repeated occurrence list inside that one pattern is allowed.
- A propagated arrival must not also be encoded as another independent
  self-generated pattern at the same node and time.
- An edge-modulation window may begin before the source peak to model a rising
  period. Reject timing only when a downstream edge begins before the event can
  reach that downstream edge's source after cumulative lags.
- Pattern time ranges may be inclusive. A positive peak must lie inside its own
  declared pattern range.
- Do not treat long storage-and-release behavior as ordinary edge propagation
  when its delay is unrelated to any directed path's cumulative lags.
- Put only genuine blocking problems in `issues`. Do not include entries whose
  conclusion is "No issue" or "No change needed".
"""


def _grounded_prompt() -> str:
    marker = "**STEP 2: SCENARIO LOGIC ASSESSMENT (Evaluating Agent 1)**"
    base = _safe_module.JUDGE_SCENARIO_PARSING_PROMPT
    return base.replace(marker, _GROUNDING + "\n" + marker, 1)


class GroundedJudgeNetworkSDEGenerator(SafeJudgeNetworkSDEGenerator):
    """Use the safe Judge response parser with an explicit invariant contract."""

    def judge_scenario_parsing(
        self,
        scenario: str,
        parsed_json: Dict[str, Any],
        iteration: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        original = _safe_module.JUDGE_SCENARIO_PARSING_PROMPT
        _safe_module.JUDGE_SCENARIO_PARSING_PROMPT = _grounded_prompt()
        try:
            return super().judge_scenario_parsing(
                scenario, parsed_json, iteration=iteration
            )
        finally:
            _safe_module.JUDGE_SCENARIO_PARSING_PROMPT = original
