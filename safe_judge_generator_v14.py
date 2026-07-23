"""STPP v14 prompt routing with fail-closed Judge response handling."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import demo_sts_sde as _base_module
import safe_judge_generator as _safe_module
from prompts.judge_scenario_parsing_prompt_stpp_v14 import (
    STPP_V14_JUDGE_SCENARIO_PARSING_PROMPT,
)
from prompts.scenario_generation_agent_prompt_stpp_v14 import (
    STPP_V14_SCENARIO_GENERATION_PROMPT,
)
from prompts.scenario_parsing_agent_prompt_stpp_v14 import (
    STPP_V14_SCENARIO_PARSING_PROMPT,
)
from safe_judge_generator import SafeJudgeNetworkSDEGenerator
from safe_judge_generator_v12 import _enforce_judgment_invariants


class EventSemanticJudgeNetworkSTPPGenerator(SafeJudgeNetworkSDEGenerator):
    """Route Agent 1, Agent 2, and Judge through the v14 prompt copies."""

    def generate_scenario_description(
        self,
        previous_scenario: str | None = None,
        previous_feedback: str | None = None,
        iteration: int = 1,
    ) -> str:
        original = _base_module.SCENARIO_GENERATION_PROMPT
        _base_module.SCENARIO_GENERATION_PROMPT = (
            STPP_V14_SCENARIO_GENERATION_PROMPT
        )
        try:
            return super().generate_scenario_description(
                previous_scenario=previous_scenario,
                previous_feedback=previous_feedback,
                iteration=iteration,
            )
        finally:
            _base_module.SCENARIO_GENERATION_PROMPT = original

    def parse_scenario_to_structured_json(
        self,
        scenario: str,
        previous_feedback: str | None = None,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        original = _base_module.SCENARIO_PARSING_PROMPT
        _base_module.SCENARIO_PARSING_PROMPT = STPP_V14_SCENARIO_PARSING_PROMPT
        try:
            return super().parse_scenario_to_structured_json(
                scenario,
                previous_feedback=previous_feedback,
                iteration=iteration,
            )
        finally:
            _base_module.SCENARIO_PARSING_PROMPT = original

    def judge_scenario_parsing(
        self,
        scenario: str,
        parsed_json: Dict[str, Any],
        iteration: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        original = _safe_module.JUDGE_SCENARIO_PARSING_PROMPT
        _safe_module.JUDGE_SCENARIO_PARSING_PROMPT = (
            STPP_V14_JUDGE_SCENARIO_PARSING_PROMPT
        )
        try:
            approved, judgment = super().judge_scenario_parsing(
                scenario, parsed_json, iteration=iteration
            )
        finally:
            _safe_module.JUDGE_SCENARIO_PARSING_PROMPT = original
        return _enforce_judgment_invariants(approved, judgment)
