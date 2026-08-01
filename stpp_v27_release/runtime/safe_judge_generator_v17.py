"""Role-routed STPP v17 agents and fail-closed final data Judge."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import demo_sts_sde as _base_module
import safe_judge_generator as _safe_module
from demo_sts_sde import LLMClientWrapper
from llm_client import LLMClient
from prompts.judge_scenario_parsing_prompt_stpp_v17 import (
    STPP_V17_JUDGE_SCENARIO_PARSING_PROMPT,
)
from prompts.judge_stpp_data_prompt_v17 import STPP_V17_JUDGE_DATA_PROMPT
from prompts.scenario_generation_agent_prompt_stpp_v17 import (
    STPP_V17_SCENARIO_GENERATION_PROMPT,
)
from prompts.scenario_parsing_agent_prompt_stpp_v17 import (
    STPP_V17_SCENARIO_PARSING_PROMPT,
)
from safe_judge_generator import SafeJudgeNetworkSDEGenerator
from safe_judge_generator_v12 import _enforce_judgment_invariants


class _TextResponse:
    def __init__(self, text: str):
        self.text = text


class TemperatureLLMClientWrapper(LLMClientWrapper):
    """Text-only wrapper with a stable per-role sampling temperature."""

    def __init__(self, client: LLMClient, temperature: float):
        super().__init__(client)
        self.temperature = float(temperature)

    def generate_content(self, contents: Any) -> _TextResponse:
        if isinstance(contents, list):
            prompt = "\n".join(str(item) for item in contents)
        else:
            prompt = str(contents)
        return _TextResponse(
            self.client.complete(prompt=prompt, temperature=self.temperature)
        )


def _model_name(wrapper: LLMClientWrapper) -> str:
    return str(getattr(wrapper.client, "model", "unknown"))


def _normalise_final_judgment(
    judgment: Any, deterministic_passed: bool
) -> Dict[str, Any]:
    output = dict(judgment) if isinstance(judgment, dict) else {}
    raw_issues = output.get("issues", [])
    issues = raw_issues if isinstance(raw_issues, list) else []
    normalised_issues = []
    for index, raw_issue in enumerate(issues, 1):
        issue = raw_issue if isinstance(raw_issue, dict) else {}
        normalised_issues.append(
            {
                "type": str(issue.get("type") or "data_usability"),
                "field": str(issue.get("field") or f"issue_{index}"),
                "problem": str(
                    issue.get("problem") or "Judge 2 reported a blocking issue"
                ),
                "evidence": str(
                    issue.get("evidence") or "See Judge 2 response and audits"
                ),
                "suggestion": str(
                    issue.get("suggestion") or "Correct and regenerate the sample"
                ),
            }
        )

    approved = (
        output.get("approved") is True
        and not normalised_issues
        and deterministic_passed
    )
    if approved:
        error_source = "none"
    else:
        error_source = str(output.get("error_source", "simulator")).lower()
        if error_source not in {"agent1", "agent2", "simulator"}:
            error_source = "simulator"
        if not normalised_issues:
            normalised_issues.append(
                {
                    "type": "judge_contract",
                    "field": "judge2_response",
                    "problem": (
                        "Judge 2 rejected or deterministic evidence failed "
                        "without a usable issue list"
                    ),
                    "evidence": "approved=false or deterministic_passed=false",
                    "suggestion": "Inspect deterministic evidence and rerun",
                }
            )

    return {
        "approved": approved,
        "error_source": error_source,
        "issues": normalised_issues,
        "confidence": str(output.get("confidence") or "unknown"),
        "overall_comment": str(
            output.get("overall_comment")
            or ("Final STPP sample approved" if approved else "Final STPP sample rejected")
        ),
        "judge_contract_consistent": (
            (approved and error_source == "none" and not normalised_issues)
            or (
                not approved
                and error_source in {"agent1", "agent2", "simulator"}
                and bool(normalised_issues)
            )
        ),
    }


class RoleRoutedJudgeNetworkSTPPGenerator(SafeJudgeNetworkSDEGenerator):
    """Use separate model clients for Agent 1, Agent 2, Judge 1 and Judge 2."""

    def __init__(
        self,
        num_nodes: int,
        logger: Any,
        agent1_client: LLMClient,
        agent2_client: LLMClient,
        judge1_client: LLMClient,
        judge2_client: LLMClient,
    ) -> None:
        super().__init__(
            num_nodes=num_nodes,
            logger=logger,
            llm_client=agent1_client,
        )
        self.agent1_model = TemperatureLLMClientWrapper(agent1_client, 0.7)
        self.agent2_model = TemperatureLLMClientWrapper(agent2_client, 0.0)
        self.judge1_model = TemperatureLLMClientWrapper(judge1_client, 0.0)
        self.judge2_model = TemperatureLLMClientWrapper(judge2_client, 0.0)

    @property
    def role_model_names(self) -> Dict[str, str]:
        return {
            "agent1": _model_name(self.agent1_model),
            "agent2": _model_name(self.agent2_model),
            "judge1": _model_name(self.judge1_model),
            "judge2": _model_name(self.judge2_model),
        }

    def generate_scenario_description(
        self,
        previous_scenario: str | None = None,
        previous_feedback: str | None = None,
        iteration: int = 1,
    ) -> str:
        original_prompt = _base_module.SCENARIO_GENERATION_PROMPT
        original_model = self.model
        _base_module.SCENARIO_GENERATION_PROMPT = (
            STPP_V17_SCENARIO_GENERATION_PROMPT
        )
        self.model = self.agent1_model
        try:
            return super().generate_scenario_description(
                previous_scenario=previous_scenario,
                previous_feedback=previous_feedback,
                iteration=iteration,
            )
        finally:
            self.model = original_model
            _base_module.SCENARIO_GENERATION_PROMPT = original_prompt

    def parse_scenario_to_structured_json(
        self,
        scenario: str,
        previous_feedback: str | None = None,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        original_prompt = _base_module.SCENARIO_PARSING_PROMPT
        original_model = self.model
        _base_module.SCENARIO_PARSING_PROMPT = STPP_V17_SCENARIO_PARSING_PROMPT
        self.model = self.agent2_model
        try:
            return super().parse_scenario_to_structured_json(
                scenario,
                previous_feedback=previous_feedback,
                iteration=iteration,
            )
        finally:
            self.model = original_model
            _base_module.SCENARIO_PARSING_PROMPT = original_prompt

    def judge_scenario_parsing(
        self,
        scenario: str,
        parsed_json: Dict[str, Any],
        iteration: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        original_prompt = _safe_module.JUDGE_SCENARIO_PARSING_PROMPT
        original_model = self.model
        _safe_module.JUDGE_SCENARIO_PARSING_PROMPT = (
            STPP_V17_JUDGE_SCENARIO_PARSING_PROMPT
        )
        self.model = self.judge1_model
        try:
            approved, judgment = super().judge_scenario_parsing(
                scenario, parsed_json, iteration=iteration
            )
        finally:
            self.model = original_model
            _safe_module.JUDGE_SCENARIO_PARSING_PROMPT = original_prompt
        return _enforce_judgment_invariants(approved, judgment)

    def judge_stpp_data(
        self,
        scenario: str,
        structured_scenario: Dict[str, Any],
        deterministic_evidence: Dict[str, Any],
        event_summary: Dict[str, Any],
        matrix_summary: Dict[str, Any],
        iteration: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Judge Agent 2: text-only final sample validation, always fail closed."""
        print("\n=== Judge Agent 2: Final STPP Data Usability Validation ===")
        models = self.role_model_names
        prompt = STPP_V17_JUDGE_DATA_PROMPT.format(
            expected_num_nodes=self.num_nodes,
            agent1_model=models["agent1"],
            agent2_model=models["agent2"],
            judge1_model=models["judge1"],
            judge2_model=models["judge2"],
            scenario=scenario,
            structured_scenario=json.dumps(
                structured_scenario, indent=2, ensure_ascii=False
            ),
            deterministic_evidence=json.dumps(
                deterministic_evidence, indent=2, ensure_ascii=False
            ),
            event_summary=json.dumps(event_summary, indent=2, ensure_ascii=False),
            matrix_summary=json.dumps(matrix_summary, indent=2, ensure_ascii=False),
        )
        input_data = {
            "scenario": scenario,
            "structured_scenario": structured_scenario,
            "deterministic_evidence": deterministic_evidence,
            "event_summary": event_summary,
            "matrix_summary": matrix_summary,
            "model": models["judge2"],
        }
        deterministic_passed = bool(
            deterministic_evidence.get("all_required_checks_passed") is True
        )
        try:
            response_text = self.judge2_model.generate_content(prompt).text
            print("Judge 2 response:")
            print(
                response_text[:800] + "..."
                if len(response_text) > 800
                else response_text
            )
            json_text = self._extract_json_from_response(response_text)
            judgment = _normalise_final_judgment(
                json.loads(json_text), deterministic_passed
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed boundary
            judgment = _normalise_final_judgment(
                {
                    "approved": False,
                    "error_source": "simulator",
                    "issues": [
                        {
                            "type": "judge_runtime",
                            "field": "judge2_call",
                            "problem": "Judge 2 call or JSON parsing failed",
                            "evidence": f"{type(exc).__name__}: {exc}",
                            "suggestion": "Retry Judge 2; do not write this sample",
                        }
                    ],
                    "confidence": "high",
                    "overall_comment": "Fail-closed because Judge 2 was unavailable",
                },
                deterministic_passed,
            )

        approved = judgment["approved"]
        print(
            "Judge Agent 2: APPROVED - final sample is usable"
            if approved
            else "Judge Agent 2: REJECTED - no data files will be written"
        )
        if self.logger:
            self.logger.log_agent_interaction(
                agent_name="Judge Agent 2: Final STPP Data Usability",
                agent_type="judge",
                iteration=iteration,
                input_data=input_data,
                output_data={"judgment": judgment},
                metadata={
                    "approved": approved,
                    "error_source": judgment["error_source"],
                    "num_issues": len(judgment["issues"]),
                    "model": models["judge2"],
                    "fail_closed": True,
                },
            )
        return approved, judgment
