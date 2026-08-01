"""Defensive Judge 1 response handling without modifying demo_sts_sde.py."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from demo_sts_sde import JUDGE_SCENARIO_PARSING_PROMPT, NetworkSDEGenerator


def _normalise_judgment(judgment: Any) -> Dict[str, Any]:
    if not isinstance(judgment, dict):
        judgment = {}
    normalised_issues = []
    raw_issues = judgment.get("issues", [])
    if not isinstance(raw_issues, list):
        raw_issues = []
    for index, raw_issue in enumerate(raw_issues, 1):
        issue = raw_issue if isinstance(raw_issue, dict) else {}
        normalised_issues.append(
            {
                **issue,
                "type": issue.get("type") or "Parsing Fidelity",
                "field": issue.get("field") or f"unspecified_issue_{index}",
                "problem": issue.get("problem")
                or "Judge did not provide details",
                "suggestion": issue.get("suggestion")
                or "Review this field against the scenario text",
            }
        )
    judgment["issues"] = normalised_issues
    judgment["approved"] = judgment.get("approved") is True
    if judgment.get("error_source") not in {"agent1", "agent2"}:
        judgment["error_source"] = "agent2"
    judgment["feedback"] = (
        judgment.get("feedback")
        or judgment.get("overall_comment")
        or "Judge requested a revision"
    )
    judgment["overall_comment"] = (
        judgment.get("overall_comment") or judgment["feedback"]
    )
    return judgment


class SafeJudgeNetworkSDEGenerator(NetworkSDEGenerator):
    """Network generator whose Judge output tolerates omitted optional fields."""

    def judge_scenario_parsing(
        self,
        scenario: str,
        parsed_json: Dict[str, Any],
        iteration: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        print("\n=== Judge Agent 1: Scenario Parsing Validation (safe) ===")
        parsed_json_str = json.dumps(parsed_json, indent=2, ensure_ascii=False)
        prompt = JUDGE_SCENARIO_PARSING_PROMPT.format(
            expected_num_nodes=self.num_nodes,
            scenario=scenario,
            parsed_json=parsed_json_str,
        )
        input_data = {
            "scenario": scenario[:500] + "..." if len(scenario) > 500 else scenario,
            "parsed_json": parsed_json,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
        }
        response = self.model.generate_content(prompt)
        response_text = response.text
        print("Judge response:")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print("-" * 50)

        json_text = self._extract_json_from_response(response_text)
        try:
            judgment = _normalise_judgment(json.loads(json_text))
        except (json.JSONDecodeError, TypeError) as exc:
            error_msg = f"Warning: Failed to parse judge response: {exc}"
            print(error_msg)
            judgment = _normalise_judgment(
                {
                    "approved": False,
                    "error_source": "agent2",
                    "feedback": "Judge response parsing failed; revise conservatively",
                    "issues": [],
                }
            )
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Judge Agent 1: Scenario Parsing Validation",
                    agent_type="judge",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"error": error_msg, "raw_response": response_text},
                    metadata={"approved": False, "error_type": "JSONDecodeError"},
                )
            return False, judgment

        approved = judgment["approved"]
        if approved:
            print("✓ Judge Agent 1: APPROVED - Parsing is accurate")
        else:
            print("✗ Judge Agent 1: REJECTED - Issues found:")
            for issue in judgment["issues"]:
                print(f"  - {issue['field']}: {issue['problem']}")
                print(f"    Suggestion: {issue['suggestion']}")
            print(f"\nOverall: {judgment['overall_comment']}")

        if self.logger:
            self.logger.log_agent_interaction(
                agent_name="Judge Agent 1: Scenario Parsing Validation",
                agent_type="judge",
                iteration=iteration,
                input_data=input_data,
                output_data={"judgment": judgment},
                metadata={
                    "approved": approved,
                    "num_issues": len(judgment["issues"]),
                    "overall_comment": judgment["overall_comment"],
                    "response_normalised": True,
                },
            )
        return approved, judgment
