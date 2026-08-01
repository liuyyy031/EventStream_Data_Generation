#!/usr/bin/env python3
"""STPP v21: route repeated day-partition failures back to Agent 1."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import demo_sts_stpp_v16 as _demo16
import demo_sts_stpp_v17 as _v17
import demo_sts_stpp_v20 as _v20
from demo_sts_stpp_v8 import _canonicalise_agent2_json_v8
from stpp_adapter_v19 import (
    audit_structured_contract_v19,
    format_contract_feedback_v19,
)


_ORIGINAL_PARSE_LOOP = _v17.parse_scenario_with_judge_audit_v16
_ORIGINAL_WRITE_V20 = _v20._write_outputs_v20  # noqa: SLF001


def _agent1_route_feedback(judgment: Dict[str, Any]) -> str:
    lines = [
        "Revise the same scenario without changing valid nodes, edges, values, "
        "time span, or spatial layout.",
        "For every DEMAND_SOURCE, declare exactly one EVENT ID family and one "
        "EVENT FULL PATH. Its schedule branches must partition DAYS 0..6 "
        "exactly once; normally use weekday [1,2,3,4,5] and weekend [0,6].",
        "Include every path stage in every branch and retain the v19 "
        "schedule-aware SCHEDULE ID/DAYS fields.",
    ]
    for issue in judgment.get("issues", []) or []:
        lines.append(
            f"- {issue.get('field', 'unspecified')}: "
            f"{issue.get('problem', 'contract violation')}"
        )
    lines.append("Return only the complete revised scenario.")
    return "\n".join(lines)


def _fingerprint(issues: List[Dict[str, Any]]) -> str:
    return json.dumps(issues, sort_keys=True, ensure_ascii=False, default=str)


def parse_scenario_with_judge_audit_v21(
    generator: Any,
    scenario: str,
    seq_len: int,
    max_agent2_attempts: int = 6,
    max_agent1_revisions: int = 4,
    repeated_causal_failures_before_revision: int = 2,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Escalate a repeated route/day contract from Agent 2 to Agent 1."""
    current_scenario = scenario
    parsed_json: Dict[str, Any] | None = None
    canonical_audit: Dict[str, Any] = {
        "changes": [],
        "num_changes": 0,
        "invalid_changes": [],
    }
    contract_audit: Dict[str, Any] = {
        "passed": False,
        "blocking_issues": [{"problem": "not yet audited"}],
    }
    last_judgment: Dict[str, Any] | None = None
    total_attempts = 0
    agent1_revisions = 0
    attempts_per_scenario: List[int] = []
    print("\n=== STPP v21 routed deterministic Agent/Judge loop ===")

    while True:
        feedback_history: List[str] = []
        scenario_attempts = 0
        repeated_causal: Dict[str, int] = {}
        repeated_route: Dict[str, int] = {}
        for agent2_attempt in range(1, max_agent2_attempts + 1):
            cumulative_feedback = None
            if feedback_history:
                cumulative_feedback = (
                    "Fix every deterministic or Judge mismatch without changing "
                    "valid scenario values:\n\n"
                    + "\n\n".join(feedback_history[-3:])
                )
            raw = generator.parse_scenario_to_structured_json(
                current_scenario,
                previous_feedback=cumulative_feedback,
                iteration=agent2_attempt,
            )
            parsed_json, canonical_audit = _canonicalise_agent2_json_v8(raw)
            contract_audit = audit_structured_contract_v19(parsed_json, seq_len)
            total_attempts += 1
            scenario_attempts += 1

            if not contract_audit["passed"]:
                feedback = format_contract_feedback_v19(contract_audit)
                placement_issues = contract_audit.get(
                    "variation_placement_issues", []
                )
                route_issues = contract_audit.get("route_contract_issues", [])
                causal_issues = contract_audit.get(
                    "temporal_contract_audit", {}
                ).get("causal_window_violations", [])

                if route_issues:
                    key = _fingerprint(route_issues)
                    repeated_route[key] = repeated_route.get(key, 0) + 1
                    if repeated_route[key] >= 2:
                        routed_feedback = (
                            "The same route/day-partition violation survived a "
                            "faithful Agent 2 retry. Revise the Agent 1 source "
                            "scenario so every demand source is explicit.\n"
                            + feedback
                        )
                        last_judgment = _demo16._deterministic_judgment(  # noqa: SLF001
                            contract_audit, "agent1", routed_feedback
                        )
                        break
                    last_judgment = _demo16._deterministic_judgment(  # noqa: SLF001
                        contract_audit, "agent2", feedback
                    )
                    feedback_history.append(feedback)
                    continue

                if placement_issues:
                    last_judgment = _demo16._deterministic_judgment(  # noqa: SLF001
                        contract_audit, "agent2", feedback
                    )
                    feedback_history.append(feedback)
                    continue

                if causal_issues:
                    key = _fingerprint(causal_issues)
                    repeated_causal[key] = repeated_causal.get(key, 0) + 1
                    if (
                        repeated_causal[key]
                        >= repeated_causal_failures_before_revision
                    ):
                        routed_feedback = (
                            "The same causal-window violation survived repeated "
                            "faithful parses, so revise the source scenario.\n"
                            + feedback
                        )
                        last_judgment = _demo16._deterministic_judgment(  # noqa: SLF001
                            contract_audit, "agent1", routed_feedback
                        )
                        break

                last_judgment = _demo16._deterministic_judgment(  # noqa: SLF001
                    contract_audit, "agent2", feedback
                )
                feedback_history.append(feedback)
                continue

            approved, judgment = generator.judge_scenario_parsing(
                current_scenario, parsed_json, iteration=agent2_attempt
            )
            last_judgment = judgment
            if approved:
                attempts_per_scenario.append(scenario_attempts)
                return current_scenario, parsed_json, {
                    "required": True,
                    "approved": True,
                    "attempts": total_attempts,
                    "used_unapproved_parse": False,
                    "agent1_revisions": agent1_revisions,
                    "agent2_attempts_per_scenario": attempts_per_scenario,
                    "termination_reason": "v21_deterministic_and_judge_approved",
                    "last_judgment": judgment,
                }, canonical_audit, contract_audit
            if judgment.get("error_source") == "agent1":
                break
            feedback_history.append(
                generator._format_feedback_for_agent2(judgment)  # noqa: SLF001
            )

        attempts_per_scenario.append(scenario_attempts)
        if (
            last_judgment
            and last_judgment.get("error_source") == "agent1"
            and agent1_revisions < max_agent1_revisions
        ):
            agent1_revisions += 1
            print(
                "\n=== STPP v21 routing repeated route/causal failure "
                "to Agent 1 ==="
            )
            current_scenario = generator.generate_scenario_description(
                previous_scenario=current_scenario,
                previous_feedback=_agent1_route_feedback(last_judgment),
                iteration=agent1_revisions + 1,
            )
            continue
        termination_reason = (
            "agent1_revision_limit"
            if last_judgment
            and last_judgment.get("error_source") == "agent1"
            else "agent2_retry_limit"
        )
        break

    if parsed_json is None:
        raise RuntimeError("Agent 2 did not return a structured scenario")
    return current_scenario, parsed_json, {
        "required": True,
        "approved": False,
        "attempts": total_attempts,
        "used_unapproved_parse": True,
        "agent1_revisions": agent1_revisions,
        "agent2_attempts_per_scenario": attempts_per_scenario,
        "termination_reason": termination_reason,
        "last_judgment": last_judgment,
    }, canonical_audit, contract_audit


def _rename_v20_to_v21(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_path = old_path.with_name(
        old_path.name.replace("_stpp_v20_", "_stpp_v21_")
    )
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v21(*args, **kwargs) -> Dict[str, str | None]:
    files = _ORIGINAL_WRITE_V20(*args, **kwargs)
    renamed = {
        key: _rename_v20_to_v21(value) if key != "visualization" else value
        for key, value in files.items()
    }
    family = "routed_schedule_aware_two_judge_stpp_v21"
    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = family
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v21"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = family
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v21"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Markdown-Edge Schedule-Aware STPP v20",
        "STReasoner Routed Schedule-Aware STPP v21",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    original_parse = _v17.parse_scenario_with_judge_audit_v16
    original_writer = _v20._write_outputs_v20  # noqa: SLF001
    added_output_argument = False
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "output_stpp_v21"])
        added_output_argument = True
    _v17.parse_scenario_with_judge_audit_v16 = (
        parse_scenario_with_judge_audit_v21
    )
    _v20._write_outputs_v20 = _write_outputs_v21  # noqa: SLF001
    try:
        return _v20.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v20 "):
            message = "v21 " + message[len("v20 ") :]
        raise RuntimeError(message) from exc
    finally:
        _v17.parse_scenario_with_judge_audit_v16 = original_parse
        _v20._write_outputs_v20 = original_writer  # noqa: SLF001
        if added_output_argument:
            del sys.argv[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
