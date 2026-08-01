#!/usr/bin/env python3
"""STPP v16 fail-closed entry point with direct declared-route sampling."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from demo_sts_sde import AgentInteractionLogger
from demo_sts_stpp_v2 import build_graph_metadata_v2
from demo_sts_stpp_v6 import repair_and_report_edge_lags_v6
from demo_sts_stpp_v8 import _canonicalise_agent2_json_v8
from demo_sts_stpp_v12 import _judge_contract_consistent
from safe_judge_generator_v16 import FailClosedRouteJudgeNetworkSTPPGenerator
from stpp_adapter_v13 import canonicalise_spatial_layout_v13
from stpp_adapter_v14 import canonicalise_temporal_contract_v14
from stpp_adapter_v16 import (
    STPPV16Config,
    audit_structured_contract_v16,
    format_contract_feedback_v16,
    simulate_stpp_for_streasoner_v16,
)


def _issues_for_judgment(audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = []
    for issue in audit.get("blocking_issues", []) or []:
        output.append(
            {
                "type": "deterministic_contract_violation",
                "field": str(
                    issue.get("field", issue.get("source_node", "unspecified"))
                ),
                "problem": str(issue.get("problem", "contract violation")),
                "suggestion": (
                    "Correct the exact field using the v16 placement and causal "
                    "window invariants."
                ),
            }
        )
    return output


def _agent1_feedback_v16(judgment: Dict[str, Any]) -> str:
    findings = []
    for issue in judgment.get("issues", []) or []:
        findings.append(
            f"- Field: {issue.get('field', 'unspecified')}\n"
            f"  Problem: {issue.get('problem', 'unspecified')}\n"
            f"  Required correction: {issue.get('suggestion', 'fix the issue')}"
        )
    return (
        "Revise the same scenario without changing valid nodes or context. "
        "Correct every v16 propagated-arrival placement and causal stage-window "
        "violation. For each stage k>0, set its TIME to the preceding stage's "
        "DESTINATION ARRIVAL WINDOW. Return only the revised scenario.\n\n"
        + "\n".join(findings)
    )


def _deterministic_judgment(
    audit: Dict[str, Any], error_source: str, feedback: str
) -> Dict[str, Any]:
    return {
        "approved": False,
        "error_source": error_source,
        "feedback": feedback,
        "issues": _issues_for_judgment(audit),
        "overall_comment": feedback,
        "judge_contract_consistent": True,
        "source": "deterministic_v16_pre_judge_audit",
    }


def parse_scenario_with_judge_audit_v16(
    generator: FailClosedRouteJudgeNetworkSTPPGenerator,
    scenario: str,
    seq_len: int,
    max_agent2_attempts: int = 6,
    max_agent1_revisions: int = 4,
    repeated_causal_failures_before_revision: int = 2,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    current_scenario = scenario
    parsed_json: Dict[str, Any] | None = None
    canonical_audit: Dict[str, Any] = {
        "changes": [], "num_changes": 0, "invalid_changes": []
    }
    contract_audit: Dict[str, Any] = {
        "passed": False, "blocking_issues": [{"problem": "not yet audited"}]
    }
    last_judgment: Dict[str, Any] | None = None
    total_attempts = 0
    agent1_revisions = 0
    attempts_per_scenario: List[int] = []
    print("\n=== STPP v16 deterministic-precheck Agent/Judge loop ===")

    while True:
        feedback_history: List[str] = []
        scenario_attempts = 0
        repeated_causal_failures: Dict[str, int] = {}
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
            contract_audit = audit_structured_contract_v16(parsed_json, seq_len)
            total_attempts += 1
            scenario_attempts += 1

            if not contract_audit["passed"]:
                feedback = format_contract_feedback_v16(contract_audit)
                placement_issues = contract_audit.get(
                    "variation_placement_issues", []
                )
                causal_issues = contract_audit.get(
                    "temporal_contract_audit", {}
                ).get("causal_window_violations", [])
                if placement_issues:
                    last_judgment = _deterministic_judgment(
                        contract_audit, "agent2", feedback
                    )
                    feedback_history.append(feedback)
                    continue
                if causal_issues:
                    fingerprint = json.dumps(
                        causal_issues, sort_keys=True, ensure_ascii=False
                    )
                    repeated_causal_failures[fingerprint] = (
                        repeated_causal_failures.get(fingerprint, 0) + 1
                    )
                    if (
                        repeated_causal_failures[fingerprint]
                        >= repeated_causal_failures_before_revision
                    ):
                        feedback = (
                            "The same causal-window violation survived repeated "
                            "faithful parses, so the source scenario must be revised.\n"
                            + feedback
                        )
                        last_judgment = _deterministic_judgment(
                            contract_audit, "agent1", feedback
                        )
                        break
                last_judgment = _deterministic_judgment(
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
                    "termination_reason": "deterministic_and_judge_approved",
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
            current_scenario = generator.generate_scenario_description(
                previous_scenario=current_scenario,
                previous_feedback=_agent1_feedback_v16(last_judgment),
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


def _write_outputs_v16(
    scenario: str,
    structured_scenario: Dict[str, Any],
    graph_metadata: Dict[str, Any],
    ts_data: np.ndarray,
    events: List[Dict[str, Any]],
    generation_info: Dict[str, Any],
    domain: str,
    output_dir: Path,
) -> Dict[str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    domain_clean = re.sub(r"[^A-Za-z0-9_-]+", "_", domain).strip("_")
    prefix = f"{domain_clean}_node{ts_data.shape[0]}_stpp_v16_{timestamp}"
    complete_data = {
        "timestamp": timestamp,
        "generator_family": "fail_closed_direct_route_stpp_v16",
        "agent1_scenario": scenario,
        "agent2_structured_scenario": structured_scenario,
        "agent3_point_process_parameters": generation_info[
            "point_process_parameters"
        ],
        "agent4_time_varying_adjacency": graph_metadata,
        "agent5_event_stream": events,
        "agent5_simulation_data": ts_data,
        "generation_info": generation_info,
        "seq_len": int(ts_data.shape[1]),
    }
    pickle_path = output_dir / f"{prefix}_data.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(complete_data, handle)
    json_path = output_dir / f"{prefix}_data.json"
    json_data = dict(complete_data)
    json_data["agent5_simulation_data"] = ts_data.tolist()
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    description_path = output_dir / f"{prefix}_results.txt"
    quality = generation_info["quality_report"]
    route = generation_info["declared_route_audit"]
    direct = generation_info.get("propagation_cycle_audit", {}).get(
        "direct_declared_route_sampling", {}
    )
    peak = generation_info["scalar_self_generated_peak_audit"]
    pre_audit = generation_info["pre_simulation_contract_audit"]
    judge = generation_info["judge1_validation"]
    with description_path.open("w", encoding="utf-8") as handle:
        handle.write("=== STReasoner Fail-Closed Direct-Route STPP v16 ===\n\n")
        handle.write(f"Generation time: {dt.datetime.now().isoformat()}\n")
        handle.write(f"Domain: {domain}\n")
        handle.write(f"Nodes: {ts_data.shape[0]}\n")
        handle.write(f"Time windows: {ts_data.shape[1]}\n")
        handle.write(f"All events: {generation_info['num_events']}\n")
        handle.write(f"Root events: {generation_info['num_root_events']}\n")
        handle.write(
            f"Propagated events: {generation_info['num_propagated_events']}\n"
        )
        handle.write(f"Judge 1 approved: {judge['approved']}\n")
        handle.write(f"Judge attempts: {judge['attempts']}\n")
        handle.write(
            f"Judge response consistent: {_judge_contract_consistent(judge)}\n"
        )
        handle.write(f"Pre-simulation contract valid: {pre_audit['passed']}\n")
        handle.write(f"Quality gates passed: {quality['passed']}\n")
        handle.write(
            "Direct propagated events created: "
            f"{direct.get('created_propagated_events', 0)}\n"
        )
        handle.write(
            "Blocked outside activation window: "
            f"{direct.get('blocked_outside_activation_window', 0)}\n"
        )
        handle.write(
            "Blocked outside arrival window: "
            f"{direct.get('blocked_outside_arrival_window', 0)}\n"
        )
        handle.write(
            "Final route violations: "
            f"{len(route['propagated_event_violations'])}\n"
        )
        handle.write(
            "Actual propagated lineages: "
            + json.dumps(route["actual_propagated_lineages"], ensure_ascii=False)
            + "\n"
        )
        handle.write(
            "Uncovered scalar self-generated peaks: "
            f"{len(peak['uncovered_scalar_self_generated_peaks'])}\n"
        )
        handle.write(
            "Quality checks: "
            + json.dumps(quality["checks"], ensure_ascii=False)
            + "\n\n"
        )
        handle.write("Pre-simulation contract audit\n-----------------------------\n")
        handle.write(json.dumps(pre_audit, indent=2, ensure_ascii=False))
        handle.write("\n\nDirect route sampling audit\n---------------------------\n")
        handle.write(json.dumps(direct, indent=2, ensure_ascii=False))
        handle.write("\n\nDeclared route audit\n--------------------\n")
        handle.write(json.dumps(route, indent=2, ensure_ascii=False))
        handle.write("\n\nDeclared route envelope audit\n-----------------------------\n")
        handle.write(
            json.dumps(
                generation_info["declared_route_envelope_audit"],
                indent=2,
                ensure_ascii=False,
            )
        )
        handle.write("\n\nScalar peak audit\n-----------------\n")
        handle.write(json.dumps(peak, indent=2, ensure_ascii=False))
        handle.write("\n\nJudge audit\n-----------\n")
        handle.write(json.dumps(judge, indent=2, ensure_ascii=False))
        handle.write("\n\nScenario\n--------\n")
        handle.write(scenario)
        handle.write("\n")
    return {
        "pickle": str(pickle_path),
        "json": str(json_path),
        "description": str(description_path),
        "visualization": None,
    }


def demo_stpp_generation_v16(
    enabled_judges: List[int] | None = None,
    enable_logging: bool = True,
    num_nodes: int = 3,
    domain: str = "traffic",
    generate_viz: bool = False,
    stpp_config: STPPV16Config | None = None,
    output_dir: str = "output_stpp_v16",
    allow_unapproved_parse: bool = False,
) -> Dict[str, Any]:
    enabled_judges = [1] if enabled_judges is None else enabled_judges
    config = stpp_config or STPPV16Config()
    if generate_viz:
        print("v16 does not generate the legacy SDE visualization.")
    logger = AgentInteractionLogger() if enable_logging else None
    generator = FailClosedRouteJudgeNetworkSTPPGenerator(
        num_nodes=num_nodes, logger=logger
    )
    generator.domain = domain
    scenario, seq_len = generator.generate_scenario_with_length_validation()
    generator.seq_len = seq_len

    if 1 in enabled_judges:
        scenario, parsed, judge_audit, agent2_audit, pre_agent_audit = (
            parse_scenario_with_judge_audit_v16(
                generator,
                scenario,
                seq_len,
                repeated_causal_failures_before_revision=(
                    config.deterministic_agent2_failures_before_agent1_revision
                ),
            )
        )
    else:
        raw = generator.parse_scenario_to_structured_json(scenario)
        parsed, agent2_audit = _canonicalise_agent2_json_v8(raw)
        pre_agent_audit = audit_structured_contract_v16(parsed, seq_len)
        judge_audit = {
            "required": False,
            "approved": None,
            "attempts": 0,
            "used_unapproved_parse": False,
            "agent1_revisions": 0,
            "agent2_attempts_per_scenario": [],
            "termination_reason": "judge_not_required",
            "last_judgment": None,
        }

    if (
        (judge_audit["required"] and judge_audit["approved"] is not True)
        or not pre_agent_audit["passed"]
    ) and not allow_unapproved_parse:
        if logger:
            logger.save_complete_log()
        last_feedback = str(
            (judge_audit.get("last_judgment") or {}).get(
                "feedback", "no Judge feedback available"
            )
        )
        raise RuntimeError(
            "v16 fail-closed: deterministic contract or Judge approval failed; "
            "no STPP data files were written. Inspect the agent log or rerun "
            "with --allow_unapproved_parse only for diagnostics. "
            f"termination_reason={judge_audit.get('termination_reason')}; "
            f"last_feedback={last_feedback}"
        )

    final_time_info = generator._extract_time_info_from_scenario(  # noqa: SLF001
        scenario
    )
    seq_len = int(final_time_info.get("calculated_seq_len") or seq_len)
    generator.seq_len = seq_len
    structured_scenario, temporal_audit = canonicalise_temporal_contract_v14(
        parsed, seq_len
    )
    structured_scenario, spatial_audit = canonicalise_spatial_layout_v13(
        structured_scenario
    )
    final_pre_audit = audit_structured_contract_v16(structured_scenario, seq_len)
    if not final_pre_audit["passed"] and not allow_unapproved_parse:
        if logger:
            logger.save_complete_log()
        raise RuntimeError(
            "v16 fail-closed: canonical structured scenario failed the final "
            "deterministic contract audit. "
            f"issues={final_pre_audit['blocking_issues']}"
        )

    sampling_minutes = float(getattr(generator, "sampling_minutes", 60))
    lag_repair = repair_and_report_edge_lags_v6(
        scenario,
        structured_scenario,
        sampling_minutes,
        config.immediate_lag_epsilon_steps,
    )
    graph_metadata = build_graph_metadata_v2(structured_scenario, seq_len)
    ts_data, events, generation_info = simulate_stpp_for_streasoner_v16(
        structured_scenario, seq_len, config
    )
    quality = generation_info["quality_report"]
    parsed_seq_len = int(structured_scenario.get("seq_len", seq_len))
    quality["checks"]["judge1_approved_or_not_required"] = (
        not judge_audit["required"] or judge_audit["approved"] is True
    )
    quality["checks"]["judge_response_contract_consistent"] = (
        _judge_contract_consistent(judge_audit)
    )
    quality["checks"]["agent2_seq_len_matches_generator"] = (
        parsed_seq_len == seq_len
    )
    quality["checks"]["agent2_canonicalisation_valid"] = not agent2_audit.get(
        "invalid_changes", []
    )
    quality["checks"]["temporal_contract_valid"] = temporal_audit["passed"]
    quality["checks"]["temporal_contract_repair_free"] = temporal_audit[
        "repair_free"
    ]
    quality["checks"]["spatial_layout_has_separated_nodes"] = spatial_audit[
        "final_valid"
    ]
    quality["checks"]["spatial_layout_repair_free"] = not spatial_audit[
        "repaired"
    ]
    quality["passed"] = all(quality["checks"].values())
    generation_info["judge1_validation"] = judge_audit
    generation_info["agent2_canonicalisation"] = agent2_audit
    generation_info["pre_generation_agent_contract_audit"] = pre_agent_audit
    generation_info["temporal_contract_audit"] = temporal_audit
    generation_info["spatial_layout_audit"] = spatial_audit
    generation_info["sampling_minutes"] = sampling_minutes
    generation_info["dt"] = sampling_minutes / 60.0
    generation_info["lag_repair"] = lag_repair
    data_files = _write_outputs_v16(
        scenario,
        structured_scenario,
        graph_metadata,
        ts_data,
        events,
        generation_info,
        domain,
        Path(output_dir),
    )
    if logger:
        logger.save_complete_log()
    return {
        "generator_family": "fail_closed_direct_route_stpp_v16",
        "agent1_scenario": scenario,
        "agent2_structured_scenario": structured_scenario,
        "agent5_event_stream": events,
        "agent5_simulation_data": ts_data,
        "generation_info": generation_info,
        "data_files": data_files,
        "logger": logger,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_nodes", type=int, default=3)
    parser.add_argument("--domain", default="traffic")
    parser.add_argument("--judges", default="1")
    parser.add_argument("--output_dir", default="output_stpp_v16")
    parser.add_argument(
        "--aggregation",
        choices=["count", "magnitude", "rolling_count", "rolling_magnitude"],
        default="magnitude",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--edge_branching_ratio", type=float, default=0.35)
    parser.add_argument("--target_immigrant_rate", type=float, default=0.30)
    parser.add_argument(
        "--allow_unapproved_parse",
        action="store_true",
        help="write diagnostic data even when deterministic/Judge checks fail",
    )
    args = parser.parse_args()
    judges = (
        []
        if args.judges.lower() == "none"
        else [int(item.strip()) for item in args.judges.split(",") if item.strip()]
    )
    config = STPPV16Config(
        aggregation=args.aggregation,
        seed=args.seed,
        edge_branching_ratio=args.edge_branching_ratio,
        target_immigrant_rate=args.target_immigrant_rate,
    )
    result = demo_stpp_generation_v16(
        enabled_judges=judges,
        num_nodes=args.num_nodes,
        domain=args.domain,
        stpp_config=config,
        output_dir=args.output_dir,
        allow_unapproved_parse=args.allow_unapproved_parse,
    )
    quality = result["generation_info"]["quality_report"]
    route = result["generation_info"]["declared_route_audit"]
    print(f"Generated {result['generation_info']['num_events']} events")
    print(f"Quality gates passed: {quality['passed']}")
    print(
        "Final declared-route violations: "
        f"{len(route['propagated_event_violations'])}"
    )
    print(f"Pickle: {result['data_files']['pickle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
