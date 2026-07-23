#!/usr/bin/env python3
"""STPP v15 entry point with exclusive declared routes and schedule windows."""

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
from safe_judge_generator_v15 import DeclaredRouteJudgeNetworkSTPPGenerator
from stpp_adapter_v13 import canonicalise_spatial_layout_v13
from stpp_adapter_v14 import canonicalise_temporal_contract_v14
from stpp_adapter_v15 import (
    STPPV15Config,
    simulate_stpp_for_streasoner_v15,
)


def _agent1_feedback_v15(judgment: Dict[str, Any]) -> str:
    findings = []
    for issue in judgment.get("issues", []) or []:
        findings.append(
            f"- Field: {issue.get('field', 'unspecified')}\n"
            f"  Problem: {issue.get('problem', 'unspecified')}\n"
            f"  Required correction: {issue.get('suggestion', 'fix the issue')}"
        )
    return (
        "Revise the same scenario without changing valid nodes or context. "
        "Obey every v15 temporal, spatial, node-local schedule, exclusive "
        "event_full_path, and complete day-partition contract. Return only "
        "the revised scenario.\n\n"
        + "\n".join(findings)
    )


def parse_scenario_with_judge_audit_v15(
    generator: DeclaredRouteJudgeNetworkSTPPGenerator,
    scenario: str,
    max_agent2_attempts: int = 6,
    max_agent1_revisions: int = 4,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    current_scenario = scenario
    parsed_json: Dict[str, Any] | None = None
    canonical_audit: Dict[str, Any] = {
        "changes": [], "num_changes": 0, "invalid_changes": []
    }
    last_judgment: Dict[str, Any] | None = None
    total_attempts = 0
    agent1_revisions = 0
    attempts_per_scenario: List[int] = []
    print("\n=== STPP v15 declared-route Agent/Judge validation loop ===")
    while True:
        feedback_history: List[str] = []
        scenario_attempts = 0
        for agent2_attempt in range(1, max_agent2_attempts + 1):
            cumulative_feedback = None
            if feedback_history:
                cumulative_feedback = (
                    "Fix every parsing mismatch without changing the scenario:\n\n"
                    + "\n\n".join(feedback_history[-3:])
                )
            raw = generator.parse_scenario_to_structured_json(
                current_scenario,
                previous_feedback=cumulative_feedback,
                iteration=agent2_attempt,
            )
            parsed_json, canonical_audit = _canonicalise_agent2_json_v8(raw)
            approved, judgment = generator.judge_scenario_parsing(
                current_scenario, parsed_json, iteration=agent2_attempt
            )
            total_attempts += 1
            scenario_attempts += 1
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
                    "termination_reason": "judge_approved",
                    "last_judgment": judgment,
                }, canonical_audit
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
                previous_feedback=_agent1_feedback_v15(last_judgment),
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
    print("WARNING: Judge 1 did not approve; output is diagnostic only.")
    return current_scenario, parsed_json, {
        "required": True,
        "approved": False,
        "attempts": total_attempts,
        "used_unapproved_parse": True,
        "agent1_revisions": agent1_revisions,
        "agent2_attempts_per_scenario": attempts_per_scenario,
        "termination_reason": termination_reason,
        "last_judgment": last_judgment,
    }, canonical_audit


def _blocked_candidate_total(generation_info: Dict[str, Any]) -> int:
    passes = generation_info.get("propagation_cycle_audit", {}).get(
        "declared_route_filter_passes", []
    )
    return sum(int(item.get("num_filtered_events", 0)) for item in passes)


def _write_outputs_v15(
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
    prefix = f"{domain_clean}_node{ts_data.shape[0]}_stpp_v15_{timestamp}"
    complete_data = {
        "timestamp": timestamp,
        "generator_family": "declared_route_schedule_stpp_v15",
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
    audit = generation_info["event_semantic_audit"]
    route = generation_info["declared_route_audit"]
    peak = generation_info["scalar_self_generated_peak_audit"]
    spatial = generation_info["spatial_layout_audit"]
    judge = generation_info["judge1_validation"]
    with description_path.open("w", encoding="utf-8") as handle:
        handle.write("=== STReasoner Declared-Route Schedule STPP v15 ===\n\n")
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
        handle.write(f"Quality gates passed: {quality['passed']}\n")
        handle.write(
            "Hawkes intensity upper bound: "
            f"{generation_info['point_process_parameters']['intensity_upper_bound']}\n"
        )
        handle.write(f"Spatial layout repaired: {spatial['repaired']}\n")
        handle.write(
            f"Blocked route/schedule candidates: {_blocked_candidate_total(generation_info)}\n"
        )
        handle.write(
            "Final route violations: "
            f"{len(route['propagated_event_violations'])}\n"
        )
        handle.write(
            "Unassigned semantic roots: "
            f"{len(route['unassigned_root_event_ids'])}\n"
        )
        handle.write(
            "Ambiguous semantic roots: "
            f"{len(route['ambiguous_root_event_ids'])}\n"
        )
        handle.write(
            "Declared scalar self-generated peaks: "
            f"{peak['num_declared_scalar_self_generated_peaks']}\n"
        )
        handle.write(
            "Uncovered scalar self-generated peaks: "
            f"{len(peak['uncovered_scalar_self_generated_peaks'])}\n"
        )
        handle.write(
            "Actual propagated lineages: "
            + json.dumps(route["actual_propagated_lineages"], ensure_ascii=False)
            + "\n"
        )
        handle.write(
            "Quality checks: "
            + json.dumps(quality["checks"], ensure_ascii=False)
            + "\n\n"
        )
        handle.write("Declared route audit\n--------------------\n")
        handle.write(json.dumps(route, indent=2, ensure_ascii=False))
        handle.write("\n\nScalar peak audit\n-----------------\n")
        handle.write(json.dumps(peak, indent=2, ensure_ascii=False))
        handle.write("\n\nEvent semantic audit\n--------------------\n")
        handle.write(json.dumps(audit, indent=2, ensure_ascii=False))
        handle.write("\n\nSpatial layout audit\n--------------------\n")
        handle.write(json.dumps(spatial, indent=2, ensure_ascii=False))
        handle.write("\n\nGraph conditional envelope audit\n--------------------------------\n")
        handle.write(
            json.dumps(
                generation_info["conditional_envelope_audit"],
                indent=2,
                ensure_ascii=False,
            )
        )
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


def demo_stpp_generation_v15(
    enabled_judges: List[int] | None = None,
    enable_logging: bool = True,
    num_nodes: int = 3,
    domain: str = "traffic",
    generate_viz: bool = False,
    stpp_config: STPPV15Config | None = None,
    output_dir: str = "output_stpp_v15",
) -> Dict[str, Any]:
    enabled_judges = [1] if enabled_judges is None else enabled_judges
    config = stpp_config or STPPV15Config()
    if generate_viz:
        print("v15 does not generate the legacy SDE visualization.")
    logger = AgentInteractionLogger() if enable_logging else None
    generator = DeclaredRouteJudgeNetworkSTPPGenerator(
        num_nodes=num_nodes, logger=logger
    )
    generator.domain = domain
    scenario, seq_len = generator.generate_scenario_with_length_validation()
    generator.seq_len = seq_len
    if 1 in enabled_judges:
        scenario, parsed, judge_audit, agent2_audit = (
            parse_scenario_with_judge_audit_v15(generator, scenario)
        )
    else:
        raw = generator.parse_scenario_to_structured_json(scenario)
        parsed, agent2_audit = _canonicalise_agent2_json_v8(raw)
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
    sampling_minutes = float(getattr(generator, "sampling_minutes", 60))
    lag_repair = repair_and_report_edge_lags_v6(
        scenario,
        structured_scenario,
        sampling_minutes,
        config.immediate_lag_epsilon_steps,
    )
    graph_metadata = build_graph_metadata_v2(structured_scenario, seq_len)
    ts_data, events, generation_info = simulate_stpp_for_streasoner_v15(
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
    quality["checks"]["no_causal_window_violations"] = not temporal_audit[
        "causal_window_violations"
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
    generation_info["temporal_contract_audit"] = temporal_audit
    generation_info["spatial_layout_audit"] = spatial_audit
    generation_info["sampling_minutes"] = sampling_minutes
    generation_info["dt"] = sampling_minutes / 60.0
    generation_info["lag_repair"] = lag_repair
    data_files = _write_outputs_v15(
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
        "generator_family": "declared_route_schedule_stpp_v15",
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
    parser.add_argument("--output_dir", default="output_stpp_v15")
    parser.add_argument(
        "--aggregation",
        choices=["count", "magnitude", "rolling_count", "rolling_magnitude"],
        default="magnitude",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--edge_branching_ratio", type=float, default=0.35)
    parser.add_argument("--target_immigrant_rate", type=float, default=0.30)
    args = parser.parse_args()
    judges = (
        []
        if args.judges.lower() == "none"
        else [int(item.strip()) for item in args.judges.split(",") if item.strip()]
    )
    config = STPPV15Config(
        aggregation=args.aggregation,
        seed=args.seed,
        edge_branching_ratio=args.edge_branching_ratio,
        target_immigrant_rate=args.target_immigrant_rate,
    )
    result = demo_stpp_generation_v15(
        enabled_judges=judges,
        num_nodes=args.num_nodes,
        domain=args.domain,
        stpp_config=config,
        output_dir=args.output_dir,
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
