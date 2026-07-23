#!/usr/bin/env python3
"""Integrated v7 entry point with safe Judge and fail-closed approval QA."""

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
from safe_judge_generator import SafeJudgeNetworkSDEGenerator
from stpp_adapter_v7 import STPPV7Config, simulate_stpp_for_streasoner_v7


def parse_scenario_with_judge_audit_v7(
    generator: SafeJudgeNetworkSDEGenerator,
    scenario: str,
    max_outer_iterations: int = 3,
    max_inner_iterations: int = 2,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    current_scenario = scenario
    previous_scenario: str | None = None
    agent1_feedback: str | None = None
    parsed_json: Dict[str, Any] | None = None
    attempts = 0
    last_judgment: Dict[str, Any] | None = None

    print("\n=== STPP v7 Agent 1 + Agent 2 + Judge 1 validation loop ===")
    for outer_iter in range(max_outer_iterations):
        print(f"\nOUTER LOOP - Iteration {outer_iter + 1}/{max_outer_iterations}")
        if outer_iter > 0 and previous_scenario and agent1_feedback:
            current_scenario = generator.generate_scenario_description(
                previous_scenario=previous_scenario,
                previous_feedback=agent1_feedback,
                iteration=outer_iter + 1,
            )

        agent2_feedback: str | None = None
        needs_scenario_revision = False
        for inner_iter in range(max_inner_iterations):
            print(f"  Inner loop - Iteration {inner_iter + 1}/{max_inner_iterations}")
            parsed_json = generator.parse_scenario_to_structured_json(
                current_scenario,
                previous_feedback=agent2_feedback,
                iteration=inner_iter + 1,
            )
            approved, judgment = generator.judge_scenario_parsing(
                current_scenario,
                parsed_json,
                iteration=inner_iter + 1,
            )
            attempts += 1
            last_judgment = judgment
            if approved:
                return current_scenario, parsed_json, {
                    "required": True,
                    "approved": True,
                    "attempts": attempts,
                    "used_unapproved_parse": False,
                    "last_judgment": judgment,
                }

            if judgment.get("error_source") == "agent1":
                previous_scenario = current_scenario
                agent1_feedback = generator._format_feedback_for_agent1(  # noqa: SLF001
                    judgment
                )
                needs_scenario_revision = True
                break

            feedback = generator._format_feedback_for_agent2(  # noqa: SLF001
                judgment
            )
            if inner_iter < max_inner_iterations - 1:
                agent2_feedback = feedback
                continue
            previous_scenario = current_scenario
            agent1_feedback = (
                "Agent 2 repeatedly failed strict parsing. Rewrite the same "
                "scenario with explicit numeric ranges, scalar or list-valued "
                "peaks, weekday restrictions, and explicit self-generated or "
                "propagated origin labels.\n\n"
                + feedback
            )
            needs_scenario_revision = True
            break

        if not needs_scenario_revision or outer_iter == max_outer_iterations - 1:
            break

    if parsed_json is None:
        raise RuntimeError("Agent 2 did not return a structured scenario")
    print(
        "WARNING: Judge 1 did not approve within the retry limit; "
        "the sample will be generated for diagnosis but fail its quality gate."
    )
    return current_scenario, parsed_json, {
        "required": True,
        "approved": False,
        "attempts": attempts,
        "used_unapproved_parse": True,
        "last_judgment": last_judgment,
    }


def _write_outputs_v7(
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
    prefix = f"{domain_clean}_node{ts_data.shape[0]}_stpp_v7_{timestamp}"
    complete_data = {
        "timestamp": timestamp,
        "generator_family": "conservative_semantic_graph_stpp_v7",
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
    with description_path.open("w", encoding="utf-8") as handle:
        quality = generation_info["quality_report"]
        judge = generation_info["judge1_validation"]
        handle.write("=== STReasoner Conservative Semantic STPP v7 ===\n\n")
        handle.write(f"Generation time: {dt.datetime.now().isoformat()}\n")
        handle.write(f"Domain: {domain}\n")
        handle.write(f"Nodes: {ts_data.shape[0]}\n")
        handle.write(f"Time windows: {ts_data.shape[1]}\n")
        handle.write(f"All events: {generation_info['num_events']}\n")
        handle.write(
            f"STPP immigrant events: {generation_info['num_stpp_immigrant_events']}\n"
        )
        handle.write(
            f"Scenario seed events: {generation_info['num_scenario_seed_events']}\n"
        )
        handle.write(
            f"Propagated events: {generation_info['num_propagated_events']}\n"
        )
        handle.write(
            "Forced propagated events: "
            f"{generation_info['num_forced_propagated_events']}\n"
        )
        handle.write(f"Judge 1 required: {judge['required']}\n")
        handle.write(f"Judge 1 approved: {judge['approved']}\n")
        handle.write(f"Judge 1 attempts: {judge['attempts']}\n")
        handle.write(
            "Agent 2 seq_len matches generator: "
            f"{generation_info['agent2_seq_len_matches_generator']}\n"
        )
        handle.write(f"Quality gates passed: {quality['passed']}\n")
        handle.write(
            "Root quotas: " + json.dumps(quality["node_root_quotas"]) + "\n"
        )
        handle.write(
            "Root counts: " + json.dumps(quality["node_root_counts"]) + "\n"
        )
        handle.write(
            "Propagation-node root events: "
            f"{quality['propagation_node_root_event_count']}\n"
        )
        handle.write(
            "Inbound-envelope max matrix ratio: "
            f"{quality['max_matrix_envelope_ratio']:.6f}\n"
        )
        handle.write(
            "Quality checks: "
            + json.dumps(quality["checks"], ensure_ascii=False)
            + "\n\n"
        )
        handle.write("Root pattern conditioning\n-------------------------\n")
        handle.write(
            json.dumps(
                generation_info["root_pattern_conditioning"],
                indent=2,
                ensure_ascii=False,
            )
        )
        handle.write("\n\nSkipped relay root seeds\n------------------------\n")
        handle.write(
            json.dumps(
                generation_info["skipped_relay_strong_edge_root_seeds"],
                indent=2,
                ensure_ascii=False,
            )
        )
        handle.write("\n\nLag repair\n----------\n")
        handle.write(
            json.dumps(generation_info["lag_repair"], indent=2, ensure_ascii=False)
        )
        handle.write("\n\nLag fidelity\n------------\n")
        handle.write(
            json.dumps(quality["lag_fidelity"], indent=2, ensure_ascii=False)
        )
        handle.write("\n\nScenario\n--------\n")
        handle.write(scenario)
        handle.write("\n")
    return {
        "pickle": str(pickle_path),
        "json": str(json_path),
        "description": str(description_path),
        "visualization": None,
    }


def demo_stpp_generation_v7(
    enabled_judges: List[int] | None = None,
    enable_logging: bool = True,
    num_nodes: int = 3,
    domain: str = "traffic",
    generate_viz: bool = False,
    stpp_config: STPPV7Config | None = None,
    output_dir: str = "output_stpp_v7",
) -> Dict[str, Any]:
    enabled_judges = [1] if enabled_judges is None else enabled_judges
    unsupported = sorted(set(enabled_judges) - {1})
    if unsupported:
        print(f"Ignoring SDE-specific judge(s): {unsupported}")
    if generate_viz:
        print("v7 does not generate the legacy SDE visualization.")
    config = stpp_config or STPPV7Config()

    logger = AgentInteractionLogger() if enable_logging else None
    generator = SafeJudgeNetworkSDEGenerator(num_nodes=num_nodes, logger=logger)
    generator.domain = domain
    scenario, seq_len = generator.generate_scenario_with_length_validation()
    generator.seq_len = seq_len
    if 1 in enabled_judges:
        scenario, structured_scenario, judge_audit = (
            parse_scenario_with_judge_audit_v7(generator, scenario)
        )
    else:
        structured_scenario = generator.parse_scenario_to_structured_json(scenario)
        judge_audit = {
            "required": False,
            "approved": None,
            "attempts": 0,
            "used_unapproved_parse": False,
            "last_judgment": None,
        }

    sampling_minutes = float(getattr(generator, "sampling_minutes", 60))
    lag_repair = repair_and_report_edge_lags_v6(
        scenario,
        structured_scenario,
        sampling_minutes,
        config.immediate_lag_epsilon_steps,
    )
    graph_metadata = build_graph_metadata_v2(structured_scenario, seq_len)
    ts_data, events, generation_info = simulate_stpp_for_streasoner_v7(
        structured_scenario, seq_len, config
    )
    quality = generation_info["quality_report"]
    parsed_seq_len = int(structured_scenario.get("seq_len", seq_len))
    seq_len_matches = parsed_seq_len == seq_len
    quality["checks"]["judge1_approved_or_not_required"] = (
        not judge_audit["required"] or judge_audit["approved"] is True
    )
    quality["checks"]["agent2_seq_len_matches_generator"] = seq_len_matches
    quality["passed"] = all(quality["checks"].values())
    generation_info["judge1_validation"] = judge_audit
    generation_info["agent2_seq_len"] = parsed_seq_len
    generation_info["agent2_seq_len_matches_generator"] = seq_len_matches
    generation_info["sampling_minutes"] = sampling_minutes
    generation_info["dt"] = sampling_minutes / 60.0
    generation_info["lag_repair"] = lag_repair
    data_files = _write_outputs_v7(
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
        "generator_family": "conservative_semantic_graph_stpp_v7",
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
    parser.add_argument("--output_dir", default="output_stpp_v7")
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
    config = STPPV7Config(
        aggregation=args.aggregation,
        seed=args.seed,
        edge_branching_ratio=args.edge_branching_ratio,
        target_immigrant_rate=args.target_immigrant_rate,
    )
    result = demo_stpp_generation_v7(
        enabled_judges=judges,
        num_nodes=args.num_nodes,
        domain=args.domain,
        stpp_config=config,
        output_dir=args.output_dir,
    )
    info = result["generation_info"]
    print(f"Generated {info['num_events']} events")
    print(f"Judge 1 approved: {info['judge1_validation']['approved']}")
    print(f"Quality gates passed: {info['quality_report']['passed']}")
    print(f"Compatibility matrix shape: {result['agent5_simulation_data'].shape}")
    print(f"Pickle: {result['data_files']['pickle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
