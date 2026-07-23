#!/usr/bin/env python3
"""STPP v9 entry point with cycle-free simple-path graph propagation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from demo_sts_sde import AgentInteractionLogger
from demo_sts_stpp_v2 import build_graph_metadata_v2
from demo_sts_stpp_v6 import repair_and_report_edge_lags_v6
from demo_sts_stpp_v8 import (
    _canonicalise_agent2_json_v8,
    parse_scenario_with_judge_audit_v8,
)
from safe_judge_generator import SafeJudgeNetworkSDEGenerator
from stpp_adapter_v9 import STPPV9Config, simulate_stpp_for_streasoner_v9


def _write_outputs_v9(
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
    prefix = f"{domain_clean}_node{ts_data.shape[0]}_stpp_v9_{timestamp}"
    complete_data = {
        "timestamp": timestamp,
        "generator_family": "cycle_free_simple_path_graph_stpp_v9",
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
        lineage = generation_info["lineage_quality"]
        cycle_audit = generation_info["propagation_cycle_audit"]
        handle.write("=== STReasoner Cycle-Free Simple-Path STPP v9 ===\n\n")
        handle.write(f"Generation time: {dt.datetime.now().isoformat()}\n")
        handle.write(f"Domain: {domain}\n")
        handle.write(f"Nodes: {ts_data.shape[0]}\n")
        handle.write(f"Time windows: {ts_data.shape[1]}\n")
        handle.write(f"All events: {generation_info['num_events']}\n")
        handle.write(f"Root events: {generation_info['num_root_events']}\n")
        handle.write(
            f"Propagated events: {generation_info['num_propagated_events']}\n"
        )
        handle.write(
            f"Scenario seed events: {generation_info['num_scenario_seed_events']}\n"
        )
        handle.write(f"Judge 1 approved: {judge['approved']}\n")
        handle.write(f"Judge 1 attempts: {judge['attempts']}\n")
        handle.write(f"Quality gates passed: {quality['passed']}\n")
        handle.write("Root quotas: " + json.dumps(quality["node_root_quotas"]) + "\n")
        handle.write("Root counts: " + json.dumps(quality["node_root_counts"]) + "\n")
        handle.write(
            "Propagation-node root events: "
            f"{quality['propagation_node_root_event_count']}\n"
        )
        handle.write(
            "Inbound-envelope max matrix ratio: "
            f"{quality['max_matrix_envelope_ratio']:.6f}\n"
        )
        handle.write(
            "Generation counts: "
            + json.dumps(lineage["generation_counts"], ensure_ascii=False)
            + "\n"
        )
        handle.write(f"Maximum lineage nodes: {lineage['max_lineage_nodes']}\n")
        handle.write(
            "Blocked immediate backtracks: "
            f"{cycle_audit['blocked_immediate_backtracks']}\n"
        )
        handle.write(
            "Blocked other lineage revisits: "
            f"{cycle_audit['blocked_lineage_revisits']}\n"
        )
        handle.write(
            "Blocked forced revisits: "
            f"{cycle_audit['blocked_forced_revisits']}\n"
        )
        handle.write(
            "Quality checks: "
            + json.dumps(quality["checks"], ensure_ascii=False)
            + "\n\n"
        )
        handle.write("Propagation cycle audit\n-----------------------\n")
        handle.write(json.dumps(cycle_audit, indent=2, ensure_ascii=False))
        handle.write("\n\nLineage quality\n---------------\n")
        handle.write(json.dumps(lineage, indent=2, ensure_ascii=False))
        handle.write("\n\nSelf-generated peak coverage\n----------------------------\n")
        handle.write(
            json.dumps(
                quality["self_generated_peak_coverage"],
                indent=2,
                ensure_ascii=False,
            )
        )
        handle.write("\n\nAgent 2 canonicalisation\n------------------------\n")
        handle.write(
            json.dumps(
                generation_info["agent2_canonicalisation"],
                indent=2,
                ensure_ascii=False,
            )
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


def demo_stpp_generation_v9(
    enabled_judges: List[int] | None = None,
    enable_logging: bool = True,
    num_nodes: int = 3,
    domain: str = "traffic",
    generate_viz: bool = False,
    stpp_config: STPPV9Config | None = None,
    output_dir: str = "output_stpp_v9",
) -> Dict[str, Any]:
    enabled_judges = [1] if enabled_judges is None else enabled_judges
    config = stpp_config or STPPV9Config()
    if generate_viz:
        print("v9 does not generate the legacy SDE visualization.")
    logger = AgentInteractionLogger() if enable_logging else None
    generator = SafeJudgeNetworkSDEGenerator(num_nodes=num_nodes, logger=logger)
    generator.domain = domain
    scenario, seq_len = generator.generate_scenario_with_length_validation()
    generator.seq_len = seq_len
    if 1 in enabled_judges:
        scenario, structured_scenario, judge_audit, canonical_audit = (
            parse_scenario_with_judge_audit_v8(generator, scenario)
        )
    else:
        raw = generator.parse_scenario_to_structured_json(scenario)
        structured_scenario, canonical_audit = _canonicalise_agent2_json_v8(raw)
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
    ts_data, events, generation_info = simulate_stpp_for_streasoner_v9(
        structured_scenario, seq_len, config
    )
    quality = generation_info["quality_report"]
    parsed_seq_len = int(structured_scenario.get("seq_len", seq_len))
    quality["checks"]["judge1_approved_or_not_required"] = (
        not judge_audit["required"] or judge_audit["approved"] is True
    )
    quality["checks"]["agent2_seq_len_matches_generator"] = parsed_seq_len == seq_len
    quality["checks"]["agent2_canonicalisation_valid"] = not canonical_audit.get(
        "invalid_changes", []
    )
    quality["passed"] = all(quality["checks"].values())
    generation_info["judge1_validation"] = judge_audit
    generation_info["agent2_canonicalisation"] = canonical_audit
    generation_info["sampling_minutes"] = sampling_minutes
    generation_info["dt"] = sampling_minutes / 60.0
    generation_info["lag_repair"] = lag_repair
    data_files = _write_outputs_v9(
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
        "generator_family": "cycle_free_simple_path_graph_stpp_v9",
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
    parser.add_argument("--output_dir", default="output_stpp_v9")
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
    config = STPPV9Config(
        aggregation=args.aggregation,
        seed=args.seed,
        edge_branching_ratio=args.edge_branching_ratio,
        target_immigrant_rate=args.target_immigrant_rate,
    )
    result = demo_stpp_generation_v9(
        enabled_judges=judges,
        num_nodes=args.num_nodes,
        domain=args.domain,
        stpp_config=config,
        output_dir=args.output_dir,
    )
    info = result["generation_info"]
    cycle_audit = info["propagation_cycle_audit"]
    print(f"Generated {info['num_events']} events")
    print(f"Judge 1 approved: {info['judge1_validation']['approved']}")
    print(f"Quality gates passed: {info['quality_report']['passed']}")
    print(
        "Blocked cyclic propagation: "
        f"{cycle_audit['blocked_immediate_backtracks'] + cycle_audit['blocked_lineage_revisits']}"
    )
    print(f"Pickle: {result['data_files']['pickle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
