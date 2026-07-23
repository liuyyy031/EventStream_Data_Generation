#!/usr/bin/env python3
"""Non-destructive v3 Stage 1 entry point with density and quality gates."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from demo_sts_sde import AgentInteractionLogger, NetworkSDEGenerator
from demo_sts_stpp import parse_scenario_with_safe_judge_loop
from demo_sts_stpp_v2 import build_graph_metadata_v2
from stpp_adapter_v3 import (
    STPPV3Config,
    simulate_stpp_for_streasoner_v3,
)


def _write_outputs_v3(
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
    prefix = f"{domain_clean}_node{ts_data.shape[0]}_stpp_v3_{timestamp}"
    complete_data = {
        "timestamp": timestamp,
        "generator_family": "quality_gated_graph_stpp_v3",
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
        handle.write("=== STReasoner Quality-Gated Graph STPP v3 ===\n\n")
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
        handle.write(f"Aggregation: {generation_info['aggregation']}\n")
        handle.write(f"Quality gates passed: {quality['passed']}\n")
        handle.write(
            "Quality checks: "
            + json.dumps(quality["checks"], ensure_ascii=False)
            + "\n\n"
        )
        handle.write("Scenario\n--------\n")
        handle.write(scenario)
        handle.write("\n\nPoint-process parameters\n------------------------\n")
        handle.write(
            json.dumps(
                generation_info["point_process_parameters"],
                indent=2,
                ensure_ascii=False,
            )
        )
        handle.write("\n")
    return {
        "pickle": str(pickle_path),
        "json": str(json_path),
        "description": str(description_path),
        "visualization": None,
    }


def demo_stpp_generation_v3(
    enabled_judges: List[int] | None = None,
    enable_logging: bool = True,
    num_nodes: int = 3,
    domain: str = "traffic",
    generate_viz: bool = False,
    stpp_config: STPPV3Config | None = None,
    output_dir: str = "output_stpp_v3",
) -> Dict[str, Any]:
    enabled_judges = [1] if enabled_judges is None else enabled_judges
    unsupported = sorted(set(enabled_judges) - {1})
    if unsupported:
        print(f"Ignoring SDE-specific judge(s): {unsupported}")
    if generate_viz:
        print("v3 does not generate the legacy SDE visualization.")

    logger = AgentInteractionLogger() if enable_logging else None
    generator = NetworkSDEGenerator(num_nodes=num_nodes, logger=logger)
    generator.domain = domain
    scenario, seq_len = generator.generate_scenario_with_length_validation()
    generator.seq_len = seq_len
    if 1 in enabled_judges:
        structured_scenario = parse_scenario_with_safe_judge_loop(
            generator, scenario, max_outer_iterations=3, max_inner_iterations=2
        )
    else:
        structured_scenario = generator.parse_scenario_to_structured_json(scenario)

    graph_metadata = build_graph_metadata_v2(structured_scenario, seq_len)
    ts_data, events, generation_info = simulate_stpp_for_streasoner_v3(
        structured_scenario, seq_len, stpp_config
    )
    generation_info["sampling_minutes"] = getattr(generator, "sampling_minutes", 60)
    generation_info["dt"] = generation_info["sampling_minutes"] / 60.0
    data_files = _write_outputs_v3(
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
        "generator_family": "quality_gated_graph_stpp_v3",
        "agent1_scenario": scenario,
        "agent2_structured_scenario": structured_scenario,
        "agent3_point_process_parameters": generation_info[
            "point_process_parameters"
        ],
        "agent4_time_varying_adjacency": graph_metadata,
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
    parser.add_argument("--output_dir", default="output_stpp_v3")
    parser.add_argument(
        "--aggregation",
        choices=[
            "count",
            "magnitude",
            "rolling_count",
            "rolling_magnitude",
            "cumulative_count",
            "cumulative_magnitude",
        ],
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
    config = STPPV3Config(
        aggregation=args.aggregation,
        seed=args.seed,
        edge_branching_ratio=args.edge_branching_ratio,
        target_immigrant_rate=args.target_immigrant_rate,
    )
    result = demo_stpp_generation_v3(
        enabled_judges=judges,
        num_nodes=args.num_nodes,
        domain=args.domain,
        stpp_config=config,
        output_dir=args.output_dir,
    )
    info = result["generation_info"]
    print(f"Generated {info['num_events']} events")
    print(f"STPP immigrants: {info['num_stpp_immigrant_events']}")
    print(f"Scenario seeds: {info['num_scenario_seed_events']}")
    print(f"Propagated events: {info['num_propagated_events']}")
    print(f"Quality gates passed: {info['quality_report']['passed']}")
    print(f"Compatibility matrix shape: {result['agent5_simulation_data'].shape}")
    print(f"Pickle: {result['data_files']['pickle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

