#!/usr/bin/env python3
"""Non-destructive v4 entry point with lag repair and v4 quality gates."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from demo_sts_sde import AgentInteractionLogger, NetworkSDEGenerator
from demo_sts_stpp import parse_scenario_with_safe_judge_loop
from demo_sts_stpp_v2 import build_graph_metadata_v2
from stpp_adapter_v4 import STPPV4Config, simulate_stpp_for_streasoner_v4


def _duration_minutes(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit.startswith("minute"):
        return value
    if unit.startswith("hour"):
        return value * 60.0
    if unit.startswith("day"):
        return value * 1440.0
    raise ValueError(f"unsupported duration unit: {unit}")


def repair_fractional_edge_lags(
    scenario: str,
    structured_scenario: Dict[str, Any],
    sampling_minutes: float,
    zero_lag_fallback_steps: float = 0.5,
) -> List[Dict[str, Any]]:
    """Recover fractional time-step lags that Agent 2 rounded down to zero."""
    if sampling_minutes <= 0:
        raise ValueError("sampling_minutes must be positive")
    report: List[Dict[str, Any]] = []
    for edge in structured_scenario.get("edges", []):
        source, target = int(edge["source"]), int(edge["target"])
        parsed_value = float(edge.get("time_lag", 0) or 0)
        edge_pattern = re.compile(
            rf"NODE\s*{source}\s*(?:→|->|to)\s*NODE\s*{target}[^\n]*",
            flags=re.IGNORECASE,
        )
        line_match = edge_pattern.search(scenario)
        source_text = line_match.group(0) if line_match else ""
        duration_match = re.search(
            r"time\s*lag\s*[:=]?\s*(\d+(?:\.\d+)?)\s*"
            r"(minutes?|hours?|days?)",
            source_text,
            flags=re.IGNORECASE,
        )
        if not duration_match:
            duration_match = re.search(
                r"(\d+(?:\.\d+)?)\s*[- ]?\s*(minutes?|hours?|days?)"
                r"\s*(?:\([^)]*\)\s*)?"
                r"(?:delay|lag|travel(?:\s*time|\s*lag)?)",
                source_text,
                flags=re.IGNORECASE,
            )
        if duration_match:
            duration = float(duration_match.group(1))
            unit = duration_match.group(2)
            repaired = _duration_minutes(duration, unit) / sampling_minutes
            repair_source = "scenario_text"
        elif parsed_value > 0:
            repaired = parsed_value
            repair_source = "agent2_structured_value"
        else:
            repaired = zero_lag_fallback_steps
            repair_source = "zero_lag_fallback"
        edge["time_lag_agent2"] = parsed_value
        edge["time_lag"] = float(repaired)
        edge["time_lag_repair_source"] = repair_source
        report.append(
            {
                "edge": f"{source}->{target}",
                "agent2_value": parsed_value,
                "repaired_value_steps": float(repaired),
                "sampling_minutes": float(sampling_minutes),
                "source": repair_source,
                "matched_text": source_text,
            }
        )
    return report


def _write_outputs_v4(
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
    prefix = f"{domain_clean}_node{ts_data.shape[0]}_stpp_v4_{timestamp}"
    complete_data = {
        "timestamp": timestamp,
        "generator_family": "lag_aware_quota_balanced_graph_stpp_v4",
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
        handle.write("=== STReasoner Lag-Aware Quota-Balanced STPP v4 ===\n\n")
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
            f"Forced propagated events: {generation_info['num_forced_propagated_events']}\n"
        )
        handle.write(f"Aggregation: {generation_info['aggregation']}\n")
        handle.write(f"Quality gates passed: {quality['passed']}\n")
        handle.write(
            "Root quotas: " + json.dumps(quality["node_root_quotas"]) + "\n"
        )
        handle.write(
            "Root counts: " + json.dumps(quality["node_root_counts"]) + "\n"
        )
        handle.write(
            "Propagation root fraction: "
            f"{quality['propagation_root_fraction']:.4f}\n"
        )
        handle.write(
            "Quality checks: "
            + json.dumps(quality["checks"], ensure_ascii=False)
            + "\n\n"
        )
        handle.write("Lag repair\n----------\n")
        handle.write(
            json.dumps(generation_info["lag_repair"], indent=2, ensure_ascii=False)
        )
        handle.write("\n\nLag fidelity\n------------\n")
        handle.write(
            json.dumps(quality["lag_fidelity"], indent=2, ensure_ascii=False)
        )
        handle.write("\n\nScenario\n--------\n")
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


def demo_stpp_generation_v4(
    enabled_judges: List[int] | None = None,
    enable_logging: bool = True,
    num_nodes: int = 3,
    domain: str = "traffic",
    generate_viz: bool = False,
    stpp_config: STPPV4Config | None = None,
    output_dir: str = "output_stpp_v4",
) -> Dict[str, Any]:
    enabled_judges = [1] if enabled_judges is None else enabled_judges
    unsupported = sorted(set(enabled_judges) - {1})
    if unsupported:
        print(f"Ignoring SDE-specific judge(s): {unsupported}")
    if generate_viz:
        print("v4 does not generate the legacy SDE visualization.")

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

    sampling_minutes = float(getattr(generator, "sampling_minutes", 60))
    lag_repair = repair_fractional_edge_lags(
        scenario, structured_scenario, sampling_minutes
    )
    graph_metadata = build_graph_metadata_v2(structured_scenario, seq_len)
    ts_data, events, generation_info = simulate_stpp_for_streasoner_v4(
        structured_scenario, seq_len, stpp_config
    )
    generation_info["sampling_minutes"] = sampling_minutes
    generation_info["dt"] = sampling_minutes / 60.0
    generation_info["lag_repair"] = lag_repair
    data_files = _write_outputs_v4(
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
        "generator_family": "lag_aware_quota_balanced_graph_stpp_v4",
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
    parser.add_argument("--output_dir", default="output_stpp_v4")
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
    config = STPPV4Config(
        aggregation=args.aggregation,
        seed=args.seed,
        edge_branching_ratio=args.edge_branching_ratio,
        target_immigrant_rate=args.target_immigrant_rate,
    )
    result = demo_stpp_generation_v4(
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
