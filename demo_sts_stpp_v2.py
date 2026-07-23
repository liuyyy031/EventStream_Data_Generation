#!/usr/bin/env python3
"""Non-destructive v2 Stage 1 entry point for graph-conditioned STPP data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from demo_sts_sde import AgentInteractionLogger, NetworkSDEGenerator
from demo_sts_stpp import parse_scenario_with_safe_judge_loop
from stpp_adapter_v2 import (
    STPPV2Config,
    _numbers_from_time_spec,
    active_time_indices,
    simulate_stpp_for_streasoner_v2,
)


EFFECT_MULTIPLIERS = {"strong": 15.0, "moderate": 7.5, "weak": 0.5}


def _edge_strings(value: Any, all_edges: Sequence[str]) -> Iterable[str]:
    if isinstance(value, list):
        candidates = [str(item) for item in value]
    elif value is None:
        candidates = []
    else:
        candidates = re.split(r"\s*[,;]\s*", str(value))
    if any("all edge" in candidate.lower() for candidate in candidates):
        yield from all_edges
    else:
        yield from candidates


def _contiguous_ranges(indices: Sequence[int]) -> List[List[int]]:
    if not indices:
        return []
    values = sorted(set(int(index) for index in indices))
    result: List[List[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append([start, previous])
            start = value
        previous = value
    result.append([start, previous])
    return result


def build_graph_metadata_v2(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Dict[str, Any]:
    """Preserve complete schedules and distinguish strong/moderate/weak."""
    num_nodes = len(structured_scenario.get("nodes", []))
    adjacency = np.zeros((num_nodes, num_nodes), dtype=float)
    all_edges: List[str] = []
    for edge in structured_scenario.get("edges", []):
        source, target = int(edge["source"]), int(edge["target"])
        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            raise ValueError(f"edge {source}->{target} references an unknown node")
        adjacency[source, target] = 0.1
        all_edges.append(f"{source}->{target}")

    output_patterns: List[Dict[str, Any]] = []
    raw_patterns = (
        structured_scenario.get("adjacency_modulation", {}).get("patterns", [])
    )
    for pattern_index, raw_pattern in enumerate(raw_patterns):
        time_spec = raw_pattern.get("time_period", raw_pattern.get("time_range"))
        indices = active_time_indices(time_spec, raw_pattern, seq_len)
        if not indices:
            continue
        effect = str(raw_pattern.get("effect", "moderate")).lower()
        multiplier = EFFECT_MULTIPLIERS.get(effect, 1.0)
        description = str(raw_pattern.get("description", ""))
        modulations: Dict[str, Dict[str, Any]] = {}
        for text in _edge_strings(raw_pattern.get("applies_to"), all_edges):
            match = re.search(r"(\d+)\s*->\s*(\d+)", text)
            if not match:
                continue
            key = f"{int(match.group(1))}->{int(match.group(2))}"
            modulations[key] = {
                "effect": effect,
                "multiplier": multiplier,
                "description": description,
            }
        if not modulations:
            continue
        for time_range in _contiguous_ranges(indices):
            output_patterns.append(
                {
                    "source_pattern_index": pattern_index,
                    "time_range": time_range,
                    "time_steps": list(range(time_range[0], time_range[1] + 1)),
                    "original_time_spec": _numbers_from_time_spec(time_spec),
                    "description": description,
                    "edge_modulations": modulations,
                }
            )

    return {
        "base_adjacency": adjacency.tolist(),
        "time_modulation": {"patterns": output_patterns},
        "used_by_simulator": True,
        "effect_multiplier_mapping": EFFECT_MULTIPLIERS,
        "note": (
            "v2 preserves every expanded time step. The same scenario schedule "
            "also conditions immigrant thinning and graph offspring generation."
        ),
    }


def _write_outputs_v2(
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
    prefix = f"{domain_clean}_node{ts_data.shape[0]}_stpp_v2_{timestamp}"
    complete_data = {
        "timestamp": timestamp,
        "generator_family": "graph_conditioned_spatial_temporal_point_process_v2",
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
        handle.write("=== STReasoner Graph-Conditioned STPP v2 ===\n\n")
        handle.write(f"Generation time: {dt.datetime.now().isoformat()}\n")
        handle.write(f"Domain: {domain}\n")
        handle.write(f"Nodes: {ts_data.shape[0]}\n")
        handle.write(f"Time windows: {ts_data.shape[1]}\n")
        handle.write(f"All events: {len(events)}\n")
        handle.write(
            f"Immigrant events: {generation_info['num_immigrant_events']}\n"
        )
        handle.write(
            f"Propagated events: {generation_info['num_propagated_events']}\n"
        )
        handle.write("Graph conditions simulator: yes\n")
        handle.write("Scenario conditions simulator: yes\n\n")
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


def demo_stpp_generation_v2(
    enabled_judges: List[int] | None = None,
    enable_logging: bool = True,
    num_nodes: int = 3,
    domain: str = "traffic",
    generate_viz: bool = False,
    stpp_config: STPPV2Config | None = None,
    output_dir: str = "output_stpp_v2",
) -> Dict[str, Any]:
    enabled_judges = [1] if enabled_judges is None else enabled_judges
    unsupported = sorted(set(enabled_judges) - {1})
    if unsupported:
        print(f"Ignoring SDE-specific judge(s): {unsupported}")
    if generate_viz:
        print("v2 does not generate the legacy SDE visualization.")

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
    ts_data, events, generation_info = simulate_stpp_for_streasoner_v2(
        structured_scenario, seq_len, stpp_config
    )
    generation_info["sampling_minutes"] = getattr(generator, "sampling_minutes", 60)
    generation_info["dt"] = generation_info["sampling_minutes"] / 60.0
    data_files = _write_outputs_v2(
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
        "generator_family": "graph_conditioned_spatial_temporal_point_process_v2",
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
    parser.add_argument("--output_dir", default="output_stpp_v2")
    parser.add_argument(
        "--aggregation",
        choices=["count", "rolling_count", "cumulative_count"],
        default="count",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--edge_branching_ratio", type=float, default=0.35)
    args = parser.parse_args()
    judges = (
        []
        if args.judges.lower() == "none"
        else [int(item.strip()) for item in args.judges.split(",") if item.strip()]
    )
    config = STPPV2Config(
        aggregation=args.aggregation,
        seed=args.seed,
        edge_branching_ratio=args.edge_branching_ratio,
    )
    result = demo_stpp_generation_v2(
        enabled_judges=judges,
        num_nodes=args.num_nodes,
        domain=args.domain,
        stpp_config=config,
        output_dir=args.output_dir,
    )
    info = result["generation_info"]
    print(f"Generated {info['num_events']} events")
    print(f"Immigrant events: {info['num_immigrant_events']}")
    print(f"Propagated events: {info['num_propagated_events']}")
    print(f"Compatibility matrix shape: {result['agent5_simulation_data'].shape}")
    print(f"Pickle: {result['data_files']['pickle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
