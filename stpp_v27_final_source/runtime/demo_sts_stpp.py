#!/usr/bin/env python3
"""Alternative STReasoner Stage 1 based on a spatial-temporal point process.

This file intentionally lives beside ``demo_sts_sde.py`` instead of replacing
it.  Agents 1/2 still create and parse a scenario, while data generation is
delegated directly to STPPG.  Raw events and the binned compatibility matrix
are both written to the output artifact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from demo_sts_sde import AgentInteractionLogger, NetworkSDEGenerator
from stpp_adapter import STPPConfig, simulate_stpp_for_streasoner


def parse_scenario_with_safe_judge_loop(
    generator: NetworkSDEGenerator,
    scenario: str,
    max_outer_iterations: int = 3,
    max_inner_iterations: int = 2,
) -> Dict[str, Any]:
    """Run Judge 1 without the upstream loop's null-feedback transition bug.

    In the original loop, exhausting Agent 2 retries changes ``error_source``
    to ``agent1`` but leaves ``agent1_feedback`` as ``None``.  The next print
    calls ``len(None)``.  This STPP-local copy always creates revision feedback
    before moving from the inner loop to the outer loop.
    """
    current_scenario = scenario
    previous_scenario: str | None = None
    agent1_feedback: str | None = None
    parsed_json: Dict[str, Any] | None = None

    print("\n=== STPP Agent 1 + Agent 2 + Judge 1 validation loop ===")
    for outer_iter in range(max_outer_iterations):
        print(
            f"\nOUTER LOOP - Iteration {outer_iter + 1}/{max_outer_iterations}"
        )
        if outer_iter > 0 and previous_scenario and agent1_feedback:
            current_scenario = generator.generate_scenario_description(
                previous_scenario=previous_scenario,
                previous_feedback=agent1_feedback,
                iteration=outer_iter + 1,
            )

        agent2_feedback: str | None = None
        needs_scenario_revision = False
        for inner_iter in range(max_inner_iterations):
            print(
                f"  Inner loop - Iteration {inner_iter + 1}/{max_inner_iterations}"
            )
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
            if approved:
                return parsed_json

            error_source = judgment.get("error_source")
            if error_source == "agent1":
                previous_scenario = current_scenario
                agent1_feedback = generator._format_feedback_for_agent1(  # noqa: SLF001
                    judgment
                )
                needs_scenario_revision = True
                break

            formatted_agent2_feedback = generator._format_feedback_for_agent2(  # noqa: SLF001
                judgment
            )
            if inner_iter < max_inner_iterations - 1:
                agent2_feedback = formatted_agent2_feedback
                continue

            # Repeated parsing failures usually indicate that the source text
            # has ambiguous ranges, paths, or mixed clock/index notation.
            previous_scenario = current_scenario
            agent1_feedback = (
                "Agent 2 repeatedly failed strict parsing. Rewrite the same "
                "scenario with explicit, non-conflicting numeric time ranges; "
                "use one representation for each peak; and distinguish event "
                "origin from the immediate upstream node.\n\n"
                + formatted_agent2_feedback
            )
            needs_scenario_revision = True
            break

        if not needs_scenario_revision or outer_iter == max_outer_iterations - 1:
            break

    if parsed_json is None:
        raise RuntimeError("Agent 2 did not return a structured scenario")
    print(
        "WARNING: Judge 1 did not approve within the retry limit; "
        "continuing with the last structurally valid parse."
    )
    return parsed_json


def _edge_strings(value: Any) -> Iterable[str]:
    if isinstance(value, list):
        for item in value:
            yield str(item)
    elif value is not None:
        for item in re.split(r"\s*[,;]\s*", str(value)):
            if item:
                yield item


def _time_range(value: Any) -> List[int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [int(value[0]), int(value[1])]
    numbers = re.findall(r"-?\d+", str(value))
    if len(numbers) >= 2:
        return [int(numbers[0]), int(numbers[1])]
    return None


def build_graph_metadata(structured_scenario: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the legacy graph schema without claiming it drives STPP events."""
    nodes = structured_scenario.get("nodes", [])
    num_nodes = len(nodes)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=float)
    for edge in structured_scenario.get("edges", []):
        source = int(edge["source"])
        target = int(edge["target"])
        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            raise ValueError(f"edge {source}->{target} references an unknown node")
        adjacency[source, target] = 0.1

    patterns: List[Dict[str, Any]] = []
    for raw_pattern in (
        structured_scenario.get("adjacency_modulation", {}).get("patterns", [])
    ):
        time_range = _time_range(
            raw_pattern.get("time_period", raw_pattern.get("time_range"))
        )
        if time_range is None:
            continue
        effect = str(raw_pattern.get("effect", "moderate")).lower()
        multiplier = 15.0 if effect == "strong" else 7.5
        description = str(raw_pattern.get("description", ""))
        edge_modulations: Dict[str, Dict[str, Any]] = {}
        for edge_text in _edge_strings(raw_pattern.get("applies_to")):
            match = re.search(r"(\d+)\s*->\s*(\d+)", edge_text)
            if match:
                key = f"{int(match.group(1))}->{int(match.group(2))}"
                edge_modulations[key] = {
                    "multiplier": multiplier,
                    "description": description,
                }
        if edge_modulations:
            patterns.append(
                {
                    "time_range": time_range,
                    "description": description,
                    "edge_modulations": edge_modulations,
                }
            )

    return {
        "base_adjacency": adjacency.tolist(),
        "time_modulation": {"patterns": patterns},
        "used_by_simulator": False,
        "note": (
            "Compatibility metadata parsed from the scenario. The upstream "
            "STPPG implementation is a continuous-space univariate Hawkes "
            "process and does not consume this node adjacency matrix."
        ),
    }


def _write_outputs(
    scenario: str,
    structured_scenario: Dict[str, Any],
    graph_metadata: Dict[str, Any],
    ts_data: np.ndarray,
    events: List[Dict[str, float | int]],
    generation_info: Dict[str, Any],
    domain: str,
    output_dir: Path,
) -> Dict[str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    domain_clean = re.sub(r"[^A-Za-z0-9_-]+", "_", domain).strip("_")
    prefix = f"{domain_clean}_node{len(structured_scenario['nodes'])}_stpp_{timestamp}"

    point_params = generation_info["point_process_parameters"]
    complete_data = {
        "timestamp": timestamp,
        "generator_family": "spatial_temporal_point_process",
        "agent1_scenario": scenario,
        "agent2_structured_scenario": structured_scenario,
        "agent3_point_process_parameters": point_params,
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
        handle.write("=== STReasoner Spatial-Temporal Point Process Generation ===\n\n")
        handle.write(f"Generation time: {dt.datetime.now().isoformat()}\n")
        handle.write(f"Domain: {domain}\n")
        handle.write(f"Nodes: {ts_data.shape[0]}\n")
        handle.write(f"Time windows: {ts_data.shape[1]}\n")
        handle.write(f"Raw events: {len(events)}\n")
        handle.write(f"Aggregation: {generation_info['aggregation']}\n")
        handle.write("Graph conditions simulator: no\n\n")
        handle.write("Scenario\n--------\n")
        handle.write(scenario)
        handle.write("\n\nPoint-process parameters\n------------------------\n")
        handle.write(json.dumps(point_params, indent=2, ensure_ascii=False))
        handle.write("\n")

    return {
        "pickle": str(pickle_path),
        "json": str(json_path),
        "description": str(description_path),
        "visualization": None,
    }


def demo_stpp_generation(
    enabled_judges: List[int] | None = None,
    enable_logging: bool = True,
    num_nodes: int = 3,
    domain: str = "traffic",
    generate_viz: bool = False,
    stpp_config: STPPConfig | None = None,
    output_dir: str = "output_stpp",
) -> Dict[str, Any]:
    """Run Agents 1/2, then generate STPP events without an SDE stage."""
    enabled_judges = [1] if enabled_judges is None else enabled_judges
    unsupported = sorted(set(enabled_judges) - {1})
    if unsupported:
        print(
            f"Ignoring judge(s) {unsupported}: Judge 2 validates SDE parameters, "
            "which do not exist in this branch."
        )
    if generate_viz:
        print("STPP compatibility branch does not generate the legacy SDE visualization.")

    logger = AgentInteractionLogger() if enable_logging else None
    scenario_agent = NetworkSDEGenerator(num_nodes=num_nodes, logger=logger)
    scenario_agent.domain = domain

    scenario, calculated_seq_len = (
        scenario_agent.generate_scenario_with_length_validation()
    )
    scenario_agent.seq_len = calculated_seq_len
    if 1 in enabled_judges:
        structured_scenario = parse_scenario_with_safe_judge_loop(
            scenario_agent,
            scenario,
            max_outer_iterations=3,
            max_inner_iterations=2,
        )
    else:
        structured_scenario = scenario_agent.parse_scenario_to_structured_json(scenario)

    graph_metadata = build_graph_metadata(structured_scenario)
    ts_data, events, generation_info = simulate_stpp_for_streasoner(
        structured_scenario,
        seq_len=calculated_seq_len,
        config=stpp_config,
    )
    generation_info["graph_conditioned"] = False
    generation_info["sampling_minutes"] = getattr(
        scenario_agent, "sampling_minutes", 60
    )
    generation_info["dt"] = generation_info["sampling_minutes"] / 60.0

    data_files = _write_outputs(
        scenario=scenario,
        structured_scenario=structured_scenario,
        graph_metadata=graph_metadata,
        ts_data=ts_data,
        events=events,
        generation_info=generation_info,
        domain=domain,
        output_dir=Path(output_dir),
    )
    if logger:
        logger.save_complete_log()

    return {
        "generator_family": "spatial_temporal_point_process",
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
    parser.add_argument("--domain", type=str, default="traffic")
    parser.add_argument("--judges", type=str, default="1")
    parser.add_argument("--output_dir", type=str, default="output_stpp")
    parser.add_argument(
        "--aggregation",
        choices=["count", "rolling_count", "cumulative_count"],
        default="count",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    judges = (
        []
        if args.judges.lower() == "none"
        else [int(item.strip()) for item in args.judges.split(",") if item.strip()]
    )
    config = STPPConfig(aggregation=args.aggregation, seed=args.seed)
    result = demo_stpp_generation(
        enabled_judges=judges,
        num_nodes=args.num_nodes,
        domain=args.domain,
        stpp_config=config,
        output_dir=args.output_dir,
    )
    print(f"Generated {len(result['agent5_event_stream'])} raw events")
    print(f"Compatibility matrix shape: {result['agent5_simulation_data'].shape}")
    print(f"Pickle: {result['data_files']['pickle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
