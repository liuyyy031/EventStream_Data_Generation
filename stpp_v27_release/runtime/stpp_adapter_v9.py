"""Cycle-free simple-path propagation for STPP v9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

import stpp_adapter_v4 as _v4_module
from stpp_adapter_v2 import _edge_key
from stpp_adapter_v3 import _strong_edge_windows
from stpp_adapter_v8 import STPPV8Config, simulate_stpp_for_streasoner_v8


@dataclass(frozen=True)
class STPPV9Config(STPPV8Config):
    prevent_lineage_node_revisit: bool = True
    prevent_immediate_edge_backtracking: bool = True


def _root_lineage(event: Mapping[str, Any]) -> Tuple[List[int], List[str]]:
    nodes = event.get("lineage_nodes")
    edges = event.get("lineage_edges")
    if isinstance(nodes, list) and nodes:
        return [int(node) for node in nodes], [str(edge) for edge in (edges or [])]
    return [int(event["node_id"])], []


def _propagate_simple_paths_v9(
    immigrants: List[Dict[str, Any]],
    outgoing: Dict[int, List[Dict[str, Any]]],
    edge_factors: Dict[str, np.ndarray],
    node_coords: np.ndarray,
    seq_len: int,
    config: STPPV9Config,
    rng: np.random.Generator,
    audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    events = list(immigrants)
    for root in events:
        root["lineage_nodes"] = [int(root["node_id"])]
        root["lineage_edges"] = []
    queue = list(events)
    next_id = max((int(event["event_id"]) for event in events), default=-1) + 1
    while queue and len(events) < config.max_events:
        parent = queue.pop(0)
        if int(parent["generation"]) >= config.max_generation:
            continue
        source = int(parent["node_id"])
        lineage_nodes, lineage_edges = _root_lineage(parent)
        for edge in outgoing.get(source, []):
            target = int(edge["target"])
            key = _edge_key(source, target)
            immediate_backtrack = (
                len(lineage_nodes) >= 2 and target == lineage_nodes[-2]
            )
            revisit = target in lineage_nodes
            if immediate_backtrack and config.prevent_immediate_edge_backtracking:
                audit["blocked_immediate_backtracks"] += 1
                audit["blocked_by_edge"][key] = (
                    audit["blocked_by_edge"].get(key, 0) + 1
                )
                continue
            if revisit and config.prevent_lineage_node_revisit:
                audit["blocked_lineage_revisits"] += 1
                audit["blocked_by_edge"][key] = (
                    audit["blocked_by_edge"].get(key, 0) + 1
                )
                continue

            factor = float(edge_factors[key][int(parent["time_index"])])
            probability = min(config.edge_branching_ratio * factor, 0.95)
            if rng.random() > probability:
                continue
            delay = max(
                0.001,
                float(rng.normal(float(edge["lag"]), config.propagation_jitter)),
            )
            child_time = float(parent["t"]) + delay
            if child_time >= seq_len:
                continue
            x, y = rng.normal(
                loc=node_coords[target], scale=config.target_spatial_jitter, size=2
            )
            child = {
                "event_id": next_id,
                "t": child_time,
                "x": float(np.clip(x, 0.0, 1.0)),
                "y": float(np.clip(y, 0.0, 1.0)),
                "node_id": target,
                "time_index": int(np.floor(child_time)),
                "event_type": "propagated",
                "parent_event_id": int(parent["event_id"]),
                "source_node_id": source,
                "generation": int(parent["generation"]) + 1,
                "edge": key,
                "edge_probability": probability,
                "configured_lag": float(edge["lag"]),
                "realized_delay": delay,
                "lineage_nodes": lineage_nodes + [target],
                "lineage_edges": lineage_edges + [key],
            }
            events.append(child)
            queue.append(child)
            next_id += 1
            if len(events) >= config.max_events:
                break
    return sorted(events, key=lambda event: (float(event["t"]), int(event["event_id"])))


def _force_strong_edge_coverage_v9(
    events: List[Dict[str, Any]],
    structured_scenario: Mapping[str, Any],
    node_coords: np.ndarray,
    seq_len: int,
    config: STPPV9Config,
    rng: np.random.Generator,
    audit: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not config.guarantee_strong_edge_coverage:
        return events, []
    windows = _strong_edge_windows(structured_scenario, seq_len)
    edge_lags = {
        _edge_key(int(edge["source"]), int(edge["target"])): max(
            float(edge.get("time_lag", 1)), 0.01
        )
        for edge in structured_scenario.get("edges", [])
    }
    output = list(events)
    forced: List[str] = []
    next_id = max((int(event["event_id"]) for event in output), default=-1) + 1
    for key, indices in sorted(windows.items(), key=lambda item: min(item[1])):
        already_covered = any(
            event.get("edge") == key
            and int(event.get("parent_time_index", -1)) in indices
            for event in output
        )
        if already_covered:
            continue
        source_text, target_text = key.split("->", 1)
        source, target = int(source_text), int(target_text)
        parents = []
        for event in output:
            if (
                int(event["node_id"]) != source
                or int(event["time_index"]) not in indices
                or int(event.get("generation", 0)) >= config.max_generation
            ):
                continue
            lineage_nodes, _ = _root_lineage(event)
            if target in lineage_nodes:
                audit["blocked_forced_revisits"] += 1
                continue
            parents.append(event)
        if not parents:
            audit["uncovered_strong_edges_without_simple_parent"].append(key)
            continue
        parent = min(parents, key=lambda event: float(event["t"]))
        lineage_nodes, lineage_edges = _root_lineage(parent)
        lag = edge_lags[key]
        delay = max(0.001, float(rng.normal(lag, config.propagation_jitter)))
        child_time = float(parent["t"]) + delay
        if child_time >= seq_len:
            audit["uncovered_strong_edges_without_simple_parent"].append(key)
            continue
        x, y = rng.normal(
            loc=node_coords[target], scale=config.target_spatial_jitter, size=2
        )
        output.append(
            {
                "event_id": next_id,
                "t": child_time,
                "x": float(np.clip(x, 0.0, 1.0)),
                "y": float(np.clip(y, 0.0, 1.0)),
                "node_id": target,
                "time_index": int(np.floor(child_time)),
                "event_type": "propagated",
                "parent_event_id": int(parent["event_id"]),
                "parent_time_index": int(parent["time_index"]),
                "source_node_id": source,
                "generation": int(parent.get("generation", 0)) + 1,
                "edge": key,
                "edge_probability": 1.0,
                "configured_lag": lag,
                "realized_delay": delay,
                "forced_by_quality_gate": True,
                "lineage_nodes": lineage_nodes + [target],
                "lineage_edges": lineage_edges + [key],
            }
        )
        forced.append(key)
        next_id += 1
    return sorted(output, key=lambda event: float(event["t"])), forced


def _lineage_quality(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    node_revisits = []
    immediate_backtracks = []
    generation_mismatches = []
    generation_counts: Dict[str, int] = {}
    for event in events:
        nodes, edges = _root_lineage(event)
        event_id = int(event["event_id"])
        generation = int(event.get("generation", 0))
        generation_counts[str(generation)] = generation_counts.get(str(generation), 0) + 1
        if len(nodes) != len(set(nodes)):
            node_revisits.append(event_id)
        if any(
            edges[index].split("->", 1)[0]
            == edges[index + 1].split("->", 1)[1]
            and edges[index].split("->", 1)[1]
            == edges[index + 1].split("->", 1)[0]
            for index in range(len(edges) - 1)
        ):
            immediate_backtracks.append(event_id)
        if generation != len(nodes) - 1:
            generation_mismatches.append(event_id)
    return {
        "node_revisit_event_ids": node_revisits,
        "immediate_backtrack_event_ids": immediate_backtracks,
        "generation_mismatch_event_ids": generation_mismatches,
        "generation_counts": generation_counts,
        "max_lineage_nodes": max(
            (len(_root_lineage(event)[0]) for event in events), default=0
        ),
    }


def simulate_stpp_for_streasoner_v9(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV9Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV9Config()
    config.validate()
    propagation_audit: Dict[str, Any] = {
        "blocked_immediate_backtracks": 0,
        "blocked_lineage_revisits": 0,
        "blocked_forced_revisits": 0,
        "blocked_by_edge": {},
        "uncovered_strong_edges_without_simple_parent": [],
    }
    original_propagate = _v4_module._propagate_events
    original_force = _v4_module._force_strong_edge_coverage
    _v4_module._propagate_events = lambda *args: _propagate_simple_paths_v9(
        *args, propagation_audit
    )
    _v4_module._force_strong_edge_coverage = (
        lambda *args: _force_strong_edge_coverage_v9(*args, propagation_audit)
    )
    try:
        ts_data, events, metadata = simulate_stpp_for_streasoner_v8(
            structured_scenario, seq_len, config
        )
    finally:
        _v4_module._propagate_events = original_propagate
        _v4_module._force_strong_edge_coverage = original_force

    lineage = _lineage_quality(events)
    quality = metadata["quality_report"]
    quality["checks"]["no_lineage_node_revisits"] = not lineage[
        "node_revisit_event_ids"
    ]
    quality["checks"]["no_immediate_edge_backtracking"] = not lineage[
        "immediate_backtrack_event_ids"
    ]
    quality["checks"]["all_lineages_are_simple_paths"] = not any(
        lineage[key]
        for key in (
            "node_revisit_event_ids",
            "immediate_backtrack_event_ids",
            "generation_mismatch_event_ids",
        )
    )
    quality["passed"] = all(quality["checks"].values())
    quality["lineage_quality"] = lineage
    metadata.update(
        {
            "simulator": "STPPG + cycle-free simple-path graph STPP v9",
            "method": (
                "demand-source roots + cycle-free simple-path graph propagation + "
                "explicit event peaks"
            ),
            "propagation_cycle_audit": propagation_audit,
            "lineage_quality": lineage,
        }
    )
    return ts_data, events, metadata
