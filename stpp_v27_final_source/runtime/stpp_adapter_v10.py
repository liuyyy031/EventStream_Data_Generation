"""Path-aware strong-edge coverage on top of cycle-free STPP v9."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import stpp_adapter_v9 as _v9_module
from stpp_adapter_v2 import _edge_key
from stpp_adapter_v3 import _demand_sources, _strong_edge_windows
from stpp_adapter_v9 import (
    STPPV9Config,
    _root_lineage,
    simulate_stpp_for_streasoner_v9,
)


@dataclass(frozen=True)
class STPPV10Config(STPPV9Config):
    allow_forced_path_support: bool = True


def _edge_maps(
    structured_scenario: Mapping[str, Any],
) -> Tuple[Dict[int, List[int]], Dict[str, float]]:
    outgoing: Dict[int, List[int]] = {}
    lags: Dict[str, float] = {}
    for edge in structured_scenario.get("edges", []):
        source = int(edge["source"])
        target = int(edge["target"])
        outgoing.setdefault(source, []).append(target)
        lags[_edge_key(source, target)] = max(
            float(edge.get("time_lag", 1)), 0.01
        )
    return outgoing, lags


def _shortest_support_paths(
    starts: Sequence[int],
    goal: int,
    forbidden: int,
    outgoing: Mapping[int, Sequence[int]],
    max_edges: int,
) -> List[List[int]]:
    paths: List[List[int]] = []
    for start in starts:
        if start == forbidden:
            continue
        queue = deque([[int(start)]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == goal:
                paths.append(path)
                break
            if len(path) - 1 >= max_edges:
                continue
            for target in outgoing.get(node, []):
                if target == forbidden or target in path:
                    continue
                queue.append(path + [int(target)])
    return sorted(paths, key=lambda path: (len(path), path))


def _support_plan(
    structured_scenario: Mapping[str, Any],
    source: int,
    target: int,
    indices: Sequence[int],
    seq_len: int,
    config: STPPV10Config,
    lags: Mapping[str, float],
    outgoing: Mapping[int, Sequence[int]],
) -> Tuple[List[int], float] | None:
    paths = _shortest_support_paths(
        _demand_sources(structured_scenario),
        source,
        target,
        outgoing,
        max(0, config.max_generation - 1),
    )
    final_lag = lags[_edge_key(source, target)]
    for path in paths:
        path_lags = [
            lags[_edge_key(path[index], path[index + 1])]
            for index in range(len(path) - 1)
        ]
        total_lag = sum(path_lags)
        for time_index in sorted(set(int(index) for index in indices)):
            source_time = float(time_index) + 0.25
            root_time = source_time - total_lag
            if root_time < 0 or source_time + final_lag >= seq_len:
                continue
            return path, root_time
    return None


def _new_support_root(
    event_id: int,
    node_id: int,
    event_time: float,
    node_coords: np.ndarray,
    config: STPPV10Config,
    rng: np.random.Generator,
    target_edge: str,
) -> Dict[str, Any]:
    x, y = rng.normal(
        loc=node_coords[node_id], scale=config.target_spatial_jitter, size=2
    )
    return {
        "event_id": event_id,
        "t": event_time,
        "x": float(np.clip(x, 0.0, 1.0)),
        "y": float(np.clip(y, 0.0, 1.0)),
        "node_id": node_id,
        "time_index": int(np.floor(event_time)),
        "event_type": "scenario_seed",
        "parent_event_id": None,
        "source_node_id": None,
        "generation": 0,
        "keep_probability": 1.0,
        "quality_gate_reason": (
            f"cycle-free strong-edge path support for {target_edge}"
        ),
        "forced_path_support_root": True,
        "forced_target_edge": target_edge,
        "lineage_nodes": [node_id],
        "lineage_edges": [],
    }


def _new_forced_child(
    event_id: int,
    parent: Mapping[str, Any],
    target: int,
    lag: float,
    node_coords: np.ndarray,
    config: STPPV10Config,
    rng: np.random.Generator,
    target_edge: str,
    is_target_edge: bool,
    is_path_support: bool,
) -> Dict[str, Any]:
    source = int(parent["node_id"])
    key = _edge_key(source, target)
    lineage_nodes, lineage_edges = _root_lineage(parent)
    child_time = float(parent["t"]) + lag
    x, y = rng.normal(
        loc=node_coords[target], scale=config.target_spatial_jitter, size=2
    )
    return {
        "event_id": event_id,
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
        "realized_delay": lag,
        "forced_by_quality_gate": True,
        "forced_path_support": is_path_support,
        "forced_target_edge": target_edge,
        "forced_strong_edge": is_target_edge,
        "lineage_nodes": lineage_nodes + [target],
        "lineage_edges": lineage_edges + [key],
    }


def _force_strong_edge_coverage_v10(
    events: List[Dict[str, Any]],
    structured_scenario: Mapping[str, Any],
    node_coords: np.ndarray,
    seq_len: int,
    config: STPPV10Config,
    rng: np.random.Generator,
    audit: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    audit.setdefault("path_support_roots_added", 0)
    audit.setdefault("path_support_events_added", 0)
    audit.setdefault("path_support_paths", [])
    audit.setdefault("uncovered_strong_edges_without_support_path", [])
    if not config.guarantee_strong_edge_coverage:
        return events, []

    windows = _strong_edge_windows(structured_scenario, seq_len)
    outgoing, edge_lags = _edge_maps(structured_scenario)
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
        lag = edge_lags[key]
        parents = []
        for event in output:
            if (
                int(event["node_id"]) != source
                or int(event["time_index"]) not in indices
                or int(event.get("generation", 0)) >= config.max_generation
                or float(event["t"]) + lag >= seq_len
            ):
                continue
            lineage_nodes, _ = _root_lineage(event)
            if target in lineage_nodes:
                audit["blocked_forced_revisits"] += 1
                continue
            parents.append(event)

        if parents:
            parent = min(parents, key=lambda event: float(event["t"]))
            child = _new_forced_child(
                next_id,
                parent,
                target,
                lag,
                node_coords,
                config,
                rng,
                key,
                True,
                False,
            )
            output.append(child)
            forced.append(key)
            next_id += 1
            continue

        if not config.allow_forced_path_support:
            audit["uncovered_strong_edges_without_simple_parent"].append(key)
            continue

        plan = _support_plan(
            structured_scenario,
            source,
            target,
            indices,
            seq_len,
            config,
            edge_lags,
            outgoing,
        )
        required_events = len(plan[0]) + 1 if plan else 0
        if plan is None or len(output) + required_events > config.max_events:
            audit["uncovered_strong_edges_without_simple_parent"].append(key)
            audit["uncovered_strong_edges_without_support_path"].append(key)
            continue

        path, root_time = plan
        root = _new_support_root(
            next_id,
            path[0],
            root_time,
            node_coords,
            config,
            rng,
            key,
        )
        output.append(root)
        created_ids = [next_id]
        audit["path_support_roots_added"] += 1
        next_id += 1
        parent: Dict[str, Any] = root
        for path_target in path[1:]:
            path_key = _edge_key(int(parent["node_id"]), path_target)
            parent = _new_forced_child(
                next_id,
                parent,
                path_target,
                edge_lags[path_key],
                node_coords,
                config,
                rng,
                key,
                False,
                True,
            )
            output.append(parent)
            created_ids.append(next_id)
            audit["path_support_events_added"] += 1
            next_id += 1

        child = _new_forced_child(
            next_id,
            parent,
            target,
            lag,
            node_coords,
            config,
            rng,
            key,
            True,
            True,
        )
        output.append(child)
        created_ids.append(next_id)
        audit["path_support_events_added"] += 1
        forced.append(key)
        next_id += 1
        audit["path_support_paths"].append(
            {
                "target_edge": key,
                "lineage_nodes": path + [target],
                "root_time": root_time,
                "created_event_ids": created_ids,
            }
        )

    return sorted(
        output, key=lambda event: (float(event["t"]), int(event["event_id"]))
    ), forced


def simulate_stpp_for_streasoner_v10(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV10Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV10Config()
    config.validate()
    original_force = _v9_module._force_strong_edge_coverage_v9
    _v9_module._force_strong_edge_coverage_v9 = _force_strong_edge_coverage_v10
    try:
        ts_data, events, metadata = simulate_stpp_for_streasoner_v9(
            structured_scenario, seq_len, config
        )
    finally:
        _v9_module._force_strong_edge_coverage_v9 = original_force

    support_roots = [
        event for event in events if event.get("forced_path_support_root") is True
    ]
    forced_events = [
        event for event in events if event.get("forced_by_quality_gate") is True
    ]
    support_reasons = [str(event["quality_gate_reason"]) for event in support_roots]
    cycle_audit = metadata["propagation_cycle_audit"]
    for path_audit in cycle_audit.get("path_support_paths", []):
        target_edge = str(path_audit["target_edge"])
        path_audit["created_event_ids"] = [
            int(event["event_id"])
            for event in events
            if event.get("forced_target_edge") == target_edge
            and (
                event.get("forced_path_support_root") is True
                or event.get("forced_path_support") is True
            )
        ]
    metadata["num_scenario_seed_events"] = sum(
        event.get("event_type") == "scenario_seed" for event in events
    )
    metadata["num_forced_propagated_events"] = len(forced_events)
    metadata["scenario_seed_reasons"] = list(
        metadata.get("scenario_seed_reasons", [])
    ) + support_reasons
    metadata.update(
        {
            "simulator": "STPPG + path-aware cycle-free graph STPP v10",
            "method": (
                "demand-source roots + cycle-free graph propagation + "
                "path-aware strong-edge support"
            ),
            "path_support_root_event_ids": [
                int(event["event_id"]) for event in support_roots
            ],
        }
    )
    return ts_data, events, metadata
