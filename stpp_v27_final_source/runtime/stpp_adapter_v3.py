"""Density-controlled, quality-gated graph STPP adapter (v3 copy)."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from stpp_adapter import _generate_one_realisation, _normalised_node_coordinates
from stpp_adapter_v2 import (
    _assign_and_thin_immigrants,
    _conditioning_arrays,
    _edge_key,
    _propagate_events,
    _scenario_edges,
    active_time_indices,
)


@dataclass(frozen=True)
class STPPV3Config:
    # Upstream STPPG parameters.
    mu: float = 1.0
    kernel_c: float = 0.05
    beta: float = 1.0
    sigma_x: float = 0.15
    sigma_y: float = 0.15
    intensity_upper_bound: float = 20.0
    min_events: int = 5
    max_attempts: int = 20

    # Scenario and graph conditioning.
    minimum_keep_probability: float = 0.25
    edge_branching_ratio: float = 0.35
    propagation_jitter: float = 0.10
    target_spatial_jitter: float = 0.04
    max_generation: int = 3
    max_events: int = 5000

    # v3 density and quality gates.
    target_immigrant_rate: float = 0.30
    max_candidate_batches: int = 30
    min_events_per_demand_source: int = 2
    guarantee_peak_coverage: bool = True
    guarantee_strong_edge_coverage: bool = True

    # Event magnitude preserves scenario units such as vehicles/hour.
    propagation_magnitude_decay: float = 0.80
    magnitude_noise: float = 0.10
    aggregation: str = "magnitude"
    rolling_window: int = 3
    seed: int | None = None

    def validate(self) -> None:
        if self.mu <= 0 or self.kernel_c <= 0 or self.beta <= 0:
            raise ValueError("mu, kernel_c, and beta must be positive")
        if self.sigma_x <= 0 or self.sigma_y <= 0:
            raise ValueError("sigma values must be positive")
        if self.intensity_upper_bound <= self.mu:
            raise ValueError("intensity_upper_bound must be greater than mu")
        if self.min_events < 1 or self.max_attempts < 1:
            raise ValueError("min_events and max_attempts must be at least 1")
        if not 0 < self.minimum_keep_probability <= 1:
            raise ValueError("minimum_keep_probability must be in (0, 1]")
        if not 0 <= self.edge_branching_ratio < 1:
            raise ValueError("edge_branching_ratio must be in [0, 1)")
        if self.max_generation < 0 or self.max_events < self.min_events:
            raise ValueError("invalid generation/event limits")
        if self.target_immigrant_rate <= 0 or self.max_candidate_batches < 1:
            raise ValueError("invalid immigrant target configuration")
        if self.min_events_per_demand_source < 1:
            raise ValueError("min_events_per_demand_source must be positive")
        if not 0 < self.propagation_magnitude_decay <= 1:
            raise ValueError("propagation_magnitude_decay must be in (0, 1]")
        if self.magnitude_noise < 0:
            raise ValueError("magnitude_noise cannot be negative")
        if self.aggregation not in {
            "count",
            "magnitude",
            "rolling_count",
            "rolling_magnitude",
            "cumulative_count",
            "cumulative_magnitude",
        }:
            raise ValueError("unsupported aggregation")


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


def _strong_edge_windows(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Dict[str, List[int]]:
    all_edges = [
        _edge_key(int(edge["source"]), int(edge["target"]))
        for edge in structured_scenario.get("edges", [])
    ]
    windows: Dict[str, set[int]] = {}
    patterns = (
        structured_scenario.get("adjacency_modulation", {}).get("patterns", [])
    )
    for pattern in patterns:
        if str(pattern.get("effect", "")).lower() != "strong":
            continue
        indices = active_time_indices(
            pattern.get("time_period", pattern.get("time_range")),
            pattern,
            seq_len,
        )
        for text in _edge_strings(pattern.get("applies_to"), all_edges):
            match = re.search(r"(\d+)\s*->\s*(\d+)", text)
            if match:
                key = _edge_key(int(match.group(1)), int(match.group(2)))
                windows.setdefault(key, set()).update(indices)
    return {key: sorted(values) for key, values in windows.items()}


def _demand_sources(structured_scenario: Mapping[str, Any]) -> List[int]:
    return [
        int(node["id"])
        for node in structured_scenario.get("nodes", [])
        if str(node.get("type", "")).lower() == "demand_source"
    ]


def _explicit_peak_windows(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Dict[int, List[int]]:
    result: Dict[int, set[int]] = {}
    drift = structured_scenario.get("drift_patterns", {}) or {}
    repeat = bool(drift.get("repeat", False))
    repeat_period = int(drift.get("repeat_period", 24) or 24)
    for node_data in drift.get("nodes", []):
        node_id = int(node_data.get("id", -1))
        for pattern in node_data.get("patterns", []):
            peak = pattern.get("peak")
            amplitude = float(pattern.get("amplitude", 0) or 0)
            if peak is None or amplitude <= 0:
                continue
            peak = int(peak)
            if (repeat or seq_len > 24) and peak < repeat_period:
                for base in range(0, seq_len, repeat_period):
                    for offset in (-1, 0, 1):
                        index = base + peak + offset
                        if 0 <= index < seq_len:
                            result.setdefault(node_id, set()).add(index)
            else:
                for index in (peak - 1, peak, peak + 1):
                    if 0 <= index < seq_len:
                        result.setdefault(node_id, set()).add(index)
    return {key: sorted(values) for key, values in result.items()}


def _node_baselines(
    structured_scenario: Mapping[str, Any], num_nodes: int
) -> np.ndarray:
    baselines = np.ones(num_nodes, dtype=float)
    drift = structured_scenario.get("drift_patterns", {}) or {}
    for node_data in drift.get("nodes", []):
        node_id = int(node_data.get("id", -1))
        if not 0 <= node_id < num_nodes:
            continue
        values = [
            float(pattern.get("baseline", 1) or 1)
            for pattern in node_data.get("patterns", [])
            if float(pattern.get("baseline", 1) or 1) > 0
        ]
        if values:
            baselines[node_id] = float(np.median(values))
    return baselines


def _reassign_root_ids(events: List[Dict[str, Any]], start: int) -> int:
    next_id = start
    for event in events:
        event["event_id"] = next_id
        next_id += 1
    return next_id


def _collect_target_immigrants(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    node_ids: Sequence[int],
    node_coords: np.ndarray,
    background: np.ndarray,
    config: STPPV3Config,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target = max(
        config.min_events,
        int(math.ceil(seq_len * len(node_ids) * config.target_immigrant_rate)),
    )
    immigrants: List[Dict[str, Any]] = []
    batches = 0
    next_id = 0
    while len(immigrants) < target and batches < config.max_candidate_batches:
        points = _generate_one_realisation(seq_len, config)  # type: ignore[arg-type]
        retained = _assign_and_thin_immigrants(
            points, node_ids, node_coords, background, config, rng  # type: ignore[arg-type]
        )
        next_id = _reassign_root_ids(retained, next_id)
        immigrants.extend(retained)
        batches += 1
    if len(immigrants) < target:
        raise RuntimeError(
            f"Only {len(immigrants)} immigrants retained; target={target} after "
            f"{batches} candidate batches"
        )
    # Keep the highest scenario-conditioned probabilities when the final batch
    # overshoots the target, then restore temporal order.
    immigrants = sorted(
        immigrants,
        key=lambda event: float(event.get("keep_probability", 0)),
        reverse=True,
    )[:target]
    immigrants.sort(key=lambda event: float(event["t"]))
    _reassign_root_ids(immigrants, 0)
    return immigrants, {
        "target_immigrant_events": target,
        "candidate_batches": batches,
    }


def _new_scenario_seed(
    event_id: int,
    node_id: int,
    allowed_indices: Sequence[int],
    node_coords: np.ndarray,
    background: np.ndarray,
    rng: np.random.Generator,
    reason: str,
) -> Dict[str, Any]:
    if not allowed_indices:
        allowed_indices = list(range(background.shape[1]))
    weights = np.asarray(
        [background[node_id, index] for index in allowed_indices], dtype=float
    )
    weights = weights / weights.sum()
    time_index = int(rng.choice(np.asarray(allowed_indices), p=weights))
    t = time_index + float(rng.random())
    x, y = rng.normal(loc=node_coords[node_id], scale=0.02, size=2)
    return {
        "event_id": event_id,
        "t": t,
        "x": float(np.clip(x, 0.0, 1.0)),
        "y": float(np.clip(y, 0.0, 1.0)),
        "node_id": node_id,
        "time_index": time_index,
        "event_type": "scenario_seed",
        "parent_event_id": None,
        "source_node_id": None,
        "generation": 0,
        "keep_probability": 1.0,
        "quality_gate_reason": reason,
    }


def _ensure_root_coverage(
    immigrants: List[Dict[str, Any]],
    structured_scenario: Mapping[str, Any],
    node_coords: np.ndarray,
    background: np.ndarray,
    seq_len: int,
    config: STPPV3Config,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    roots = list(immigrants)
    next_id = max((int(event["event_id"]) for event in roots), default=-1) + 1
    added_reasons: List[str] = []

    for node_id in _demand_sources(structured_scenario):
        count = sum(int(event["node_id"]) == node_id for event in roots)
        for _ in range(max(0, config.min_events_per_demand_source - count)):
            reason = f"minimum demand-source coverage for node {node_id}"
            roots.append(
                _new_scenario_seed(
                    next_id,
                    node_id,
                    list(range(seq_len)),
                    node_coords,
                    background,
                    rng,
                    reason,
                )
            )
            added_reasons.append(reason)
            next_id += 1

    if config.guarantee_peak_coverage:
        for node_id, indices in _explicit_peak_windows(
            structured_scenario, seq_len
        ).items():
            covered = any(
                int(event["node_id"]) == node_id
                and int(event["time_index"]) in indices
                for event in roots
            )
            if not covered:
                reason = f"explicit peak coverage for node {node_id}"
                roots.append(
                    _new_scenario_seed(
                        next_id,
                        node_id,
                        indices,
                        node_coords,
                        background,
                        rng,
                        reason,
                    )
                )
                added_reasons.append(reason)
                next_id += 1

    if config.guarantee_strong_edge_coverage:
        for key, indices in _strong_edge_windows(structured_scenario, seq_len).items():
            source = int(key.split("->", 1)[0])
            covered = any(
                int(event["node_id"]) == source
                and int(event["time_index"]) in indices
                for event in roots
            )
            if not covered:
                reason = f"strong-edge parent coverage for {key}"
                roots.append(
                    _new_scenario_seed(
                        next_id,
                        source,
                        indices,
                        node_coords,
                        background,
                        rng,
                        reason,
                    )
                )
                added_reasons.append(reason)
                next_id += 1

    roots.sort(key=lambda event: float(event["t"]))
    # IDs remain creation IDs until the final lineage-preserving compaction.
    return roots, {
        "num_scenario_seed_events": sum(
            event["event_type"] == "scenario_seed" for event in roots
        ),
        "scenario_seed_reasons": added_reasons,
    }


def _force_strong_edge_coverage(
    events: List[Dict[str, Any]],
    structured_scenario: Mapping[str, Any],
    node_coords: np.ndarray,
    seq_len: int,
    config: STPPV3Config,
    rng: np.random.Generator,
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
        parents = [
            event
            for event in output
            if int(event["node_id"]) == source
            and int(event["time_index"]) in indices
            and int(event.get("generation", 0)) < config.max_generation
        ]
        if not parents:
            continue
        parent = min(parents, key=lambda event: float(event["t"]))
        lag = edge_lags[key]
        delay = max(0.001, float(rng.normal(lag, config.propagation_jitter)))
        child_time = float(parent["t"]) + delay
        if child_time >= seq_len:
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
            }
        )
        forced.append(key)
        next_id += 1
    return sorted(output, key=lambda event: float(event["t"])), forced


def _annotate_parent_times(events: List[Dict[str, Any]]) -> None:
    by_id = {int(event["event_id"]): event for event in events}
    for event in events:
        parent_id = event.get("parent_event_id")
        if parent_id is not None and int(parent_id) in by_id:
            event["parent_time_index"] = int(by_id[int(parent_id)]["time_index"])


def _assign_magnitudes(
    events: List[Dict[str, Any]],
    baselines: np.ndarray,
    background: np.ndarray,
    config: STPPV3Config,
    rng: np.random.Generator,
    magnitude_unit: str,
) -> None:
    by_id: Dict[int, Dict[str, Any]] = {}
    for event in sorted(events, key=lambda item: float(item["t"])):
        parent_id = event.get("parent_event_id")
        if parent_id is None:
            expected = float(
                baselines[int(event["node_id"])]
                * background[int(event["node_id"]), int(event["time_index"])]
            )
        else:
            parent = by_id.get(int(parent_id))
            expected = (
                float(parent["magnitude"]) * config.propagation_magnitude_decay
                if parent is not None
                else float(baselines[int(event["node_id"])])
            )
        noise_scale = max(expected * config.magnitude_noise, 1e-6)
        event["magnitude"] = max(0.01, float(rng.normal(expected, noise_scale)))
        event["magnitude_unit"] = magnitude_unit
        by_id[int(event["event_id"])] = event


def _compact_event_ids(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(events, key=lambda event: (float(event["t"]), int(event["event_id"])))
    mapping = {int(event["event_id"]): index for index, event in enumerate(ordered)}
    for index, event in enumerate(ordered):
        parent_id = event.get("parent_event_id")
        event["event_id"] = index
        if parent_id is not None:
            event["parent_event_id"] = mapping[int(parent_id)]
    return ordered


def _aggregate_v3(
    events: Sequence[Mapping[str, Any]],
    num_nodes: int,
    seq_len: int,
    config: STPPV3Config,
) -> np.ndarray:
    use_magnitude = "magnitude" in config.aggregation
    values = np.zeros((num_nodes, seq_len), dtype=float)
    for event in events:
        amount = float(event["magnitude"]) if use_magnitude else 1.0
        values[int(event["node_id"]), int(event["time_index"])] += amount
    if config.aggregation.startswith("rolling_"):
        kernel = np.ones(config.rolling_window, dtype=float)
        values = np.vstack([np.convolve(row, kernel, mode="same") for row in values])
    elif config.aggregation.startswith("cumulative_"):
        values = np.cumsum(values, axis=1)
    return values


def _quality_report(
    events: Sequence[Mapping[str, Any]],
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    target: int,
    min_events_per_demand_source: int,
) -> Dict[str, Any]:
    roots = [event for event in events if event.get("parent_event_id") is None]
    demand_coverage = {
        str(node_id): sum(int(event["node_id"]) == node_id for event in roots)
        for node_id in _demand_sources(structured_scenario)
    }
    strong_windows = _strong_edge_windows(structured_scenario, seq_len)
    strong_coverage = {
        key: any(
            event.get("edge") == key
            and int(event.get("parent_time_index", -1)) in indices
            for event in events
        )
        for key, indices in strong_windows.items()
    }
    peak_windows = _explicit_peak_windows(structured_scenario, seq_len)
    peak_coverage = {
        str(node_id): any(
            int(event["node_id"]) == node_id
            and int(event["time_index"]) in indices
            and event.get("parent_event_id") is None
            for event in events
        )
        for node_id, indices in peak_windows.items()
    }
    nonzero_cells = len(
        {(int(event["node_id"]), int(event["time_index"])) for event in events}
    )
    checks = {
        "target_immigrant_density_met": len(roots) >= target,
        "demand_source_minimums_met": all(
            value >= min_events_per_demand_source
            for value in demand_coverage.values()
        ),
        "all_explicit_peaks_covered": all(peak_coverage.values()),
        "all_strong_edges_covered": all(strong_coverage.values()),
        "has_propagated_events": any(
            event.get("parent_event_id") is not None for event in events
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "demand_source_root_counts": demand_coverage,
        "explicit_peak_coverage": peak_coverage,
        "strong_edge_coverage": strong_coverage,
        "nonzero_cell_rate": nonzero_cells / max(1, seq_len * len(structured_scenario.get("nodes", []))),
    }


def simulate_stpp_for_streasoner_v3(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV3Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    config = config or STPPV3Config()
    config.validate()
    if config.seed is not None:
        np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    node_ids, node_coords = _normalised_node_coordinates(structured_scenario)
    if node_ids != list(range(len(node_ids))):
        raise ValueError("v3 requires contiguous node IDs 0..N-1")
    outgoing, edge_keys = _scenario_edges(structured_scenario, len(node_ids))
    background, edge_factors = _conditioning_arrays(
        structured_scenario, len(node_ids), seq_len, edge_keys
    )
    immigrants, density_info = _collect_target_immigrants(
        structured_scenario,
        seq_len,
        node_ids,
        node_coords,
        background,
        config,
        rng,
    )
    roots, seed_info = _ensure_root_coverage(
        immigrants,
        structured_scenario,
        node_coords,
        background,
        seq_len,
        config,
        rng,
    )
    events = _propagate_events(
        roots,
        outgoing,
        edge_factors,
        node_coords,
        seq_len,
        config,  # type: ignore[arg-type]
        rng,
    )
    _annotate_parent_times(events)
    events, forced_edges = _force_strong_edge_coverage(
        events,
        structured_scenario,
        node_coords,
        seq_len,
        config,
        rng,
    )
    _annotate_parent_times(events)
    baselines = _node_baselines(structured_scenario, len(node_ids))
    magnitude_unit = str(
        structured_scenario.get("variable", "scenario variable units per time window")
    )
    _assign_magnitudes(
        events, baselines, background, config, rng, magnitude_unit
    )
    events = _compact_event_ids(events)
    ts_data = _aggregate_v3(events, len(node_ids), seq_len, config)

    target = int(density_info["target_immigrant_events"])
    quality = _quality_report(
        events,
        structured_scenario,
        seq_len,
        target,
        config.min_events_per_demand_source,
    )
    if config.aggregation == "count":
        expected_total = float(len(events))
    elif config.aggregation == "magnitude":
        expected_total = float(sum(float(event["magnitude"]) for event in events))
    else:
        expected_total = None
    if expected_total is not None:
        total_matches = bool(np.isclose(float(ts_data.sum()), expected_total))
        quality["checks"]["aggregation_total_matches"] = total_matches
        quality["passed"] = bool(quality["passed"] and total_matches)
    propagated = sum(event.get("parent_event_id") is not None for event in events)
    metadata: Dict[str, Any] = {
        "simulator": "STPPG + STReasoner density-controlled marked branching v3",
        "simulator_commit": "be65d949e475c636a34fc6216044be94e139f50f",
        "method": "pooled STPPG immigrants + scenario seeds + graph offspring",
        "integration_method": "density control + quality-gated marked branching + aggregation",
        "event_format": [
            "event_id",
            "t",
            "x",
            "y",
            "node_id",
            "time_index",
            "event_type",
            "parent_event_id",
            "source_node_id",
            "generation",
            "magnitude",
            "magnitude_unit",
        ],
        "aggregation": config.aggregation,
        "dt": 1.0,
        "num_events": len(events),
        "num_root_events": len(events) - propagated,
        "num_stpp_immigrant_events": len(immigrants),
        "num_scenario_seed_events": seed_info["num_scenario_seed_events"],
        "num_propagated_events": propagated,
        "num_forced_propagated_events": len(forced_edges),
        "forced_strong_edges": forced_edges,
        "graph_conditioned": True,
        "scenario_temporal_conditioned": True,
        "magnitude_conditioned": True,
        "magnitude_unit": magnitude_unit,
        "quality_report": quality,
        "density_control": density_info,
        "scenario_seed_reasons": seed_info["scenario_seed_reasons"],
        "node_coordinates": {
            str(node_id): {"x": float(coord[0]), "y": float(coord[1])}
            for node_id, coord in zip(node_ids, node_coords)
        },
        "point_process_parameters": asdict(config),
        "background_profile": background.tolist(),
        "edge_excitation_profiles": {
            key: values.tolist() for key, values in edge_factors.items()
        },
        "event_magnitude_total": float(
            sum(float(event["magnitude"]) for event in events)
        ),
        "matrix_total": float(ts_data.sum()),
    }
    return ts_data, events, metadata
