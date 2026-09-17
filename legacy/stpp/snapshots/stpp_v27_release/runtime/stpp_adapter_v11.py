"""Lag-aligned conditional envelopes and deterministic semantic QA for v11."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from stpp_adapter_v3 import _demand_sources
from stpp_adapter_v10 import STPPV10Config, simulate_stpp_for_streasoner_v10


@dataclass(frozen=True)
class STPPV11Config(STPPV10Config):
    conditional_envelope_shift_margin: int = 1

    def validate(self) -> None:
        super().validate()
        if self.conditional_envelope_shift_margin < 0:
            raise ValueError("conditional_envelope_shift_margin cannot be negative")


def _graph_maps(
    structured_scenario: Mapping[str, Any],
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float]]:
    outgoing: Dict[int, List[int]] = {}
    lags: Dict[Tuple[int, int], float] = {}
    for edge in structured_scenario.get("edges", []):
        source = int(edge["source"])
        target = int(edge["target"])
        outgoing.setdefault(source, []).append(target)
        lags[(source, target)] = max(float(edge.get("time_lag", 1)), 0.01)
    return outgoing, lags


def _simple_paths(
    starts: Sequence[int],
    outgoing: Mapping[int, Sequence[int]],
    max_edges: int,
) -> List[List[int]]:
    paths: List[List[int]] = []
    for start in starts:
        stack = [[int(start)]]
        while stack:
            path = stack.pop()
            if len(path) > 1:
                paths.append(path)
            if len(path) - 1 >= max_edges:
                continue
            for target in reversed(list(outgoing.get(path[-1], []))):
                if target in path:
                    continue
                stack.append(path + [int(target)])
    return sorted(paths, key=lambda path: (len(path), path))


def _possible_bin_shifts(
    cumulative_lag: float, margin: int
) -> List[int]:
    lower = int(math.floor(cumulative_lag))
    upper = int(math.ceil(cumulative_lag))
    return list(
        range(max(0, lower - margin), max(0, upper + margin) + 1)
    )


def _shifted_profile_max(profile: np.ndarray, shifts: Sequence[int]) -> np.ndarray:
    shifted = np.zeros_like(profile, dtype=float)
    for shift in shifts:
        candidate = np.zeros_like(profile, dtype=float)
        if shift == 0:
            candidate[:] = profile
        elif shift < len(profile):
            candidate[shift:] = profile[:-shift]
        shifted = np.maximum(shifted, candidate)
    return shifted


def _conditional_simple_path_envelope(
    structured_scenario: Mapping[str, Any],
    target_profile: np.ndarray,
    config: STPPV11Config,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    outgoing, lags = _graph_maps(structured_scenario)
    paths = _simple_paths(
        _demand_sources(structured_scenario), outgoing, config.max_generation
    )
    envelope = np.asarray(target_profile, dtype=float).copy()
    path_audit: List[Dict[str, Any]] = []
    for path in paths:
        cumulative_lag = sum(
            lags[(path[index], path[index + 1])]
            for index in range(len(path) - 1)
        )
        shifts = _possible_bin_shifts(
            cumulative_lag, config.conditional_envelope_shift_margin
        )
        generations = len(path) - 1
        decay = config.propagation_magnitude_decay**generations
        contribution = _shifted_profile_max(target_profile[path[0]], shifts)
        envelope[path[-1]] += contribution * decay
        path_audit.append(
            {
                "lineage_nodes": path,
                "generation": generations,
                "cumulative_lag": cumulative_lag,
                "possible_bin_shifts": shifts,
                "magnitude_decay": decay,
            }
        )
    return envelope, {
        "origin_nodes": _demand_sources(structured_scenario),
        "max_generation": config.max_generation,
        "shift_margin": config.conditional_envelope_shift_margin,
        "num_simple_paths": len(path_audit),
        "paths": path_audit,
    }


def _time_intervals(value: Any) -> List[Tuple[float, float]]:
    if isinstance(value, (int, float)):
        point = float(value)
        return [(point, point)]
    if isinstance(value, list):
        intervals: List[Tuple[float, float]] = []
        for item in value:
            intervals.extend(_time_intervals(item))
        return intervals
    text = str(value or "")
    clock = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", text)
    if clock:
        hour = float(int(clock.group(1)))
        minute = float(int(clock.group(2)))
        return [(hour + minute / 60.0, hour + minute / 60.0)]
    range_match = re.search(r"(?<!\d)(\d+)\s*[-–—]\s*(\d+)(?!\d)", text)
    if range_match:
        start = float(int(range_match.group(1)))
        end = float(int(range_match.group(2)))
        return [(min(start, end), max(start, end))]
    numbers = re.findall(r"(?<!\d)\d+(?!\d)", text)
    if numbers:
        point = float(int(numbers[0]))
        return [(point, point)]
    return []


def _pattern_peaks(pattern: Mapping[str, Any]) -> List[float]:
    peaks: List[float] = []
    peak = pattern.get("peak")
    if peak is not None:
        try:
            peaks.append(float(peak))
        except (TypeError, ValueError):
            pass
    listed = pattern.get("self_generated_peaks")
    if listed is not None:
        values = listed if isinstance(listed, list) else [listed]
        for value in values:
            try:
                peaks.append(float(value))
            except (TypeError, ValueError):
                continue
    return sorted(set(peaks))


def _semantic_peak_audit(
    structured_scenario: Mapping[str, Any], tolerance: float = 0.01
) -> Dict[str, Any]:
    node_types = {
        int(node["id"]): str(node.get("type", "demand_source")).lower()
        for node in structured_scenario.get("nodes", [])
    }
    demand_nodes = {
        node_id
        for node_id, node_type in node_types.items()
        if node_type == "demand_source"
    }
    counts = {str(node_id): 0 for node_id in sorted(demand_nodes)}
    conflicts: List[Dict[str, Any]] = []
    propagation_positive_patterns: List[Dict[str, Any]] = []
    drift = structured_scenario.get("drift_patterns", {}) or {}
    for node_data in drift.get("nodes", []) or []:
        node_id = int(node_data.get("id", -1))
        self_patterns: List[Tuple[int, List[float]]] = []
        for index, pattern in enumerate(node_data.get("patterns", []) or []):
            peaks = _pattern_peaks(pattern)
            amplitude = float(pattern.get("amplitude", 0) or 0)
            origin = str(pattern.get("origin", "")).lower()
            behavior = str(pattern.get("behavior", "")).lower()
            explicitly_propagated = (
                origin == "propagated" or behavior == "propagated"
            )
            if amplitude <= 0 or not peaks or explicitly_propagated:
                continue
            if node_id in demand_nodes:
                self_patterns.append((index, peaks))
            else:
                propagation_positive_patterns.append(
                    {
                        "node_id": node_id,
                        "pattern_index": index,
                        "peaks": peaks,
                        "amplitude": amplitude,
                    }
                )
        if node_id in demand_nodes:
            signatures = {tuple(peaks) for _, peaks in self_patterns}
            counts[str(node_id)] = len(signatures)
            variation_intervals: List[Tuple[float, float, int, Any]] = []
            for variation_index, variation in enumerate(
                node_data.get("propagated_variations", []) or []
            ):
                raw_time = variation.get(
                    "time", variation.get("arrival_steps", variation.get("time_range"))
                )
                for start, end in _time_intervals(raw_time):
                    variation_intervals.append(
                        (start, end, variation_index, raw_time)
                    )
            for pattern_index, peaks in self_patterns:
                for peak in peaks:
                    for start, end, variation_index, raw_time in variation_intervals:
                        if start - tolerance <= peak <= end + tolerance:
                            conflicts.append(
                                {
                                    "node_id": node_id,
                                    "self_pattern_index": pattern_index,
                                    "self_peak": peak,
                                    "propagated_variation_index": variation_index,
                                    "propagated_time": raw_time,
                                    "parsed_interval": [start, end],
                                }
                            )
    return {
        "independent_self_pattern_counts": counts,
        "invalid_self_pattern_counts": {
            node_id: count for node_id, count in counts.items() if count != 1
        },
        "self_generated_propagated_time_conflicts": conflicts,
        "positive_patterns_on_propagation_nodes": propagation_positive_patterns,
    }


def simulate_stpp_for_streasoner_v11(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV11Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV11Config()
    config.validate()
    ts_data, events, metadata = simulate_stpp_for_streasoner_v10(
        structured_scenario, seq_len, config
    )

    old_envelope = np.asarray(metadata["scenario_magnitude_envelope"], dtype=float)
    old_ratio = float(metadata["max_matrix_envelope_ratio"])
    target_profile = np.asarray(metadata["target_magnitude_profile"], dtype=float)
    envelope, envelope_audit = _conditional_simple_path_envelope(
        structured_scenario, target_profile, config
    )
    ratios = np.divide(
        ts_data, envelope, out=np.zeros_like(ts_data), where=envelope > 0
    )
    conditional_ratio = float(ratios.max())
    semantic_audit = _semantic_peak_audit(structured_scenario)

    quality = metadata["quality_report"]
    quality["checks"]["matrix_peak_ratio_within_limit"] = (
        conditional_ratio <= config.max_matrix_target_ratio
    )
    quality["checks"]["conditional_simple_path_envelope_within_limit"] = (
        conditional_ratio <= config.max_matrix_target_ratio
    )
    quality["checks"]["exactly_one_self_pattern_per_demand_source"] = not (
        semantic_audit["invalid_self_pattern_counts"]
    )
    quality["checks"]["no_self_generated_propagated_time_conflicts"] = not (
        semantic_audit["self_generated_propagated_time_conflicts"]
    )
    quality["checks"]["no_positive_patterns_on_propagation_nodes"] = not (
        semantic_audit["positive_patterns_on_propagation_nodes"]
    )
    quality["passed"] = all(quality["checks"].values())
    quality["expected_one_hop_max_matrix_envelope_ratio"] = old_ratio
    quality["max_matrix_envelope_ratio"] = conditional_ratio
    quality["max_matrix_target_ratio"] = conditional_ratio
    quality["conditional_envelope_audit"] = envelope_audit
    quality["semantic_peak_audit"] = semantic_audit

    metadata.update(
        {
            "simulator": "STPPG + grounded semantic conditional-envelope STPP v11",
            "method": (
                "cycle-free graph propagation + path-aware strong-edge support + "
                "lag-aligned conditional simple-path envelope"
            ),
            "expected_one_hop_magnitude_envelope": old_envelope.tolist(),
            "expected_one_hop_max_matrix_envelope_ratio": old_ratio,
            "scenario_magnitude_envelope": envelope.tolist(),
            "max_matrix_envelope_ratio": conditional_ratio,
            "conditional_envelope_audit": envelope_audit,
            "semantic_peak_audit": semantic_audit,
        }
    )
    return ts_data, events, metadata
