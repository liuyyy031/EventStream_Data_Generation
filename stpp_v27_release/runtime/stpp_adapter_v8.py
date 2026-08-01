"""Explicit event-peak support layered on conservative semantic STPP v7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import stpp_adapter_v4 as _v4_module
import stpp_adapter_v7 as _v7_module
from stpp_adapter_v2 import _conditioning_arrays as _conditioning_arrays_v2
from stpp_adapter_v3 import _new_scenario_seed, _node_baselines
from stpp_adapter_v7 import STPPV7Config, simulate_stpp_for_streasoner_v7


_BASE_V7_ENSURE_ROOT_COVERAGE = (
    _v7_module._ensure_root_coverage_without_relay_seeds
)


@dataclass(frozen=True)
class STPPV8Config(STPPV7Config):
    self_generated_peak_factor: float = 3.0

    def validate(self) -> None:
        super().validate()
        if self.self_generated_peak_factor <= 1:
            raise ValueError("self_generated_peak_factor must be greater than 1")


def _self_generated_peaks(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Dict[int, List[int]]:
    demand_nodes = {
        int(node["id"])
        for node in structured_scenario.get("nodes", [])
        if str(node.get("type", "")).lower() == "demand_source"
    }
    peaks: Dict[int, set[int]] = {}
    drift = structured_scenario.get("drift_patterns", {}) or {}
    for node_data in drift.get("nodes", []) or []:
        node_id = int(node_data.get("id", -1))
        if node_id not in demand_nodes:
            continue
        for pattern in node_data.get("patterns", []) or []:
            values = pattern.get("self_generated_peaks")
            if values is None:
                continue
            if not isinstance(values, list):
                values = [values]
            for raw_peak in values:
                try:
                    peak = int(raw_peak)
                except (TypeError, ValueError):
                    continue
                if 0 <= peak < seq_len:
                    peaks.setdefault(node_id, set()).add(peak)
    return {node: sorted(values) for node, values in peaks.items()}


def _conditioning_arrays_v8(
    structured_scenario: Mapping[str, Any],
    num_nodes: int,
    seq_len: int,
    edge_keys: set[str],
    config: STPPV8Config,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    background, edge_factors = _conditioning_arrays_v2(
        structured_scenario, num_nodes, seq_len, edge_keys
    )
    for node_id, peaks in _self_generated_peaks(
        structured_scenario, seq_len
    ).items():
        for peak in peaks:
            background[node_id, peak] = max(
                background[node_id, peak], config.self_generated_peak_factor
            )
    return background, edge_factors


def _ensure_root_coverage_v8(
    immigrants: List[Dict[str, Any]],
    structured_scenario: Mapping[str, Any],
    node_coords: np.ndarray,
    background: np.ndarray,
    seq_len: int,
    config: STPPV8Config,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    roots, info = _BASE_V7_ENSURE_ROOT_COVERAGE(
        immigrants,
        structured_scenario,
        node_coords,
        background,
        seq_len,
        config,
        rng,
    )
    next_id = max((int(event["event_id"]) for event in roots), default=-1) + 1
    reasons = list(info["scenario_seed_reasons"])
    for node_id, peaks in _self_generated_peaks(
        structured_scenario, seq_len
    ).items():
        for peak in peaks:
            covered = any(
                event.get("parent_event_id") is None
                and int(event["node_id"]) == node_id
                and int(event["time_index"]) == peak
                for event in roots
            )
            if covered:
                continue
            reason = f"explicit self-generated event peak for node {node_id} at {peak}"
            roots.append(
                _new_scenario_seed(
                    next_id,
                    node_id,
                    [peak],
                    node_coords,
                    background,
                    rng,
                    reason,
                )
            )
            reasons.append(reason)
            next_id += 1
    roots.sort(key=lambda event: float(event["t"]))
    info["num_scenario_seed_events"] = sum(
        event["event_type"] == "scenario_seed" for event in roots
    )
    info["scenario_seed_reasons"] = reasons
    return roots, info


def simulate_stpp_for_streasoner_v8(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV8Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV8Config()
    config.validate()
    original_conditioning = _v4_module._conditioning_arrays
    original_v7_ensure = _v7_module._ensure_root_coverage_without_relay_seeds
    _v4_module._conditioning_arrays = lambda scenario, nodes, length, keys: (
        _conditioning_arrays_v8(scenario, nodes, length, keys, config)
    )
    _v7_module._ensure_root_coverage_without_relay_seeds = _ensure_root_coverage_v8
    try:
        ts_data, events, metadata = simulate_stpp_for_streasoner_v7(
            structured_scenario, seq_len, config
        )
    finally:
        _v4_module._conditioning_arrays = original_conditioning
        _v7_module._ensure_root_coverage_without_relay_seeds = original_v7_ensure

    expected_peaks = _self_generated_peaks(structured_scenario, seq_len)
    coverage = {
        str(node_id): {
            str(peak): any(
                event.get("parent_event_id") is None
                and int(event["node_id"]) == node_id
                and int(event["time_index"]) == peak
                for event in events
            )
            for peak in peaks
        }
        for node_id, peaks in expected_peaks.items()
    }
    consumed_count = sum(len(peaks) for peaks in expected_peaks.values())
    self_peak_lists_declared = any(
        bool(pattern.get("self_generated_peaks"))
        for node_data in (
            structured_scenario.get("drift_patterns", {}).get("nodes", []) or []
        )
        for pattern in node_data.get("patterns", []) or []
    )
    quality = metadata["quality_report"]
    quality["checks"]["self_generated_peak_lists_consumed"] = (
        not self_peak_lists_declared or consumed_count > 0
    )
    quality["checks"]["all_self_generated_peaks_covered"] = (
        not self_peak_lists_declared
        or bool(coverage)
        and all(all(node_coverage.values()) for node_coverage in coverage.values())
    )
    quality["passed"] = all(quality["checks"].values())
    quality["self_generated_peak_coverage"] = coverage
    quality["num_self_generated_peaks"] = consumed_count

    baselines = _node_baselines(structured_scenario, ts_data.shape[0])
    metadata.update(
        {
            "simulator": "STPPG + explicit event-peak semantic STPP v8",
            "method": (
                "demand-source roots + graph-only relay events + explicit "
                "self-generated event peak coverage"
            ),
            "self_generated_peaks": expected_peaks,
            "self_generated_peak_lists_declared": self_peak_lists_declared,
            "self_generated_peak_factor": config.self_generated_peak_factor,
            "self_generated_peak_targets": {
                str(node_id): {
                    str(peak): float(
                        baselines[node_id] * config.self_generated_peak_factor
                    )
                    for peak in peaks
                }
                for node_id, peaks in expected_peaks.items()
            },
        }
    )
    return ts_data, events, metadata
