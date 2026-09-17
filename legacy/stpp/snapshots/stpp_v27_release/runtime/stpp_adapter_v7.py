"""Conservative semantic conditioning and fail-closed QA for STPP v7."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import stpp_adapter_v4 as _v4_module
from stpp_adapter_v3 import (
    _demand_sources,
    _explicit_peak_windows,
    _new_scenario_seed,
    _strong_edge_windows,
)
from stpp_adapter_v5 import simulate_stpp_for_streasoner_v5
from stpp_adapter_v6 import STPPV6Config
from stpp_adapter_v6_listfix import _normalise_peak_lists


@dataclass(frozen=True)
class STPPV7Config(STPPV6Config):
    propagation_root_weight: float = 0.0


def _variation_centres(node_data: Mapping[str, Any]) -> List[float]:
    centres: List[float] = []
    for variation in node_data.get("propagated_variations", []) or []:
        values = [
            int(value)
            for value in re.findall(r"\d+", str(variation.get("time", "")))
        ]
        if len(values) >= 2:
            centres.append((values[0] + values[1]) / 2.0)
        elif values:
            centres.append(float(values[0]))
    return centres


def _condition_root_patterns_v7(
    structured_scenario: Mapping[str, Any], config: STPPV7Config
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter only explicitly propagated peaks; retain ambiguous demand peaks."""
    conditioned = copy.deepcopy(dict(structured_scenario))
    node_types = {
        int(node["id"]): str(node.get("type", "demand_source")).lower()
        for node in conditioned.get("nodes", [])
    }
    audit: Dict[str, Any] = {
        "excluded_patterns": [],
        "retained_self_generated_patterns": [],
        "ambiguous_demand_pattern_overlaps": [],
        "remaining_explicit_propagated_patterns": [],
    }
    drift = conditioned.get("drift_patterns", {}) or {}
    for node_data in drift.get("nodes", []) or []:
        node_id = int(node_data.get("id", -1))
        node_type = node_types.get(node_id, "demand_source")
        propagated_centres = _variation_centres(node_data)
        retained = []
        for pattern in node_data.get("patterns", []) or []:
            peak_value = pattern.get("peak")
            amplitude = float(pattern.get("amplitude", 0) or 0)
            positive_peak = peak_value is not None and amplitude > 0
            origin = str(pattern.get("origin", "")).lower()
            behavior = str(pattern.get("behavior", "")).lower()
            explicitly_propagated = (
                origin == "propagated" or behavior == "propagated"
            )
            reason = None
            if positive_peak and node_type == "propagation":
                reason = "pure_propagation_node"
            elif positive_peak and explicitly_propagated:
                reason = "explicitly_marked_propagated_pattern"

            record = {
                "node_id": node_id,
                "node_type": node_type,
                "peak": peak_value,
                "amplitude": amplitude,
                "time_range": pattern.get("time_range"),
                "origin": pattern.get("origin"),
                "behavior": pattern.get("behavior"),
            }
            if reason:
                record["reason"] = reason
                audit["excluded_patterns"].append(record)
                continue

            retained.append(pattern)
            if positive_peak:
                audit["retained_self_generated_patterns"].append(record)
                peak = float(peak_value)
                if node_type != "propagation" and any(
                    abs(peak - centre)
                    <= config.propagated_peak_match_tolerance
                    for centre in propagated_centres
                ):
                    warning = dict(record)
                    warning["propagated_variation_centres"] = propagated_centres
                    warning["action"] = "retained; explicit origin is required to remove"
                    audit["ambiguous_demand_pattern_overlaps"].append(warning)
            if positive_peak and (
                node_type == "propagation" or explicitly_propagated
            ):
                audit["remaining_explicit_propagated_patterns"].append(record)
        node_data["patterns"] = retained

    audit["num_excluded_patterns"] = len(audit["excluded_patterns"])
    audit["num_ambiguous_demand_pattern_overlaps"] = len(
        audit["ambiguous_demand_pattern_overlaps"]
    )
    return conditioned, audit


def _ensure_root_coverage_without_relay_seeds(
    immigrants: List[Dict[str, Any]],
    structured_scenario: Mapping[str, Any],
    node_coords: np.ndarray,
    background: np.ndarray,
    seq_len: int,
    config: STPPV7Config,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    roots = list(immigrants)
    next_id = max((int(event["event_id"]) for event in roots), default=-1) + 1
    added_reasons: List[str] = []
    skipped_relay_edges: List[str] = []
    propagation_nodes = {
        int(node["id"])
        for node in structured_scenario.get("nodes", [])
        if str(node.get("type", "")).lower() == "propagation"
    }

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
            if node_id in propagation_nodes:
                continue
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
        for key, indices in _strong_edge_windows(
            structured_scenario, seq_len
        ).items():
            source = int(key.split("->", 1)[0])
            if source in propagation_nodes:
                skipped_relay_edges.append(key)
                continue
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
    return roots, {
        "num_scenario_seed_events": sum(
            event["event_type"] == "scenario_seed" for event in roots
        ),
        "scenario_seed_reasons": added_reasons,
        "skipped_relay_strong_edge_root_seeds": skipped_relay_edges,
    }


def simulate_stpp_for_streasoner_v7(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV7Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV7Config()
    config.validate()
    peak_normalised, peak_audit = _normalise_peak_lists(structured_scenario)
    conditioned, root_audit = _condition_root_patterns_v7(
        peak_normalised, config
    )

    original_ensure = _v4_module._ensure_root_coverage
    _v4_module._ensure_root_coverage = _ensure_root_coverage_without_relay_seeds
    try:
        ts_data, events, metadata = simulate_stpp_for_streasoner_v5(
            conditioned, seq_len, config
        )
    finally:
        _v4_module._ensure_root_coverage = original_ensure

    propagation_nodes = {
        int(node["id"])
        for node in structured_scenario.get("nodes", [])
        if str(node.get("type", "")).lower() == "propagation"
    }
    propagation_roots = [
        event
        for event in events
        if event.get("parent_event_id") is None
        and int(event["node_id"]) in propagation_nodes
    ]
    skipped_relay_edge_seeds = sorted(
        key
        for key in _strong_edge_windows(conditioned, seq_len)
        if int(key.split("->", 1)[0]) in propagation_nodes
    )
    quality = metadata["quality_report"]
    quality["checks"]["peak_lists_normalised"] = not peak_audit[
        "invalid_peak_values"
    ]
    quality["checks"]["no_roots_on_pure_propagation_nodes"] = not (
        propagation_roots
    )
    quality["checks"]["explicit_propagated_patterns_excluded_from_roots"] = not (
        root_audit["remaining_explicit_propagated_patterns"]
    )
    quality["passed"] = all(quality["checks"].values())
    quality["propagation_node_root_event_count"] = len(propagation_roots)
    quality["peak_list_normalisation"] = peak_audit
    quality["root_pattern_conditioning"] = root_audit

    metadata.update(
        {
            "simulator": "STPPG + conservative semantic graph STPP v7",
            "method": (
                "demand-source roots + graph-only relay events + conservative "
                "origin-aware peak filtering"
            ),
            "root_conditioned_structured_scenario": conditioned,
            "peak_list_normalisation": peak_audit,
            "root_pattern_conditioning": root_audit,
            "skipped_relay_strong_edge_root_seeds": skipped_relay_edge_seeds,
        }
    )
    return ts_data, events, metadata
