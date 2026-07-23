"""Lag-aware, quota-balanced, magnitude-calibrated STPP adapter (v4)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from stpp_adapter import _generate_one_realisation, _normalised_node_coordinates
from stpp_adapter_v2 import (
    _assign_and_thin_immigrants,
    _conditioning_arrays,
    _propagate_events,
    _scenario_edges,
)
from stpp_adapter_v3 import (
    STPPV3Config,
    _aggregate_v3,
    _annotate_parent_times,
    _compact_event_ids,
    _demand_sources,
    _ensure_root_coverage,
    _force_strong_edge_coverage,
    _node_baselines,
    _quality_report,
)


@dataclass(frozen=True)
class STPPV4Config(STPPV3Config):
    demand_source_root_weight: float = 1.0
    propagation_root_weight: float = 0.25
    max_propagation_root_fraction: float = 0.35
    lag_mean_tolerance: float = 0.15
    max_matrix_target_ratio: float = 3.0
    max_root_magnitude_relative_error: float = 1e-8

    def validate(self) -> None:
        super().validate()
        if self.demand_source_root_weight <= 0:
            raise ValueError("demand_source_root_weight must be positive")
        if self.propagation_root_weight < 0:
            raise ValueError("propagation_root_weight cannot be negative")
        if not 0 <= self.max_propagation_root_fraction <= 1:
            raise ValueError("max_propagation_root_fraction must be in [0, 1]")
        if self.lag_mean_tolerance < 0:
            raise ValueError("lag_mean_tolerance cannot be negative")
        if self.max_matrix_target_ratio <= 0:
            raise ValueError("max_matrix_target_ratio must be positive")


def _largest_remainder_quotas(weights: np.ndarray, total: int) -> np.ndarray:
    if total < 1 or float(weights.sum()) <= 0:
        raise ValueError("cannot allocate root-event quotas")
    exact = weights / weights.sum() * total
    quotas = np.floor(exact).astype(int)
    remaining = total - int(quotas.sum())
    order = np.argsort(-(exact - quotas))
    for index in order[:remaining]:
        quotas[int(index)] += 1
    return quotas


def _root_quotas(
    structured_scenario: Mapping[str, Any],
    baselines: np.ndarray,
    target: int,
    config: STPPV4Config,
) -> np.ndarray:
    node_types = {
        int(node["id"]): str(node.get("type", "demand_source")).lower()
        for node in structured_scenario.get("nodes", [])
    }
    weights = np.asarray(
        [
            baselines[node_id]
            * (
                config.propagation_root_weight
                if node_types.get(node_id) == "propagation"
                else config.demand_source_root_weight
            )
            for node_id in range(len(baselines))
        ],
        dtype=float,
    )
    quotas = _largest_remainder_quotas(weights, target)
    demand_sources = _demand_sources(structured_scenario)
    for node_id in demand_sources:
        if quotas[node_id] >= config.min_events_per_demand_source:
            continue
        deficit = config.min_events_per_demand_source - quotas[node_id]
        donors = np.argsort(-quotas)
        for donor in donors:
            donor = int(donor)
            if donor == node_id:
                continue
            transferable = max(0, quotas[donor] - 1)
            moved = min(deficit, transferable)
            quotas[donor] -= moved
            quotas[node_id] += moved
            deficit -= moved
            if deficit == 0:
                break
        if deficit:
            raise ValueError("unable to satisfy demand-source root quota")
    return quotas


def _collect_quota_immigrants(
    seq_len: int,
    node_ids: Sequence[int],
    node_coords: np.ndarray,
    background: np.ndarray,
    quotas: np.ndarray,
    config: STPPV4Config,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pools: Dict[int, List[Dict[str, Any]]] = {
        node_id: [] for node_id in range(len(node_ids))
    }
    batches = 0
    while batches < config.max_candidate_batches:
        if all(len(pools[node]) >= int(quotas[node]) for node in pools):
            break
        points = _generate_one_realisation(seq_len, config)  # type: ignore[arg-type]
        retained = _assign_and_thin_immigrants(
            points,
            node_ids,
            node_coords,
            background,
            config,  # type: ignore[arg-type]
            rng,
        )
        for event in retained:
            pools[int(event["node_id"])].append(event)
        batches += 1

    shortages = {
        str(node): int(quotas[node]) - len(pools[node])
        for node in pools
        if len(pools[node]) < int(quotas[node])
    }
    if shortages:
        raise RuntimeError(
            f"root quotas not met after {batches} candidate batches: {shortages}"
        )

    selected: List[Dict[str, Any]] = []
    for node_id, pool in pools.items():
        pool.sort(
            key=lambda event: float(event.get("keep_probability", 0)), reverse=True
        )
        selected.extend(pool[: int(quotas[node_id])])
    selected.sort(key=lambda event: float(event["t"]))
    for event_id, event in enumerate(selected):
        event["event_id"] = event_id
    return selected, {
        "target_immigrant_events": int(quotas.sum()),
        "candidate_batches": batches,
        "root_event_quotas": {
            str(node_id): int(value) for node_id, value in enumerate(quotas)
        },
        "retained_candidate_pool_sizes": {
            str(node_id): len(pool) for node_id, pool in pools.items()
        },
    }


def _calibrate_magnitudes(
    events: List[Dict[str, Any]],
    baselines: np.ndarray,
    background: np.ndarray,
    config: STPPV4Config,
    rng: np.random.Generator,
    magnitude_unit: str,
) -> Dict[str, Any]:
    roots_by_cell: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for event in events:
        if event.get("parent_event_id") is None:
            key = (int(event["node_id"]), int(event["time_index"]))
            roots_by_cell.setdefault(key, []).append(event)

    max_relative_error = 0.0
    for (node_id, time_index), roots in roots_by_cell.items():
        target = float(baselines[node_id] * background[node_id, time_index])
        raw_weights = np.maximum(
            rng.normal(1.0, config.magnitude_noise, size=len(roots)), 0.01
        )
        weights = raw_weights / raw_weights.sum()
        for event, weight in zip(roots, weights):
            event["magnitude"] = float(target * weight)
            event["magnitude_unit"] = magnitude_unit
            event["cell_target_magnitude"] = target
        assigned = sum(float(event["magnitude"]) for event in roots)
        relative_error = abs(assigned - target) / max(target, 1e-12)
        max_relative_error = max(max_relative_error, relative_error)

    by_id: Dict[int, Dict[str, Any]] = {}
    for event in sorted(events, key=lambda item: float(item["t"])):
        parent_id = event.get("parent_event_id")
        if parent_id is not None:
            parent = by_id.get(int(parent_id))
            expected = (
                float(parent["magnitude"]) * config.propagation_magnitude_decay
                if parent is not None
                else float(baselines[int(event["node_id"])])
            )
            event["magnitude"] = max(
                0.01,
                float(rng.normal(expected, max(expected * config.magnitude_noise, 1e-6))),
            )
            event["magnitude_unit"] = magnitude_unit
        by_id[int(event["event_id"])] = event
    return {"max_root_magnitude_relative_error": max_relative_error}


def _lag_report(
    events: Sequence[Mapping[str, Any]],
    structured_scenario: Mapping[str, Any],
    config: STPPV4Config,
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for event in events:
        if event.get("edge"):
            groups.setdefault(str(event["edge"]), []).append(event)
    configured_edges = {
        f'{int(edge["source"])}->{int(edge["target"])}': max(
            float(edge.get("time_lag", 1)), 0.01
        )
        for edge in structured_scenario.get("edges", [])
    }
    details: Dict[str, Any] = {}
    for key, configured in configured_edges.items():
        group = groups.get(key, [])
        if not group:
            details[key] = {
                "events": 0,
                "configured_mean": configured,
                "realized_mean": None,
                "absolute_error": None,
                "tolerance": max(config.lag_mean_tolerance, configured * 0.25),
                "passed": False,
                "reason": "no propagated event was generated for this edge",
            }
            continue
        realized = float(np.mean([float(event["realized_delay"]) for event in group]))
        absolute_error = abs(realized - configured)
        tolerance = max(config.lag_mean_tolerance, configured * 0.25)
        details[key] = {
            "events": len(group),
            "configured_mean": configured,
            "realized_mean": realized,
            "absolute_error": absolute_error,
            "tolerance": tolerance,
            "passed": absolute_error <= tolerance,
        }
    return {
        "passed": bool(configured_edges)
        and all(item["passed"] for item in details.values()),
        "configured_edge_count": len(configured_edges),
        "observed_edge_count": sum(bool(groups.get(key)) for key in configured_edges),
        "edges": details,
    }


def simulate_stpp_for_streasoner_v4(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV4Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    config = config or STPPV4Config()
    config.validate()
    if config.seed is not None:
        np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    node_ids, node_coords = _normalised_node_coordinates(structured_scenario)
    if node_ids != list(range(len(node_ids))):
        raise ValueError("v4 requires contiguous node IDs 0..N-1")
    baselines = _node_baselines(structured_scenario, len(node_ids))
    target = max(
        config.min_events,
        int(math.ceil(seq_len * len(node_ids) * config.target_immigrant_rate)),
    )
    quotas = _root_quotas(structured_scenario, baselines, target, config)
    outgoing, edge_keys = _scenario_edges(structured_scenario, len(node_ids))
    background, edge_factors = _conditioning_arrays(
        structured_scenario, len(node_ids), seq_len, edge_keys
    )
    immigrants, density_info = _collect_quota_immigrants(
        seq_len,
        node_ids,
        node_coords,
        background,
        quotas,
        config,
        rng,
    )
    roots, seed_info = _ensure_root_coverage(
        immigrants,
        structured_scenario,
        node_coords,
        background,
        seq_len,
        config,  # type: ignore[arg-type]
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
        config,  # type: ignore[arg-type]
        rng,
    )
    _annotate_parent_times(events)
    magnitude_unit = str(
        structured_scenario.get("variable", "scenario variable units per time window")
    )
    magnitude_report = _calibrate_magnitudes(
        events, baselines, background, config, rng, magnitude_unit
    )
    events = _compact_event_ids(events)
    ts_data = _aggregate_v3(
        events, len(node_ids), seq_len, config  # type: ignore[arg-type]
    )

    quality = _quality_report(
        events,
        structured_scenario,
        seq_len,
        int(quotas.sum()),
        config.min_events_per_demand_source,
    )
    root_counts = {
        str(node_id): sum(
            event.get("parent_event_id") is None
            and int(event["node_id"]) == node_id
            for event in events
        )
        for node_id in range(len(node_ids))
    }
    quota_checks = {
        str(node_id): root_counts[str(node_id)] >= int(quotas[node_id])
        for node_id in range(len(node_ids))
    }
    propagation_nodes = {
        int(node["id"])
        for node in structured_scenario.get("nodes", [])
        if str(node.get("type", "")).lower() == "propagation"
    }
    total_roots = sum(root_counts.values())
    propagation_roots = sum(root_counts[str(node)] for node in propagation_nodes)
    propagation_root_fraction = propagation_roots / max(1, total_roots)
    lag_report = _lag_report(events, structured_scenario, config)
    target_profile = baselines[:, None] * background
    ratios = np.divide(
        ts_data,
        target_profile,
        out=np.zeros_like(ts_data),
        where=target_profile > 0,
    )
    max_matrix_target_ratio = float(ratios.max())

    extra_checks = {
        "all_node_root_quotas_met": all(quota_checks.values()),
        "propagation_root_fraction_within_limit": (
            propagation_root_fraction <= config.max_propagation_root_fraction
        ),
        "all_edge_lag_means_within_tolerance": lag_report["passed"],
        "root_magnitudes_calibrated": (
            magnitude_report["max_root_magnitude_relative_error"]
            <= config.max_root_magnitude_relative_error
        ),
        "matrix_peak_ratio_within_limit": (
            max_matrix_target_ratio <= config.max_matrix_target_ratio
        ),
    }
    quality["checks"].update(extra_checks)
    if config.aggregation == "count":
        aggregation_matches = bool(np.isclose(float(ts_data.sum()), len(events)))
    elif config.aggregation == "magnitude":
        aggregation_matches = bool(
            np.isclose(
                float(ts_data.sum()),
                sum(float(event["magnitude"]) for event in events),
            )
        )
    else:
        aggregation_matches = True
    quality["checks"]["aggregation_total_matches"] = aggregation_matches
    quality["passed"] = all(quality["checks"].values())
    quality["node_root_quotas"] = density_info["root_event_quotas"]
    quality["node_root_counts"] = root_counts
    quality["node_root_quota_checks"] = quota_checks
    quality["propagation_root_fraction"] = propagation_root_fraction
    quality["lag_fidelity"] = lag_report
    quality["magnitude_calibration"] = magnitude_report
    quality["max_matrix_target_ratio"] = max_matrix_target_ratio

    propagated = sum(event.get("parent_event_id") is not None for event in events)
    metadata: Dict[str, Any] = {
        "simulator": "STPPG + lag-aware quota-balanced graph branching v4",
        "simulator_commit": "be65d949e475c636a34fc6216044be94e139f50f",
        "method": "node-quota STPPG roots + calibrated marks + lag-aware graph offspring",
        "integration_method": "quota control + quality-gated marked branching + calibrated aggregation",
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
        "target_magnitude_profile": target_profile.tolist(),
        "event_magnitude_total": float(
            sum(float(event["magnitude"]) for event in events)
        ),
        "matrix_total": float(ts_data.sum()),
    }
    return ts_data, events, metadata
