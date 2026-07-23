"""Scenario- and graph-conditioned STPP adapter (non-destructive v2 copy).

STPPG still generates the continuous immigrant events.  This adapter then:

1. assigns each immigrant to the nearest scenario node;
2. thins immigrants using scenario temporal profiles; and
3. creates marked offspring along directed graph edges using their time lags.

The result is an event stream whose node marks and propagation history are
explicit, while retaining the dense ``node x time`` compatibility matrix.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from stpp_adapter import (
    _generate_one_realisation,
    _normalised_node_coordinates,
)


@dataclass(frozen=True)
class STPPV2Config:
    # Upstream STPPG parameters.
    mu: float = 1.0
    kernel_c: float = 0.05
    beta: float = 1.0
    sigma_x: float = 0.15
    sigma_y: float = 0.15
    intensity_upper_bound: float = 20.0
    min_events: int = 5
    max_attempts: int = 20

    # STReasoner graph-conditioning parameters.
    minimum_keep_probability: float = 0.25
    edge_branching_ratio: float = 0.35
    propagation_jitter: float = 0.10
    target_spatial_jitter: float = 0.04
    max_generation: int = 3
    max_events: int = 5000

    aggregation: str = "count"
    rolling_window: int = 3
    seed: int | None = None

    def validate(self) -> None:
        if self.mu <= 0 or self.kernel_c <= 0 or self.beta <= 0:
            raise ValueError("mu, kernel_c, and beta must be positive")
        if self.sigma_x <= 0 or self.sigma_y <= 0:
            raise ValueError("sigma_x and sigma_y must be positive")
        if self.intensity_upper_bound <= self.mu:
            raise ValueError("intensity_upper_bound must be greater than mu")
        if self.min_events < 1 or self.max_attempts < 1:
            raise ValueError("min_events and max_attempts must be at least 1")
        if not 0 < self.minimum_keep_probability <= 1:
            raise ValueError("minimum_keep_probability must be in (0, 1]")
        if not 0 <= self.edge_branching_ratio < 1:
            raise ValueError("edge_branching_ratio must be in [0, 1)")
        if self.propagation_jitter < 0 or self.target_spatial_jitter < 0:
            raise ValueError("jitter values cannot be negative")
        if self.max_generation < 0 or self.max_events < self.min_events:
            raise ValueError("invalid max_generation or max_events")
        if self.aggregation not in {"count", "rolling_count", "cumulative_count"}:
            raise ValueError("unsupported aggregation")
        if self.rolling_window < 1:
            raise ValueError("rolling_window must be at least 1")


def _numbers_from_time_spec(value: Any) -> List[int]:
    """Preserve explicit hour lists and expand textual inclusive ranges."""
    if isinstance(value, Mapping):
        value = value.get("hour_range", value.get("time_range", []))
    if isinstance(value, (list, tuple)):
        values: List[int] = []
        for item in value:
            if isinstance(item, (list, tuple, Mapping)):
                values.extend(_numbers_from_time_spec(item))
            else:
                try:
                    values.append(int(item))
                except (TypeError, ValueError):
                    values.extend(_numbers_from_time_spec(str(item)))
        return sorted(set(values))

    text = str(value)
    values = []
    for start_text, end_text in re.findall(r"(\d+)\s*-\s*(\d+)", text):
        start, end = int(start_text), int(end_text)
        low, high = sorted((start, end))
        values.extend(range(low, high + 1))
    if not values:
        values = [int(item) for item in re.findall(r"\d+", text)]
    return sorted(set(values))


def _day_filter(pattern: Mapping[str, Any]) -> set[int] | None:
    """Return day indices where day 0 is Sunday, or None for all days."""
    explicit = pattern.get("days", pattern.get("day_range"))
    if explicit is not None:
        values = _numbers_from_time_spec(explicit)
        if (
            "days" not in pattern
            and isinstance(explicit, (list, tuple))
            and len(explicit) == 2
            and len(values) == 2
        ):
            low, high = sorted(values)
            values = list(range(low, high + 1))
        # Scenario prompts commonly use Monday=1 ... Sunday=7.
        return {value % 7 for value in values}

    description = str(pattern.get("description", "")).lower()
    if "weekday" in description or "monday-friday" in description:
        return {1, 2, 3, 4, 5}
    if "weekend" in description or "saturday-sunday" in description:
        return {0, 6}
    return None


def active_time_indices(
    time_spec: Any,
    pattern: Mapping[str, Any],
    seq_len: int,
) -> List[int]:
    """Expand a scenario time specification without dropping list members."""
    values = _numbers_from_time_spec(time_spec)
    if not values:
        return []
    days = _day_filter(pattern)
    is_hour_of_day = seq_len > 24 and max(values) <= 23
    if is_hour_of_day:
        allowed_hours = set(values)
        return [
            index
            for index in range(seq_len)
            if index % 24 in allowed_hours
            and (days is None or (index // 24) % 7 in days)
        ]
    return [value for value in values if 0 <= value < seq_len]


def _effect_factor(effect: Any, *, for_branching: bool) -> float:
    value = str(effect or "moderate").lower()
    if for_branching:
        return {"strong": 1.8, "moderate": 1.2, "weak": 0.5}.get(value, 1.0)
    return {"strong": 3.0, "moderate": 1.5, "weak": 0.5}.get(value, 1.0)


def _edge_key(source: int, target: int) -> str:
    return f"{source}->{target}"


def _edge_strings(value: Any) -> Iterable[str]:
    if isinstance(value, list):
        for item in value:
            yield str(item)
    elif value is not None:
        yield from re.split(r"\s*[,;]\s*", str(value))


def _scenario_edges(
    structured_scenario: Mapping[str, Any],
    num_nodes: int,
) -> Tuple[Dict[int, List[Dict[str, Any]]], set[str]]:
    outgoing: Dict[int, List[Dict[str, Any]]] = {node: [] for node in range(num_nodes)}
    keys: set[str] = set()
    for edge in structured_scenario.get("edges", []):
        source, target = int(edge["source"]), int(edge["target"])
        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            raise ValueError(f"edge {source}->{target} references an unknown node")
        lag = max(float(edge.get("time_lag", 1)), 0.01)
        outgoing[source].append({"target": target, "lag": lag})
        keys.add(_edge_key(source, target))
    return outgoing, keys


def _conditioning_arrays(
    structured_scenario: Mapping[str, Any],
    num_nodes: int,
    seq_len: int,
    edge_keys: set[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Create background and edge excitation factors from parsed scenario data."""
    background = np.ones((num_nodes, seq_len), dtype=float)
    edge_factors = {key: np.ones(seq_len, dtype=float) for key in edge_keys}

    patterns = (
        structured_scenario.get("adjacency_modulation", {}).get("patterns", [])
    )
    for pattern in patterns:
        indices = active_time_indices(
            pattern.get("time_period", pattern.get("time_range")),
            pattern,
            seq_len,
        )
        if not indices:
            continue
        background_factor = _effect_factor(pattern.get("effect"), for_branching=False)
        branching_factor = _effect_factor(pattern.get("effect"), for_branching=True)
        for text in _edge_strings(pattern.get("applies_to")):
            match = re.search(r"(\d+)\s*->\s*(\d+)", text)
            if not match:
                continue
            source, target = int(match.group(1)), int(match.group(2))
            key = _edge_key(source, target)
            if 0 <= source < num_nodes:
                current = background[source, indices]
                if background_factor >= 1.0:
                    background[source, indices] = np.maximum(
                        current, background_factor
                    )
                else:
                    # Weak periods lower only untouched baseline slots. Strong
                    # or moderate overlapping periods retain precedence.
                    background[source, indices] = np.where(
                        current == 1.0, background_factor, current
                    )
            if key in edge_factors:
                current = edge_factors[key][indices]
                if branching_factor >= 1.0:
                    edge_factors[key][indices] = np.maximum(
                        current, branching_factor
                    )
                else:
                    edge_factors[key][indices] = np.where(
                        current == 1.0, branching_factor, current
                    )

    # Add explicit self-generated peaks from drift patterns. Peaks below 24 in
    # multi-day scenarios are interpreted as hour-of-day and repeated daily.
    drift = structured_scenario.get("drift_patterns", {}) or {}
    for node_data in drift.get("nodes", []):
        node_id = int(node_data.get("id", -1))
        if not 0 <= node_id < num_nodes:
            continue
        for pattern in node_data.get("patterns", []):
            peak = pattern.get("peak")
            amplitude = float(pattern.get("amplitude", 0) or 0)
            baseline = max(float(pattern.get("baseline", 1) or 1), 1e-6)
            if peak is None or amplitude <= 0:
                continue
            peak = int(peak)
            peak_factor = min(1.0 + amplitude / baseline, 8.0)
            for index in range(seq_len):
                local_index = index % 24 if seq_len > 24 and peak < 24 else index
                distance = abs(local_index - peak)
                if seq_len > 24 and peak < 24:
                    distance = min(distance, 24 - distance)
                boost = 1.0 + (peak_factor - 1.0) * np.exp(
                    -0.5 * (distance / 1.5) ** 2
                )
                background[node_id, index] = max(
                    background[node_id, index], boost
                )

    return background, edge_factors


def _assign_and_thin_immigrants(
    points: np.ndarray,
    node_ids: Sequence[int],
    node_coords: np.ndarray,
    background: np.ndarray,
    config: STPPV2Config,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    next_id = 0
    maxima = np.maximum(background.max(axis=1), 1e-12)
    for t, x, y in points:
        node_index = int(np.argmin(np.sum((node_coords - [x, y]) ** 2, axis=1)))
        time_index = min(max(int(np.floor(t)), 0), background.shape[1] - 1)
        relative = float(background[node_index, time_index] / maxima[node_index])
        keep_probability = config.minimum_keep_probability + (
            1.0 - config.minimum_keep_probability
        ) * relative
        event = {
            "event_id": next_id,
            "t": float(t),
            "x": float(x),
            "y": float(y),
            "node_id": int(node_ids[node_index]),
            "time_index": time_index,
            "event_type": "immigrant",
            "parent_event_id": None,
            "source_node_id": None,
            "generation": 0,
            "keep_probability": float(keep_probability),
        }
        candidates.append((keep_probability, event))
        next_id += 1

    retained = [event for probability, event in candidates if rng.random() <= probability]
    if len(retained) < config.min_events:
        retained_ids = {event["event_id"] for event in retained}
        for _, event in sorted(candidates, key=lambda item: item[0], reverse=True):
            if event["event_id"] not in retained_ids:
                retained.append(event)
                retained_ids.add(event["event_id"])
            if len(retained) >= config.min_events:
                break
    return retained


def _propagate_events(
    immigrants: List[Dict[str, Any]],
    outgoing: Dict[int, List[Dict[str, Any]]],
    edge_factors: Dict[str, np.ndarray],
    node_coords: np.ndarray,
    seq_len: int,
    config: STPPV2Config,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    events = list(immigrants)
    queue = list(immigrants)
    next_id = max((event["event_id"] for event in events), default=-1) + 1
    while queue and len(events) < config.max_events:
        parent = queue.pop(0)
        if int(parent["generation"]) >= config.max_generation:
            continue
        source = int(parent["node_id"])
        for edge in outgoing.get(source, []):
            target = int(edge["target"])
            key = _edge_key(source, target)
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
            }
            events.append(child)
            queue.append(child)
            next_id += 1
            if len(events) >= config.max_events:
                break
    return sorted(events, key=lambda event: (event["t"], event["event_id"]))


def _aggregate(
    events: Sequence[Mapping[str, Any]],
    num_nodes: int,
    seq_len: int,
    config: STPPV2Config,
) -> np.ndarray:
    counts = np.zeros((num_nodes, seq_len), dtype=float)
    for event in events:
        counts[int(event["node_id"]), int(event["time_index"])] += 1.0
    if config.aggregation == "rolling_count":
        kernel = np.ones(config.rolling_window, dtype=float)
        counts = np.vstack([np.convolve(row, kernel, mode="same") for row in counts])
    elif config.aggregation == "cumulative_count":
        counts = np.cumsum(counts, axis=1)
    return counts


def simulate_stpp_for_streasoner_v2(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV2Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    config = config or STPPV2Config()
    config.validate()
    if config.seed is not None:
        np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    node_ids, node_coords = _normalised_node_coordinates(structured_scenario)
    if node_ids != list(range(len(node_ids))):
        raise ValueError(
            "v2 requires contiguous node IDs 0..N-1 so graph marks and "
            "coordinate rows have the same index"
        )
    outgoing, edge_keys = _scenario_edges(structured_scenario, len(node_ids))
    background, edge_factors = _conditioning_arrays(
        structured_scenario, len(node_ids), seq_len, edge_keys
    )
    points = _generate_one_realisation(seq_len, config)  # type: ignore[arg-type]
    immigrants = _assign_and_thin_immigrants(
        points, node_ids, node_coords, background, config, rng
    )
    events = _propagate_events(
        immigrants,
        outgoing,
        edge_factors,
        node_coords,
        seq_len,
        config,
        rng,
    )
    ts_data = _aggregate(events, len(node_ids), seq_len, config)

    propagated = sum(event["event_type"] == "propagated" for event in events)
    metadata: Dict[str, Any] = {
        "simulator": "meowoodie/Spatio-Temporal-Point-Process-Simulator + STReasoner marked branching v2",
        "simulator_commit": "be65d949e475c636a34fc6216044be94e139f50f",
        "method": "STPPG immigrants + scenario thinning + graph-marked offspring",
        "integration_method": "STPP thinning + marked network branching + time binning",
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
        ],
        "aggregation": config.aggregation,
        "dt": 1.0,
        "num_events": len(events),
        "num_immigrant_events": len(immigrants),
        "num_propagated_events": propagated,
        "graph_conditioned": True,
        "scenario_temporal_conditioned": True,
        "node_coordinates": {
            str(node_id): {"x": float(coord[0]), "y": float(coord[1])}
            for node_id, coord in zip(node_ids, node_coords)
        },
        "point_process_parameters": asdict(config),
        "background_profile": background.tolist(),
        "edge_excitation_profiles": {
            key: values.tolist() for key, values in edge_factors.items()
        },
    }
    return ts_data, events, metadata
