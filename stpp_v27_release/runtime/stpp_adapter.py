"""Compatibility adapter for the upstream STPPG simulator.

The upstream project emits padded tensors of continuous events ``(t, x, y)``.
STReasoner's existing QA builders consume a dense ``node x time`` matrix.  This
module keeps both representations: the original events are preserved and a
deterministic binning step produces the compatibility matrix.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


UPSTREAM_DIR = (
    Path(__file__).resolve().parents[3]
    / "external"
    / "Spatio-Temporal-Point-Process-Simulator"
)


def _load_upstream():
    """Import the unmodified upstream module from its vendored directory."""
    if not (UPSTREAM_DIR / "stppg.py").is_file():
        raise FileNotFoundError(
            "STPPG source is missing. Expected: "
            f"{UPSTREAM_DIR / 'stppg.py'}"
        )
    upstream = str(UPSTREAM_DIR)
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from stppg import (  # type: ignore[import-not-found]
        HawkesLam,
        SpatialTemporalPointProcess,
        StdDiffusionKernel,
    )

    return HawkesLam, SpatialTemporalPointProcess, StdDiffusionKernel


@dataclass(frozen=True)
class STPPConfig:
    """Parameters for a single spatial-temporal Hawkes realization."""

    mu: float = 1.0
    kernel_c: float = 0.05
    beta: float = 1.0
    sigma_x: float = 0.15
    sigma_y: float = 0.15
    intensity_upper_bound: float = 20.0
    min_events: int = 5
    max_attempts: int = 20
    aggregation: str = "count"
    rolling_window: int = 3
    seed: int | None = None

    def validate(self) -> None:
        if self.mu <= 0:
            raise ValueError("mu must be positive")
        if self.kernel_c <= 0 or self.beta <= 0:
            raise ValueError("kernel_c and beta must be positive")
        if self.sigma_x <= 0 or self.sigma_y <= 0:
            raise ValueError("sigma_x and sigma_y must be positive")
        if self.intensity_upper_bound <= self.mu:
            raise ValueError("intensity_upper_bound must be greater than mu")
        if self.min_events < 1 or self.max_attempts < 1:
            raise ValueError("min_events and max_attempts must be at least 1")
        if self.aggregation not in {"count", "rolling_count", "cumulative_count"}:
            raise ValueError(
                "aggregation must be count, rolling_count, or cumulative_count"
            )
        if self.rolling_window < 1:
            raise ValueError("rolling_window must be at least 1")


def _normalised_node_coordinates(
    structured_scenario: Mapping[str, Any],
) -> Tuple[List[int], np.ndarray]:
    """Return node IDs and coordinates scaled into the unit square."""
    nodes = structured_scenario.get("nodes", [])
    if not nodes:
        raise ValueError("structured_scenario must contain at least one node")

    node_ids = [int(node["id"]) for node in nodes]
    layout = structured_scenario.get("spatial_layout", {}) or {}
    raw = []
    for index, node_id in enumerate(node_ids):
        item = layout.get(str(node_id), layout.get(node_id))
        if isinstance(item, Mapping) and "x" in item and "y" in item:
            raw.append([float(item["x"]), float(item["y"])])
        else:
            angle = 2.0 * np.pi * index / max(len(node_ids), 1)
            raw.append([np.cos(angle), np.sin(angle)])

    coords = np.asarray(raw, dtype=float)
    for axis in (0, 1):
        low = float(coords[:, axis].min())
        high = float(coords[:, axis].max())
        if high == low:
            coords[:, axis] = 0.5
        else:
            coords[:, axis] = 0.05 + 0.90 * (coords[:, axis] - low) / (high - low)
    return node_ids, coords


def _generate_one_realisation(seq_len: int, config: STPPConfig) -> np.ndarray:
    """Generate events with a bounded retry loop around upstream thinning."""
    HawkesLam, SpatialTemporalPointProcess, StdDiffusionKernel = _load_upstream()
    kernel = StdDiffusionKernel(
        C=config.kernel_c,
        beta=config.beta,
        sigma_x=config.sigma_x,
        sigma_y=config.sigma_y,
    )
    lam = HawkesLam(config.mu, kernel, maximum=config.intensity_upper_bound)
    process = SpatialTemporalPointProcess(lam)

    for _ in range(config.max_attempts):
        candidates = process._homogeneous_poisson_sampling(  # noqa: SLF001
            T=[0.0, float(seq_len)], S=[[0.0, 1.0], [0.0, 1.0]]
        )
        points = process._inhomogeneous_poisson_thinning(  # noqa: SLF001
            candidates, verbose=False
        )
        if points is not None and len(points) >= config.min_events:
            return np.asarray(points, dtype=float)

    raise RuntimeError(
        "STPPG could not produce a valid realization within max_attempts. "
        "Increase intensity_upper_bound or adjust mu/kernel parameters."
    )


def _aggregate_events(
    points: np.ndarray,
    node_ids: Sequence[int],
    node_coords: np.ndarray,
    seq_len: int,
    config: STPPConfig,
) -> Tuple[np.ndarray, List[Dict[str, float | int]]]:
    counts = np.zeros((len(node_ids), seq_len), dtype=float)
    events: List[Dict[str, float | int]] = []

    for t, x, y in points:
        node_index = int(np.argmin(np.sum((node_coords - [x, y]) ** 2, axis=1)))
        time_index = min(max(int(np.floor(t)), 0), seq_len - 1)
        counts[node_index, time_index] += 1.0
        events.append(
            {
                "t": float(t),
                "x": float(x),
                "y": float(y),
                "node_id": int(node_ids[node_index]),
                "time_index": time_index,
            }
        )

    if config.aggregation == "rolling_count":
        kernel = np.ones(config.rolling_window, dtype=float)
        counts = np.vstack([np.convolve(row, kernel, mode="same") for row in counts])
    elif config.aggregation == "cumulative_count":
        counts = np.cumsum(counts, axis=1)

    return counts, events


def simulate_stpp_for_streasoner(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPConfig | None = None,
) -> Tuple[np.ndarray, List[Dict[str, float | int]], Dict[str, Any]]:
    """Generate a raw event stream and a STReasoner-compatible dense matrix."""
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    config = config or STPPConfig()
    config.validate()

    if config.seed is not None:
        np.random.seed(config.seed)

    node_ids, node_coords = _normalised_node_coordinates(structured_scenario)
    points = _generate_one_realisation(seq_len, config)
    ts_data, events = _aggregate_events(
        points, node_ids, node_coords, seq_len, config
    )

    metadata: Dict[str, Any] = {
        "simulator": "meowoodie/Spatio-Temporal-Point-Process-Simulator",
        "simulator_commit": "be65d949e475c636a34fc6216044be94e139f50f",
        "method": "spatial-temporal Hawkes process via Poisson thinning",
        "integration_method": "STPP thinning + nearest-node/time-window binning",
        "event_format": ["t", "x", "y", "node_id", "time_index"],
        "aggregation": config.aggregation,
        "dt": 1.0,
        "num_events": len(events),
        "node_coordinates": {
            str(node_id): {"x": float(coord[0]), "y": float(coord[1])}
            for node_id, coord in zip(node_ids, node_coords)
        },
        "point_process_parameters": asdict(config),
    }
    return ts_data, events, metadata

