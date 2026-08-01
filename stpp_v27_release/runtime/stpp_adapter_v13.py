"""Spatial-support safeguards and a valid Hawkes bound layered on STPP v12."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from stpp_adapter_v12 import (
    STPPV12Config,
    simulate_stpp_for_streasoner_v12,
)


@dataclass(frozen=True)
class STPPV13Config(STPPV12Config):
    # v12 server evidence observed Hawkes intensity 76.13 with a bound of 20.
    intensity_upper_bound: float = 128.0


def _normalised_distances(points: np.ndarray) -> Tuple[np.ndarray, float | None]:
    normalised = np.asarray(points, dtype=float).copy()
    for axis in (0, 1):
        low = float(normalised[:, axis].min())
        high = float(normalised[:, axis].max())
        if high == low:
            normalised[:, axis] = 0.5
        else:
            normalised[:, axis] = (
                0.05 + 0.90 * (normalised[:, axis] - low) / (high - low)
            )
    if len(normalised) < 2:
        return normalised, None
    minimum = min(
        float(np.linalg.norm(normalised[left] - normalised[right]))
        for left in range(len(normalised))
        for right in range(left + 1, len(normalised))
    )
    return normalised, minimum


def _circle_layout(node_ids: List[int]) -> Dict[str, Dict[str, float]]:
    if len(node_ids) == 1:
        return {str(node_ids[0]): {"x": 0.5, "y": 0.5}}
    output: Dict[str, Dict[str, float]] = {}
    for index, node_id in enumerate(node_ids):
        angle = 2.0 * math.pi * index / len(node_ids)
        output[str(node_id)] = {
            "x": float(0.5 + 0.4 * math.cos(angle)),
            "y": float(0.5 + 0.4 * math.sin(angle)),
        }
    return output


def canonicalise_spatial_layout_v13(
    structured_scenario: Mapping[str, Any],
    minimum_separation: float = 0.05,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Replace missing/duplicate layouts with a deterministic separated circle."""
    canonical = copy.deepcopy(dict(structured_scenario))
    node_ids = [int(node["id"]) for node in canonical.get("nodes", []) or []]
    layout = canonical.get("spatial_layout", {}) or {}
    issues: List[Dict[str, Any]] = []
    raw_points: List[List[float]] = []
    for node_id in node_ids:
        item = layout.get(str(node_id), layout.get(node_id))
        if not isinstance(item, Mapping) or "x" not in item or "y" not in item:
            issues.append(
                {
                    "node_id": node_id,
                    "problem": "missing x/y coordinate",
                }
            )
            continue
        try:
            x = float(item["x"])
            y = float(item["y"])
        except (TypeError, ValueError):
            issues.append(
                {"node_id": node_id, "problem": "non-numeric x/y coordinate"}
            )
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            issues.append(
                {"node_id": node_id, "problem": "non-finite x/y coordinate"}
            )
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            issues.append(
                {"node_id": node_id, "problem": "coordinate outside [0,1]"}
            )
        raw_points.append([x, y])

    original_minimum: float | None = None
    if len(raw_points) == len(node_ids) and raw_points:
        _, original_minimum = _normalised_distances(np.asarray(raw_points))
        if original_minimum is not None and original_minimum < minimum_separation:
            issues.append(
                {
                    "problem": "duplicate or near-duplicate normalized coordinates",
                    "minimum_separation": original_minimum,
                    "required_separation": minimum_separation,
                }
            )

    repaired = bool(issues)
    if repaired:
        canonical["spatial_layout"] = _circle_layout(node_ids)
    final_points = np.asarray(
        [
            [
                canonical["spatial_layout"][str(node_id)]["x"],
                canonical["spatial_layout"][str(node_id)]["y"],
            ]
            for node_id in node_ids
        ],
        dtype=float,
    )
    _, final_minimum = _normalised_distances(final_points)
    final_valid = final_minimum is None or final_minimum >= minimum_separation
    return canonical, {
        "node_ids": node_ids,
        "minimum_required_separation": minimum_separation,
        "original_minimum_normalized_separation": original_minimum,
        "final_minimum_normalized_separation": final_minimum,
        "input_issues": issues,
        "repaired": repaired,
        "repair_method": "deterministic_circle" if repaired else None,
        "final_valid": final_valid,
    }


def simulate_stpp_for_streasoner_v13(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV13Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV13Config()
    config.validate()
    canonical, spatial_audit = canonicalise_spatial_layout_v13(
        structured_scenario
    )
    ts_data, events, metadata = simulate_stpp_for_streasoner_v12(
        canonical, seq_len, config
    )
    quality = metadata["quality_report"]
    quality["checks"]["spatial_layout_has_separated_nodes"] = spatial_audit[
        "final_valid"
    ]
    quality["checks"]["spatial_layout_repair_free"] = not spatial_audit[
        "repaired"
    ]
    quality["passed"] = all(quality["checks"].values())
    quality["spatial_layout_audit"] = spatial_audit
    metadata.update(
        {
            "simulator": "STPPG + spatially grounded temporal-contract STPP v13",
            "method": (
                "v12 temporal-contract STPP + separated nearest-node support + "
                "validated Hawkes intensity bound"
            ),
            "spatial_layout_audit": spatial_audit,
        }
    )
    return ts_data, events, metadata
