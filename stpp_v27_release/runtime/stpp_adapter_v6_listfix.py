"""List-valued peak compatibility layer for the STPP v6 adapter."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from stpp_adapter_v6 import STPPV6Config, simulate_stpp_for_streasoner_v6


def _normalise_peak_lists(
    structured_scenario: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Expand each JSON peak list into scalar-peak pattern copies."""
    normalised = copy.deepcopy(dict(structured_scenario))
    audit: Dict[str, Any] = {
        "expanded_patterns": [],
        "invalid_peak_values": [],
    }
    drift = normalised.get("drift_patterns", {}) or {}
    for node_data in drift.get("nodes", []) or []:
        node_id = int(node_data.get("id", -1))
        expanded_patterns: List[Dict[str, Any]] = []
        for pattern_index, pattern in enumerate(node_data.get("patterns", []) or []):
            peak_value = pattern.get("peak")
            if not isinstance(peak_value, list):
                expanded_patterns.append(pattern)
                continue

            valid_peaks: List[int] = []
            for raw_peak in peak_value:
                try:
                    peak = int(raw_peak)
                except (TypeError, ValueError):
                    audit["invalid_peak_values"].append(
                        {
                            "node_id": node_id,
                            "pattern_index": pattern_index,
                            "value": raw_peak,
                        }
                    )
                    continue
                valid_peaks.append(peak)
                scalar_pattern = copy.deepcopy(pattern)
                scalar_pattern["peak"] = peak
                expanded_patterns.append(scalar_pattern)

            audit["expanded_patterns"].append(
                {
                    "node_id": node_id,
                    "pattern_index": pattern_index,
                    "original_peaks": peak_value,
                    "expanded_peaks": valid_peaks,
                }
            )
            if not valid_peaks:
                no_peak_pattern = copy.deepcopy(pattern)
                no_peak_pattern["peak"] = None
                no_peak_pattern["amplitude"] = 0
                expanded_patterns.append(no_peak_pattern)
        node_data["patterns"] = expanded_patterns

    audit["num_expanded_source_patterns"] = len(audit["expanded_patterns"])
    audit["num_invalid_peak_values"] = len(audit["invalid_peak_values"])
    return normalised, audit


def simulate_stpp_for_streasoner_v6_listfix(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV6Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    normalised_scenario, audit = _normalise_peak_lists(structured_scenario)
    ts_data, events, metadata = simulate_stpp_for_streasoner_v6(
        normalised_scenario, seq_len, config
    )
    quality = metadata["quality_report"]
    quality["checks"]["peak_lists_normalised"] = not audit[
        "invalid_peak_values"
    ]
    quality["passed"] = all(quality["checks"].values())
    quality["peak_list_normalisation"] = audit
    metadata["peak_list_normalisation"] = audit
    metadata["peak_normalised_structured_scenario"] = normalised_scenario
    return ts_data, events, metadata
