"""Root/propagation semantic separation for graph-conditioned STPP v6."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from stpp_adapter_v5 import STPPV5Config, simulate_stpp_for_streasoner_v5


@dataclass(frozen=True)
class STPPV6Config(STPPV5Config):
    # A pure relay receives events through graph edges; it is not an exogenous
    # point-process source.
    propagation_root_weight: float = 0.0
    propagated_peak_match_tolerance: int = 1

    def validate(self) -> None:
        super().validate()
        if self.propagated_peak_match_tolerance < 0:
            raise ValueError("propagated_peak_match_tolerance cannot be negative")


def _mentioned_time_steps(value: Any) -> List[int]:
    text = str(value or "")
    return [
        int(match)
        for match in re.findall(
            r"(?:(?:step|time)\s*)?(\d+)(?=\s*(?:-|to|,|$|\)))",
            text,
            flags=re.IGNORECASE,
        )
    ]


def _root_conditioned_scenario(
    structured_scenario: Mapping[str, Any], config: STPPV6Config
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Remove derived propagation peaks from the exogenous-root profile."""
    conditioned = copy.deepcopy(dict(structured_scenario))
    node_types = {
        int(node["id"]): str(node.get("type", "demand_source")).lower()
        for node in conditioned.get("nodes", [])
    }
    audit: Dict[str, Any] = {
        "excluded_patterns": [],
        "retained_positive_amplitude_patterns": [],
        "propagated_steps_by_node": {},
        "remaining_disallowed_patterns": [],
    }
    drift = conditioned.get("drift_patterns", {}) or {}
    for node_data in drift.get("nodes", []):
        node_id = int(node_data.get("id", -1))
        node_type = node_types.get(node_id, "demand_source")
        propagated_steps: List[int] = []
        for variation in node_data.get("propagated_variations", []) or []:
            propagated_steps.extend(_mentioned_time_steps(variation.get("time")))
        propagated_steps = sorted(set(propagated_steps))
        audit["propagated_steps_by_node"][str(node_id)] = propagated_steps

        retained = []
        for pattern in node_data.get("patterns", []) or []:
            peak_value = pattern.get("peak")
            amplitude = float(pattern.get("amplitude", 0) or 0)
            positive_peak = peak_value is not None and amplitude > 0
            reason = None
            if positive_peak and node_type == "propagation":
                reason = "pure_propagation_node"
            elif positive_peak and propagated_steps:
                peak = int(peak_value)
                if any(
                    abs(peak - step) <= config.propagated_peak_match_tolerance
                    for step in propagated_steps
                ):
                    reason = "matches_propagated_variation_time"

            record = {
                "node_id": node_id,
                "node_type": node_type,
                "peak": peak_value,
                "amplitude": amplitude,
                "time_range": pattern.get("time_range"),
            }
            if reason:
                record["reason"] = reason
                record["matched_propagated_steps"] = propagated_steps
                audit["excluded_patterns"].append(record)
            else:
                retained.append(pattern)
                if positive_peak:
                    audit["retained_positive_amplitude_patterns"].append(record)
                    peak = int(peak_value)
                    matches_propagated = any(
                        abs(peak - step)
                        <= config.propagated_peak_match_tolerance
                        for step in propagated_steps
                    )
                    if node_type == "propagation" or matches_propagated:
                        audit["remaining_disallowed_patterns"].append(record)
        node_data["patterns"] = retained

    audit["num_excluded_patterns"] = len(audit["excluded_patterns"])
    audit["num_retained_positive_amplitude_patterns"] = len(
        audit["retained_positive_amplitude_patterns"]
    )
    return conditioned, audit


def simulate_stpp_for_streasoner_v6(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV6Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV6Config()
    config.validate()
    conditioned_scenario, audit = _root_conditioned_scenario(
        structured_scenario, config
    )
    ts_data, events, metadata = simulate_stpp_for_streasoner_v5(
        conditioned_scenario, seq_len, config
    )

    propagation_nodes = {
        int(node["id"])
        for node in structured_scenario.get("nodes", [])
        if str(node.get("type", "")).lower() == "propagation"
    }
    propagation_root_events = [
        event
        for event in events
        if event.get("parent_event_id") is None
        and int(event["node_id"]) in propagation_nodes
    ]
    quality = metadata["quality_report"]
    quality["checks"]["no_roots_on_pure_propagation_nodes"] = not (
        propagation_root_events
    )
    quality["checks"]["propagated_patterns_not_used_as_root_peaks"] = not audit[
        "remaining_disallowed_patterns"
    ]
    quality["passed"] = all(quality["checks"].values())
    quality["propagation_node_root_event_count"] = len(propagation_root_events)
    quality["root_pattern_conditioning"] = audit

    metadata.update(
        {
            "simulator": "STPPG + root/propagation-separated graph STPP v6",
            "method": (
                "demand-source STPPG roots + graph-only relay propagation + "
                "inbound-aware magnitude QA"
            ),
            "integration_method": (
                "root semantic filtering + quota control + marked branching"
            ),
            "root_conditioned_structured_scenario": conditioned_scenario,
            "root_pattern_conditioning": audit,
        }
    )
    return ts_data, events, metadata
