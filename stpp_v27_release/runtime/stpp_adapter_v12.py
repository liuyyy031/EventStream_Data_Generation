"""Explicit time-axis contracts and temporal QA layered on STPP v11."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from stpp_adapter_v11 import STPPV11Config, simulate_stpp_for_streasoner_v11


@dataclass(frozen=True)
class STPPV12Config(STPPV11Config):
    """v12 changes semantics and QA while retaining v11 numerical controls."""


def _range_endpoints(value: Any) -> Tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    match = re.fullmatch(r"\s*(\d+)\s*[-–—]\s*(\d+)\s*", str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _normalise_days(value: Any) -> List[int] | None:
    if not isinstance(value, list):
        return None
    try:
        days = sorted(set(int(item) for item in value))
    except (TypeError, ValueError):
        return None
    return days


def canonicalise_temporal_contract_v12(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split wrapping windows deterministically and report every repair."""
    canonical = copy.deepcopy(dict(structured_scenario))
    changes: List[Dict[str, Any]] = []
    adjacency = canonical.setdefault("adjacency_modulation", {})
    patterns = adjacency.get("patterns", []) or []
    normalised_patterns: List[Dict[str, Any]] = []
    for index, raw_pattern in enumerate(patterns):
        pattern = copy.deepcopy(raw_pattern)
        endpoints = _range_endpoints(
            pattern.get("time_period", pattern.get("time_range"))
        )
        if endpoints is None or endpoints[0] <= endpoints[1]:
            normalised_patterns.append(pattern)
            continue
        start, end = endpoints
        axis = canonical.get("time_axis", {}) or {}
        period = axis.get("repeat_period")
        if period is None:
            period = (canonical.get("drift_patterns", {}) or {}).get(
                "repeat_period"
            )
        try:
            period = int(period)
        except (TypeError, ValueError):
            period = 24
        first = copy.deepcopy(pattern)
        second = copy.deepcopy(pattern)
        first["time_period"] = f"{start}-{period - 1}"
        second["time_period"] = f"0-{end}"
        first["crosses_midnight"] = False
        second["crosses_midnight"] = False
        first["split_segment"] = "end_of_cycle"
        second["split_segment"] = "start_of_cycle"
        normalised_patterns.extend([first, second])
        changes.append(
            {
                "field": f"adjacency_modulation.patterns[{index}].time_period",
                "before": pattern.get("time_period", pattern.get("time_range")),
                "after": [first["time_period"], second["time_period"]],
                "reason": "split wrapping inclusive range at repeat boundary",
            }
        )
    adjacency["patterns"] = normalised_patterns
    audit = audit_temporal_contract_v12(canonical, seq_len)
    audit["changes"] = changes
    audit["num_changes"] = len(changes)
    audit["repair_free"] = not changes
    return canonical, audit


def audit_temporal_contract_v12(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Dict[str, Any]:
    """Reject ambiguous coordinates, lost day filters, and invalid schedules."""
    blocking: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    axis = structured_scenario.get("time_axis", {}) or {}
    mode = str(axis.get("mode", "")).lower()
    drift = structured_scenario.get("drift_patterns", {}) or {}
    repeat = drift.get("repeat")
    period_raw = axis.get("repeat_period", drift.get("repeat_period"))
    try:
        period = int(period_raw) if period_raw is not None else None
    except (TypeError, ValueError):
        period = None

    if mode not in {"cyclic_local", "absolute"}:
        blocking.append(
            {"field": "time_axis.mode", "problem": "missing or invalid mode"}
        )
    if str(axis.get("interval_semantics", "")).lower() != "inclusive":
        blocking.append(
            {
                "field": "time_axis.interval_semantics",
                "problem": "v12 requires explicit inclusive intervals",
            }
        )
    if mode == "cyclic_local":
        if repeat is not True:
            blocking.append(
                {
                    "field": "drift_patterns.repeat",
                    "problem": "cyclic_local requires repeat=true",
                }
            )
        if period is None or period < 1 or period > seq_len:
            blocking.append(
                {
                    "field": "time_axis.repeat_period",
                    "problem": "invalid cyclic repeat period",
                }
            )
        if str(axis.get("week_start", "")).lower() != "sunday":
            blocking.append(
                {
                    "field": "time_axis.week_start",
                    "problem": "v12 code contract uses Sunday as cycle day 0",
                }
            )
        expected_day_index = {
            "Sunday": 0,
            "Monday": 1,
            "Tuesday": 2,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 5,
            "Saturday": 6,
        }
        if axis.get("day_index") != expected_day_index:
            blocking.append(
                {
                    "field": "time_axis.day_index",
                    "problem": "day index mapping is missing or inconsistent",
                }
            )
    if mode == "absolute" and repeat is not False:
        blocking.append(
            {
                "field": "drift_patterns.repeat",
                "problem": "absolute mode requires repeat=false",
            }
        )

    allowed_effects = {"strong", "moderate", "weak"}
    event_stages: Dict[str, List[Tuple[int, int, int, str]]] = {}
    edges = {
        (int(edge["source"]), int(edge["target"])): float(
            edge.get("time_lag", 1)
        )
        for edge in structured_scenario.get("edges", []) or []
    }
    for node_index, node_data in enumerate(drift.get("nodes", []) or []):
        for pattern_index, pattern in enumerate(node_data.get("patterns", []) or []):
            prefix = (
                f"drift_patterns.nodes[{node_index}].patterns[{pattern_index}]"
            )
            endpoints = _range_endpoints(pattern.get("time_range"))
            if endpoints is None:
                blocking.append(
                    {"field": f"{prefix}.time_range", "problem": "invalid range"}
                )
                continue
            start, end = endpoints
            upper = period if mode == "cyclic_local" else seq_len
            if start > end or start < 0 or upper is None or end >= upper:
                blocking.append(
                    {
                        "field": f"{prefix}.time_range",
                        "problem": "node pattern range is outside its time axis",
                    }
                )
            peak = pattern.get("peak")
            amplitude = float(pattern.get("amplitude", 0) or 0)
            if peak is not None and amplitude > 0:
                try:
                    peak_value = int(peak)
                except (TypeError, ValueError):
                    blocking.append(
                        {"field": f"{prefix}.peak", "problem": "peak is not integer"}
                    )
                else:
                    if not start <= peak_value <= end:
                        blocking.append(
                            {
                                "field": f"{prefix}.peak",
                                "problem": "positive peak lies outside pattern range",
                            }
                        )
        for variation_index, variation in enumerate(
            node_data.get("propagated_variations", []) or []
        ):
            prefix = (
                f"drift_patterns.nodes[{node_index}]."
                f"propagated_variations[{variation_index}]"
            )
            path = variation.get("path")
            cumulative = variation.get("cumulative_lag")
            if not isinstance(path, list) or len(path) < 2 or cumulative is None:
                blocking.append(
                    {
                        "field": prefix,
                        "problem": "propagated variation lacks path/cumulative_lag",
                    }
                )
                continue
            try:
                path_nodes = [int(item) for item in path]
                declared_cumulative = float(cumulative)
                calculated_cumulative = sum(
                    edges[(path_nodes[pos], path_nodes[pos + 1])]
                    for pos in range(len(path_nodes) - 1)
                )
            except (KeyError, TypeError, ValueError):
                blocking.append(
                    {
                        "field": f"{prefix}.path",
                        "problem": "path contains an undeclared directed edge",
                    }
                )
            else:
                if abs(declared_cumulative - calculated_cumulative) > 1e-9:
                    blocking.append(
                        {
                            "field": f"{prefix}.cumulative_lag",
                            "problem": (
                                "declared cumulative lag does not equal edge-lag sum"
                            ),
                        }
                    )
    patterns = (
        structured_scenario.get("adjacency_modulation", {}).get("patterns", [])
        or []
    )
    for index, pattern in enumerate(patterns):
        prefix = f"adjacency_modulation.patterns[{index}]"
        endpoints = _range_endpoints(
            pattern.get("time_period", pattern.get("time_range"))
        )
        if endpoints is None:
            blocking.append(
                {"field": f"{prefix}.time_period", "problem": "unparseable range"}
            )
            continue
        start, end = endpoints
        if start > end:
            blocking.append(
                {
                    "field": f"{prefix}.time_period",
                    "problem": "wrapping range must be split",
                }
            )
        if mode == "cyclic_local" and period is not None:
            if start < 0 or end >= period:
                blocking.append(
                    {
                        "field": f"{prefix}.time_period",
                        "problem": "cyclic range is outside repeat period",
                    }
                )
            days = _normalise_days(pattern.get("days"))
            if days is None or not days or any(day < 0 or day > 6 for day in days):
                blocking.append(
                    {
                        "field": f"{prefix}.days",
                        "problem": "cyclic modulation requires explicit days 0..6",
                    }
                )
        if mode == "absolute" and (start < 0 or end >= seq_len):
            blocking.append(
                {
                    "field": f"{prefix}.time_period",
                    "problem": "absolute range is outside sequence",
                }
            )
        effect = str(pattern.get("effect", "")).lower()
        if effect not in allowed_effects:
            blocking.append(
                {
                    "field": f"{prefix}.effect",
                    "problem": "effect must be strong, moderate, or weak",
                }
            )
        description = str(pattern.get("description", "")).lower()
        if "overnight" in description and "weekend" in description:
            blocking.append(
                {
                    "field": prefix,
                    "problem": (
                        "daily overnight and weekend-all-day semantics must be "
                        "separate patterns"
                    ),
                }
            )
        event_id = str(pattern.get("event_id", "")).strip()
        path = pattern.get("path")
        stage = pattern.get("path_stage")
        if event_id and isinstance(path, list) and stage is not None:
            try:
                stage_int = int(stage)
                source = int(path[stage_int])
                target = int(path[stage_int + 1])
                applies_to = pattern.get("applies_to")
                applies_values = (
                    [str(item) for item in applies_to]
                    if isinstance(applies_to, list)
                    else [str(applies_to)]
                )
                expected_edge = f"{source}->{target}"
                if expected_edge not in applies_values:
                    blocking.append(
                        {
                            "field": f"{prefix}.applies_to",
                            "problem": "applies_to does not match path stage edge",
                        }
                    )
                event_stages.setdefault(event_id, []).append(
                    (stage_int, start, end, expected_edge)
                )
            except (IndexError, TypeError, ValueError):
                blocking.append(
                    {
                        "field": f"{prefix}.path_stage",
                        "problem": "stage does not identify an edge in path",
                    }
                )
        elif effect in {"strong", "moderate"}:
            blocking.append(
                {
                    "field": prefix,
                    "problem": (
                        "strong/moderate modulation requires event_id, path, "
                        "and path_stage"
                    ),
                }
            )

    causal_violations: List[Dict[str, Any]] = []
    for event_id, stages in event_stages.items():
        ordered = sorted(stages)
        for previous, current in zip(ordered, ordered[1:]):
            previous_stage, previous_start, _, previous_edge = previous
            current_stage, current_start, _, current_edge = current
            if current_stage != previous_stage + 1:
                continue
            source, target = [int(item) for item in previous_edge.split("->")]
            earliest = previous_start + edges.get((source, target), 1.0)
            comparable_current = float(current_start)
            if (
                mode == "cyclic_local"
                and period is not None
                and comparable_current < previous_start
            ):
                comparable_current += period
            if comparable_current + 1e-9 < earliest:
                violation = {
                    "event_id": event_id,
                    "previous_edge": previous_edge,
                    "current_edge": current_edge,
                    "current_start": current_start,
                    "current_start_unwrapped": comparable_current,
                    "earliest_legal_start": earliest,
                }
                causal_violations.append(violation)
                blocking.append(
                    {
                        "field": f"adjacency_modulation.event_id={event_id}",
                        "problem": "downstream window starts before causal arrival",
                    }
                )

    return {
        "mode": mode or None,
        "repeat": repeat,
        "repeat_period": period,
        "interval_semantics": axis.get("interval_semantics"),
        "week_start": axis.get("week_start"),
        "blocking_issues": blocking,
        "warnings": warnings,
        "causal_window_violations": causal_violations,
        "passed": not blocking,
    }


def simulate_stpp_for_streasoner_v12(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV12Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV12Config()
    config.validate()
    canonical, temporal_audit = canonicalise_temporal_contract_v12(
        structured_scenario, seq_len
    )
    ts_data, events, metadata = simulate_stpp_for_streasoner_v11(
        canonical, seq_len, config
    )
    quality = metadata["quality_report"]
    quality["checks"]["temporal_contract_valid"] = temporal_audit["passed"]
    quality["checks"]["temporal_contract_repair_free"] = temporal_audit[
        "repair_free"
    ]
    quality["checks"]["no_causal_window_violations"] = not temporal_audit[
        "causal_window_violations"
    ]
    quality["passed"] = all(quality["checks"].values())
    quality["temporal_contract_audit"] = temporal_audit
    metadata.update(
        {
            "simulator": "STPPG + explicit temporal-contract STPP v12",
            "method": (
                "v11 conditional simple-path STPP + explicit cyclic/absolute "
                "time-axis validation"
            ),
            "temporal_contract_audit": temporal_audit,
        }
    )
    return ts_data, events, metadata
