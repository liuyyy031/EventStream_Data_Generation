"""Node-local arrival semantics and schedule-aware causal QA for STPP v14."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from stpp_adapter_v12 import (
    _range_endpoints,
    audit_temporal_contract_v12,
    canonicalise_temporal_contract_v12,
)
from stpp_adapter_v13 import (
    STPPV13Config,
    canonicalise_spatial_layout_v13,
    simulate_stpp_for_streasoner_v13,
)


@dataclass(frozen=True)
class STPPV14Config(STPPV13Config):
    """v14 retains v13 numerical parameters and changes semantic validation."""


def _ranges(value: Any) -> List[Tuple[int, int]]:
    if isinstance(value, (int, float)):
        point = int(value)
        return [(point, point)]
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            return [(int(value[0]), int(value[1]))]
        output: List[Tuple[int, int]] = []
        for item in value:
            output.extend(_ranges(item))
        return output
    endpoints = _range_endpoints(value)
    if endpoints is not None:
        return [endpoints]
    numbers = re.findall(r"(?<!\d)\d+(?!\d)", str(value or ""))
    if len(numbers) == 1:
        point = int(numbers[0])
        return [(point, point)]
    return []


def _shifted_ranges(
    start: int,
    end: int,
    lag: float,
    mode: str,
    period: int | None,
) -> List[Tuple[int, int]]:
    if not float(lag).is_integer():
        return []
    shift = int(lag)
    shifted_start = start + shift
    shifted_end = end + shift
    if mode != "cyclic_local" or period is None:
        return [(shifted_start, shifted_end)]
    start_cycle = shifted_start // period
    end_cycle = shifted_end // period
    local_start = shifted_start % period
    local_end = shifted_end % period
    if start_cycle == end_cycle:
        return [(local_start, local_end)]
    output = [(local_start, period - 1), (0, local_end)]
    return sorted(output)


def _edge_map(structured_scenario: Mapping[str, Any]) -> Dict[Tuple[int, int], float]:
    return {
        (int(edge["source"]), int(edge["target"])): float(
            edge.get("time_lag", 1)
        )
        for edge in structured_scenario.get("edges", []) or []
    }


def audit_temporal_contract_v14(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Dict[str, Any]:
    """Replace v12's full-path/day-mixing checks with v14 semantics."""
    base = audit_temporal_contract_v12(structured_scenario, seq_len)
    blocking = []
    for issue in base.get("blocking_issues", []):
        field = str(issue.get("field", ""))
        if "propagated_variations" in field:
            continue
        if field.startswith("adjacency_modulation.event_id="):
            continue
        blocking.append(issue)

    mode = str(base.get("mode") or "")
    period = base.get("repeat_period")
    edges = _edge_map(structured_scenario)
    nodes = structured_scenario.get("nodes", []) or []
    node_types = {
        int(node["id"]): str(node.get("type", "")).lower() for node in nodes
    }
    outgoing = {node_id: 0 for node_id in node_types}
    for source, _ in edges:
        outgoing[source] = outgoing.get(source, 0) + 1
    demand_without_outgoing = sorted(
        node_id
        for node_id, node_type in node_types.items()
        if node_type == "demand_source" and outgoing.get(node_id, 0) == 0
    )
    for node_id in demand_without_outgoing:
        blocking.append(
            {
                "field": f"nodes[{node_id}]",
                "problem": "demand_source has no outgoing edge",
            }
        )

    drift = structured_scenario.get("drift_patterns", {}) or {}
    origin_peaks: Dict[int, int] = {}
    ambient_origin_violations: List[Dict[str, Any]] = []
    for node_data in drift.get("nodes", []) or []:
        node_id = int(node_data.get("id", -1))
        for pattern_index, pattern in enumerate(node_data.get("patterns", []) or []):
            amplitude = float(pattern.get("amplitude", 0) or 0)
            peak = pattern.get("peak")
            if amplitude > 0 and peak is not None:
                try:
                    origin_peaks[node_id] = int(peak)
                except (TypeError, ValueError):
                    pass
            if (
                node_types.get(node_id) == "propagation"
                and str(pattern.get("origin", "")).lower() == "self_generated"
            ):
                violation = {
                    "node_id": node_id,
                    "pattern_index": pattern_index,
                    "origin": pattern.get("origin"),
                }
                ambient_origin_violations.append(violation)
                blocking.append(
                    {
                        "field": (
                            f"drift_patterns.nodes[{node_id}]."
                            f"patterns[{pattern_index}].origin"
                        ),
                        "problem": (
                            "propagation baseline must be ambient_baseline, "
                            "not self_generated"
                        ),
                    }
                )

    variation_violations: List[Dict[str, Any]] = []
    for node_data in drift.get("nodes", []) or []:
        receiving_node = int(node_data.get("id", -1))
        for variation_index, variation in enumerate(
            node_data.get("propagated_variations", []) or []
        ):
            prefix = (
                f"drift_patterns.nodes[{receiving_node}]."
                f"propagated_variations[{variation_index}]"
            )
            arrival_path = variation.get("arrival_path")
            full_path = variation.get("event_full_path")
            cumulative = variation.get("cumulative_lag_to_node")
            source = variation.get("source")
            problems: List[str] = []
            try:
                arrival = [int(item) for item in arrival_path]
                full = [int(item) for item in full_path]
                source_id = int(source)
                declared_lag = float(cumulative)
            except (TypeError, ValueError):
                problems.append("missing or invalid node-local path fields")
                arrival = []
                full = []
                source_id = -1
                declared_lag = -1.0
            if arrival:
                if len(arrival) < 2:
                    problems.append("arrival_path must contain at least one edge")
                if arrival[0] != source_id:
                    problems.append("arrival_path does not start at source")
                if arrival[-1] != receiving_node:
                    problems.append("arrival_path does not end at receiving node")
                if full[: len(arrival)] != arrival:
                    problems.append("arrival_path is not a prefix of event_full_path")
                try:
                    calculated_lag = sum(
                        edges[(arrival[pos], arrival[pos + 1])]
                        for pos in range(len(arrival) - 1)
                    )
                    for pos in range(len(full) - 1):
                        edges[(full[pos], full[pos + 1])]
                except KeyError:
                    problems.append("path contains an undeclared directed edge")
                else:
                    if abs(calculated_lag - declared_lag) > 1e-9:
                        problems.append(
                            "cumulative_lag_to_node does not equal arrival_path lag sum"
                        )
                    peak = origin_peaks.get(source_id)
                    arrival_ranges = _ranges(variation.get("time"))
                    if peak is not None and arrival_ranges:
                        expected = peak + int(round(calculated_lag))
                        if mode == "cyclic_local" and period:
                            expected %= int(period)
                        if not any(start <= expected <= end for start, end in arrival_ranges):
                            problems.append(
                                "arrival time does not equal source peak plus cumulative lag"
                            )
            if problems:
                violation = {"field": prefix, "problems": problems}
                variation_violations.append(violation)
                for problem in problems:
                    blocking.append({"field": prefix, "problem": problem})

    schedule_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    schedule_days: Dict[Tuple[str, str], Tuple[int, ...]] = {}
    schedule_metadata_violations: List[Dict[str, Any]] = []
    scheduled_demand_origins: set[int] = set()
    patterns = (
        structured_scenario.get("adjacency_modulation", {}).get("patterns", [])
        or []
    )
    for index, pattern in enumerate(patterns):
        prefix = f"adjacency_modulation.patterns[{index}]"
        event_id = str(pattern.get("event_id", "")).strip()
        schedule_id = str(pattern.get("schedule_id", "")).strip()
        path = pattern.get("path")
        stage = pattern.get("path_stage")
        days_raw = pattern.get("days")
        endpoints = _range_endpoints(pattern.get("time_period"))
        problems: List[str] = []
        if not event_id or not schedule_id:
            problems.append("event_id and schedule_id are required")
        try:
            path_nodes = [int(item) for item in path]
            stage_int = int(stage)
            days = tuple(sorted(set(int(item) for item in days_raw)))
            source = path_nodes[stage_int]
            target = path_nodes[stage_int + 1]
        except (IndexError, TypeError, ValueError):
            problems.append("invalid path, path_stage, or days")
            path_nodes = []
            stage_int = -1
            days = ()
            source = target = -1
        if endpoints is None:
            problems.append("invalid edge activation window")
            start = end = -1
        else:
            start, end = endpoints
        if path_nodes:
            edge_key = (source, target)
            if edge_key not in edges:
                problems.append("path_stage identifies an undeclared edge")
            if str(pattern.get("applies_to")) != f"{source}->{target}":
                problems.append("applies_to does not match path_stage")
            expected_arrival = _shifted_ranges(
                start, end, edges.get(edge_key, 0), mode, period
            )
            actual_arrival = sorted(_ranges(pattern.get("destination_arrival_period")))
            if actual_arrival != expected_arrival:
                problems.append(
                    "destination_arrival_period is not activation window plus edge lag"
                )
        if event_id and schedule_id:
            key = (event_id, schedule_id)
            previous_days = schedule_days.setdefault(key, days)
            if previous_days != days:
                problems.append("one schedule_id is reused across different days")
            schedule_groups.setdefault(key, []).append(
                {
                    "stage": stage_int,
                    "start": start,
                    "end": end,
                    "edge": (source, target),
                    "path": path_nodes,
                    "days": days,
                    "field": prefix,
                }
            )
            if stage_int == 0 and path_nodes:
                scheduled_demand_origins.add(path_nodes[0])
        if problems:
            violation = {"field": prefix, "problems": problems}
            schedule_metadata_violations.append(violation)
            for problem in problems:
                blocking.append({"field": prefix, "problem": problem})

    causal_violations: List[Dict[str, Any]] = []
    for (event_id, schedule_id), records in schedule_groups.items():
        ordered = sorted(records, key=lambda item: item["stage"])
        if not ordered:
            continue
        expected_stages = list(range(max(len(ordered[0]["path"]) - 1, 0)))
        actual_stages = [record["stage"] for record in ordered]
        if actual_stages != expected_stages:
            blocking.append(
                {
                    "field": f"schedule={schedule_id}",
                    "problem": "schedule does not contain every path stage exactly once",
                }
            )
        for previous, current in zip(ordered, ordered[1:]):
            lag = edges.get(previous["edge"], 0.0)
            earliest = float(previous["start"]) + lag
            comparable = float(current["start"])
            if mode == "cyclic_local" and period and comparable < previous["start"]:
                comparable += int(period)
            if comparable + 1e-9 < earliest:
                violation = {
                    "event_id": event_id,
                    "schedule_id": schedule_id,
                    "previous_edge": f"{previous['edge'][0]}->{previous['edge'][1]}",
                    "current_edge": f"{current['edge'][0]}->{current['edge'][1]}",
                    "current_start": current["start"],
                    "earliest_legal_start": earliest,
                }
                causal_violations.append(violation)
                blocking.append(
                    {
                        "field": f"schedule={schedule_id}",
                        "problem": "downstream edge activates before causal arrival",
                    }
                )

    demand_without_schedules = sorted(
        node_id
        for node_id, node_type in node_types.items()
        if node_type == "demand_source" and node_id not in scheduled_demand_origins
    )
    for node_id in demand_without_schedules:
        blocking.append(
            {
                "field": f"nodes[{node_id}]",
                "problem": "demand_source has no stage-0 propagation schedule",
            }
        )

    base.update(
        {
            "blocking_issues": blocking,
            "causal_window_violations": causal_violations,
            "demand_sources_without_outgoing_edges": demand_without_outgoing,
            "demand_sources_without_event_schedules": demand_without_schedules,
            "propagated_variation_violations": variation_violations,
            "schedule_metadata_violations": schedule_metadata_violations,
            "ambient_baseline_origin_violations": ambient_origin_violations,
            "passed": not blocking,
            "audit_semantics": (
                "node-local arrival_path + (event_id,schedule_id) causal grouping"
            ),
        }
    )
    return base


def canonicalise_temporal_contract_v14(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    canonical, base_audit = canonicalise_temporal_contract_v12(
        structured_scenario, seq_len
    )
    audit = audit_temporal_contract_v14(canonical, seq_len)
    audit["changes"] = base_audit.get("changes", [])
    audit["num_changes"] = base_audit.get("num_changes", 0)
    audit["repair_free"] = base_audit.get("repair_free", True)
    return canonical, audit


def simulate_stpp_for_streasoner_v14(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV14Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV14Config()
    config.validate()
    temporal_canonical, temporal_audit = canonicalise_temporal_contract_v14(
        structured_scenario, seq_len
    )
    canonical, spatial_audit = canonicalise_spatial_layout_v13(
        temporal_canonical
    )
    ts_data, events, metadata = simulate_stpp_for_streasoner_v13(
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
    quality["checks"]["all_demand_sources_have_outgoing_edges"] = not (
        temporal_audit["demand_sources_without_outgoing_edges"]
    )
    quality["checks"]["all_demand_sources_have_event_schedules"] = not (
        temporal_audit["demand_sources_without_event_schedules"]
    )
    quality["checks"]["propagated_variation_contract_valid"] = not (
        temporal_audit["propagated_variation_violations"]
    )
    quality["checks"]["schedule_metadata_valid"] = not temporal_audit[
        "schedule_metadata_violations"
    ]
    quality["checks"]["propagation_baselines_are_ambient"] = not temporal_audit[
        "ambient_baseline_origin_violations"
    ]
    quality["checks"]["spatial_layout_has_separated_nodes"] = spatial_audit[
        "final_valid"
    ]
    quality["checks"]["spatial_layout_repair_free"] = not spatial_audit[
        "repaired"
    ]
    quality["passed"] = all(quality["checks"].values())
    quality["temporal_contract_audit"] = temporal_audit
    quality["event_semantic_audit"] = temporal_audit
    metadata.update(
        {
            "simulator": "STPPG + node-local schedule-aware STPP v14",
            "method": (
                "v13 spatial STPP + node-local arrival paths + schedule-aware "
                "causal validation"
            ),
            "temporal_contract_audit": temporal_audit,
            "event_semantic_audit": temporal_audit,
            "spatial_layout_audit": spatial_audit,
        }
    )
    return ts_data, events, metadata
