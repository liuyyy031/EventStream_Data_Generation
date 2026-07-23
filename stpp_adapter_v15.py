"""Declared-event-route and schedule-window enforcement for STPP v15."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import stpp_adapter_v9 as _v9_module
import stpp_adapter_v10 as _v10_module
from stpp_adapter_v14 import STPPV14Config, _ranges, simulate_stpp_for_streasoner_v14


@dataclass(frozen=True)
class STPPV15Config(STPPV14Config):
    """v15 keeps v14 numerics and enforces semantic routes before aggregation."""

    enforce_declared_event_routes: bool = True
    enforce_schedule_activation_windows: bool = True
    enforce_destination_arrival_windows: bool = True


def _contains(ranges: Sequence[Tuple[int, int]], value: int) -> bool:
    for start, end in ranges:
        if start <= end and start <= value <= end:
            return True
        if start > end and (value >= start or value <= end):
            return True
    return False


def _clock(structured_scenario: Mapping[str, Any]) -> Tuple[int, int]:
    time_axis = structured_scenario.get("time_axis", {}) or {}
    drift = structured_scenario.get("drift_patterns", {}) or {}
    period = int(
        time_axis.get("repeat_period") or drift.get("repeat_period") or 24
    )
    return max(period, 1), 7


def _day_and_local(time_index: int, period: int, week_size: int) -> Tuple[int, int]:
    return (int(time_index) // period) % week_size, int(time_index) % period


def _route_contracts(
    structured_scenario: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build one complete route contract for each event/schedule branch."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    patterns = (
        structured_scenario.get("adjacency_modulation", {}).get("patterns", [])
        or []
    )
    for index, raw in enumerate(patterns):
        pattern = dict(raw)
        event_id = str(pattern.get("event_id", "")).strip()
        schedule_id = str(pattern.get("schedule_id", "")).strip()
        if event_id and schedule_id:
            pattern["_pattern_index"] = index
            groups.setdefault((event_id, schedule_id), []).append(pattern)

    contracts: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    for (event_id, schedule_id), records in sorted(groups.items()):
        paths = {
            tuple(int(node) for node in record.get("path", []) or [])
            for record in records
        }
        day_sets = {
            tuple(sorted(set(int(day) for day in record.get("days", []) or [])))
            for record in records
        }
        if len(paths) != 1 or len(day_sets) != 1:
            issues.append(
                {
                    "event_id": event_id,
                    "schedule_id": schedule_id,
                    "problem": "one schedule branch has inconsistent path or days",
                }
            )
            continue
        path = next(iter(paths))
        days = next(iter(day_sets))
        if len(path) < 2:
            issues.append(
                {
                    "event_id": event_id,
                    "schedule_id": schedule_id,
                    "problem": "event_full_path must contain at least one edge",
                }
            )
            continue
        stages: Dict[int, Dict[str, Any]] = {}
        duplicate_stages: List[int] = []
        for record in records:
            try:
                stage = int(record.get("path_stage"))
            except (TypeError, ValueError):
                stage = -1
            if stage in stages:
                duplicate_stages.append(stage)
            stages[stage] = {
                "time_ranges": _ranges(record.get("time_period")),
                "arrival_ranges": _ranges(
                    record.get("destination_arrival_period")
                ),
                "effect": str(record.get("effect", "")),
                "applies_to": str(record.get("applies_to", "")),
                "pattern_index": int(record["_pattern_index"]),
            }
        expected = set(range(len(path) - 1))
        if set(stages) != expected or duplicate_stages:
            issues.append(
                {
                    "event_id": event_id,
                    "schedule_id": schedule_id,
                    "problem": "schedule branch must contain each path stage once",
                    "expected_stages": sorted(expected),
                    "actual_stages": sorted(stages),
                    "duplicate_stages": sorted(set(duplicate_stages)),
                }
            )
            continue
        contracts.append(
            {
                "event_id": event_id,
                "schedule_id": schedule_id,
                "path": list(path),
                "days": list(days),
                "stages": stages,
            }
        )

    demand_sources = {
        int(node["id"])
        for node in structured_scenario.get("nodes", []) or []
        if str(node.get("type", "")).lower() == "demand_source"
    }
    for source in sorted(demand_sources):
        source_contracts = [
            contract for contract in contracts if int(contract["path"][0]) == source
        ]
        event_ids = {contract["event_id"] for contract in source_contracts}
        paths = {tuple(contract["path"]) for contract in source_contracts}
        day_counts = Counter(
            day for contract in source_contracts for day in contract["days"]
        )
        missing_days = [day for day in range(7) if day_counts.get(day, 0) == 0]
        overlapping_days = [
            day for day in range(7) if day_counts.get(day, 0) > 1
        ]
        if len(event_ids) != 1:
            issues.append(
                {
                    "source_node": source,
                    "problem": "demand source must have exactly one event_id family",
                    "event_ids": sorted(event_ids),
                }
            )
        if len(paths) != 1:
            issues.append(
                {
                    "source_node": source,
                    "problem": "all schedule branches must share one full path",
                    "paths": [list(path) for path in sorted(paths)],
                }
            )
        if missing_days or overlapping_days:
            issues.append(
                {
                    "source_node": source,
                    "problem": "schedule branches must partition days 0..6 exactly once",
                    "missing_days": missing_days,
                    "overlapping_days": overlapping_days,
                }
            )
    return contracts, issues


def _select_root_contract(
    root: Mapping[str, Any],
    contracts: Sequence[Dict[str, Any]],
    period: int,
    week_size: int,
) -> Tuple[Dict[str, Any] | None, bool]:
    node_id = int(root["node_id"])
    day, local = _day_and_local(int(root["time_index"]), period, week_size)
    candidates = [
        contract
        for contract in contracts
        if int(contract["path"][0]) == node_id and day in contract["days"]
    ]
    if len(candidates) == 1:
        return candidates[0], False
    matching_window = [
        contract
        for contract in candidates
        if _contains(contract["stages"][0]["time_ranges"], local)
    ]
    if len(matching_window) == 1:
        return matching_window[0], len(candidates) > 1
    if candidates:
        chosen = sorted(
            candidates, key=lambda item: (item["event_id"], item["schedule_id"])
        )[0]
        return chosen, True
    return None, False


def _annotate_root(
    event: Dict[str, Any], contract: Dict[str, Any] | None, ambiguous: bool,
    period: int, week_size: int,
) -> None:
    day, local = _day_and_local(int(event["time_index"]), period, week_size)
    event["semantic_root_event_id"] = int(event["event_id"])
    event["semantic_path_stage"] = 0
    event["scenario_event_id"] = contract["event_id"] if contract else None
    event["schedule_id"] = contract["schedule_id"] if contract else None
    event["declared_event_full_path"] = list(contract["path"]) if contract else []
    event["semantic_assignment_basis"] = "origin_node_and_day_branch"
    event["semantic_assignment_ambiguous"] = bool(ambiguous)
    event["origin_in_schedule_window"] = bool(
        contract
        and day in contract["days"]
        and _contains(contract["stages"][0]["time_ranges"], local)
    )


def _apply_route_contracts(
    events: List[Dict[str, Any]],
    structured_scenario: Mapping[str, Any],
    config: STPPV15Config,
    *,
    filter_invalid: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Annotate events and optionally remove undeclared or off-schedule offspring."""
    contracts, contract_issues = _route_contracts(structured_scenario)
    period, week_size = _clock(structured_scenario)
    ordered = sorted(
        events,
        key=lambda event: (
            int(event.get("generation", 0)),
            float(event["t"]),
            int(event["event_id"]),
        ),
    )
    by_id = {int(event["event_id"]): event for event in ordered}
    assignment: Dict[int, Dict[str, Any] | None] = {}
    kept_ids: set[int] = set()
    output: List[Dict[str, Any]] = []
    ambiguous_roots: List[int] = []
    unassigned_roots: List[int] = []
    blocked = Counter()
    violations: List[Dict[str, Any]] = []

    for event in ordered:
        event_id = int(event["event_id"])
        parent_id_raw = event.get("parent_event_id")
        if parent_id_raw is None:
            contract, ambiguous = _select_root_contract(
                event, contracts, period, week_size
            )
            assignment[event_id] = contract
            _annotate_root(event, contract, ambiguous, period, week_size)
            if contract is None:
                unassigned_roots.append(event_id)
            if ambiguous:
                ambiguous_roots.append(event_id)
            kept_ids.add(event_id)
            output.append(event)
            continue

        parent_id = int(parent_id_raw)
        parent = by_id.get(parent_id)
        contract = assignment.get(parent_id)
        assignment[event_id] = contract
        reasons: List[str] = []
        if parent is None or parent_id not in kept_ids:
            reasons.append("parent was removed by the declared-route filter")
        if contract is None:
            reasons.append("root has no unique event/schedule contract")

        lineage = [int(node) for node in event.get("lineage_nodes", []) or []]
        generation = int(event.get("generation", 0))
        stage = generation - 1
        if contract is not None:
            declared_path = [int(node) for node in contract["path"]]
            if not lineage or declared_path[: len(lineage)] != lineage:
                reasons.append("lineage is not a prefix of event_full_path")
            stage_contract = contract["stages"].get(stage)
            if stage_contract is None:
                reasons.append("propagation exceeds the declared full path")
            elif parent is not None:
                parent_day, parent_local = _day_and_local(
                    int(parent["time_index"]), period, week_size
                )
                child_day, child_local = _day_and_local(
                    int(event["time_index"]), period, week_size
                )
                activation_ok = bool(
                    parent_day in contract["days"]
                    and _contains(stage_contract["time_ranges"], parent_local)
                )
                arrival_ok = bool(
                    _contains(stage_contract["arrival_ranges"], child_local)
                )
                if config.enforce_schedule_activation_windows and not activation_ok:
                    reasons.append("parent is outside the stage activation window")
                if config.enforce_destination_arrival_windows and not arrival_ok:
                    reasons.append("child is outside destination_arrival_period")
                event["parent_schedule_day"] = parent_day
                event["child_schedule_day"] = child_day
                event["parent_local_time_index"] = parent_local
                event["child_local_time_index"] = child_local
                event["schedule_activation_match"] = activation_ok
                event["destination_arrival_match"] = arrival_ok

            event["semantic_root_event_id"] = int(
                parent.get("semantic_root_event_id", parent_id)
                if parent is not None else parent_id
            )
            event["scenario_event_id"] = contract["event_id"]
            event["schedule_id"] = contract["schedule_id"]
            event["declared_event_full_path"] = declared_path
            event["semantic_path_stage"] = generation
            event["semantic_assignment_basis"] = "inherited_from_root"
            event["semantic_assignment_ambiguous"] = bool(
                parent.get("semantic_assignment_ambiguous", False)
                if parent is not None else False
            )
        else:
            event["scenario_event_id"] = None
            event["schedule_id"] = None
            event["declared_event_full_path"] = []
            event["semantic_path_stage"] = generation

        if reasons:
            violations.append({"event_id": event_id, "reasons": reasons})
            for reason in reasons:
                blocked[reason] += 1
            if filter_invalid:
                continue
        kept_ids.add(event_id)
        output.append(event)

    actual_lineages = Counter(
        "->".join(str(node) for node in event.get("lineage_nodes", []) or [])
        for event in output
        if event.get("parent_event_id") is not None
    )
    contract_summary = [
        {
            "event_id": contract["event_id"],
            "schedule_id": contract["schedule_id"],
            "path": contract["path"],
            "days": contract["days"],
        }
        for contract in contracts
    ]
    audit = {
        "contracts": contract_summary,
        "contract_issues": contract_issues,
        "ambiguous_root_event_ids": ambiguous_roots,
        "unassigned_root_event_ids": unassigned_roots,
        "propagated_event_violations": violations,
        "blocked_candidate_counts": dict(sorted(blocked.items())),
        "num_input_events": len(events),
        "num_output_events": len(output),
        "num_filtered_events": len(events) - len(output),
        "actual_propagated_lineages": dict(sorted(actual_lineages.items())),
        "passed": not (
            contract_issues or ambiguous_roots or unassigned_roots or violations
        ),
        "filter_invalid": filter_invalid,
    }
    return sorted(
        output, key=lambda event: (float(event["t"]), int(event["event_id"]))
    ), audit


def _scalar_peak_audit(
    structured_scenario: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    period, _ = _clock(structured_scenario)
    declared: List[Dict[str, Any]] = []
    for node_data in (
        structured_scenario.get("drift_patterns", {}).get("nodes", []) or []
    ):
        node_id = int(node_data.get("id", -1))
        for pattern_index, pattern in enumerate(node_data.get("patterns", []) or []):
            peak = pattern.get("peak")
            amplitude = float(pattern.get("amplitude", 0) or 0)
            if (
                str(pattern.get("origin", "")).lower() == "self_generated"
                and amplitude > 0
                and isinstance(peak, (int, float))
            ):
                declared.append(
                    {
                        "node_id": node_id,
                        "pattern_index": pattern_index,
                        "peak": int(peak),
                    }
                )
    roots = [event for event in events if event.get("parent_event_id") is None]
    uncovered: List[Dict[str, Any]] = []
    for item in declared:
        covered = any(
            int(event["node_id"]) == item["node_id"]
            and int(event["time_index"]) % period == item["peak"] % period
            for event in roots
        )
        item["covered"] = covered
        if not covered:
            uncovered.append(dict(item))
    return {
        "declared_scalar_self_generated_peaks": declared,
        "num_declared_scalar_self_generated_peaks": len(declared),
        "uncovered_scalar_self_generated_peaks": uncovered,
        "passed": bool(declared) and not uncovered,
    }


def simulate_stpp_for_streasoner_v15(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV15Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV15Config()
    config.validate()
    original_propagate = _v9_module._propagate_simple_paths_v9
    original_v10_force = _v10_module._force_strong_edge_coverage_v10

    def propagate_with_contract(*args: Any) -> List[Dict[str, Any]]:
        raw_events = original_propagate(*args)
        propagation_audit = args[-1]
        filtered, route_pass = _apply_route_contracts(
            raw_events, structured_scenario, config, filter_invalid=True
        )
        propagation_audit.setdefault("declared_route_filter_passes", []).append(
            route_pass
        )
        return filtered

    def force_with_contract(*args: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
        raw_events, forced = original_v10_force(*args)
        propagation_audit = args[-1]
        filtered, route_pass = _apply_route_contracts(
            raw_events, structured_scenario, config, filter_invalid=True
        )
        propagation_audit.setdefault("declared_route_filter_passes", []).append(
            route_pass
        )
        retained_forced_edges = {
            str(event.get("edge"))
            for event in filtered
            if event.get("forced_by_quality_gate") is True
        }
        return filtered, [edge for edge in forced if edge in retained_forced_edges]

    if config.enforce_declared_event_routes:
        _v9_module._propagate_simple_paths_v9 = propagate_with_contract
        _v10_module._force_strong_edge_coverage_v10 = force_with_contract
    try:
        ts_data, events, metadata = simulate_stpp_for_streasoner_v14(
            structured_scenario, seq_len, config
        )
    finally:
        _v9_module._propagate_simple_paths_v9 = original_propagate
        _v10_module._force_strong_edge_coverage_v10 = original_v10_force

    # IDs are compacted below v9, so annotate once more using final IDs.
    events, route_audit = _apply_route_contracts(
        events, structured_scenario, config, filter_invalid=False
    )
    peak_audit = _scalar_peak_audit(structured_scenario, events)
    all_semantic_ids = all(
        event.get("scenario_event_id") and event.get("schedule_id")
        for event in events
    )
    route_contracts_valid = not route_audit["contract_issues"]
    roots_unambiguous = not (
        route_audit["ambiguous_root_event_ids"]
        or route_audit["unassigned_root_event_ids"]
    )
    actual_routes_valid = not route_audit["propagated_event_violations"]

    quality = metadata["quality_report"]
    quality["checks"].update(
        {
            "declared_event_route_contracts_valid": route_contracts_valid,
            "all_roots_have_unambiguous_semantic_schedule": roots_unambiguous,
            "all_events_have_semantic_ids": all_semantic_ids,
            "all_propagated_lineages_match_declared_paths": actual_routes_valid,
            "all_propagated_events_match_schedule_windows": actual_routes_valid,
            "all_scalar_self_generated_peaks_covered": peak_audit["passed"],
        }
    )
    quality["passed"] = all(quality["checks"].values())
    quality["declared_route_audit"] = route_audit
    quality["scalar_self_generated_peak_audit"] = peak_audit

    temporal_audit = dict(metadata.get("event_semantic_audit", {}))
    temporal_audit.update(
        {
            "declared_route_audit": route_audit,
            "scalar_self_generated_peak_audit": peak_audit,
            "passed": bool(temporal_audit.get("passed", True))
            and route_contracts_valid
            and roots_unambiguous
            and all_semantic_ids
            and actual_routes_valid
            and peak_audit["passed"],
            "audit_semantics": (
                "v14 node-local schedules + v15 event-level exclusive routes"
            ),
        }
    )
    metadata.update(
        {
            "simulator": "STPPG + declared-event-route STPP v15",
            "method": (
                "v14 node-local schedule-aware STPP + pre-aggregation exclusive "
                "event_full_path and schedule-window filtering"
            ),
            "event_semantic_audit": temporal_audit,
            "declared_route_audit": route_audit,
            "scalar_self_generated_peak_audit": peak_audit,
        }
    )
    quality["event_semantic_audit"] = temporal_audit
    return ts_data, events, metadata
