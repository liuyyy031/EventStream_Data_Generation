"""Schedule-aware propagated-variation contract layered over STPP v16."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

import stpp_adapter_v14 as _v14_module
import stpp_adapter_v16 as _v16_module
from stpp_adapter_v14 import _ranges, audit_temporal_contract_v14
from stpp_adapter_v15 import _contains, _route_contracts
from stpp_adapter_v16 import STPPV16Config


_SCALAR_PEAK_ERROR = "arrival time does not equal source peak plus cumulative lag"


def _days(value: Any) -> Tuple[int, ...]:
    try:
        return tuple(sorted(set(int(day) for day in (value or []))))
    except (TypeError, ValueError):
        return ()


def _time_inside_ranges(value: Any, legal_ranges: List[Tuple[int, int]]) -> bool:
    actual = _ranges(value)
    if not actual or not legal_ranges:
        return False
    return all(
        any(_contains([legal], start) and _contains([legal], end) for legal in legal_ranges)
        for start, end in actual
    )


def _remove_scalar_peak_false_positives(audit: Dict[str, Any]) -> Dict[str, Any]:
    """Remove v14's node-global peak check; v19 validates per schedule below."""
    output = copy.deepcopy(audit)
    output["blocking_issues"] = [
        issue
        for issue in output.get("blocking_issues", []) or []
        if issue.get("problem") != _SCALAR_PEAK_ERROR
    ]
    cleaned = []
    for violation in output.get("propagated_variation_violations", []) or []:
        item = dict(violation)
        item["problems"] = [
            problem
            for problem in item.get("problems", []) or []
            if problem != _SCALAR_PEAK_ERROR
        ]
        if item["problems"]:
            cleaned.append(item)
    output["propagated_variation_violations"] = cleaned
    return output


def _schedule_pattern_issues(
    structured_scenario: Mapping[str, Any],
    contracts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    contract_by_schedule = {
        str(contract["schedule_id"]): contract for contract in contracts
    }
    expected = {
        (int(contract["path"][0]), str(contract["schedule_id"]))
        for contract in contracts
    }
    actual: Counter[Tuple[int, str]] = Counter()
    drift_nodes = (
        structured_scenario.get("drift_patterns", {}).get("nodes", []) or []
    )
    for node_data in drift_nodes:
        node_id = int(node_data.get("id", -1))
        for index, pattern in enumerate(node_data.get("patterns", []) or []):
            amplitude = float(pattern.get("amplitude", 0) or 0)
            peak = pattern.get("peak")
            if amplitude <= 0 or peak is None:
                continue
            field = f"drift_patterns.nodes[{node_id}].patterns[{index}]"
            schedule_id = str(pattern.get("schedule_id", "")).strip()
            if not schedule_id:
                issues.append(
                    {
                        "field": field,
                        "problem": "positive self-generated pattern lacks schedule_id",
                    }
                )
                continue
            contract = contract_by_schedule.get(schedule_id)
            actual[(node_id, schedule_id)] += 1
            if contract is None:
                issues.append(
                    {
                        "field": field,
                        "problem": "self-generated pattern references unknown schedule_id",
                    }
                )
                continue
            if int(contract["path"][0]) != node_id:
                issues.append(
                    {
                        "field": field,
                        "problem": "self-generated pattern schedule belongs to another origin",
                    }
                )
            if _days(pattern.get("days")) != tuple(contract["days"]):
                issues.append(
                    {
                        "field": field,
                        "problem": "self-generated pattern days differ from schedule days",
                    }
                )
            if not _contains(contract["stages"][0]["time_ranges"], int(peak)):
                issues.append(
                    {
                        "field": field,
                        "problem": "source peak is outside schedule stage-0 activation window",
                    }
                )
    for key in sorted(expected):
        if actual.get(key, 0) == 0:
            issues.append(
                {
                    "field": f"source_node={key[0]}, schedule_id={key[1]}",
                    "problem": "schedule branch has no positive self-generated pattern",
                }
            )
        elif actual[key] > 1:
            issues.append(
                {
                    "field": f"source_node={key[0]}, schedule_id={key[1]}",
                    "problem": "schedule branch has duplicate self-generated patterns",
                    "count": actual[key],
                }
            )
    return issues


def _schedule_variation_issues(
    structured_scenario: Mapping[str, Any],
    contracts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    contract_map = {
        (str(contract["event_id"]), str(contract["schedule_id"])): contract
        for contract in contracts
    }
    drift_nodes = (
        structured_scenario.get("drift_patterns", {}).get("nodes", []) or []
    )
    for node_data in drift_nodes:
        receiving_node = int(node_data.get("id", -1))
        for index, variation in enumerate(
            node_data.get("propagated_variations", []) or []
        ):
            field = (
                f"drift_patterns.nodes[{receiving_node}]."
                f"propagated_variations[{index}]"
            )
            event_id = str(variation.get("event_id", "")).strip()
            schedule_id = str(variation.get("schedule_id", "")).strip()
            if not schedule_id:
                issues.append(
                    {"field": field, "problem": "propagated variation lacks schedule_id"}
                )
                continue
            contract = contract_map.get((event_id, schedule_id))
            if contract is None:
                issues.append(
                    {
                        "field": field,
                        "problem": "variation references unknown event/schedule branch",
                    }
                )
                continue
            if _days(variation.get("days")) != tuple(contract["days"]):
                issues.append(
                    {"field": field, "problem": "variation days differ from schedule days"}
                )
            path = [int(node) for node in contract["path"]]
            if receiving_node not in path[1:]:
                issues.append(
                    {
                        "field": field,
                        "problem": "receiving node is not on the schedule full path",
                    }
                )
                continue
            stage = path.index(receiving_node) - 1
            legal_ranges = contract["stages"][stage]["arrival_ranges"]
            if not _time_inside_ranges(variation.get("time"), legal_ranges):
                issues.append(
                    {
                        "field": field,
                        "problem": (
                            "variation time is outside matching schedule "
                            "destination_arrival_period"
                        ),
                        "legal_ranges": legal_ranges,
                        "actual_time": variation.get("time"),
                    }
                )
    return issues


def audit_temporal_contract_v19(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Dict[str, Any]:
    """Validate branch-specific peaks/arrivals instead of one peak per node."""
    base = _remove_scalar_peak_false_positives(
        audit_temporal_contract_v14(structured_scenario, seq_len)
    )
    contracts, _ = _route_contracts(structured_scenario)
    schedule_pattern_issues = _schedule_pattern_issues(
        structured_scenario, contracts
    )
    schedule_variation_issues = _schedule_variation_issues(
        structured_scenario, contracts
    )
    base["blocking_issues"].extend(schedule_pattern_issues)
    base["blocking_issues"].extend(schedule_variation_issues)
    base["schedule_pattern_issues"] = schedule_pattern_issues
    base["schedule_variation_issues"] = schedule_variation_issues
    base["passed"] = not base["blocking_issues"]
    base["audit_semantics"] = (
        "v19 (event_id,schedule_id,receiving_node) branch-aware timing"
    )
    return base


def audit_structured_contract_v19(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Dict[str, Any]:
    temporal = audit_temporal_contract_v19(structured_scenario, seq_len)
    contracts, route_contract_issues = _route_contracts(structured_scenario)
    placement_issues: List[Dict[str, Any]] = []
    actual: Counter[Tuple[str, str, int]] = Counter()
    drift_nodes = (
        structured_scenario.get("drift_patterns", {}).get("nodes", []) or []
    )
    for node_data in drift_nodes:
        receiving_node = int(node_data.get("id", -1))
        for index, variation in enumerate(
            node_data.get("propagated_variations", []) or []
        ):
            field = (
                f"drift_patterns.nodes[{receiving_node}]."
                f"propagated_variations[{index}]"
            )
            event_id = str(variation.get("event_id", "")).strip()
            schedule_id = str(variation.get("schedule_id", "")).strip()
            try:
                arrival_path = [
                    int(node) for node in variation.get("arrival_path", []) or []
                ]
            except (TypeError, ValueError):
                arrival_path = []
            if not arrival_path:
                placement_issues.append(
                    {"field": field, "problem": "arrival_path is missing or invalid"}
                )
                continue
            if arrival_path[-1] != receiving_node:
                placement_issues.append(
                    {
                        "field": field,
                        "problem": "variation is stored under the wrong receiving node",
                    }
                )
                continue
            if not event_id or not schedule_id:
                placement_issues.append(
                    {
                        "field": field,
                        "problem": "event_id and schedule_id are required",
                    }
                )
                continue
            actual[(event_id, schedule_id, receiving_node)] += 1

    expected = {
        (str(contract["event_id"]), str(contract["schedule_id"]), int(node))
        for contract in contracts
        for node in contract["path"][1:]
    }
    for key, count in sorted(actual.items()):
        label = f"event_id={key[0]}, schedule_id={key[1]}, receiving_node={key[2]}"
        if count > 1:
            placement_issues.append(
                {
                    "field": label,
                    "problem": "schedule-specific propagated arrival is duplicated",
                    "count": count,
                }
            )
        if key not in expected:
            placement_issues.append(
                {"field": label, "problem": "arrival is not declared by this schedule"}
            )
    for key in sorted(expected):
        if actual.get(key, 0) == 0:
            placement_issues.append(
                {
                    "field": (
                        f"event_id={key[0]}, schedule_id={key[1]}, "
                        f"receiving_node={key[2]}"
                    ),
                    "problem": "schedule receiving node has no propagated arrival entry",
                }
            )

    blocking = list(temporal.get("blocking_issues", []))
    blocking.extend(route_contract_issues)
    blocking.extend(placement_issues)
    output = {
        "passed": not blocking,
        "blocking_issues": blocking,
        "temporal_contract_audit": temporal,
        "route_contract_issues": route_contract_issues,
        "variation_placement_issues": placement_issues,
        "expected_event_schedule_receivers": [
            {
                "event_id": event_id,
                "schedule_id": schedule_id,
                "receiving_node": node,
            }
            for event_id, schedule_id, node in sorted(expected)
        ],
        "actual_event_schedule_receiver_counts": {
            f"{event_id}@{schedule_id}@{node}": count
            for (event_id, schedule_id, node), count in sorted(actual.items())
        },
    }
    if blocking:
        print("\n=== STPP v19 deterministic schedule-aware audit failed ===")
        for issue in blocking:
            field = issue.get("field", issue.get("source_node", "unspecified"))
            print(f"  - {field}: {issue.get('problem', 'contract violation')}")
    return output


def format_contract_feedback_v19(audit: Mapping[str, Any]) -> str:
    lines = [
        "Deterministic v19 schedule-aware validation failed. Fix every listed "
        "field without changing already valid nodes, edges, paths, or values."
    ]
    for issue in audit.get("blocking_issues", []) or []:
        field = issue.get("field", issue.get("source_node", "unspecified"))
        lines.append(f"- {field}: {issue.get('problem', 'contract violation')}")
    lines.extend(
        [
            "Uniqueness key: (event_id, schedule_id, receiving_node).",
            "Every positive source pattern and propagated variation requires "
            "schedule_id and days matching adjacency_modulation.",
            "Validate variation time against that schedule stage's "
            "destination_arrival_period, not a node-global source peak.",
        ]
    )
    return "\n".join(lines)


def canonicalise_temporal_contract_v19(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    original = _v14_module.audit_temporal_contract_v14
    _v14_module.audit_temporal_contract_v14 = audit_temporal_contract_v19
    try:
        return _v14_module.canonicalise_temporal_contract_v14(
            structured_scenario, seq_len
        )
    finally:
        _v14_module.audit_temporal_contract_v14 = original


def simulate_stpp_for_streasoner_v19(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV16Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    original_temporal = _v14_module.audit_temporal_contract_v14
    original_structured = _v16_module.audit_structured_contract_v16
    _v14_module.audit_temporal_contract_v14 = audit_temporal_contract_v19
    _v16_module.audit_structured_contract_v16 = audit_structured_contract_v19
    try:
        ts_data, events, metadata = _v16_module.simulate_stpp_for_streasoner_v16(
            structured_scenario, seq_len, config
        )
    finally:
        _v14_module.audit_temporal_contract_v14 = original_temporal
        _v16_module.audit_structured_contract_v16 = original_structured
    metadata["simulator"] = "STPPG + schedule-aware direct-route STPP v19"
    metadata["method"] = (
        "v16 direct next-edge sampling + v19 branch-aware semantic audit"
    )
    metadata["schedule_aware_contract_audit"] = audit_structured_contract_v19(
        structured_scenario, seq_len
    )
    return ts_data, events, metadata
