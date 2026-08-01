"""Fail-closed contract audit and direct declared-route sampling for STPP v16."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import stpp_adapter_v9 as _v9_module
import stpp_adapter_v10 as _v10_module
from stpp_adapter_v2 import _edge_key
from stpp_adapter_v3 import _strong_edge_windows
from stpp_adapter_v9 import _root_lineage
from stpp_adapter_v14 import (
    audit_temporal_contract_v14,
    simulate_stpp_for_streasoner_v14,
)
from stpp_adapter_v15 import (
    STPPV15Config,
    _annotate_root,
    _apply_route_contracts,
    _clock,
    _contains,
    _day_and_local,
    _route_contracts,
    _scalar_peak_audit,
    _select_root_contract,
)


@dataclass(frozen=True)
class STPPV16Config(STPPV15Config):
    """v16 samples only legal next hops instead of filtering descendants later."""

    deterministic_agent2_failures_before_agent1_revision: int = 2


def audit_structured_contract_v16(
    structured_scenario: Mapping[str, Any], seq_len: int
) -> Dict[str, Any]:
    """Deterministically validate placement, route branches, and v14 causality."""
    temporal = audit_temporal_contract_v14(structured_scenario, seq_len)
    contracts, route_contract_issues = _route_contracts(structured_scenario)
    placement_issues: List[Dict[str, Any]] = []
    actual = Counter()

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
                        "problem": (
                            "variation is stored under the wrong node; it must be "
                            f"stored only under node {arrival_path[-1]}=arrival_path[-1]"
                        ),
                        "container_node": receiving_node,
                        "arrival_path_end": arrival_path[-1],
                    }
                )
                continue
            actual[(event_id, receiving_node)] += 1

    event_paths: Dict[str, Tuple[int, ...]] = {}
    for contract in contracts:
        event_id = str(contract["event_id"])
        path = tuple(int(node) for node in contract["path"])
        previous = event_paths.setdefault(event_id, path)
        if previous != path:
            placement_issues.append(
                {
                    "field": f"event_id={event_id}",
                    "problem": "one event_id has multiple full paths",
                }
            )

    expected = {
        (event_id, node)
        for event_id, path in event_paths.items()
        for node in path[1:]
    }
    for key, count in sorted(actual.items()):
        if count > 1:
            placement_issues.append(
                {
                    "field": f"event_id={key[0]}, receiving_node={key[1]}",
                    "problem": "propagated arrival is duplicated",
                    "count": count,
                }
            )
        if key not in expected:
            placement_issues.append(
                {
                    "field": f"event_id={key[0]}, receiving_node={key[1]}",
                    "problem": "arrival is not declared by event_full_path",
                }
            )
    for event_id, node in sorted(expected):
        if actual.get((event_id, node), 0) == 0:
            placement_issues.append(
                {
                    "field": f"event_id={event_id}, receiving_node={node}",
                    "problem": "declared receiving node has no propagated arrival entry",
                }
            )

    blocking = list(temporal.get("blocking_issues", []))
    blocking.extend(route_contract_issues)
    blocking.extend(placement_issues)
    return {
        "passed": not blocking,
        "blocking_issues": blocking,
        "temporal_contract_audit": temporal,
        "route_contract_issues": route_contract_issues,
        "variation_placement_issues": placement_issues,
        "expected_event_receivers": [
            {"event_id": event_id, "receiving_node": node}
            for event_id, node in sorted(expected)
        ],
        "actual_event_receiver_counts": {
            f"{event_id}@{node}": count
            for (event_id, node), count in sorted(actual.items())
        },
    }


def format_contract_feedback_v16(audit: Mapping[str, Any]) -> str:
    lines = [
        "Deterministic v16 validation failed. Fix every listed field; do not "
        "change already valid nodes, edges, paths, or values."
    ]
    for issue in audit.get("blocking_issues", []) or []:
        field = issue.get("field", issue.get("source_node", "unspecified"))
        problem = issue.get("problem", "contract violation")
        lines.append(f"- {field}: {problem}")
    lines.append(
        "Placement invariant: store each propagated_variation only under "
        "arrival_path[-1], exactly once per (event_id, receiving_node)."
    )
    lines.append(
        "Causal invariant: downstream TIME must start no earlier than the "
        "preceding stage start plus the preceding edge lag."
    )
    return "\n".join(lines)


def _contract_map(
    structured_scenario: Mapping[str, Any],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    contracts, _ = _route_contracts(structured_scenario)
    return {
        (str(contract["event_id"]), str(contract["schedule_id"])): contract
        for contract in contracts
    }


def _parent_contract(
    event: Mapping[str, Any],
    contracts: Mapping[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any] | None:
    return contracts.get(
        (str(event.get("scenario_event_id", "")), str(event.get("schedule_id", "")))
    )


def _next_stage(
    parent: Mapping[str, Any], contract: Mapping[str, Any]
) -> Tuple[int, int, int, Dict[str, Any]] | None:
    stage = int(parent.get("generation", 0))
    path = [int(node) for node in contract["path"]]
    if stage < 0 or stage >= len(path) - 1:
        return None
    return path[stage], path[stage + 1], stage, contract["stages"][stage]


def _stage_time_matches(
    parent: Mapping[str, Any],
    child_time_index: int,
    contract: Mapping[str, Any],
    stage_contract: Mapping[str, Any],
    period: int,
    week_size: int,
) -> Tuple[bool, bool, int, int]:
    parent_day, parent_local = _day_and_local(
        int(parent["time_index"]), period, week_size
    )
    _, child_local = _day_and_local(child_time_index, period, week_size)
    activation_ok = bool(
        parent_day in contract["days"]
        and _contains(stage_contract["time_ranges"], parent_local)
    )
    arrival_ok = _contains(stage_contract["arrival_ranges"], child_local)
    return activation_ok, arrival_ok, parent_local, child_local


def _propagate_declared_routes_v16(
    immigrants: List[Dict[str, Any]],
    outgoing: Dict[int, List[Dict[str, Any]]],
    edge_factors: Dict[str, np.ndarray],
    node_coords: np.ndarray,
    seq_len: int,
    config: STPPV16Config,
    rng: np.random.Generator,
    audit: Dict[str, Any],
    structured_scenario: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    events = list(immigrants)
    contracts_list, contract_issues = _route_contracts(structured_scenario)
    contracts = {
        (str(item["event_id"]), str(item["schedule_id"])): item
        for item in contracts_list
    }
    period, week_size = _clock(structured_scenario)
    direct = audit.setdefault(
        "direct_declared_route_sampling",
        {
            "contract_issues": contract_issues,
            "unassigned_root_event_ids": [],
            "ambiguous_root_event_ids": [],
            "stopped_at_declared_path_end": 0,
            "blocked_outside_activation_window": 0,
            "blocked_outside_arrival_window": 0,
            "blocked_missing_declared_edge": 0,
            "probability_rejections": 0,
            "created_propagated_events": 0,
        },
    )
    for root in events:
        contract, ambiguous = _select_root_contract(
            root, contracts_list, period, week_size
        )
        _annotate_root(root, contract, ambiguous, period, week_size)
        if contract is None:
            direct["unassigned_root_event_ids"].append(int(root["event_id"]))
        if ambiguous:
            direct["ambiguous_root_event_ids"].append(int(root["event_id"]))

    queue = list(events)
    next_id = max((int(event["event_id"]) for event in events), default=-1) + 1
    while queue and len(events) < config.max_events:
        parent = queue.pop(0)
        if int(parent.get("generation", 0)) >= config.max_generation:
            continue
        contract = _parent_contract(parent, contracts)
        if contract is None:
            continue
        next_stage = _next_stage(parent, contract)
        if next_stage is None:
            direct["stopped_at_declared_path_end"] += 1
            continue
        source, target, stage, stage_contract = next_stage
        if int(parent["node_id"]) != source:
            direct["blocked_missing_declared_edge"] += 1
            continue
        matching_edges = [
            edge for edge in outgoing.get(source, []) if int(edge["target"]) == target
        ]
        if not matching_edges:
            direct["blocked_missing_declared_edge"] += 1
            continue
        edge = matching_edges[0]
        key = _edge_key(source, target)
        delay = max(
            0.001,
            float(rng.normal(float(edge["lag"]), config.propagation_jitter)),
        )
        child_time = float(parent["t"]) + delay
        if child_time >= seq_len:
            continue
        child_time_index = int(np.floor(child_time))
        activation_ok, arrival_ok, parent_local, child_local = _stage_time_matches(
            parent,
            child_time_index,
            contract,
            stage_contract,
            period,
            week_size,
        )
        if config.enforce_schedule_activation_windows and not activation_ok:
            direct["blocked_outside_activation_window"] += 1
            continue
        if config.enforce_destination_arrival_windows and not arrival_ok:
            direct["blocked_outside_arrival_window"] += 1
            continue
        factor = float(edge_factors[key][int(parent["time_index"])])
        probability = min(config.edge_branching_ratio * factor, 0.95)
        if rng.random() > probability:
            direct["probability_rejections"] += 1
            continue
        lineage_nodes, lineage_edges = _root_lineage(parent)
        x, y = rng.normal(
            loc=node_coords[target], scale=config.target_spatial_jitter, size=2
        )
        child = {
            "event_id": next_id,
            "t": child_time,
            "x": float(np.clip(x, 0.0, 1.0)),
            "y": float(np.clip(y, 0.0, 1.0)),
            "node_id": target,
            "time_index": child_time_index,
            "event_type": "propagated",
            "parent_event_id": int(parent["event_id"]),
            "source_node_id": source,
            "generation": int(parent.get("generation", 0)) + 1,
            "edge": key,
            "edge_probability": probability,
            "configured_lag": float(edge["lag"]),
            "realized_delay": delay,
            "lineage_nodes": lineage_nodes + [target],
            "lineage_edges": lineage_edges + [key],
            "semantic_root_event_id": int(parent["semantic_root_event_id"]),
            "semantic_path_stage": stage + 1,
            "scenario_event_id": contract["event_id"],
            "schedule_id": contract["schedule_id"],
            "declared_event_full_path": list(contract["path"]),
            "semantic_assignment_basis": "inherited_from_root",
            "semantic_assignment_ambiguous": bool(
                parent.get("semantic_assignment_ambiguous", False)
            ),
            "parent_local_time_index": parent_local,
            "child_local_time_index": child_local,
            "schedule_activation_match": activation_ok,
            "destination_arrival_match": arrival_ok,
        }
        events.append(child)
        queue.append(child)
        next_id += 1
        direct["created_propagated_events"] += 1
    return sorted(events, key=lambda event: (float(event["t"]), int(event["event_id"])))


def _force_declared_strong_edges_v16(
    events: List[Dict[str, Any]],
    structured_scenario: Mapping[str, Any],
    node_coords: np.ndarray,
    seq_len: int,
    config: STPPV16Config,
    rng: np.random.Generator,
    audit: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not config.guarantee_strong_edge_coverage:
        return events, []
    windows = _strong_edge_windows(structured_scenario, seq_len)
    contracts = _contract_map(structured_scenario)
    period, week_size = _clock(structured_scenario)
    edge_lags = {
        _edge_key(int(edge["source"]), int(edge["target"])): max(
            float(edge.get("time_lag", 1)), 0.01
        )
        for edge in structured_scenario.get("edges", []) or []
    }
    output = list(events)
    forced: List[str] = []
    next_id = max((int(event["event_id"]) for event in output), default=-1) + 1
    direct = audit.setdefault("direct_declared_route_sampling", {})
    direct.setdefault("uncovered_strong_edges_without_declared_parent", [])

    for key, indices in sorted(windows.items(), key=lambda item: min(item[1])):
        if any(
            event.get("edge") == key
            and int(event.get("parent_time_index", -1)) in indices
            for event in output
        ):
            continue
        source_text, target_text = key.split("->", 1)
        source, target = int(source_text), int(target_text)
        lag = edge_lags[key]
        parents: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], int, int]] = []
        for parent in output:
            if (
                int(parent["node_id"]) != source
                or int(parent["time_index"]) not in indices
                or int(parent.get("generation", 0)) >= config.max_generation
            ):
                continue
            contract = _parent_contract(parent, contracts)
            if contract is None:
                continue
            next_stage = _next_stage(parent, contract)
            if next_stage is None or next_stage[0:2] != (source, target):
                continue
            _, _, stage, stage_contract = next_stage
            child_time = float(parent["t"]) + lag
            if child_time >= seq_len:
                continue
            child_index = int(np.floor(child_time))
            activation_ok, arrival_ok, parent_local, child_local = (
                _stage_time_matches(
                    parent,
                    child_index,
                    contract,
                    stage_contract,
                    period,
                    week_size,
                )
            )
            if activation_ok and arrival_ok:
                parents.append(
                    (parent, contract, stage_contract, parent_local, child_local)
                )
        if not parents or len(output) >= config.max_events:
            direct["uncovered_strong_edges_without_declared_parent"].append(key)
            audit.setdefault("uncovered_strong_edges_without_simple_parent", []).append(
                key
            )
            continue
        parent, contract, _, parent_local, child_local = min(
            parents, key=lambda item: float(item[0]["t"])
        )
        lineage_nodes, lineage_edges = _root_lineage(parent)
        child_time = float(parent["t"]) + lag
        x, y = rng.normal(
            loc=node_coords[target], scale=config.target_spatial_jitter, size=2
        )
        output.append(
            {
                "event_id": next_id,
                "t": child_time,
                "x": float(np.clip(x, 0.0, 1.0)),
                "y": float(np.clip(y, 0.0, 1.0)),
                "node_id": target,
                "time_index": int(np.floor(child_time)),
                "event_type": "propagated",
                "parent_event_id": int(parent["event_id"]),
                "parent_time_index": int(parent["time_index"]),
                "source_node_id": source,
                "generation": int(parent.get("generation", 0)) + 1,
                "edge": key,
                "edge_probability": 1.0,
                "configured_lag": lag,
                "realized_delay": lag,
                "forced_by_quality_gate": True,
                "forced_strong_edge": True,
                "lineage_nodes": lineage_nodes + [target],
                "lineage_edges": lineage_edges + [key],
                "semantic_root_event_id": int(parent["semantic_root_event_id"]),
                "semantic_path_stage": int(parent.get("generation", 0)) + 1,
                "scenario_event_id": contract["event_id"],
                "schedule_id": contract["schedule_id"],
                "declared_event_full_path": list(contract["path"]),
                "semantic_assignment_basis": "inherited_from_root",
                "semantic_assignment_ambiguous": False,
                "parent_local_time_index": parent_local,
                "child_local_time_index": child_local,
                "schedule_activation_match": True,
                "destination_arrival_match": True,
            }
        )
        forced.append(key)
        next_id += 1
    return sorted(
        output, key=lambda event: (float(event["t"]), int(event["event_id"]))
    ), forced


def _declared_route_envelope_audit(
    structured_scenario: Mapping[str, Any],
    route_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    contracts, issues = _route_contracts(structured_scenario)
    prefixes = {
        tuple(int(node) for node in contract["path"][:end])
        for contract in contracts
        for end in range(2, len(contract["path"]) + 1)
    }
    return {
        "contract_issues": issues,
        "declared_propagation_prefixes": [list(path) for path in sorted(prefixes)],
        "actual_propagated_lineages": route_audit.get(
            "actual_propagated_lineages", {}
        ),
        "passed": not issues and bool(prefixes),
    }


def simulate_stpp_for_streasoner_v16(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV16Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV16Config()
    config.validate()
    pre_audit = audit_structured_contract_v16(structured_scenario, seq_len)
    original_propagate = _v9_module._propagate_simple_paths_v9
    original_v10_force = _v10_module._force_strong_edge_coverage_v10

    def direct_propagate(*args: Any) -> List[Dict[str, Any]]:
        return _propagate_declared_routes_v16(
            *args, structured_scenario=structured_scenario
        )

    def direct_force(*args: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
        return _force_declared_strong_edges_v16(*args)

    _v9_module._propagate_simple_paths_v9 = direct_propagate
    _v10_module._force_strong_edge_coverage_v10 = direct_force
    try:
        ts_data, events, metadata = simulate_stpp_for_streasoner_v14(
            structured_scenario, seq_len, config
        )
    finally:
        _v9_module._propagate_simple_paths_v9 = original_propagate
        _v10_module._force_strong_edge_coverage_v10 = original_v10_force

    events, route_audit = _apply_route_contracts(
        events, structured_scenario, config, filter_invalid=False
    )
    peak_audit = _scalar_peak_audit(structured_scenario, events)
    declared_envelope = _declared_route_envelope_audit(
        structured_scenario, route_audit
    )
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
            "pre_simulation_contract_valid": pre_audit["passed"],
            "declared_event_route_contracts_valid": route_contracts_valid,
            "all_roots_have_unambiguous_semantic_schedule": roots_unambiguous,
            "all_events_have_semantic_ids": all_semantic_ids,
            "all_propagated_lineages_match_declared_paths": actual_routes_valid,
            "all_propagated_events_match_schedule_windows": actual_routes_valid,
            "all_scalar_self_generated_peaks_covered": peak_audit["passed"],
            "declared_route_envelope_valid": declared_envelope["passed"],
        }
    )
    quality["passed"] = all(quality["checks"].values())
    quality["pre_simulation_contract_audit"] = pre_audit
    quality["declared_route_audit"] = route_audit
    quality["scalar_self_generated_peak_audit"] = peak_audit

    temporal_audit = dict(metadata.get("event_semantic_audit", {}))
    temporal_audit.update(
        {
            "pre_simulation_contract_audit": pre_audit,
            "declared_route_audit": route_audit,
            "scalar_self_generated_peak_audit": peak_audit,
            "passed": bool(temporal_audit.get("passed", True))
            and pre_audit["passed"]
            and route_contracts_valid
            and roots_unambiguous
            and all_semantic_ids
            and actual_routes_valid
            and peak_audit["passed"],
            "audit_semantics": (
                "v16 deterministic placement/causality + direct declared-route sampling"
            ),
        }
    )
    metadata.update(
        {
            "simulator": "STPPG + direct declared-route STPP v16",
            "method": (
                "fail-closed deterministic contract audit + direct next-edge "
                "event_full_path sampling"
            ),
            "event_semantic_audit": temporal_audit,
            "pre_simulation_contract_audit": pre_audit,
            "declared_route_audit": route_audit,
            "declared_route_envelope_audit": declared_envelope,
            "scalar_self_generated_peak_audit": peak_audit,
        }
    )
    quality["event_semantic_audit"] = temporal_audit
    return ts_data, events, metadata
