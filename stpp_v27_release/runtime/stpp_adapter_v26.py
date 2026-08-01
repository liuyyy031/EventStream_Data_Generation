"""Deterministic Agent-2 propagated-variation placement for STPP v26."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Tuple

from demo_sts_stpp_v8 import _canonicalise_agent2_json_v8


def canonicalise_agent2_json_v26(
    raw: Mapping[str, Any] | str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Store each variation under the node at the end of its arrival path."""
    structured, base_audit = _canonicalise_agent2_json_v8(raw)
    output = copy.deepcopy(structured)
    audit = copy.deepcopy(base_audit)
    drift_nodes = output.get("drift_patterns", {}).get("nodes", []) or []
    node_by_id: Dict[int, Dict[str, Any]] = {}
    for node_data in drift_nodes:
        try:
            node_by_id[int(node_data.get("id"))] = node_data
        except (TypeError, ValueError):
            continue

    pending: Dict[int, List[Dict[str, Any]]] = {
        node_id: [] for node_id in node_by_id
    }
    placement_repairs: List[Dict[str, Any]] = []
    for stored_node, node_data in node_by_id.items():
        retained: List[Dict[str, Any]] = []
        variations = list(node_data.get("propagated_variations", []) or [])
        for index, raw_variation in enumerate(variations):
            variation = dict(raw_variation)
            try:
                arrival_path = [
                    int(node) for node in variation.get("arrival_path", []) or []
                ]
            except (TypeError, ValueError):
                arrival_path = []
            receiving_node = arrival_path[-1] if arrival_path else None
            if (
                receiving_node is None
                or receiving_node == stored_node
                or receiving_node not in node_by_id
            ):
                retained.append(variation)
                continue
            pending[receiving_node].append(variation)
            placement_repairs.append(
                {
                    "field": (
                        f"drift_patterns.nodes[{stored_node}]."
                        f"propagated_variations[{index}]"
                    ),
                    "operation": "relocate_to_arrival_path_terminal",
                    "from_node": stored_node,
                    "to_node": receiving_node,
                    "event_id": variation.get("event_id"),
                    "schedule_id": variation.get("schedule_id"),
                    "arrival_path": arrival_path,
                    "reason": (
                        "propagated_variation must be stored under "
                        "arrival_path[-1]"
                    ),
                }
            )
        node_data["propagated_variations"] = retained

    for receiving_node, variations in pending.items():
        if variations:
            node_by_id[receiving_node].setdefault(
                "propagated_variations", []
            ).extend(variations)

    existing_changes = list(audit.get("changes", []) or [])
    audit["changes"] = existing_changes + placement_repairs
    audit["num_changes"] = len(audit["changes"])
    audit["placement_repairs"] = placement_repairs
    audit["num_placement_repairs"] = len(placement_repairs)
    audit["placement_semantics"] = "stored_node == arrival_path[-1]"
    return output, audit
