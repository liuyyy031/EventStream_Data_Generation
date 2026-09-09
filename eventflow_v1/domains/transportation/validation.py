"""Transportation-specific rule and topology validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ...core.models import EpisodeBundle
from ...core.type_registry import EventTypeRegistry
from ...core.validation import ValidationResult, compute_episode_metrics
from .calibration import load_calibration_profile, recompute_lag
from .event_types import load_event_type_registry
from .network import TransportNetwork


class TransportationValidator:
    def __init__(
        self,
        network: TransportNetwork,
        simulation_config: Dict[str, Any],
        network_config: Dict[str, Any],
        calibration_profile: Dict[str, Any] | None = None,
        event_type_registry: EventTypeRegistry | None = None,
    ):
        spec_path = Path(__file__).with_name("domain_spec.json")
        with spec_path.open("r", encoding="utf-8") as handle:
            self.spec = json.load(handle)
        self.network = network
        self.simulation_config = simulation_config
        self.network_config = network_config
        self.calibration_profile = calibration_profile or load_calibration_profile()
        self.event_type_registry = event_type_registry or load_event_type_registry()
        self.edge_by_id = {edge.edge_id: edge for edge in network.edges}
        registry_ids = set(self.event_type_registry.event_types)
        for rule_id, rule in self.spec["relation_rules"].items():
            referenced = set(rule.get("source_type_ids", [])).union(
                rule.get("target_type_ids", [])
            )
            missing_types = sorted(referenced.difference(registry_ids))
            if missing_types:
                raise ValueError(
                    f"Domain rule {rule_id} references unregistered event types: {', '.join(missing_types)}"
                )

    def validate(self, bundle: EpisodeBundle) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        checks = {
            "valid_event_type_contract": True,
            "valid_domain_rules": True,
            "valid_topology_use": True,
            "bounded_event_branching": True,
            "bounded_topology_degree": True,
            "complete_episode": True,
            "calibrated_lag_provenance": True,
            "event_role_support": True,
            "valid_transport_constraints": True,
        }
        event_by_id = {event.event_id: event for event in bundle.events}

        for event in bundle.events:
            registry_errors = self.event_type_registry.validate_event(
                event, self.network.entity_by_id
            )
            if registry_errors:
                checks["valid_event_type_contract"] = False
                errors.extend(registry_errors)
            road_id = _single_role_entity(event, "affected_road")
            if road_id and "lanes_blocked" in event.attributes:
                road = self.network.entity_by_id.get(road_id)
                available = int(road.attributes.get("lanes", 0)) if road else 0
                blocked = int(event.attributes["lanes_blocked"])
                if blocked > available:
                    checks["valid_transport_constraints"] = False
                    errors.append(
                        f"Event {event.event_id} blocks {blocked} lanes on a {available}-lane road"
                    )

        rules: Dict[str, Dict[str, object]] = self.spec["relation_rules"]
        incoming_by_target: Dict[str, List[Any]] = {}
        for relation in bundle.relations:
            incoming_by_target.setdefault(relation.target_event_id, []).append(relation)
            rule = rules.get(relation.rule_id)
            if rule is None:
                checks["valid_domain_rules"] = False
                errors.append(f"Relation {relation.relation_id} uses unknown rule {relation.rule_id}")
                continue
            source = event_by_id[relation.source_event_id]
            target = event_by_id[relation.target_event_id]
            if source.event_type_id not in rule["source_type_ids"]:
                checks["valid_domain_rules"] = False
                errors.append(
                    f"Rule {relation.rule_id} rejects source type {source.event_type_id}"
                )
            if target.event_type_id not in rule["target_type_ids"]:
                checks["valid_domain_rules"] = False
                errors.append(
                    f"Rule {relation.rule_id} rejects target type {target.event_type_id}"
                )
            if relation.relation_type not in rule["relation_types"]:
                checks["valid_domain_rules"] = False
                errors.append(f"Rule {relation.rule_id} rejects relation type {relation.relation_type}")
            if rule.get("requires_network_edge"):
                edge_id = relation.attributes.get("network_edge_id")
                edge = self.edge_by_id.get(str(edge_id))
                if edge is None:
                    checks["valid_topology_use"] = False
                    errors.append(f"Propagation relation {relation.relation_id} lacks a valid edge")
                elif (
                    _single_role_entity(source, "affected_road") != edge.source_entity_id
                    or _single_role_entity(target, "affected_road") != edge.target_entity_id
                ):
                    checks["valid_topology_use"] = False
                    errors.append(
                        f"Propagation relation {relation.relation_id} disagrees with network edge {edge_id}"
                    )
            if relation.relation_type in {
                "propagates_downstream",
                "transitions_to_recovery",
            }:
                self._validate_calibrated_lag(
                    relation, source, target, errors, checks
                )

        for event in bundle.events:
            incoming = incoming_by_target.get(event.event_id, [])
            if event.event_role == "derived" and not any(
                relation.relation_class == "causal" for relation in incoming
            ):
                checks["event_role_support"] = False
                errors.append(
                    f"Derived event {event.event_id} lacks an incoming causal relation"
                )
            if event.event_role == "recovery" and not any(
                relation.relation_class == "transition"
                and relation.relation_type == "transitions_to_recovery"
                for relation in incoming
            ):
                checks["event_role_support"] = False
                errors.append(
                    f"Recovery event {event.event_id} lacks an incoming recovery transition"
                )

        metrics = compute_episode_metrics(bundle)
        max_children = int(self.simulation_config.get("max_children_per_event", 3))
        if int(metrics["max_propagation_children"]) > max_children:
            checks["bounded_event_branching"] = False
            errors.append(
                f"Episode has {metrics['max_propagation_children']} propagation children "
                f"from one event; configured maximum is {max_children}"
            )

        max_topology_degree = int(self.network_config.get("max_topology_out_degree", 6))
        observed_topology_degree = max(
            (len(edges) for edges in self.network.outgoing.values()), default=0
        )
        if observed_topology_degree > max_topology_degree:
            checks["bounded_topology_degree"] = False
            errors.append(
                f"Network out-degree {observed_topology_degree} exceeds configured "
                f"maximum {max_topology_degree}"
            )

        if bool(metrics["truncated"]):
            checks["complete_episode"] = False
            message = (
                f"Episode is truncated: {metrics['termination_reason']}; "
                f"pending frontier={metrics['pending_propagation_frontier_count']}, "
                f"skipped recoveries={metrics['skipped_recovery_candidate_count']}"
            )
            if bool(self.simulation_config.get("reject_truncated_episodes", True)):
                errors.append(message)
            else:
                warnings.append(message)

        return ValidationResult(not errors, errors, warnings, checks)

    def _validate_calibrated_lag(
        self,
        relation: Any,
        source: Any,
        target: Any,
        errors: List[str],
        checks: Dict[str, bool],
    ) -> None:
        actual_lag = round(target.time_offset_seconds - source.time_offset_seconds, 3)
        recorded_lag = relation.attributes.get("lag_seconds")
        provenance = relation.attributes.get("lag_calibration")
        if recorded_lag is None or not isinstance(provenance, dict):
            checks["calibrated_lag_provenance"] = False
            errors.append(
                f"Relation {relation.relation_id} lacks calibrated lag provenance"
            )
            return
        if provenance.get("profile_id") != self.calibration_profile["profile_id"]:
            checks["calibrated_lag_provenance"] = False
            errors.append(
                f"Relation {relation.relation_id} uses an unexpected calibration profile"
            )
            return
        try:
            recomputed_lag = recompute_lag(provenance)
        except (KeyError, TypeError, ValueError) as exc:
            checks["calibrated_lag_provenance"] = False
            errors.append(
                f"Relation {relation.relation_id} has invalid lag provenance: {exc}"
            )
            return
        if abs(float(recorded_lag) - actual_lag) > 0.002:
            checks["calibrated_lag_provenance"] = False
            errors.append(
                f"Relation {relation.relation_id} lag_seconds disagrees with event time"
            )
        if abs(recomputed_lag - actual_lag) > 0.01:
            checks["calibrated_lag_provenance"] = False
            errors.append(
                f"Relation {relation.relation_id} lag cannot be reproduced from calibration inputs"
            )


def _single_role_entity(event: Any, role: str) -> str | None:
    entity_ids = event.participant_ids_for_role(role)
    return entity_ids[0] if len(entity_ids) == 1 else None
