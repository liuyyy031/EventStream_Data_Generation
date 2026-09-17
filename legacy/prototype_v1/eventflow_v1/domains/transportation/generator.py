"""Continuous-time road-transportation episode generation.

Topology defines propagation eligibility only.  Event relations are emitted
only when a named DomainSpec rule is executed.  Event names and attribute
contracts are resolved through the versioned transportation type registry.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ...core.models import EpisodeBundle, Event, EventParticipant, Mechanism, Relation
from ...core.type_registry import EventTypeRegistry
from .calibration import load_calibration_profile, propagation_lag, recovery_lag
from .event_types import (
    CONGESTION_EASING,
    CONGESTION_ONSET,
    FLOODING_ONSET,
    HEAVY_RAIN_ONSET,
    NORMAL_FLOW_RESTORED,
    ROAD_CLOSURE_STARTED,
    ROAD_DEBRIS,
    ROADWORKS_STARTED,
    TRAFFIC_SIGNAL_FAILURE,
    VEHICLE_BREAKDOWN,
    VEHICLE_COLLISION,
    load_event_type_registry,
)
from .network import TransportNetwork


ROOT_RELATION_RULES = {
    VEHICLE_COLLISION: "INCIDENT_TO_CONGESTION",
    VEHICLE_BREAKDOWN: "INCIDENT_TO_CONGESTION",
    ROAD_DEBRIS: "OBSTRUCTION_TO_CONGESTION",
    TRAFFIC_SIGNAL_FAILURE: "EQUIPMENT_FAULT_TO_CONGESTION",
    ROAD_CLOSURE_STARTED: "MANAGEMENT_ACTION_TO_CONGESTION",
    ROADWORKS_STARTED: "MANAGEMENT_ACTION_TO_CONGESTION",
    HEAVY_RAIN_ONSET: "ENVIRONMENT_TO_CONGESTION",
    FLOODING_ONSET: "ENVIRONMENT_TO_CONGESTION",
}


class TransportationEpisodeGenerator:
    def __init__(
        self,
        network: TransportNetwork,
        simulation_config: Dict[str, Any],
        calibration_profile: Dict[str, Any] | None = None,
        event_type_registry: EventTypeRegistry | None = None,
    ):
        self.network = network
        self.config = simulation_config
        self.calibration_profile = calibration_profile or load_calibration_profile()
        self.event_type_registry = event_type_registry or load_event_type_registry()

    def generate(self, episode_index: int, seed: int) -> EpisodeBundle:
        rng = random.Random(seed)
        episode_id = f"transport_ep_{episode_index:07d}"
        start = datetime.fromisoformat(str(self.config["start_time"]))
        duration = float(self.config["duration_seconds"])
        scenario = rng.choice(list(self.config["scenario_families"]))
        max_events = int(self.config.get("max_events_per_episode", 24))
        max_depth = int(self.config.get("max_propagation_depth", 5))
        expected_children = float(self.config.get("expected_children_per_event", 1.15))
        max_children = int(self.config.get("max_children_per_event", 3))
        recovery_probability = float(self.config.get("recovery_probability", 0.75))
        full_recovery_probability = float(
            self.config.get("full_recovery_probability", 0.35)
        )

        road_candidates = [
            road_id for road_id in self.network.road_ids if self.network.outgoing.get(road_id)
        ] or self.network.road_ids
        root_road_id = rng.choice(road_candidates)
        region_id = self.network.region_ids[0]

        events: List[Event] = []
        relations: List[Relation] = []
        mechanisms: List[Mechanism] = []
        executed_rules: List[Dict[str, Any]] = []
        event_counter = 0
        relation_counter = 0
        mechanism_counter = 0

        def add_event(
            event_type_id: str,
            offset: float,
            participants: List[EventParticipant],
            attributes: Dict[str, Any],
            role: str,
            rule_id: str,
            provenance_extra: Optional[Dict[str, Any]] = None,
        ) -> Event:
            nonlocal event_counter
            event_counter += 1
            bounded_offset = round(max(0.0, min(duration, offset)), 3)
            event = Event(
                event_id=f"{episode_id}_event_{event_counter:04d}",
                episode_id=episode_id,
                domain="transportation",
                event_type_id=event_type_id,
                event_time=(start + timedelta(seconds=bounded_offset)).isoformat(),
                time_offset_seconds=bounded_offset,
                participants=participants,
                attributes=attributes,
                event_role=role,
                observation={
                    "observability": "observed",
                    "observation_source": "synthetic_eventflow_generator",
                    "observation_delay_seconds": 0.0,
                },
                provenance={
                    "generator": "eventflow_v1.4",
                    "rule_id": rule_id,
                    **(provenance_extra or {}),
                },
            )
            events.append(event)
            return event

        def add_relation(
            source: Event,
            target: Event,
            relation_class: str,
            relation_type: str,
            rule_id: str,
            mechanism_group_id: Optional[str] = None,
            attributes: Optional[Dict[str, Any]] = None,
        ) -> Relation:
            nonlocal relation_counter
            relation_counter += 1
            relation = Relation(
                relation_id=f"{episode_id}_relation_{relation_counter:04d}",
                episode_id=episode_id,
                source_event_id=source.event_id,
                target_event_id=target.event_id,
                relation_class=relation_class,
                relation_type=relation_type,
                status="ground_truth",
                rule_id=rule_id,
                mechanism_group_id=mechanism_group_id,
                attributes=attributes or {},
                provenance={"recorded_when_rule_executed": True},
            )
            relations.append(relation)
            return relation

        root_offset = rng.uniform(duration * 0.05, duration * 0.18)
        environmental_event: Optional[Event] = None
        trigger_event: Optional[Event] = None

        if scenario == "mixed_scenario":
            environmental_event = _add_heavy_rain(
                add_event, rng, region_id, max(0.0, root_offset - rng.uniform(120.0, 600.0))
            )
            trigger_event = _add_collision(
                add_event, rng, root_road_id, root_offset, _road_lanes(self.network, root_road_id)
            )
        elif scenario == "weather_disruption":
            if rng.random() < 0.65:
                environmental_event = _add_heavy_rain(
                    add_event, rng, region_id, root_offset
                )
            else:
                environmental_event = _add_flooding(
                    add_event, rng, root_road_id, root_offset
                )
        elif scenario == "accident_propagation":
            trigger_event = _add_collision(
                add_event, rng, root_road_id, root_offset, _road_lanes(self.network, root_road_id)
            )
        elif scenario == "vehicle_breakdown":
            trigger_event = _add_breakdown(
                add_event, rng, root_road_id, root_offset, _road_lanes(self.network, root_road_id)
            )
        elif scenario == "road_obstruction":
            trigger_event = _add_road_debris(add_event, rng, root_road_id, root_offset)
        elif scenario == "signal_failure":
            trigger_event = _add_signal_failure(add_event, rng, root_road_id, root_offset)
        elif scenario == "road_closure":
            trigger_event = (
                _add_road_closure(add_event, rng, root_road_id, root_offset)
                if rng.random() < 0.65
                else _add_roadworks(add_event, rng, root_road_id, root_offset)
            )
        else:
            raise ValueError(f"Unsupported transportation scenario_family: {scenario}")

        primary_roots = [
            event for event in (environmental_event, trigger_event) if event is not None
        ]
        capacity_reduction = _combined_capacity_reduction(primary_roots)
        baseline_vc_ratio = float(
            self.network.entity_by_id[root_road_id].attributes.get(
                "baseline_volume_capacity_ratio", 0.75
            )
        )
        severity, severity_provenance = _initial_congestion_severity(
            capacity_reduction, baseline_vc_ratio, rng
        )
        initial_delay = rng.triangular(45.0, 600.0, 180.0)
        initial_congestion = add_event(
            CONGESTION_ONSET,
            root_offset + initial_delay,
            [_participant(root_road_id, "affected_road")],
            {
                "severity": severity,
                "propagation_depth": 0,
                "trigger_capacity_reduction_fraction": capacity_reduction,
                "baseline_volume_capacity_ratio": round(baseline_vc_ratio, 6),
            },
            "derived",
            "INITIAL_CONGESTION",
            {"severity_calculation": severity_provenance},
        )

        if scenario == "mixed_scenario" and trigger_event and environmental_event:
            mechanism_counter += 1
            mechanism_id = f"{episode_id}_mechanism_{mechanism_counter:04d}"
            joint_rule = "COLLISION_AND_RAIN_TO_SEVERE_CONGESTION"
            left = add_relation(
                trigger_event,
                initial_congestion,
                "causal",
                "jointly_causes_congestion",
                joint_rule,
                mechanism_id,
            )
            right = add_relation(
                environmental_event,
                initial_congestion,
                "causal",
                "jointly_causes_congestion",
                joint_rule,
                mechanism_id,
            )
            mechanisms.append(
                Mechanism(
                    mechanism_group_id=mechanism_id,
                    episode_id=episode_id,
                    target_event_id=initial_congestion.event_id,
                    combination="all_of",
                    member_relation_ids=[left.relation_id, right.relation_id],
                    rule_id=joint_rule,
                    description=(
                        "The vehicle collision and heavy-rain onset jointly reduce capacity "
                        "and produce the initial congestion-onset event."
                    ),
                )
            )
            executed_rules.append({"rule_id": joint_rule, "target": initial_congestion.event_id})
        else:
            parent = trigger_event or environmental_event
            if parent is None:  # defensive; every supported scenario creates one
                raise RuntimeError("Scenario produced no primary root event")
            rule_id = ROOT_RELATION_RULES[parent.event_type_id]
            relation_type = (
                "weather_induces_congestion"
                if parent.event_type_id in {HEAVY_RAIN_ONSET, FLOODING_ONSET}
                else "initiates_congestion"
            )
            add_relation(
                parent,
                initial_congestion,
                "causal",
                relation_type,
                rule_id,
            )
            executed_rules.append({"rule_id": rule_id, "target": initial_congestion.event_id})

        queue: List[Tuple[Event, int]] = [(initial_congestion, 0)]
        visited_roads = {root_road_id}
        congestion_events = [initial_congestion]
        truncated = False
        pending_propagation_frontier_count = 0
        weather_active = environmental_event is not None
        while queue:
            if len(events) >= max_events:
                truncated = True
                pending_propagation_frontier_count += len(queue)
                break
            parent, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            source_road = _single_participant(parent, "affected_road")
            candidates = [
                edge
                for edge in self.network.outgoing.get(source_road, [])
                if edge.target_entity_id not in visited_roads
            ]
            rng.shuffle(candidates)
            per_edge_probability = min(1.0, expected_children / max(1, len(candidates)))
            selected = [
                edge for edge in candidates if rng.random() < per_edge_probability
            ][:max_children]
            if depth == 0 and candidates and not selected:
                selected = [candidates[0]]
            for edge_index, edge in enumerate(selected):
                if len(events) >= max_events:
                    truncated = True
                    pending_propagation_frontier_count += len(selected) - edge_index + len(queue)
                    break
                source_entity = self.network.entity_by_id[source_road]
                baseline_vc = float(
                    source_entity.attributes.get("baseline_volume_capacity_ratio", 0.75)
                )
                lag, lag_provenance = propagation_lag(
                    free_flow_seconds=float(edge.attributes["free_flow_travel_seconds"]),
                    baseline_volume_capacity_ratio=baseline_vc,
                    congestion_severity=float(parent.attributes["severity"]),
                    rain_active=weather_active,
                    rng=rng,
                    profile=self.calibration_profile,
                )
                target_offset = parent.time_offset_seconds + lag
                if target_offset >= duration:
                    continue
                visited_roads.add(edge.target_entity_id)
                child_severity = round(
                    max(0.15, float(parent.attributes["severity"]) * rng.uniform(0.68, 0.96)),
                    4,
                )
                child = add_event(
                    CONGESTION_ONSET,
                    target_offset,
                    [_participant(edge.target_entity_id, "affected_road")],
                    {"severity": child_severity, "propagation_depth": depth + 1},
                    "derived",
                    "CONGESTION_PROPAGATES_DOWNSTREAM",
                )
                add_relation(
                    parent,
                    child,
                    "causal",
                    "propagates_downstream",
                    "CONGESTION_PROPAGATES_DOWNSTREAM",
                    attributes={
                        "network_edge_id": edge.edge_id,
                        "lag_seconds": round(lag, 3),
                        "lag_calibration": lag_provenance,
                    },
                )
                executed_rules.append(
                    {
                        "rule_id": "CONGESTION_PROPAGATES_DOWNSTREAM",
                        "source": parent.event_id,
                        "target": child.event_id,
                        "network_edge_id": edge.edge_id,
                    }
                )
                queue.append((child, depth + 1))
                congestion_events.append(child)

        planned_recoveries: List[Tuple[Event, float, float, Dict[str, Any]]] = []
        for congestion in list(congestion_events):
            if rng.random() >= recovery_probability:
                continue
            recovery_delay, recovery_provenance = recovery_lag(
                scenario_family=scenario,
                congestion_severity=float(congestion.attributes["severity"]),
                rng=rng,
                profile=self.calibration_profile,
            )
            recovery_offset = congestion.time_offset_seconds + recovery_delay
            if recovery_offset < duration:
                planned_recoveries.append(
                    (congestion, recovery_offset, recovery_delay, recovery_provenance)
                )

        available_recovery_slots = max(0, max_events - len(events))
        skipped_recovery_candidate_count = max(
            0, len(planned_recoveries) - available_recovery_slots
        )
        if skipped_recovery_candidate_count:
            truncated = True
        for congestion, recovery_offset, recovery_delay, recovery_provenance in planned_recoveries[
            :available_recovery_slots
        ]:
            full_recovery = rng.random() < full_recovery_probability
            recovery_fraction = 1.0 if full_recovery else round(rng.uniform(0.45, 0.9), 4)
            recovery_type = NORMAL_FLOW_RESTORED if full_recovery else CONGESTION_EASING
            recovery = add_event(
                recovery_type,
                recovery_offset,
                [_participant(_single_participant(congestion, "affected_road"), "affected_road")],
                {
                    "recovered_from_severity": congestion.attributes["severity"],
                    "recovery_fraction": recovery_fraction,
                },
                "recovery",
                "CONGESTION_TO_RECOVERY",
            )
            add_relation(
                congestion,
                recovery,
                "transition",
                "transitions_to_recovery",
                "CONGESTION_TO_RECOVERY",
                attributes={
                    "lag_seconds": round(recovery_delay, 3),
                    "lag_calibration": recovery_provenance,
                },
            )
            executed_rules.append(
                {
                    "rule_id": "CONGESTION_TO_RECOVERY",
                    "source": congestion.event_id,
                    "target": recovery.event_id,
                }
            )

        background_count = _poisson(
            rng,
            float(self.config.get("background_root_rate_per_hour", 0.0)) * duration / 3600.0,
        )
        unused_roads = [road for road in self.network.road_ids if road not in visited_roads]
        rng.shuffle(unused_roads)
        background_limit = min(background_count, max(0, max_events - len(events)))
        skipped_background_root_count = max(0, background_count - background_limit)
        if skipped_background_root_count:
            truncated = True
        independent_root_events: List[Event] = []
        for road_id in unused_roads[:background_limit]:
            if len(events) >= max_events:
                break
            independent_root_events.append(
                _add_background_incident(add_event, rng, road_id, rng.uniform(0.0, duration))
            )

        events.sort(key=lambda event: (event.time_offset_seconds, event.event_id))
        involved_entity_ids = sorted(
            {entity_id for event in events for entity_id in event.participant_ids}
        )
        program = {
            "program_version": "episode-program-v2",
            "seed": seed,
            "scenario_family": scenario,
            "root_event_ids": [event.event_id for event in events if event.event_role == "root"],
            "primary_root_event_ids": [event.event_id for event in primary_roots],
            "independent_root_event_ids": [event.event_id for event in independent_root_events],
            "rules_executed": executed_rules,
            "relationship_policy": "record_only_when_domain_rule_executes",
            "topology_role": "propagation_eligibility_only",
            "parameters": {
                "expected_children_per_event": expected_children,
                "max_children_per_event": max_children,
                "max_propagation_depth": max_depth,
                "recovery_probability": recovery_probability,
                "full_recovery_probability": full_recovery_probability,
                "calibration_profile_id": self.calibration_profile["profile_id"],
                "calibration_status": self.calibration_profile["status"],
                "event_type_registry_id": self.event_type_registry.registry_id,
                "event_type_registry_version": self.event_type_registry.registry_version,
            },
            "termination": {
                "reason": "event_limit_reached" if truncated else "completed",
                "truncated": truncated,
                "event_count": len(events),
                "event_limit": max_events,
                "pending_propagation_frontier_count": pending_propagation_frontier_count,
                "skipped_recovery_candidate_count": skipped_recovery_candidate_count,
                "skipped_background_root_count": skipped_background_root_count,
            },
        }
        return EpisodeBundle(
            episode_id=episode_id,
            network_id=self.network.network_id,
            domain="transportation",
            scenario_family=scenario,
            start_time=start.isoformat(),
            duration_seconds=duration,
            involved_entity_ids=involved_entity_ids,
            program=program,
            events=events,
            relations=relations,
            mechanisms=mechanisms,
        )


def _participant(entity_id: str, role: str) -> EventParticipant:
    return EventParticipant(entity_id=entity_id, role=role)


def _single_participant(event: Event, role: str) -> str:
    entity_ids = event.participant_ids_for_role(role)
    if len(entity_ids) != 1:
        raise ValueError(f"Event {event.event_id} requires exactly one {role} participant")
    return entity_ids[0]


def _add_collision(
    add_event: Any,
    rng: random.Random,
    road_id: str,
    offset: float,
    lanes_available: int,
) -> Event:
    severity = round(rng.uniform(0.45, 0.95), 4)
    reduction = round(rng.uniform(0.28, 0.72) * severity, 4)
    return add_event(
        VEHICLE_COLLISION,
        offset,
        [_participant(road_id, "affected_road")],
        {
            "severity": severity,
            "lanes_blocked": rng.randint(1, max(1, min(2, lanes_available))),
            "capacity_reduction_fraction": min(0.95, reduction),
            "vehicles_involved": rng.choice([1, 2, 2, 3, 4]),
        },
        "root",
        "EXOGENOUS_COLLISION_ROOT",
    )


def _add_breakdown(
    add_event: Any,
    rng: random.Random,
    road_id: str,
    offset: float,
    lanes_available: int,
) -> Event:
    severity = round(rng.uniform(0.2, 0.65), 4)
    shoulder_available = rng.random() < 0.65
    reduction = rng.uniform(0.08, 0.32) * (0.7 if shoulder_available else 1.0)
    return add_event(
        VEHICLE_BREAKDOWN,
        offset,
        [_participant(road_id, "affected_road")],
        {
            "severity": severity,
            "lanes_blocked": 0 if shoulder_available else min(1, lanes_available),
            "capacity_reduction_fraction": round(reduction, 4),
            "shoulder_available": shoulder_available,
        },
        "root",
        "EXOGENOUS_BREAKDOWN_ROOT",
    )


def _add_road_debris(add_event: Any, rng: random.Random, road_id: str, offset: float) -> Event:
    obstruction = round(rng.uniform(0.15, 0.65), 4)
    return add_event(
        ROAD_DEBRIS,
        offset,
        [_participant(road_id, "affected_road")],
        {
            "severity": round(rng.uniform(0.25, 0.75), 4),
            "obstruction_fraction": obstruction,
            "capacity_reduction_fraction": round(obstruction * rng.uniform(0.65, 0.95), 4),
        },
        "root",
        "EXOGENOUS_OBSTRUCTION_ROOT",
    )


def _add_signal_failure(add_event: Any, rng: random.Random, road_id: str, offset: float) -> Event:
    severity = round(rng.uniform(0.3, 0.8), 4)
    return add_event(
        TRAFFIC_SIGNAL_FAILURE,
        offset,
        [_participant(road_id, "affected_road")],
        {
            "severity": severity,
            "affected_movements": rng.randint(1, 4),
            "capacity_reduction_fraction": round(rng.uniform(0.12, 0.45) * severity, 4),
        },
        "root",
        "EXOGENOUS_SIGNAL_FAILURE_ROOT",
    )


def _add_heavy_rain(add_event: Any, rng: random.Random, region_id: str, offset: float) -> Event:
    intensity = round(rng.uniform(18.0, 55.0), 2)
    return add_event(
        HEAVY_RAIN_ONSET,
        offset,
        [_participant(region_id, "affected_region")],
        {
            "intensity_mm_per_hour": intensity,
            "coverage": "regional",
            "capacity_reduction_fraction": round(min(0.35, 0.05 + intensity / 250.0), 4),
        },
        "root",
        "EXOGENOUS_HEAVY_RAIN_ROOT",
    )


def _add_flooding(add_event: Any, rng: random.Random, road_id: str, offset: float) -> Event:
    severity = round(rng.uniform(0.45, 0.95), 4)
    return add_event(
        FLOODING_ONSET,
        offset,
        [_participant(road_id, "affected_road")],
        {
            "severity": severity,
            "water_depth_cm": round(rng.uniform(5.0, 40.0), 2),
            "capacity_reduction_fraction": round(rng.uniform(0.3, 0.8) * severity, 4),
        },
        "root",
        "EXOGENOUS_FLOODING_ROOT",
    )


def _add_road_closure(add_event: Any, rng: random.Random, road_id: str, offset: float) -> Event:
    closure_fraction = round(rng.uniform(0.5, 1.0), 4)
    return add_event(
        ROAD_CLOSURE_STARTED,
        offset,
        [_participant(road_id, "affected_road")],
        {
            "closure_fraction": closure_fraction,
            "planned": False,
            "capacity_reduction_fraction": closure_fraction,
        },
        "root",
        "OPERATOR_ROAD_CLOSURE_ROOT",
    )


def _add_roadworks(add_event: Any, rng: random.Random, road_id: str, offset: float) -> Event:
    closure_fraction = round(rng.uniform(0.2, 0.65), 4)
    return add_event(
        ROADWORKS_STARTED,
        offset,
        [_participant(road_id, "affected_road")],
        {
            "closure_fraction": closure_fraction,
            "planned": True,
            "capacity_reduction_fraction": closure_fraction,
        },
        "root",
        "PLANNED_ROADWORKS_ROOT",
    )


def _add_background_incident(
    add_event: Any,
    rng: random.Random,
    road_id: str,
    offset: float,
    lanes_available: int = 1,
) -> Event:
    if rng.random() < 0.6:
        severity = round(rng.uniform(0.1, 0.35), 4)
        return add_event(
            VEHICLE_COLLISION,
            offset,
            [_participant(road_id, "affected_road")],
            {
                "severity": severity,
                "lanes_blocked": 1,
                "capacity_reduction_fraction": round(rng.uniform(0.05, 0.18), 4),
                "vehicles_involved": rng.choice([1, 2]),
            },
            "root",
            "INDEPENDENT_BACKGROUND_ROOT",
            {"independent_from_primary_flow": True},
        )
    severity = round(rng.uniform(0.1, 0.3), 4)
    return add_event(
        VEHICLE_BREAKDOWN,
        offset,
        [_participant(road_id, "affected_road")],
        {
            "severity": severity,
            "lanes_blocked": 0,
            "capacity_reduction_fraction": round(rng.uniform(0.03, 0.12), 4),
            "shoulder_available": True,
        },
        "root",
        "INDEPENDENT_BACKGROUND_ROOT",
        {"independent_from_primary_flow": True},
    )


def _combined_capacity_reduction(events: List[Event]) -> float:
    remaining = 1.0
    for event in events:
        remaining *= 1.0 - float(event.attributes.get("capacity_reduction_fraction", 0.0))
    return round(1.0 - remaining, 6)


def _road_lanes(network: TransportNetwork, road_id: str) -> int:
    return max(1, int(network.entity_by_id[road_id].attributes.get("lanes", 1)))


def _initial_congestion_severity(
    capacity_reduction_fraction: float,
    baseline_vc_ratio: float,
    rng: random.Random,
) -> Tuple[float, Dict[str, Any]]:
    usable_capacity = max(0.05, 1.0 - capacity_reduction_fraction)
    effective_vc_ratio = baseline_vc_ratio / usable_capacity
    noise = rng.uniform(-0.05, 0.05)
    raw = 0.15 + 0.45 * capacity_reduction_fraction + 0.25 * min(effective_vc_ratio, 2.0) + noise
    severity = round(max(0.15, min(0.98, raw)), 4)
    return severity, {
        "model": "synthetic_capacity_pressure_v1",
        "formula": "severity=clamp(0.15+0.45*capacity_reduction+0.25*min(effective_vc,2)+noise,0.15,0.98)",
        "inputs": {
            "capacity_reduction_fraction": round(capacity_reduction_fraction, 6),
            "baseline_volume_capacity_ratio": round(baseline_vc_ratio, 6),
            "effective_volume_capacity_ratio": round(effective_vc_ratio, 6),
            "noise": round(noise, 6),
        },
        "status": "transparent_synthetic_prior_pending_empirical_fit",
    }


def _poisson(rng: random.Random, mean: float) -> int:
    if mean <= 0.0:
        return 0
    threshold = math.exp(-mean)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1
