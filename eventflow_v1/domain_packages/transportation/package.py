"""Transportation rules implemented as a plug-in for the neutral scheduler.

The bundled parameters are transparent synthetic priors.  They validate the
generation architecture but do not establish real-world statistical fidelity.
"""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any, Dict, List

from generation_core.domain import (
    CandidateSpec,
    CandidateUpdate,
    EpisodeContext,
    ObservationPlan,
    RelationSpec,
)
from generation_core.domain_spec import LoadedDomainSpec, load_domain_spec
from generation_core.mechanisms import load_mechanism_registry
from generation_core.models import (
    Candidate,
    ContextEvidence,
    ContextRelation,
    Entity,
    EpisodeResult,
    EventParticipant,
    EventRecord,
    StatePredicateEvidence,
    TemporalExtent,
)
from generation_core.temporal import (
    ConditionalLogNormalModel,
    DeterministicDelayModel,
    ExponentialArrivalModel,
    PiecewiseExponentialHazardModel,
    TemporalModelRegistry,
    build_standard_temporal_registry,
)
from generation_core.topology import (
    SparseHeterogeneousTopologyGenerator,
    relation_is_active,
)


COLLISION = "transportation.road.incident.vehicle_collision"
CONGESTION = "transportation.road.traffic_state.congestion_onset"
RECOVERY = "transportation.road.traffic_state.normal_flow_restored"
HEAVY_RAIN_START = "transportation.weather.heavy_rain.start"
HEAVY_RAIN_END = "transportation.weather.heavy_rain.end"
ROAD_CLOSURE_START = "transportation.road.operation.closure.start"
ROAD_CLOSURE_END = "transportation.road.operation.closure.end"

EVENT_TYPE_IDS = (
    COLLISION,
    CONGESTION,
    RECOVERY,
    HEAVY_RAIN_START,
    HEAVY_RAIN_END,
    ROAD_CLOSURE_START,
    ROAD_CLOSURE_END,
)

SCENARIO_FAMILIES = (
    "accident_propagation",
    "weather_disruption",
    "planned_closure",
    "compound_weather_accident",
)


class TransportationPackage:
    domain_id = "transportation"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.domain_spec = load_domain_spec(Path(__file__).with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(
            Path(__file__).with_name("mechanism_specs.json"),
            known_event_type_ids=EVENT_TYPE_IDS,
            known_context_relation_type_ids=self.domain_spec.context_relation_type_ids,
        )
        supplied = config or {}
        topology_supplied = supplied.get("topology_profile", {})
        self.config = {
            "base_seed": int(supplied.get("base_seed", 20260716)),
            "context_count": int(
                supplied.get("context_count", supplied.get("network_count", 4))
            ),
            "nodes_per_context": int(supplied.get("nodes_per_context", 100)),
            "topology_profile": {
                "expected_out_degree_range": list(
                    topology_supplied.get("expected_out_degree_range", [1.4, 3.2])
                ),
                "max_out_degree": int(
                    topology_supplied.get("max_out_degree", 8)
                ),
                "community_count_range": list(
                    topology_supplied.get("community_count_range", [3, 10])
                ),
                "within_community_bias_range": list(
                    topology_supplied.get("within_community_bias_range", [1.5, 5.0])
                ),
                "degree_sigma_range": list(
                    topology_supplied.get("degree_sigma_range", [0.45, 1.0])
                ),
                "hub_fraction_range": list(
                    topology_supplied.get("hub_fraction_range", [0.02, 0.08])
                ),
                "hub_multiplier_range": list(
                    topology_supplied.get("hub_multiplier_range", [2.0, 5.0])
                ),
            },
            "duration_seconds": float(supplied.get("duration_seconds", 7200.0)),
            "start_time": str(
                supplied.get("start_time", "2026-07-16T08:00:00+08:00")
            ),
            "root_rate_per_hour": float(supplied.get("root_rate_per_hour", 1.5)),
            "max_root_events": int(supplied.get("max_root_events", 3)),
            "max_propagation_depth": int(
                supplied.get("max_propagation_depth", 4)
            ),
            "scenario_weights": dict(
                supplied.get(
                    "scenario_weights",
                    {
                        "accident_propagation": 0.35,
                        "weather_disruption": 0.25,
                        "planned_closure": 0.2,
                        "compound_weather_accident": 0.2,
                    },
                )
            ),
        }
        if self.config["nodes_per_context"] < 2:
            raise ValueError("nodes_per_context must be at least 2")
        if self.config["context_count"] <= 0:
            raise ValueError("context_count must be positive")
        _validate_topology_profile(self.config["topology_profile"])
        unknown_scenarios = set(self.config["scenario_weights"]) - set(
            SCENARIO_FAMILIES
        )
        if unknown_scenarios:
            raise ValueError(f"Unknown scenario families: {sorted(unknown_scenarios)}")
        if not self.config["scenario_weights"] or any(
            float(weight) < 0.0
            for weight in self.config["scenario_weights"].values()
        ):
            raise ValueError("scenario_weights must be non-empty and non-negative")
        if sum(float(value) for value in self.config["scenario_weights"].values()) <= 0.0:
            raise ValueError("At least one scenario weight must be positive")
        self._context_cache: Dict[
            int,
            tuple[
                List[Entity],
                List[ContextRelation],
                Dict[str, Any],
            ],
        ] = {}

    def catalog(self) -> Dict[str, Any]:
        """Describe the open, namespaced domain vocabulary used by this package.

        Event types are registry entries rather than a closed language-level
        enum.  New leaf types can be added without changing the common record
        contract, while their attributes remain explicitly documented here.
        """

        return {
            "domain": self.domain_id,
            "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_pending_empirical_fit",
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {
                "mode": "deterministic_grounded_projection",
                "free_form_causal_completion": False,
                "alignment_unit": "sentence_to_event_or_relation_ids",
            },
            "entity_types": [
                {
                    "entity_type_id": "transportation.region",
                    "parent_type_id": None,
                    "required_attributes": ["name"],
                },
                {
                    "entity_type_id": "transportation.road.segment",
                    "parent_type_id": "transportation.infrastructure",
                    "required_attributes": [
                        "region_id",
                        "length_km",
                        "speed_limit_kph",
                        "capacity_vehicles_per_hour",
                        "baseline_volume_capacity_ratio",
                    ],
                },
            ],
            "event_types": [
                {
                    "event_type_id": COLLISION,
                    "parent_type_id": "transportation.road.incident",
                    "required_attributes": ["severity", "lanes_blocked"],
                },
                {
                    "event_type_id": CONGESTION,
                    "parent_type_id": "transportation.road.traffic_state",
                    "required_attributes": ["severity", "propagation_depth"],
                },
                {
                    "event_type_id": RECOVERY,
                    "parent_type_id": "transportation.road.traffic_state",
                    "required_attributes": ["recovered_from_severity"],
                },
                {
                    "event_type_id": HEAVY_RAIN_START,
                    "parent_type_id": "transportation.weather.precipitation",
                    "required_attributes": ["severity", "coverage_fraction"],
                },
                {
                    "event_type_id": HEAVY_RAIN_END,
                    "parent_type_id": "transportation.weather.precipitation",
                    "required_attributes": ["ended_severity"],
                },
                {
                    "event_type_id": ROAD_CLOSURE_START,
                    "parent_type_id": "transportation.road.operation.closure",
                    "required_attributes": ["closure_reason", "planned"],
                },
                {
                    "event_type_id": ROAD_CLOSURE_END,
                    "parent_type_id": "transportation.road.operation.closure",
                    "required_attributes": ["closure_reason", "planned"],
                },
            ],
            "context_relation_types": [
                {
                    "relation_type_id": "transportation.road.downstream_of",
                    "directed": True,
                    "meaning": "physical_or_operational_downstream_reachability_only",
                    "does_not_imply_event_parenthood": True,
                }
            ],
            "event_relation_types": [
                {
                    "relation_type_id": "transportation.initiates_congestion",
                    "relation_class": "causal",
                },
                {
                    "relation_type_id": "transportation.propagates_downstream",
                    "relation_class": "causal",
                },
                {
                    "relation_type_id": "transportation.transitions_to_recovery",
                    "relation_class": "transition",
                },
                {
                    "relation_type_id": "transportation.weather_induces_congestion",
                    "relation_class": "statistical_influence",
                },
                {
                    "relation_type_id": "transportation.weather_contributes_to_collision",
                    "relation_class": "statistical_influence",
                },
                {
                    "relation_type_id": "transportation.weather_contributes_to_congestion",
                    "relation_class": "statistical_influence",
                },
                {
                    "relation_type_id": "transportation.weather_clears",
                    "relation_class": "transition",
                },
                {
                    "relation_type_id": "transportation.closure_reopens",
                    "relation_class": "transition",
                },
            ],
            "mechanisms": self.mechanism_registry.descriptors(),
        }

    def render_episode_text(self, result: EpisodeResult) -> Dict[str, Any]:
        event_labels = {
            COLLISION: "a vehicle collision",
            CONGESTION: "congestion onset",
            RECOVERY: "normal traffic flow restoration",
            HEAVY_RAIN_START: "heavy rain onset",
            HEAVY_RAIN_END: "heavy rain clearance",
            ROAD_CLOSURE_START: "a planned road closure",
            ROAD_CLOSURE_END: "the planned road reopening",
        }
        relation_labels = {
            "transportation.initiates_congestion": "initiated",
            "transportation.propagates_downstream": "propagated to",
            "transportation.transitions_to_recovery": "transitioned to",
            "transportation.weather_induces_congestion": "induced",
            "transportation.weather_contributes_to_collision": "contributed to",
            "transportation.weather_contributes_to_congestion": "jointly contributed to",
            "transportation.weather_clears": "ended with",
            "transportation.closure_reopens": "ended with",
        }
        sentences: List[str] = []
        claims: List[Dict[str, Any]] = []
        if not result.events:
            sentences.append("No event occurrence was generated inside this observation window.")
        for event in result.events:
            participants = ", ".join(
                f"{item.role}={item.entity_id}" for item in event.participants
            )
            attributes = ", ".join(
                f"{key}={value}" for key, value in sorted(event.attributes.items())
            )
            sentence = (
                f"At +{event.temporal.occurrence_start_offset_seconds:.3f} seconds, "
                f"{event_labels[event.event_type_id]} occurred "
                f"({participants}; {attributes})."
            )
            sentences.append(sentence)
            claims.append(
                {
                    "claim_id": f"{result.episode_id}_text_claim_{len(claims) + 1:06d}",
                    "sentence_index": len(sentences) - 1,
                    "event_ids": [event.event_id],
                    "relation_ids": [],
                    "evidence_fields": [
                        "event_type_id",
                        "temporal.occurrence_start_offset_seconds",
                        "participants",
                        "attributes",
                    ],
                }
            )
        for relation in result.event_relations:
            sentence = (
                f"Generated rule {relation.rule_id} states that event "
                f"{relation.source_event_id} {relation_labels[relation.relation_type_id]} "
                f"event {relation.target_event_id}, with an occurrence-start lag of "
                f"{relation.temporal_link.lag_seconds:.3f} seconds."
            )
            sentences.append(sentence)
            claims.append(
                {
                    "claim_id": f"{result.episode_id}_text_claim_{len(claims) + 1:06d}",
                    "sentence_index": len(sentences) - 1,
                    "event_ids": [
                        relation.source_event_id,
                        relation.target_event_id,
                    ],
                    "relation_ids": [relation.relation_id],
                    "evidence_fields": [
                        "rule_id",
                        "relation_type_id",
                        "source_event_id",
                        "target_event_id",
                        "temporal_link",
                    ],
                }
            )
        return {
            "alignment_id": f"{result.episode_id}_text_alignment",
            "episode_id": result.episode_id,
            "domain": result.domain,
            "projection_mode": "deterministic_grounded_projection",
            "sentences": sentences,
            "text": " ".join(sentences),
            "claims": claims,
        }

    def create_episode_context(self, episode_index: int, seed: int) -> EpisodeContext:
        context_index = episode_index % self.config["context_count"]
        context_id = f"transport_context_{context_index:03d}"
        if context_index not in self._context_cache:
            self._context_cache[context_index] = _build_transport_context(
                context_id,
                self.config["nodes_per_context"],
                self.config["base_seed"] + context_index * 7919,
                self.config["duration_seconds"],
                self.config["topology_profile"],
                self.domain_spec,
            )
        entities, relations, topology = self._context_cache[context_index]
        road_ids = [entity.entity_id for entity in entities if entity.entity_type_id == "transportation.road.segment"]
        scenario_rng = random.Random(seed ^ 0x5EED5EED)
        scenario_names = list(self.config["scenario_weights"])
        scenario_family = scenario_rng.choices(
            scenario_names,
            weights=[self.config["scenario_weights"][name] for name in scenario_names],
            k=1,
        )[0]
        return EpisodeContext(
            episode_id=f"transport_episode_{episode_index:07d}",
            context_id=context_id,
            domain=self.domain_id,
            start_time=self.config["start_time"],
            duration_seconds=self.config["duration_seconds"],
            entities=entities,
            context_relations=relations,
            context_attributes={"topology_profile": topology},
            episode_attributes={
                "scenario_family": scenario_family,
                "topology_profile_id": topology["profile_id"],
            },
            state={
                "topology_profile_id": topology["profile_id"],
                "road_ids": road_ids,
                "road_status": {road_id: "normal" for road_id in road_ids},
                "road_severity": {road_id: 0.0 for road_id in road_ids},
                "root_events_fired": 0,
                "scenario_family": scenario_family,
                "active_weather_event_id": None,
                "weather_severity": 0.0,
            },
        )

    def seed_candidates(
        self, context: EpisodeContext, rng: random.Random
    ) -> List[CandidateSpec]:
        scenario = context.state["scenario_family"]
        road_id = rng.choice(context.state["road_ids"])
        if scenario == "accident_propagation":
            return [self._root_spec(context, road_id, 0.0, 0)]
        if scenario in {"weather_disruption", "compound_weather_accident"}:
            return [self._weather_root_spec(context)]
        if scenario == "planned_closure":
            scheduled = rng.uniform(
                0.08 * context.duration_seconds,
                0.25 * context.duration_seconds,
            )
            return [
                CandidateSpec(
                    mechanism_id="transportation.planned_closure_start",
                    target_event_type_id=ROAD_CLOSURE_START,
                    parent_event_ids=[],
                    participants=[EventParticipant(road_id, "affected_road")],
                    temporal_model_ref="core.scheduled",
                    temporal_inputs={"scheduled_offset_seconds": scheduled},
                    attributes={
                        "candidate_kind": "root_closure",
                        "target_road_id": road_id,
                    },
                    provenance={
                        "independent_root": True,
                        "schedule_policy": "synthetic_planned_window",
                        "fit_status": "synthetic_prior",
                    },
                )
            ]
        raise ValueError(f"Unsupported scenario family: {scenario}")

    def revalidate_candidate(
        self,
        candidate: Candidate,
        context: EpisodeContext,
        event_by_id: Dict[str, EventRecord],
        now: float,
    ) -> CandidateUpdate:
        del event_by_id
        kind = candidate.attributes.get("candidate_kind")
        if kind in {"root_incident", "root_weather"}:
            return CandidateUpdate(valid=True)
        road_status = context.state["road_status"]
        if kind == "root_closure":
            road_id = str(candidate.attributes["target_road_id"])
            return CandidateUpdate(
                valid=road_status[road_id] == "normal",
                reason="planned_closure_target_is_not_available",
            )
        if kind in {"weather_congestion", "weather_conditioned_collision"}:
            road_id = str(candidate.attributes["target_road_id"])
            weather_parent_id = str(candidate.attributes["weather_parent_event_id"])
            active_weather = context.state["active_weather_event_id"]
            return CandidateUpdate(
                valid=(
                    active_weather == weather_parent_id
                    and road_status[road_id] == "normal"
                ),
                reason="weather_or_target_state_changed",
            )
        if kind == "weather_clear":
            weather_parent_id = str(candidate.attributes["weather_parent_event_id"])
            return CandidateUpdate(
                valid=context.state["active_weather_event_id"] == weather_parent_id,
                reason="weather_episode_was_already_cleared",
            )
        if kind == "closure_end":
            road_id = str(candidate.attributes["target_road_id"])
            return CandidateUpdate(
                valid=road_status[road_id] == "closed",
                reason="road_is_no_longer_closed",
            )
        if kind == "initial_congestion":
            road_id = str(candidate.attributes["target_road_id"])
            return CandidateUpdate(
                valid=road_status[road_id] == "normal",
                reason="target_road_is_not_normal",
            )
        if kind == "propagation":
            source_road = str(candidate.attributes["source_road_id"])
            target_road = str(candidate.attributes["target_road_id"])
            relation_ids = candidate.context_evidence.context_relation_ids
            relation = (
                context.relation_index.get(relation_ids[0])
                if len(relation_ids) == 1
                else None
            )
            relation_valid = bool(
                relation is not None
                and relation.source_entity_id == source_road
                and relation.target_entity_id == target_road
                and relation.relation_type_id == "transportation.road.downstream_of"
                and relation_is_active(relation, now)
            )
            source_valid = road_status[source_road] == "congested"
            target_valid = road_status[target_road] == "normal"
            valid = relation_valid and source_valid and target_valid
            severity = float(context.state["road_severity"][source_road])
            return CandidateUpdate(
                valid=valid,
                reason="propagation_precondition_became_false",
                temporal_inputs={
                    **candidate.temporal_inputs,
                    "severity": severity,
                },
                context_evidence=ContextEvidence(
                    context_relation_ids=list(relation_ids),
                    entity_ids=[source_road, target_road],
                    evaluated_at_offset_seconds=now,
                    state_predicates=[
                        StatePredicateEvidence(
                            source_road,
                            "road_status",
                            "equals",
                            "congested",
                            road_status[source_road],
                            source_valid,
                        ),
                        StatePredicateEvidence(
                            target_road,
                            "road_status",
                            "equals",
                            "normal",
                            road_status[target_road],
                            target_valid,
                        ),
                    ],
                ),
            )
        if kind == "recovery":
            road_id = str(candidate.attributes["source_road_id"])
            valid = road_status[road_id] == "congested"
            severity = float(context.state["road_severity"][road_id])
            return CandidateUpdate(
                valid=valid,
                reason="road_is_no_longer_congested",
                temporal_inputs={"severity": severity},
            )
        return CandidateUpdate(valid=False, reason="unknown_candidate_kind")

    def materialize_event(
        self,
        event_id: str,
        candidate: Candidate,
        context: EpisodeContext,
        rng: random.Random,
    ) -> EventRecord:
        kind = candidate.attributes["candidate_kind"]
        road_id = candidate.attributes.get("target_road_id") or candidate.attributes.get("source_road_id")
        if kind == "root_weather":
            region_id = f"{context.context_id}_region_0000"
            severity = round(rng.uniform(0.45, 0.95), 4)
            participants = [EventParticipant(region_id, "affected_region")]
            attributes = {
                "severity": severity,
                "coverage_fraction": round(rng.uniform(0.35, 0.9), 4),
            }
            role = "root"
        elif kind == "weather_clear":
            region_id = f"{context.context_id}_region_0000"
            participants = [EventParticipant(region_id, "affected_region")]
            attributes = {
                "ended_severity": round(float(context.state["weather_severity"]), 4)
            }
            role = "transition"
        elif kind in {"root_closure", "closure_end"}:
            road_id = str(road_id)
            participants = [EventParticipant(road_id, "affected_road")]
            attributes = {"closure_reason": "planned_maintenance", "planned": True}
            role = "root" if kind == "root_closure" else "transition"
        elif kind in {"root_incident", "weather_conditioned_collision"}:
            road_id = str(road_id)
            severity = round(rng.uniform(0.35, 0.9), 4)
            attributes = {
                "severity": severity,
                "lanes_blocked": max(1, int(round(severity * 2.0))),
            }
            participants = [EventParticipant(road_id, "affected_road")]
            role = "root" if kind == "root_incident" else "derived"
        elif kind in {"initial_congestion", "propagation", "weather_congestion"}:
            road_id = str(road_id)
            parent_severity = float(candidate.attributes.get("parent_severity", 0.55))
            severity = round(max(0.15, parent_severity * rng.uniform(0.72, 0.96)), 4)
            attributes = {
                "severity": severity,
                "propagation_depth": int(candidate.attributes.get("depth", 0)),
            }
            participants = [EventParticipant(road_id, "affected_road")]
            role = "derived"
        elif kind == "recovery":
            road_id = str(road_id)
            attributes = {
                "recovered_from_severity": round(
                    float(context.state["road_severity"][road_id]), 4
                )
            }
            participants = [EventParticipant(road_id, "affected_road")]
            role = "transition"
        else:
            raise ValueError(f"Unsupported candidate kind: {kind}")
        return EventRecord(
            event_id=event_id,
            episode_id=context.episode_id,
            domain=self.domain_id,
            event_type_id=candidate.target_event_type_id,
            temporal=TemporalExtent(float(candidate.scheduled_time)),
            participants=participants,
            attributes=attributes,
            event_role=role,
            provenance={
                "mechanism_id": candidate.mechanism_id,
                "candidate_id": candidate.candidate_id,
                "fit_status": "transparent_synthetic_prior_pending_empirical_fit",
            },
        )

    def apply_event(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
    ) -> None:
        road_id = event.participants[0].entity_id
        if event.event_type_id == HEAVY_RAIN_START:
            context.state["active_weather_event_id"] = event.event_id
            context.state["weather_severity"] = float(event.attributes["severity"])
        elif event.event_type_id == HEAVY_RAIN_END:
            context.state["active_weather_event_id"] = None
            context.state["weather_severity"] = 0.0
        elif event.event_type_id == ROAD_CLOSURE_START:
            context.state["road_status"][road_id] = "closed"
        elif event.event_type_id == ROAD_CLOSURE_END:
            context.state["road_status"][road_id] = "normal"
        elif event.event_type_id == COLLISION:
            context.state["root_events_fired"] += 1
        elif event.event_type_id == CONGESTION:
            context.state["road_status"][road_id] = "congested"
            context.state["road_severity"][road_id] = float(event.attributes["severity"])
        elif event.event_type_id == RECOVERY:
            context.state["road_status"][road_id] = "normal"
            context.state["road_severity"][road_id] = 0.0

    def spawn_candidates(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
        event_by_id: Dict[str, EventRecord],
        rng: random.Random,
    ) -> List[CandidateSpec]:
        del event_by_id
        specs: List[CandidateSpec] = []
        road_id = event.participants[0].entity_id
        if event.event_type_id == HEAVY_RAIN_START:
            severity = float(event.attributes["severity"])
            specs.append(
                CandidateSpec(
                    mechanism_id="transportation.weather_clearance",
                    target_event_type_id=HEAVY_RAIN_END,
                    parent_event_ids=[event.event_id],
                    participants=list(event.participants),
                    temporal_model_ref="transportation.weather_duration.prior_v1",
                    temporal_inputs={"severity": severity},
                    attributes={
                        "candidate_kind": "weather_clear",
                        "weather_parent_event_id": event.event_id,
                    },
                    provenance={"fit_status": "synthetic_prior"},
                    phase_priority=10,
                )
            )
            target_road = rng.choice(context.state["road_ids"])
            if context.state["scenario_family"] == "compound_weather_accident":
                specs.append(
                    CandidateSpec(
                        mechanism_id="transportation.weather_to_collision",
                        target_event_type_id=COLLISION,
                        parent_event_ids=[event.event_id],
                        participants=[EventParticipant(target_road, "affected_road")],
                        temporal_model_ref="transportation.weather_to_collision.prior_v1",
                        temporal_inputs={"weather_severity": severity},
                        attributes={
                            "candidate_kind": "weather_conditioned_collision",
                            "target_road_id": target_road,
                            "weather_parent_event_id": event.event_id,
                        },
                        provenance={"fit_status": "synthetic_prior"},
                    )
                )
            else:
                specs.append(
                    CandidateSpec(
                        mechanism_id="transportation.weather_to_congestion",
                        target_event_type_id=CONGESTION,
                        parent_event_ids=[event.event_id],
                        participants=[EventParticipant(target_road, "affected_road")],
                        temporal_model_ref="transportation.weather_to_congestion.prior_v1",
                        temporal_inputs={"weather_severity": severity},
                        attributes={
                            "candidate_kind": "weather_congestion",
                            "target_road_id": target_road,
                            "weather_parent_event_id": event.event_id,
                            "parent_severity": severity,
                            "depth": 0,
                        },
                        provenance={"fit_status": "synthetic_prior"},
                    )
                )
        elif event.event_type_id == ROAD_CLOSURE_START:
            specs.append(
                CandidateSpec(
                    mechanism_id="transportation.planned_closure_end",
                    target_event_type_id=ROAD_CLOSURE_END,
                    parent_event_ids=[event.event_id],
                    participants=list(event.participants),
                    temporal_model_ref="transportation.closure_duration.prior_v1",
                    temporal_inputs={},
                    attributes={
                        "candidate_kind": "closure_end",
                        "target_road_id": road_id,
                    },
                    provenance={"fit_status": "synthetic_prior"},
                    phase_priority=10,
                )
            )
        elif event.event_type_id == COLLISION:
            road = context.entity_by_id[road_id]
            weather_parent_id = context.state["active_weather_event_id"]
            parents = [event.event_id]
            combination = "single"
            if weather_parent_id is not None:
                parents = [str(weather_parent_id), event.event_id]
                combination = "all_of"
            specs.append(
                CandidateSpec(
                    mechanism_id=(
                        "transportation.weather_collision_joint_to_congestion"
                        if weather_parent_id is not None
                        else "transportation.incident_to_congestion"
                    ),
                    target_event_type_id=CONGESTION,
                    parent_event_ids=parents,
                    participants=[EventParticipant(road_id, "affected_road")],
                    temporal_model_ref="transportation.incident_to_congestion.prior_v1",
                    temporal_inputs={
                        "severity": float(event.attributes["severity"]),
                        "volume_capacity_ratio": float(
                            road.attributes["baseline_volume_capacity_ratio"]
                        ),
                        "weather_severity": (
                            float(context.state["weather_severity"])
                            if weather_parent_id is not None
                            else 0.0
                        ),
                    },
                    attributes={
                        "candidate_kind": "initial_congestion",
                        "target_road_id": road_id,
                        "parent_severity": float(event.attributes["severity"]),
                        "depth": 0,
                        "collision_parent_event_id": event.event_id,
                        "weather_parent_event_id": weather_parent_id,
                    },
                    provenance={"fit_status": "synthetic_prior"},
                    combination=combination,
                )
            )
            root_index = int(candidate.attributes.get("root_index", 0)) + 1
            if (
                candidate.attributes.get("candidate_kind") == "root_incident"
                and root_index < self.config["max_root_events"]
            ):
                unused = [
                    item
                    for item in context.state["road_ids"]
                    if context.state["road_status"][item] == "normal" and item != road_id
                ]
                if unused:
                    specs.append(
                        self._root_spec(context, rng.choice(unused), event.temporal.occurrence_start_offset_seconds, root_index)
                    )
        elif event.event_type_id == CONGESTION:
            depth = int(event.attributes.get("propagation_depth", 0))
            severity = float(event.attributes["severity"])
            if depth < self.config["max_propagation_depth"]:
                neighbors = context.neighbors(
                    road_id,
                    "transportation.road.downstream_of",
                    at_time=event.temporal.occurrence_start_offset_seconds,
                )
                for neighbor in neighbors[:3]:
                    edge = neighbor.relation
                    target = neighbor.neighbor_entity_id
                    if context.state["road_status"][target] != "normal":
                        continue
                    specs.append(
                        CandidateSpec(
                            mechanism_id="transportation.congestion_propagation",
                            target_event_type_id=CONGESTION,
                            parent_event_ids=[event.event_id],
                            participants=[EventParticipant(target, "affected_road")],
                            temporal_model_ref="transportation.propagation_hazard.prior_v1",
                            temporal_inputs={
                                "severity": severity,
                                "free_flow_seconds": edge.attributes["free_flow_seconds"],
                                "hazard_start_delay_seconds": edge.attributes[
                                    "free_flow_seconds"
                                ],
                            },
                            context_evidence=ContextEvidence(
                                context_relation_ids=[edge.relation_id],
                                entity_ids=[road_id, target],
                                evaluated_at_offset_seconds=(
                                    event.temporal.occurrence_start_offset_seconds
                                ),
                                state_predicates=[
                                    StatePredicateEvidence(
                                        road_id,
                                        "road_status",
                                        "equals",
                                        "congested",
                                        context.state["road_status"][road_id],
                                        context.state["road_status"][road_id]
                                        == "congested",
                                    ),
                                    StatePredicateEvidence(
                                        target,
                                        "road_status",
                                        "equals",
                                        "normal",
                                        context.state["road_status"][target],
                                        context.state["road_status"][target] == "normal",
                                    ),
                                ],
                            ),
                            attributes={
                                "candidate_kind": "propagation",
                                "source_road_id": road_id,
                                "target_road_id": target,
                                "parent_severity": severity,
                                "depth": depth + 1,
                            },
                            provenance={"fit_status": "synthetic_prior"},
                        )
                    )
            specs.append(
                CandidateSpec(
                    mechanism_id="transportation.congestion_recovery",
                    target_event_type_id=RECOVERY,
                    parent_event_ids=[event.event_id],
                    participants=[EventParticipant(road_id, "affected_road")],
                    temporal_model_ref="transportation.recovery_hazard.prior_v1",
                    temporal_inputs={"severity": severity},
                    attributes={
                        "candidate_kind": "recovery",
                        "source_road_id": road_id,
                    },
                    provenance={"fit_status": "synthetic_prior"},
                    phase_priority=10,
                )
            )
        return specs

    def relation_specs(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
        event_by_id: Dict[str, EventRecord],
    ) -> List[RelationSpec]:
        del context, event_by_id
        if not candidate.parent_event_ids:
            return []
        kind = candidate.attributes["candidate_kind"]
        if kind == "initial_congestion":
            relation_types = {
                candidate.attributes.get("collision_parent_event_id"):
                    "transportation.initiates_congestion",
                candidate.attributes.get("weather_parent_event_id"):
                    "transportation.weather_contributes_to_congestion",
            }
            return [
                RelationSpec(
                    source_event_id=parent_id,
                    relation_class=(
                        "statistical_influence"
                        if parent_id
                        == candidate.attributes.get("weather_parent_event_id")
                        else "causal"
                    ),
                    relation_type_id=relation_types[parent_id],
                    rule_id=candidate.mechanism_id,
                    mechanism_group_id=(
                        candidate.candidate_id
                        if len(candidate.parent_event_ids) > 1
                        else None
                    ),
                    context_evidence=copy.deepcopy(candidate.context_evidence),
                )
                for parent_id in candidate.parent_event_ids
            ]
        elif kind == "propagation":
            relation_class, relation_type = "causal", "transportation.propagates_downstream"
        elif kind == "recovery":
            relation_class, relation_type = "transition", "transportation.transitions_to_recovery"
        elif kind == "weather_congestion":
            relation_class, relation_type = (
                "statistical_influence",
                "transportation.weather_induces_congestion",
            )
        elif kind == "weather_conditioned_collision":
            relation_class, relation_type = (
                "statistical_influence",
                "transportation.weather_contributes_to_collision",
            )
        elif kind == "weather_clear":
            relation_class, relation_type = "transition", "transportation.weather_clears"
        elif kind == "closure_end":
            relation_class, relation_type = "transition", "transportation.closure_reopens"
        else:
            return []
        return [
            RelationSpec(
                source_event_id=parent_id,
                relation_class=relation_class,
                relation_type_id=relation_type,
                rule_id=candidate.mechanism_id,
                mechanism_group_id=(
                    candidate.candidate_id if len(candidate.parent_event_ids) > 1 else None
                ),
                context_evidence=copy.deepcopy(candidate.context_evidence),
            )
            for parent_id in candidate.parent_event_ids
        ]

    def observation_plan(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
    ) -> ObservationPlan:
        del event, candidate, context
        return ObservationPlan(
            observation_model_ref="transportation.observation_delay.prior_v1",
            recording_model_ref="core.immediate",
            source="synthetic_traffic_monitoring_system",
        )

    def _root_spec(
        self,
        context: EpisodeContext,
        road_id: str,
        activation_time: float,
        root_index: int,
    ) -> CandidateSpec:
        return CandidateSpec(
            mechanism_id="transportation.exogenous_incident_arrival",
            target_event_type_id=COLLISION,
            parent_event_ids=[],
            participants=[EventParticipant(road_id, "affected_road")],
            activation_time=activation_time,
            temporal_model_ref="transportation.root_arrival.prior_v1",
            temporal_inputs={
                "rate_per_second": self.config["root_rate_per_hour"] / 3600.0
            },
            attributes={
                "candidate_kind": "root_incident",
                "target_road_id": road_id,
                "root_index": root_index,
            },
            provenance={
                "independent_root": True,
                "fit_status": "synthetic_prior",
                "context_id": context.context_id,
            },
        )

    def _weather_root_spec(self, context: EpisodeContext) -> CandidateSpec:
        region_id = f"{context.context_id}_region_0000"
        return CandidateSpec(
            mechanism_id="transportation.exogenous_weather_arrival",
            target_event_type_id=HEAVY_RAIN_START,
            parent_event_ids=[],
            participants=[EventParticipant(region_id, "affected_region")],
            temporal_model_ref="transportation.weather_arrival.prior_v1",
            temporal_inputs={"rate_per_second": 2.0 / 3600.0},
            attributes={"candidate_kind": "root_weather"},
            provenance={
                "independent_root": True,
                "fit_status": "synthetic_prior",
                "context_id": context.context_id,
            },
        )


def build_transportation_temporal_models() -> TemporalModelRegistry:
    registry = build_standard_temporal_registry()
    registry.register(
        ExponentialArrivalModel(
            "transportation.root_arrival.prior_v1",
            default_rate_per_second=1.5 / 3600.0,
        )
    )
    registry.register(
        ConditionalLogNormalModel(
            "transportation.incident_to_congestion.prior_v1",
            intercept=math.log(150.0),
            sigma=0.35,
            coefficients={
                "severity": -0.35,
                "volume_capacity_ratio": -0.25,
                "weather_severity": -0.2,
            },
            minimum_seconds=20.0,
            maximum_seconds=1800.0,
        )
    )
    registry.register(
        PiecewiseExponentialHazardModel(
            "transportation.propagation_hazard.prior_v1",
            baseline_rate_per_second=1.0 / 260.0,
            coefficients={"severity": 0.8},
        )
    )
    registry.register(
        PiecewiseExponentialHazardModel(
            "transportation.recovery_hazard.prior_v1",
            baseline_rate_per_second=1.0 / 1800.0,
            coefficients={"severity": -0.9},
        )
    )
    registry.register(
        ConditionalLogNormalModel(
            "transportation.observation_delay.prior_v1",
            intercept=math.log(8.0),
            sigma=0.25,
            coefficients={},
            minimum_seconds=1.0,
            maximum_seconds=60.0,
        )
    )
    registry.register(
        ExponentialArrivalModel(
            "transportation.weather_arrival.prior_v1",
            default_rate_per_second=2.0 / 3600.0,
        )
    )
    registry.register(
        ConditionalLogNormalModel(
            "transportation.weather_duration.prior_v1",
            intercept=math.log(2400.0),
            sigma=0.4,
            coefficients={"severity": 0.25},
            minimum_seconds=600.0,
            maximum_seconds=10_800.0,
        )
    )
    registry.register(
        ConditionalLogNormalModel(
            "transportation.weather_to_congestion.prior_v1",
            intercept=math.log(420.0),
            sigma=0.45,
            coefficients={"weather_severity": -0.5},
            minimum_seconds=30.0,
            maximum_seconds=3600.0,
        )
    )
    registry.register(
        ConditionalLogNormalModel(
            "transportation.weather_to_collision.prior_v1",
            intercept=math.log(600.0),
            sigma=0.5,
            coefficients={"weather_severity": -0.45},
            minimum_seconds=45.0,
            maximum_seconds=3600.0,
        )
    )
    registry.register(
        ConditionalLogNormalModel(
            "transportation.closure_duration.prior_v1",
            intercept=math.log(1800.0),
            sigma=0.3,
            coefficients={},
            minimum_seconds=600.0,
            maximum_seconds=7200.0,
        )
    )
    return registry


def _build_transport_context(
    context_id: str,
    node_count: int,
    seed: int,
    duration_seconds: float,
    topology_config: Dict[str, Any],
    domain_spec: LoadedDomainSpec,
) -> tuple[
    List[Entity],
    List[ContextRelation],
    Dict[str, Any],
]:
    rng = random.Random(seed)
    entities: List[Entity] = [
        Entity(
            entity_id=f"{context_id}_region_0000",
            context_id=context_id,
            domain="transportation",
            entity_type_id="transportation.region",
            attributes={"name": "synthetic_region"},
            provenance={"generation_role": "context_container"},
        )
    ]
    road_ids: List[str] = []
    for index in range(node_count):
        road_id = f"{context_id}_road_{index:06d}"
        road_ids.append(road_id)
        length_km = round(rng.uniform(0.3, 2.5), 4)
        speed_kph = rng.choice([40, 50, 60, 70, 80])
        capacity = rng.randint(900, 2400)
        vc_ratio = rng.uniform(0.35, 1.05)
        entities.append(
            Entity(
                entity_id=road_id,
                context_id=context_id,
                domain="transportation",
                entity_type_id="transportation.road.segment",
                attributes={
                    "region_id": f"{context_id}_region_0000",
                    "length_km": length_km,
                    "speed_limit_kph": speed_kph,
                    "capacity_vehicles_per_hour": capacity,
                    "baseline_volume_capacity_ratio": round(vc_ratio, 6),
                    "functional_class": rng.choice(
                        ["local", "collector", "arterial", "express"]
                    ),
                },
                provenance={"fit_status": "synthetic_prior"},
            )
        )

    profile_rng = random.Random(seed ^ 0x70F0109)
    community_low, community_high = topology_config["community_count_range"]
    community_minimum = max(1, min(node_count, int(community_low)))
    community_maximum = max(
        community_minimum,
        min(node_count, int(community_high)),
    )
    community_count = profile_rng.randint(
        community_minimum,
        community_maximum,
    )
    sampled_profile = {
        "profile_id": f"{context_id}_irregular_profile",
        "generator_family": "constrained_degree_weighted_block",
        "expected_out_degree": _sample_profile_range(
            topology_config, "expected_out_degree_range", profile_rng
        ),
        "max_out_degree": int(topology_config["max_out_degree"]),
        "community_count": community_count,
        "within_community_bias": _sample_profile_range(
            topology_config, "within_community_bias_range", profile_rng
        ),
        "degree_sigma": _sample_profile_range(
            topology_config, "degree_sigma_range", profile_rng
        ),
        "hub_fraction": _sample_profile_range(
            topology_config, "hub_fraction_range", profile_rng
        ),
        "hub_multiplier": _sample_profile_range(
            topology_config, "hub_multiplier_range", profile_rng
        ),
        "connectivity_constraint": "weakly_connected_for_this_relation_layer",
        "spatial_coordinates_required": False,
    }

    def relation_attributes(
        source: Entity, target: Entity, edge_rng: random.Random
    ) -> Dict[str, Any]:
        source_length = float(source.attributes["length_km"])
        target_length = float(target.attributes["length_km"])
        effective_distance = max(
            0.05,
            (source_length + target_length)
            * 0.5
            * edge_rng.uniform(0.35, 1.35),
        )
        effective_speed = min(
            float(source.attributes["speed_limit_kph"]),
            float(target.attributes["speed_limit_kph"]),
        )
        return {
            "distance_km": round(effective_distance, 6),
            "free_flow_seconds": round(
                effective_distance / effective_speed * 3600.0, 6
            ),
            "coupling_strength": round(edge_rng.uniform(0.25, 1.0), 6),
        }

    layer = domain_spec.compile_relation_layers(
        parameter_overrides={
            "transportation.road.downstream_of": {
                "expected_out_degree": sampled_profile["expected_out_degree"],
                "max_out_degree": sampled_profile["max_out_degree"],
                "community_count": sampled_profile["community_count"],
                "within_community_bias": sampled_profile["within_community_bias"],
                "degree_sigma": sampled_profile["degree_sigma"],
                "hub_fraction": sampled_profile["hub_fraction"],
                "hub_multiplier": sampled_profile["hub_multiplier"],
            }
        },
        attribute_builders={
            "transportation.downstream_relation.v1": relation_attributes
        },
    )[0]
    generated = SparseHeterogeneousTopologyGenerator().generate(
        context_id=context_id,
        domain="transportation",
        entities=entities,
        layers=[layer],
        seed=seed ^ 0x1A2B3C4D,
        observation_window_seconds=duration_seconds,
    )
    sampled_profile["observed_statistics"] = generated.statistics
    return entities, generated.relations, sampled_profile


def _sample_profile_range(
    profile: Dict[str, Any], key: str, rng: random.Random
) -> float:
    lower, upper = (float(value) for value in profile[key])
    return round(rng.uniform(lower, upper), 6)


def _validate_topology_profile(profile: Dict[str, Any]) -> None:
    range_keys = (
        "expected_out_degree_range",
        "community_count_range",
        "within_community_bias_range",
        "degree_sigma_range",
        "hub_fraction_range",
        "hub_multiplier_range",
    )
    for key in range_keys:
        values = profile.get(key)
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"{key} must contain [minimum, maximum]")
        lower, upper = (float(value) for value in values)
        if lower > upper:
            raise ValueError(f"{key} minimum exceeds maximum")
    if float(profile["expected_out_degree_range"][0]) < 1.0:
        raise ValueError(
            "A weakly connected directed layer needs expected_out_degree >= 1"
        )
    if int(profile["max_out_degree"]) < 2:
        raise ValueError("max_out_degree must be at least two")
    if not 0.0 <= float(profile["hub_fraction_range"][0]) <= 1.0:
        raise ValueError("hub_fraction_range must lie in [0, 1]")
    if not 0.0 <= float(profile["hub_fraction_range"][1]) <= 1.0:
        raise ValueError("hub_fraction_range must lie in [0, 1]")
