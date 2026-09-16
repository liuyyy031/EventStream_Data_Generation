"""Synthetic healthcare workflows for the domain-neutral event scheduler.

The package generates auditable inpatient monitoring and review workflows.  It
does not diagnose patients, recommend treatment, or claim empirical clinical
realism. Numeric and temporal parameters are transparent synthetic priors.
"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from generation_core.domain import (
    CandidateSpec,
    CandidateUpdate,
    EpisodeContext,
    ObservationPlan,
    RelationSpec,
)
from generation_core.domain_spec import load_domain_spec
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
    TemporalModelRegistry,
    build_standard_temporal_registry,
)
from generation_core.topology import SparseHeterogeneousTopologyGenerator


TEMPERATURE = "healthcare.observation.vital_sign.temperature_recorded"
OXYGEN_SATURATION = "healthcare.observation.vital_sign.oxygen_saturation_recorded"
SPECIMEN_COLLECTION = "healthcare.specimen.collection.completed"
LAB_RESULT = "healthcare.observation.laboratory.result_recorded"
THRESHOLD_FLAG = "healthcare.finding.configured_threshold_flag.created"
MONITORING_ALERT = "healthcare.workflow.monitoring_alert.created"
CLINICAL_REVIEW = "healthcare.workflow.clinical_review.completed"
CARE_PLAN_ADJUSTED = "healthcare.intervention.care_plan.adjusted"
FOLLOW_UP = "healthcare.observation.follow_up.recorded"
FOLLOW_UP_STATUS = "healthcare.finding.follow_up_status.assessed"

EVENT_TYPE_IDS = (
    TEMPERATURE,
    OXYGEN_SATURATION,
    SPECIMEN_COLLECTION,
    LAB_RESULT,
    THRESHOLD_FLAG,
    MONITORING_ALERT,
    CLINICAL_REVIEW,
    CARE_PLAN_ADJUSTED,
    FOLLOW_UP,
    FOLLOW_UP_STATUS,
)

SCENARIO_FAMILIES = (
    "temperature_escalation",
    "oxygen_desaturation_response",
    "laboratory_abnormality_review",
    "combined_vital_sign_escalation",
    "routine_observation",
)

RELATION_TEXT_RENDERINGS = {
    "healthcare.collection_produces_result": {
        "surface_predicate": "entered the laboratory workflow that produced",
        "asserted_relation_class": "workflow",
    },
    "healthcare.observation_derives_threshold_flag": {
        "surface_predicate": "was deterministically evaluated into",
        "asserted_relation_class": "derivation",
    },
    "healthcare.flag_derives_alert": {
        "surface_predicate": "was used by the configured rule to create",
        "asserted_relation_class": "derivation",
    },
    "healthcare.alert_routes_to_review": {
        "surface_predicate": "entered the workflow that routed to",
        "asserted_relation_class": "workflow",
    },
    "healthcare.review_routes_to_care_plan": {
        "surface_predicate": "was followed in the configured workflow by",
        "asserted_relation_class": "workflow",
    },
    "healthcare.care_plan_schedules_follow_up": {
        "surface_predicate": "scheduled the workflow follow-up",
        "asserted_relation_class": "workflow",
    },
    "healthcare.routine_observation_schedules_follow_up": {
        "surface_predicate": "scheduled the routine follow-up",
        "asserted_relation_class": "workflow",
    },
    "healthcare.follow_up_derives_status": {
        "surface_predicate": "was deterministically assessed as",
        "asserted_relation_class": "derivation",
    },
}


class HealthcarePackage:
    domain_id = "healthcare"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.domain_spec = load_domain_spec(Path(__file__).with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(
            Path(__file__).with_name("mechanism_specs.json"),
            known_event_type_ids=EVENT_TYPE_IDS,
            known_context_relation_type_ids=self.domain_spec.context_relation_type_ids,
        )
        supplied = config or {}
        self.config = {
            "base_seed": int(supplied.get("base_seed", 20260916)),
            "context_count": int(supplied.get("context_count", 4)),
            "nodes_per_context": int(supplied.get("nodes_per_context", 1000)),
            "duration_seconds": float(supplied.get("duration_seconds", 28_800.0)),
            "start_time": str(supplied.get("start_time", "2026-09-16T08:00:00+08:00")),
            "scenario_weights": dict(
                supplied.get(
                    "scenario_weights",
                    {
                        "temperature_escalation": 0.25,
                        "oxygen_desaturation_response": 0.25,
                        "laboratory_abnormality_review": 0.2,
                        "combined_vital_sign_escalation": 0.2,
                        "routine_observation": 0.1,
                    },
                )
            ),
        }
        if self.config["nodes_per_context"] < 12:
            raise ValueError("Healthcare contexts require at least 12 entities")
        if self.config["context_count"] <= 0:
            raise ValueError("context_count must be positive")
        if self.config["duration_seconds"] <= 0.0:
            raise ValueError("duration_seconds must be positive")
        unknown = set(self.config["scenario_weights"]) - set(SCENARIO_FAMILIES)
        if unknown:
            raise ValueError(f"Unknown healthcare scenarios: {sorted(unknown)}")
        if not self.config["scenario_weights"] or any(
            float(value) < 0.0 for value in self.config["scenario_weights"].values()
        ):
            raise ValueError("scenario_weights must be non-empty and non-negative")
        if sum(float(value) for value in self.config["scenario_weights"].values()) <= 0.0:
            raise ValueError("At least one scenario weight must be positive")
        self._context_cache: Dict[int, tuple[List[Entity], List[ContextRelation], Dict[str, Any]]] = {}

    def catalog(self) -> Dict[str, Any]:
        entity_types = [
            ("healthcare.patient", ["synthetic_patient_index", "reference_profile_id"]),
            ("healthcare.encounter", ["encounter_class", "status"]),
            ("healthcare.device", ["device_kind", "measurement_capabilities"]),
            ("healthcare.specimen", ["specimen_kind", "status"]),
            ("healthcare.practitioner", ["practitioner_role"]),
            ("healthcare.location", ["location_kind"]),
        ]
        event_types = [
            (TEMPERATURE, "healthcare.observation.vital_sign", ["measurement_kind", "numeric_value", "unit", "interpretation", "reference_profile_id"]),
            (OXYGEN_SATURATION, "healthcare.observation.vital_sign", ["measurement_kind", "numeric_value", "unit", "interpretation", "reference_profile_id"]),
            (SPECIMEN_COLLECTION, "healthcare.specimen.collection", ["specimen_kind", "collection_method"]),
            (LAB_RESULT, "healthcare.observation.laboratory", ["measurement_kind", "numeric_value", "unit", "interpretation", "reference_profile_id"]),
            (THRESHOLD_FLAG, "healthcare.finding", ["flag_code", "source_measurement_kind", "interpretation", "reference_profile_id"]),
            (MONITORING_ALERT, "healthcare.workflow.alert", ["signal_kinds", "urgency_level", "rule_profile_id"]),
            (CLINICAL_REVIEW, "healthcare.workflow.review", ["review_outcome", "signal_count"]),
            (CARE_PLAN_ADJUSTED, "healthcare.intervention.care_plan", ["action_class", "prescriptive_detail_included"]),
            (FOLLOW_UP, "healthcare.observation", ["measurement_kind", "numeric_value", "unit", "interpretation", "reference_profile_id"]),
            (FOLLOW_UP_STATUS, "healthcare.finding", ["status", "assessment_basis", "reference_profile_id"]),
        ]
        return {
            "domain": self.domain_id,
            "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_pending_clinical_calibration",
            "clinical_safety_scope": {
                "diagnostic_claims": False,
                "medication_or_dose_recommendations": False,
                "treatment_effectiveness_claims": False,
                "intended_use": "llm_event_stream_understanding_research",
            },
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {
                "mode": "deterministic_grounded_projection",
                "free_form_causal_completion": False,
                "alignment_unit": "sentence_to_event_or_relation_ids",
            },
            "entity_types": [
                {"entity_type_id": type_id, "parent_type_id": None, "required_attributes": attributes}
                for type_id, attributes in entity_types
            ],
            "event_types": [
                {"event_type_id": type_id, "parent_type_id": parent, "required_attributes": attributes}
                for type_id, parent, attributes in event_types
            ],
            "context_relation_types": [
                {
                    "relation_type_id": relation_id,
                    "directed": True,
                    "meaning": "typed_care_context_only",
                    "does_not_imply_event_parenthood": True,
                }
                for relation_id in self.domain_spec.context_relation_type_ids
            ],
            "event_relation_types": [
                {
                    "relation_type_id": relation_id,
                    "relation_class": rendering["asserted_relation_class"],
                    "text_rendering": dict(rendering),
                }
                for relation_id, rendering in RELATION_TEXT_RENDERINGS.items()
            ],
            "mechanisms": self.mechanism_registry.descriptors(),
        }

    def render_episode_text(self, result: EpisodeResult) -> Dict[str, Any]:
        labels = {
            TEMPERATURE: "a temperature observation",
            OXYGEN_SATURATION: "an oxygen-saturation observation",
            SPECIMEN_COLLECTION: "a specimen collection",
            LAB_RESULT: "a laboratory observation",
            THRESHOLD_FLAG: "a configured-threshold flag",
            MONITORING_ALERT: "a monitoring alert",
            CLINICAL_REVIEW: "a clinical workflow review",
            CARE_PLAN_ADJUSTED: "a non-prescriptive care-plan adjustment",
            FOLLOW_UP: "a follow-up observation",
            FOLLOW_UP_STATUS: "a follow-up status assessment",
        }
        sentences: List[str] = []
        claims: List[Dict[str, Any]] = []
        if not result.events:
            sentences.append("No event occurrence was generated inside this observation window.")
        for event in result.events:
            participants = ", ".join(f"{item.role}={item.entity_id}" for item in event.participants)
            attributes = ", ".join(f"{key}={value}" for key, value in sorted(event.attributes.items()))
            sentences.append(
                f"At +{event.temporal.occurrence_start_offset_seconds:.3f} seconds, "
                f"{labels[event.event_type_id]} occurred ({participants}; {attributes})."
            )
            claims.append({
                "claim_id": f"{result.episode_id}_text_claim_{len(claims) + 1:06d}",
                "sentence_index": len(sentences) - 1,
                "event_ids": [event.event_id],
                "relation_ids": [],
                "evidence_fields": ["event_type_id", "temporal.occurrence_start_offset_seconds", "participants", "attributes"],
            })
        for relation in result.event_relations:
            rendering = RELATION_TEXT_RENDERINGS[relation.relation_type_id]
            predicate = rendering["surface_predicate"]
            sentences.append(
                f"Generated rule {relation.rule_id} states that event {relation.source_event_id} "
                f"{predicate} event {relation.target_event_id}, with an occurrence-start lag of "
                f"{relation.temporal_link.lag_seconds:.3f} seconds."
            )
            claims.append({
                "claim_id": f"{result.episode_id}_text_claim_{len(claims) + 1:06d}",
                "sentence_index": len(sentences) - 1,
                "event_ids": [relation.source_event_id, relation.target_event_id],
                "relation_ids": [relation.relation_id],
                "relation_assertions": [{
                    "relation_id": relation.relation_id,
                    "asserted_relation_class": rendering["asserted_relation_class"],
                    "surface_predicate": predicate,
                }],
                "evidence_fields": ["rule_id", "relation_type_id", "source_event_id", "target_event_id", "temporal_link"],
            })
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
        context_id = f"healthcare_context_{context_index:03d}"
        if context_index not in self._context_cache:
            self._context_cache[context_index] = _build_healthcare_context(
                context_id,
                self.config["nodes_per_context"],
                self.config["base_seed"] + context_index * 7919,
                self.config["duration_seconds"],
                self.domain_spec,
            )
        entities, relations, topology = self._context_cache[context_index]
        rng = random.Random(seed ^ 0x4845414C5448)
        scenarios = list(self.config["scenario_weights"])
        scenario = rng.choices(
            scenarios,
            weights=[self.config["scenario_weights"][name] for name in scenarios],
            k=1,
        )[0]
        bundle = _choose_care_bundle(entities, relations, rng)
        return EpisodeContext(
            episode_id=f"healthcare_episode_{episode_index:07d}",
            context_id=context_id,
            domain=self.domain_id,
            start_time=self.config["start_time"],
            duration_seconds=self.config["duration_seconds"],
            entities=entities,
            context_relations=relations,
            context_attributes={"topology_profile": topology},
            episode_attributes={
                "scenario_family": scenario,
                "reference_profile_id": "synthetic_monitoring_profile_v1",
                "clinical_realism_claimed": False,
            },
            state={
                "scenario_family": scenario,
                "bundle": bundle,
                "active_flag_event_ids": {},
                "alert_event_id": None,
                "review_event_id": None,
                "care_plan_event_id": None,
                "primary_measurement_kind": None,
                "workload_level": round(rng.uniform(0.2, 0.9), 4),
            },
        )

    def seed_candidates(self, context: EpisodeContext, rng: random.Random) -> List[CandidateSpec]:
        del rng
        scenario = context.state["scenario_family"]
        initial_time = 0.08 * context.duration_seconds
        if scenario == "temperature_escalation":
            return [self._measurement_root(context, "temperature", initial_time)]
        if scenario == "oxygen_desaturation_response":
            return [self._measurement_root(context, "oxygen_saturation", initial_time)]
        if scenario == "combined_vital_sign_escalation":
            return [
                self._measurement_root(context, "temperature", initial_time),
                self._measurement_root(context, "oxygen_saturation", initial_time + 45.0),
            ]
        if scenario == "laboratory_abnormality_review":
            return [self._specimen_root(context, initial_time)]
        if scenario == "routine_observation":
            return [self._measurement_root(context, "temperature", initial_time)]
        raise ValueError(f"Unsupported healthcare scenario: {scenario}")

    def revalidate_candidate(
        self,
        candidate: Candidate,
        context: EpisodeContext,
        event_by_id: Dict[str, EventRecord],
        now: float,
    ) -> CandidateUpdate:
        kind = str(candidate.attributes.get("candidate_kind"))
        if kind in {"root_measurement", "root_specimen_collection"}:
            return CandidateUpdate(valid=True)
        parents_exist = all(parent_id in event_by_id for parent_id in candidate.parent_event_ids)
        predicates = [StatePredicateEvidence(None, "parent_events_materialized", "equals", True, parents_exist, parents_exist)]
        if kind == "monitoring_alert":
            valid = parents_exist and context.state["alert_event_id"] is None
            predicates.append(StatePredicateEvidence(None, "alert_event_id", "equals", None, context.state["alert_event_id"], context.state["alert_event_id"] is None))
        elif kind == "clinical_review":
            valid = parents_exist and context.state["review_event_id"] is None
        elif kind == "care_plan_adjustment":
            valid = parents_exist and context.state["care_plan_event_id"] is None
        else:
            valid = parents_exist
        evidence = copy.deepcopy(candidate.context_evidence)
        evidence.evaluated_at_offset_seconds = now
        evidence.state_predicates = predicates
        return CandidateUpdate(valid=valid, reason="healthcare_workflow_precondition_became_false", context_evidence=evidence)

    def materialize_event(
        self,
        event_id: str,
        candidate: Candidate,
        context: EpisodeContext,
        rng: random.Random,
    ) -> EventRecord:
        kind = str(candidate.attributes["candidate_kind"])
        measurement_kind = candidate.attributes.get("measurement_kind")
        if kind == "root_measurement":
            attributes = _measurement_attributes(
                str(measurement_kind),
                abnormal=(context.state["scenario_family"] != "routine_observation"),
                rng=rng,
                phase="initial",
            )
            role = "root"
        elif kind == "root_specimen_collection":
            attributes = {"specimen_kind": "synthetic_blood_specimen", "collection_method": "configured_venous_collection_workflow"}
            role = "root"
        elif kind == "lab_result":
            attributes = _measurement_attributes("synthetic_inflammation_marker", abnormal=True, rng=rng, phase="initial")
            role = "workflow_result"
        elif kind == "threshold_flag":
            attributes = {
                "flag_code": "outside_configured_reference_band",
                "source_measurement_kind": str(measurement_kind),
                "interpretation": "configured_abnormal",
                "reference_profile_id": "synthetic_monitoring_profile_v1",
            }
            role = "derived"
        elif kind == "monitoring_alert":
            signal_kinds = list(candidate.attributes["signal_kinds"])
            attributes = {"signal_kinds": signal_kinds, "urgency_level": 2 if len(signal_kinds) > 1 else 1, "rule_profile_id": "synthetic_escalation_policy_v1"}
            role = "derived"
        elif kind == "clinical_review":
            attributes = {"review_outcome": "monitor_and_reassess", "signal_count": int(candidate.attributes["signal_count"])}
            role = "workflow"
        elif kind == "care_plan_adjustment":
            attributes = {"action_class": "non_prescriptive_monitoring_plan_adjustment", "prescriptive_detail_included": False}
            role = "workflow"
        elif kind in {"follow_up_observation", "routine_follow_up"}:
            attributes = _measurement_attributes(str(measurement_kind or "clinical_status_index"), abnormal=False, rng=rng, phase="follow_up")
            role = "observation"
        elif kind == "follow_up_status":
            attributes = {"status": "within_configured_reference_band", "assessment_basis": "immediately_preceding_follow_up_observation", "reference_profile_id": "synthetic_monitoring_profile_v1"}
            role = "derived"
        else:
            raise ValueError(f"Unsupported healthcare candidate kind: {kind}")
        return EventRecord(
            event_id=event_id,
            episode_id=context.episode_id,
            domain=self.domain_id,
            event_type_id=candidate.target_event_type_id,
            temporal=TemporalExtent(float(candidate.scheduled_time)),
            participants=list(candidate.participants),
            attributes=attributes,
            event_role=role,
            provenance={
                "mechanism_id": candidate.mechanism_id,
                "candidate_id": candidate.candidate_id,
                "fit_status": "transparent_synthetic_prior_pending_clinical_calibration",
                "not_for_clinical_decision_making": True,
            },
        )

    def apply_event(self, event: EventRecord, candidate: Candidate, context: EpisodeContext) -> None:
        kind = str(candidate.attributes["candidate_kind"])
        if kind == "threshold_flag":
            measurement_kind = str(candidate.attributes["measurement_kind"])
            context.state["active_flag_event_ids"][measurement_kind] = event.event_id
            if context.state["primary_measurement_kind"] is None:
                context.state["primary_measurement_kind"] = measurement_kind
        elif kind == "monitoring_alert":
            context.state["alert_event_id"] = event.event_id
        elif kind == "clinical_review":
            context.state["review_event_id"] = event.event_id
        elif kind == "care_plan_adjustment":
            context.state["care_plan_event_id"] = event.event_id

    def spawn_candidates(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
        event_by_id: Dict[str, EventRecord],
        rng: random.Random,
    ) -> List[CandidateSpec]:
        del event_by_id, rng
        kind = str(candidate.attributes["candidate_kind"])
        bundle = context.state["bundle"]
        patient_id = bundle["patient_id"]
        encounter_id = bundle["encounter_id"]
        device_id = bundle["device_id"]
        practitioner_id = bundle["practitioner_id"]
        common = [EventParticipant(patient_id, "subject"), EventParticipant(encounter_id, "encounter")]
        now = event.temporal.occurrence_start_offset_seconds
        if kind == "root_specimen_collection":
            return [CandidateSpec(
                mechanism_id="healthcare.specimen_to_lab_result",
                target_event_type_id=LAB_RESULT,
                parent_event_ids=[event.event_id],
                participants=common + [EventParticipant(bundle["specimen_id"], "specimen")],
                temporal_model_ref="healthcare.lab_turnaround.prior_v1",
                temporal_inputs={"priority": 1.0},
                attributes={"candidate_kind": "lab_result", "measurement_kind": "synthetic_inflammation_marker"},
                context_evidence=_evidence(context, ["specimen_relation_id"], [bundle["specimen_id"], patient_id], now),
                provenance=_synthetic_provenance(),
            )]
        if kind in {"root_measurement", "lab_result"}:
            measurement_kind = str(candidate.attributes.get("measurement_kind", event.attributes.get("measurement_kind")))
            if kind == "root_measurement" and context.state["scenario_family"] == "routine_observation":
                return [CandidateSpec(
                    mechanism_id="healthcare.routine_repeat_observation",
                    target_event_type_id=FOLLOW_UP,
                    parent_event_ids=[event.event_id],
                    participants=common + [EventParticipant(device_id, "measurement_device")],
                    temporal_model_ref="core.scheduled",
                    temporal_inputs={"scheduled_offset_seconds": now + 3600.0},
                    combination="single",
                    attributes={"candidate_kind": "routine_follow_up", "measurement_kind": measurement_kind},
                    context_evidence=_evidence(context, ["device_relation_id"], [device_id, patient_id], now),
                    provenance=_synthetic_provenance(policy=True),
                )]
            return [CandidateSpec(
                mechanism_id="healthcare.observation_to_threshold_flag",
                target_event_type_id=THRESHOLD_FLAG,
                parent_event_ids=[event.event_id],
                participants=common,
                temporal_model_ref="core.immediate",
                temporal_inputs={},
                attributes={"candidate_kind": "threshold_flag", "measurement_kind": measurement_kind},
                provenance=_synthetic_provenance(policy=True),
                phase_priority=10,
            )]
        if kind == "threshold_flag":
            flags = dict(context.state["active_flag_event_ids"])
            combined = context.state["scenario_family"] == "combined_vital_sign_escalation"
            if combined and len(flags) < 2:
                return []
            parent_ids = sorted(flags.values()) if combined else [event.event_id]
            signal_kinds = sorted(flags) if combined else [str(candidate.attributes["measurement_kind"])]
            return [CandidateSpec(
                mechanism_id="healthcare.multisignal_flags_to_alert" if len(parent_ids) > 1 else "healthcare.flag_to_alert",
                target_event_type_id=MONITORING_ALERT,
                parent_event_ids=parent_ids,
                participants=common,
                temporal_model_ref="healthcare.alert_processing.policy_v1",
                temporal_inputs={},
                combination="all_of" if len(parent_ids) > 1 else "single",
                attributes={"candidate_kind": "monitoring_alert", "signal_kinds": signal_kinds},
                provenance=_synthetic_provenance(policy=True),
                phase_priority=10,
            )]
        if kind == "monitoring_alert":
            urgency = 2.0 if len(candidate.attributes["signal_kinds"]) > 1 else 1.0
            return [CandidateSpec(
                mechanism_id="healthcare.alert_to_clinical_review",
                target_event_type_id=CLINICAL_REVIEW,
                parent_event_ids=[event.event_id],
                participants=common + [EventParticipant(practitioner_id, "reviewer")],
                temporal_model_ref="healthcare.review_delay.prior_v1",
                temporal_inputs={"urgency_level": urgency, "workload_level": context.state["workload_level"]},
                attributes={"candidate_kind": "clinical_review", "signal_count": len(candidate.attributes["signal_kinds"]), "urgency_level": urgency},
                context_evidence=_evidence(context, ["practitioner_relation_id"], [practitioner_id, patient_id], now),
                provenance=_synthetic_provenance(),
            )]
        if kind == "clinical_review":
            return [CandidateSpec(
                mechanism_id="healthcare.review_to_care_plan_adjustment",
                target_event_type_id=CARE_PLAN_ADJUSTED,
                parent_event_ids=[event.event_id],
                participants=common + [EventParticipant(practitioner_id, "responsible_practitioner")],
                temporal_model_ref="healthcare.care_plan_delay.prior_v1",
                temporal_inputs={"urgency_level": float(candidate.attributes["urgency_level"])},
                attributes={"candidate_kind": "care_plan_adjustment"},
                provenance=_synthetic_provenance(),
            )]
        if kind == "care_plan_adjustment":
            measurement_kind = str(context.state["primary_measurement_kind"] or "clinical_status_index")
            return [CandidateSpec(
                mechanism_id="healthcare.care_plan_to_follow_up_observation",
                target_event_type_id=FOLLOW_UP,
                parent_event_ids=[event.event_id],
                participants=common + [EventParticipant(device_id, "measurement_device")],
                temporal_model_ref="healthcare.follow_up_delay.prior_v1",
                temporal_inputs={"urgency_level": 1.0},
                attributes={"candidate_kind": "follow_up_observation", "measurement_kind": measurement_kind},
                context_evidence=_evidence(context, ["device_relation_id"], [device_id, patient_id], now),
                provenance=_synthetic_provenance(),
            )]
        if kind in {"follow_up_observation", "routine_follow_up"}:
            return [CandidateSpec(
                mechanism_id="healthcare.follow_up_to_status_assessment",
                target_event_type_id=FOLLOW_UP_STATUS,
                parent_event_ids=[event.event_id],
                participants=common,
                temporal_model_ref="core.immediate",
                temporal_inputs={},
                attributes={"candidate_kind": "follow_up_status"},
                provenance=_synthetic_provenance(policy=True),
                phase_priority=10,
            )]
        return []

    def relation_specs(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
        event_by_id: Dict[str, EventRecord],
    ) -> List[RelationSpec]:
        del event, context, event_by_id
        if not candidate.parent_event_ids:
            return []
        mapping = {
            "lab_result": ("workflow", "healthcare.collection_produces_result"),
            "threshold_flag": ("derivation", "healthcare.observation_derives_threshold_flag"),
            "monitoring_alert": ("derivation", "healthcare.flag_derives_alert"),
            "clinical_review": ("workflow", "healthcare.alert_routes_to_review"),
            "care_plan_adjustment": ("workflow", "healthcare.review_routes_to_care_plan"),
            "follow_up_observation": ("workflow", "healthcare.care_plan_schedules_follow_up"),
            "routine_follow_up": ("workflow", "healthcare.routine_observation_schedules_follow_up"),
            "follow_up_status": ("derivation", "healthcare.follow_up_derives_status"),
        }
        kind = str(candidate.attributes["candidate_kind"])
        if kind not in mapping:
            return []
        relation_class, relation_type = mapping[kind]
        return [RelationSpec(
            source_event_id=parent_id,
            relation_class=relation_class,
            relation_type_id=relation_type,
            rule_id=candidate.mechanism_id,
            mechanism_group_id=candidate.candidate_id if len(candidate.parent_event_ids) > 1 else None,
            context_evidence=copy.deepcopy(candidate.context_evidence),
        ) for parent_id in candidate.parent_event_ids]

    def observation_plan(self, event: EventRecord, candidate: Candidate, context: EpisodeContext) -> ObservationPlan:
        del candidate, context
        if event.event_type_id in {TEMPERATURE, OXYGEN_SATURATION, LAB_RESULT, FOLLOW_UP}:
            return ObservationPlan(
                observation_model_ref="healthcare.measurement_observation_delay.prior_v1",
                recording_model_ref="healthcare.recording_delay.prior_v1",
                source="synthetic_clinical_information_system",
            )
        return ObservationPlan(observation_model_ref="core.immediate", recording_model_ref="core.immediate", source="synthetic_clinical_workflow_system")

    def _measurement_root(self, context: EpisodeContext, measurement_kind: str, scheduled_time: float) -> CandidateSpec:
        bundle = context.state["bundle"]
        event_type = TEMPERATURE if measurement_kind == "temperature" else OXYGEN_SATURATION
        mechanism = "healthcare.scheduled_temperature_observation" if measurement_kind == "temperature" else "healthcare.scheduled_oxygen_saturation_observation"
        return CandidateSpec(
            mechanism_id=mechanism,
            target_event_type_id=event_type,
            parent_event_ids=[],
            participants=[
                EventParticipant(bundle["patient_id"], "subject"),
                EventParticipant(bundle["encounter_id"], "encounter"),
                EventParticipant(bundle["device_id"], "measurement_device"),
            ],
            temporal_model_ref="core.scheduled",
            temporal_inputs={"scheduled_offset_seconds": scheduled_time},
            combination="none",
            attributes={"candidate_kind": "root_measurement", "measurement_kind": measurement_kind},
            context_evidence=_evidence(context, ["device_relation_id", "encounter_relation_id"], [bundle["device_id"], bundle["patient_id"], bundle["encounter_id"]], 0.0),
            provenance=_synthetic_provenance(policy=True),
        )

    def _specimen_root(self, context: EpisodeContext, scheduled_time: float) -> CandidateSpec:
        bundle = context.state["bundle"]
        return CandidateSpec(
            mechanism_id="healthcare.scheduled_specimen_collection",
            target_event_type_id=SPECIMEN_COLLECTION,
            parent_event_ids=[],
            participants=[
                EventParticipant(bundle["patient_id"], "subject"),
                EventParticipant(bundle["encounter_id"], "encounter"),
                EventParticipant(bundle["specimen_id"], "specimen"),
            ],
            temporal_model_ref="core.scheduled",
            temporal_inputs={"scheduled_offset_seconds": scheduled_time},
            combination="none",
            attributes={"candidate_kind": "root_specimen_collection"},
            context_evidence=_evidence(context, ["specimen_relation_id", "encounter_relation_id"], [bundle["specimen_id"], bundle["patient_id"], bundle["encounter_id"]], 0.0),
            provenance=_synthetic_provenance(policy=True),
        )


def build_healthcare_temporal_models() -> TemporalModelRegistry:
    registry = build_standard_temporal_registry()
    registry.register(DeterministicDelayModel("healthcare.alert_processing.policy_v1", 15.0))
    registry.register(ConditionalLogNormalModel("healthcare.lab_turnaround.prior_v1", intercept=math.log(1800.0), sigma=0.4, coefficients={"priority": -0.35}, minimum_seconds=300.0, maximum_seconds=7200.0))
    registry.register(ConditionalLogNormalModel("healthcare.review_delay.prior_v1", intercept=math.log(600.0), sigma=0.5, coefficients={"urgency_level": -0.35, "workload_level": 0.25}, minimum_seconds=60.0, maximum_seconds=3600.0))
    registry.register(ConditionalLogNormalModel("healthcare.care_plan_delay.prior_v1", intercept=math.log(300.0), sigma=0.4, coefficients={"urgency_level": -0.25}, minimum_seconds=30.0, maximum_seconds=1800.0))
    registry.register(ConditionalLogNormalModel("healthcare.follow_up_delay.prior_v1", intercept=math.log(1800.0), sigma=0.4, coefficients={"urgency_level": -0.2}, minimum_seconds=600.0, maximum_seconds=7200.0))
    registry.register(ConditionalLogNormalModel("healthcare.measurement_observation_delay.prior_v1", intercept=math.log(8.0), sigma=0.3, coefficients={}, minimum_seconds=1.0, maximum_seconds=60.0))
    registry.register(ConditionalLogNormalModel("healthcare.recording_delay.prior_v1", intercept=math.log(20.0), sigma=0.45, coefficients={}, minimum_seconds=2.0, maximum_seconds=300.0))
    return registry


def _build_healthcare_context(context_id: str, node_count: int, seed: int, duration_seconds: float, domain_spec: Any) -> tuple[List[Entity], List[ContextRelation], Dict[str, Any]]:
    rng = random.Random(seed)
    type_counts = _allocate_entity_counts(node_count)
    entities: List[Entity] = []
    for type_name, count in type_counts.items():
        for index in range(count):
            entity_id = f"{context_id}_{type_name}_{index:06d}"
            if type_name == "patient":
                attributes = {"synthetic_patient_index": index, "reference_profile_id": "synthetic_monitoring_profile_v1"}
            elif type_name == "encounter":
                attributes = {"encounter_class": "inpatient", "status": "active"}
            elif type_name == "device":
                attributes = {"device_kind": "multiparameter_monitor", "measurement_capabilities": ["temperature", "oxygen_saturation", "clinical_status_index"]}
            elif type_name == "specimen":
                attributes = {"specimen_kind": "synthetic_blood_specimen", "status": "available_for_configured_collection"}
            elif type_name == "practitioner":
                attributes = {"practitioner_role": rng.choice(["registered_nurse", "physician", "clinical_reviewer"])}
            else:
                attributes = {"location_kind": rng.choice(["general_ward", "observation_unit", "step_down_unit"])}
            entities.append(Entity(
                entity_id=entity_id,
                context_id=context_id,
                domain="healthcare",
                entity_type_id=f"healthcare.{type_name}",
                attributes=attributes,
                provenance={"fit_status": "synthetic_structure", "not_a_real_person_or_care_record": True},
            ))
    generated = SparseHeterogeneousTopologyGenerator().generate(
        context_id=context_id,
        domain="healthcare",
        entities=entities,
        layers=domain_spec.compile_relation_layers(),
        seed=seed ^ 0x4845414C,
        observation_window_seconds=duration_seconds,
    )
    relations = list(generated.relations)
    supplemental = _ensure_minimum_care_bundle(context_id, entities, relations)
    relations.extend(supplemental)
    profile = {
        "profile_id": f"{context_id}_typed_care_context",
        "generator_family": "sparse_typed_relation_layers",
        "global_connectivity_required": False,
        "spatial_coordinates_required": False,
        "entity_type_counts": type_counts,
        "supplemental_minimum_bundle_relation_count": len(supplemental),
        "observed_statistics": {**generated.statistics, "relation_count_after_minimum_bundle": len(relations)},
    }
    return entities, relations, profile


def _allocate_entity_counts(node_count: int) -> Dict[str, int]:
    weights = {"patient": 0.34, "encounter": 0.25, "device": 0.15, "specimen": 0.12, "practitioner": 0.08, "location": 0.06}
    counts = {name: max(1, int(node_count * weight)) for name, weight in weights.items()}
    counts["patient"] += node_count - sum(counts.values())
    if counts["patient"] <= 0:
        raise ValueError("nodes_per_context is too small for healthcare entity types")
    return counts


def _ensure_minimum_care_bundle(context_id: str, entities: List[Entity], relations: List[ContextRelation]) -> List[ContextRelation]:
    by_type: Dict[str, List[Entity]] = defaultdict(list)
    for entity in entities:
        by_type[entity.entity_type_id].append(entity)
    patient = by_type["healthcare.patient"][0]
    endpoints = [
        ("healthcare.encounter_has_patient", by_type["healthcare.encounter"][0], patient),
        ("healthcare.device_monitors_patient", by_type["healthcare.device"][0], patient),
        ("healthcare.specimen_from_patient", by_type["healthcare.specimen"][0], patient),
        ("healthcare.practitioner_responsible_for_patient", by_type["healthcare.practitioner"][0], patient),
        ("healthcare.encounter_at_location", by_type["healthcare.encounter"][0], by_type["healthcare.location"][0]),
    ]
    existing = {(item.relation_type_id, item.source_entity_id, item.target_entity_id) for item in relations}
    supplemental: List[ContextRelation] = []
    for index, (relation_type, source, target) in enumerate(endpoints, 1):
        key = (relation_type, source.entity_id, target.entity_id)
        if key not in existing:
            supplemental.append(ContextRelation(
                relation_id=f"{context_id}_minimum_bundle_relation_{index:02d}",
                context_id=context_id,
                domain="healthcare",
                relation_type_id=relation_type,
                source_entity_id=source.entity_id,
                target_entity_id=target.entity_id,
                directed=True,
                provenance={"generation_family": "minimum_typed_care_bundle", "synthetic_structure": True},
            ))
    return supplemental


def _choose_care_bundle(entities: List[Entity], relations: List[ContextRelation], rng: random.Random) -> Dict[str, str]:
    incoming: Dict[tuple[str, str], List[ContextRelation]] = defaultdict(list)
    outgoing: Dict[tuple[str, str], List[ContextRelation]] = defaultdict(list)
    for relation in relations:
        incoming[(relation.target_entity_id, relation.relation_type_id)].append(relation)
        outgoing[(relation.source_entity_id, relation.relation_type_id)].append(relation)
    patients = [item.entity_id for item in entities if item.entity_type_id == "healthcare.patient"]
    eligible = []
    for patient_id in patients:
        groups = {
            "encounter": incoming[(patient_id, "healthcare.encounter_has_patient")],
            "device": incoming[(patient_id, "healthcare.device_monitors_patient")],
            "specimen": incoming[(patient_id, "healthcare.specimen_from_patient")],
            "practitioner": incoming[(patient_id, "healthcare.practitioner_responsible_for_patient")],
        }
        if all(groups.values()):
            eligible.append((patient_id, groups))
    if not eligible:
        raise ValueError("Healthcare context has no complete typed care bundle")
    patient_id, groups = rng.choice(eligible)
    encounter_relation = rng.choice(groups["encounter"])
    device_relation = rng.choice(groups["device"])
    specimen_relation = rng.choice(groups["specimen"])
    practitioner_relation = rng.choice(groups["practitioner"])
    encounter_id = encounter_relation.source_entity_id
    locations = outgoing[(encounter_id, "healthcare.encounter_at_location")]
    return {
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "device_id": device_relation.source_entity_id,
        "specimen_id": specimen_relation.source_entity_id,
        "practitioner_id": practitioner_relation.source_entity_id,
        "location_id": rng.choice(locations).target_entity_id if locations else "",
        "encounter_relation_id": encounter_relation.relation_id,
        "device_relation_id": device_relation.relation_id,
        "specimen_relation_id": specimen_relation.relation_id,
        "practitioner_relation_id": practitioner_relation.relation_id,
    }


def _evidence(context: EpisodeContext, bundle_relation_keys: List[str], entity_ids: List[str], evaluated_at: float) -> ContextEvidence:
    bundle = context.state["bundle"]
    return ContextEvidence(
        context_relation_ids=[bundle[key] for key in bundle_relation_keys],
        entity_ids=list(dict.fromkeys(entity_ids)),
        evaluated_at_offset_seconds=evaluated_at,
    )


def _measurement_attributes(measurement_kind: str, *, abnormal: bool, rng: random.Random, phase: str) -> Dict[str, Any]:
    if measurement_kind == "temperature":
        value = rng.uniform(38.1, 39.4) if abnormal else rng.uniform(36.4, 37.5)
        unit = "Cel"
    elif measurement_kind == "oxygen_saturation":
        value = rng.uniform(86.0, 92.5) if abnormal else rng.uniform(95.0, 99.0)
        unit = "%"
    elif measurement_kind == "synthetic_inflammation_marker":
        value = rng.uniform(12.0, 30.0) if abnormal else rng.uniform(1.0, 9.0)
        unit = "synthetic_unit"
    else:
        value = rng.uniform(0.65, 0.95) if not abnormal else rng.uniform(0.1, 0.45)
        unit = "synthetic_index"
    return {
        "measurement_kind": measurement_kind,
        "numeric_value": round(value, 3),
        "unit": unit,
        "interpretation": "configured_abnormal" if abnormal else "within_configured_band",
        "reference_profile_id": "synthetic_monitoring_profile_v1",
        "measurement_phase": phase,
    }


def _synthetic_provenance(*, policy: bool = False) -> Dict[str, Any]:
    return {"fit_status": "policy_defined" if policy else "synthetic_prior", "not_for_clinical_decision_making": True}
