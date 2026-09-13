"""Deterministic discrete-event scheduler with auditable candidate clocks."""

from __future__ import annotations

import copy
import heapq
import math
import random
from itertools import count
from typing import Dict, List, Tuple

from .domain import CandidateSpec, DomainPackage, EpisodeContext, RelationSpec
from .models import (
    Candidate,
    CandidateStatus,
    EventRecord,
    EventRelation,
    EpisodeResult,
    RiskAlternativeEvidence,
    RiskSetRecord,
    TemporalLink,
)
from .temporal import TemporalModelRegistry
from .validation import validate_episode_result


class SimulationEngine:
    """Run one episode without embedding any domain-specific semantics."""

    def __init__(
        self,
        domain_package: DomainPackage,
        temporal_models: TemporalModelRegistry,
        *,
        max_events: int = 10_000,
    ) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.domain_package = domain_package
        self.temporal_models = temporal_models
        self.max_events = max_events

    def run(self, episode_index: int, seed: int) -> EpisodeResult:
        rng = random.Random(seed)
        context = self.domain_package.create_episode_context(episode_index, seed)
        end_time = float(context.duration_seconds)
        candidates: Dict[str, Candidate] = {}
        events: List[EventRecord] = []
        event_by_id: Dict[str, EventRecord] = {}
        relations: List[EventRelation] = []
        risk_sets: List[RiskSetRecord] = []
        heap: List[Tuple[float, int, int, int, str, int]] = []
        sequence = count()
        candidate_counter = count(1)
        event_counter = count(1)
        relation_counter = count(1)
        risk_set_counter = count(1)
        last_processed_time = 0.0

        def push(candidate: Candidate) -> None:
            if candidate.scheduled_time is None:
                return
            heapq.heappush(
                heap,
                (
                    float(candidate.scheduled_time),
                    int(candidate.phase_priority),
                    int(candidate.domain_priority),
                    next(sequence),
                    candidate.candidate_id,
                    candidate.revision,
                ),
            )

        def activate(spec: CandidateSpec) -> Candidate:
            activation = _resolve_activation_time(spec, event_by_id)
            candidate = Candidate(
                candidate_id=(
                    f"{context.episode_id}_candidate_{next(candidate_counter):06d}"
                ),
                episode_id=context.episode_id,
                mechanism_id=spec.mechanism_id,
                target_event_type_id=spec.target_event_type_id,
                parent_event_ids=list(spec.parent_event_ids),
                participants=list(spec.participants),
                activation_time=activation,
                temporal_model_ref=spec.temporal_model_ref,
                temporal_inputs=copy.deepcopy(spec.temporal_inputs),
                combination=spec.combination,
                phase_priority=spec.phase_priority,
                domain_priority=spec.domain_priority,
                context_evidence=copy.deepcopy(spec.context_evidence),
                attributes=copy.deepcopy(spec.attributes),
                provenance=copy.deepcopy(spec.provenance),
            )
            self.temporal_models.initialize(candidate, rng)
            if (
                candidate.scheduled_time is not None
                and candidate.scheduled_time < candidate.activation_time - 1e-9
            ):
                raise ValueError(
                    f"Candidate {candidate.candidate_id} is scheduled before activation"
                )
            candidates[candidate.candidate_id] = candidate
            push(candidate)
            return candidate

        for spec in self.domain_package.seed_candidates(context, rng):
            activate(spec)

        while heap and len(events) < self.max_events:
            due, _, _, _, candidate_id, revision = heapq.heappop(heap)
            candidate = candidates[candidate_id]
            if candidate.status is not CandidateStatus.SCHEDULED:
                continue
            if revision != candidate.revision:
                continue
            if candidate.scheduled_time is None or abs(candidate.scheduled_time - due) > 1e-8:
                continue
            if due > end_time:
                break
            last_processed_time = due

            # Recheck every currently active alternative at the decision time.
            # This keeps expired context links and stale state predicates out of
            # the risk set instead of recording them as possible next events.
            for active in list(candidates.values()):
                if (
                    active.status is not CandidateStatus.SCHEDULED
                    or active.activation_time > due + 1e-9
                ):
                    continue
                decision = self.domain_package.revalidate_candidate(
                    active, context, event_by_id, due
                )
                if decision.context_evidence is not None:
                    active.context_evidence = copy.deepcopy(decision.context_evidence)
                if not decision.valid:
                    _terminate_candidate(
                        active, decision.reason, decision.superseded_by_event_id
                    )
            if candidate.status is not CandidateStatus.SCHEDULED:
                continue

            event_id = f"{context.episode_id}_event_{next(event_counter):06d}"
            risk_set = _build_risk_set(
                risk_set_id=(
                    f"{context.episode_id}_risk_set_{next(risk_set_counter):06d}"
                ),
                event_id=event_id,
                selected=candidate,
                candidates=candidates,
                at_time=due,
                temporal_models=self.temporal_models,
            )
            risk_sets.append(risk_set)
            candidate.risk_set_id = risk_set.risk_set_id
            candidate.selection_evidence = {
                "selection_mode": risk_set.selection_mode,
                **_selected_factor_summary(risk_set.factorization),
            }
            event = self.domain_package.materialize_event(
                event_id, candidate, context, rng
            )
            if abs(event.temporal.occurrence_start_offset_seconds - due) > 1e-8:
                raise ValueError(
                    f"Domain package changed scheduled occurrence time for {candidate_id}"
                )
            _apply_observation_plan(
                event,
                candidate,
                context,
                self.domain_package,
                self.temporal_models,
                rng,
            )
            event.provenance["risk_set_id"] = risk_set.risk_set_id
            event.provenance["selection_mode"] = risk_set.selection_mode
            candidate.status = CandidateStatus.FIRED
            candidate.fired_event_id = event.event_id
            candidate.terminal_reason = "candidate_fired"
            events.append(event)
            event_by_id[event.event_id] = event

            for spec in self.domain_package.relation_specs(
                event, candidate, context, event_by_id
            ):
                relations.append(
                    _materialize_relation(
                        f"{context.episode_id}_relation_{next(relation_counter):06d}",
                        context,
                        event,
                        candidate,
                        spec,
                        event_by_id,
                    )
                )

            self.domain_package.apply_event(event, candidate, context)

            # Existing stochastic clocks are revalidated after every state
            # change.  Time-varying hazards preserve their original random
            # threshold and accumulated hazard instead of redrawing a delay.
            for active in list(candidates.values()):
                if active.status is not CandidateStatus.SCHEDULED:
                    continue
                update = self.domain_package.revalidate_candidate(
                    active, context, event_by_id, due
                )
                if update.context_evidence is not None:
                    active.context_evidence = copy.deepcopy(update.context_evidence)
                if not update.valid:
                    _terminate_candidate(
                        active, update.reason, update.superseded_by_event_id
                    )
                    continue
                model = self.temporal_models.get(active.temporal_model_ref)
                if model.covariate_mode.value == "time_varying_hazard":
                    inputs = update.temporal_inputs or active.temporal_inputs
                    self.temporal_models.update(active, due, inputs)
                    active.revision += 1
                    push(active)

            for spec in self.domain_package.spawn_candidates(
                event, candidate, context, event_by_id, rng
            ):
                activate(spec)

        termination_reason = (
            "event_limit_reached" if len(events) >= self.max_events else "observation_window_ended"
        )
        censoring_time = (
            last_processed_time
            if termination_reason == "event_limit_reached"
            else end_time
        )
        for candidate in candidates.values():
            if candidate.status is CandidateStatus.SCHEDULED:
                candidate.status = CandidateStatus.RIGHT_CENSORED
                candidate.censored_at = censoring_time
                candidate.terminal_reason = termination_reason
                candidate.revision += 1

        result = EpisodeResult(
            episode_id=context.episode_id,
            domain=context.domain,
            context_id=context.context_id,
            start_time=context.start_time,
            duration_seconds=context.duration_seconds,
            entities=context.entities,
            context_relations=context.context_relations,
            events=events,
            event_relations=relations,
            candidates=list(candidates.values()),
            risk_sets=risk_sets,
            final_state=copy.deepcopy(context.state),
            termination_reason=termination_reason,
            context_attributes=copy.deepcopy(context.context_attributes),
            episode_attributes=copy.deepcopy(context.episode_attributes),
        )
        result.validation = validate_episode_result(
            result, self.domain_package.catalog()
        )
        return result


def _build_risk_set(
    *,
    risk_set_id: str,
    event_id: str,
    selected: Candidate,
    candidates: Dict[str, Candidate],
    at_time: float,
    temporal_models: TemporalModelRegistry,
) -> RiskSetRecord:
    active = sorted(
        (
            candidate
            for candidate in candidates.values()
            if candidate.status is CandidateStatus.SCHEDULED
            and candidate.activation_time <= at_time + 1e-9
        ),
        key=lambda item: item.candidate_id,
    )
    alternatives: List[RiskAlternativeEvidence] = []
    for candidate in active:
        measure = temporal_models.measure(candidate, at_time)
        alternatives.append(
            RiskAlternativeEvidence(
                candidate_id=candidate.candidate_id,
                mechanism_id=candidate.mechanism_id,
                target_event_type_id=candidate.target_event_type_id,
                participant_entity_ids=sorted(
                    {item.entity_id for item in candidate.participants}
                ),
                parent_event_ids=list(candidate.parent_event_ids),
                temporal_model_ref=candidate.temporal_model_ref,
                scheduled_time=candidate.scheduled_time,
                clock_measure=measure,
            )
        )

    selected_alternative = next(
        item for item in alternatives if item.candidate_id == selected.candidate_id
    )
    due_atoms = [
        item
        for item in alternatives
        if item.clock_measure.get("is_due_atom") is True
    ]
    selected_kind = selected_alternative.clock_measure.get("clock_kind")
    factorization: Dict[str, object]
    if due_atoms:
        selection_mode = (
            "deterministic_atom_priority"
            if selected_kind == "deterministic_atom"
            else "mixed_or_discrete_atom_priority"
        )
        factorization = {
            "basis": "point_mass_or_policy_clock",
            "continuous_hazard_factorization_applicable": False,
            "selected_event_type_id": selected.target_event_type_id,
            "selected_entity_ids": selected_alternative.participant_entity_ids,
            "explanation": (
                "A due point mass is resolved by the declared scheduler priorities; "
                "no artificial continuous density is assigned."
            ),
        }
    else:
        continuous = [
            item
            for item in alternatives
            if item.clock_measure.get("clock_kind") == "stochastic_continuous"
            and item.clock_measure.get("hazard_per_second") is not None
        ]
        total_hazard = sum(
            float(item.clock_measure["hazard_per_second"])
            for item in continuous
        )
        if selected_kind == "stochastic_continuous" and total_hazard > 0.0:
            selection_mode = "cause_specific_competing_hazards"
            for item in continuous:
                item.candidate_probability_given_time = (
                    float(item.clock_measure["hazard_per_second"])
                    / total_hazard
                )
            selected_hazard = float(
                selected_alternative.clock_measure["hazard_per_second"]
            )
            type_hazard = sum(
                float(item.clock_measure["hazard_per_second"])
                for item in continuous
                if item.target_event_type_id == selected.target_event_type_id
            )
            selected_entities = tuple(selected_alternative.participant_entity_ids)
            entity_hazard = sum(
                float(item.clock_measure["hazard_per_second"])
                for item in continuous
                if item.target_event_type_id == selected.target_event_type_id
                and tuple(item.participant_entity_ids) == selected_entities
            )
            joint_log_survival = sum(
                math.log(
                    max(
                        float(item.clock_measure.get("survival_probability", 1.0)),
                        1e-300,
                    )
                )
                for item in continuous
            )
            joint_survival = math.exp(max(-745.0, joint_log_survival))
            factorization = {
                "basis": "one_shared_competing-risk_set",
                "continuous_hazard_factorization_applicable": True,
                "history_conditioning": "candidate_activation_parents_context_and_state",
                "time": {
                    "total_hazard_per_second": total_hazard,
                    "joint_survival_probability": joint_survival,
                    "next_event_time_density_per_second": total_hazard
                    * joint_survival,
                },
                "type_given_time": {
                    "selected_event_type_id": selected.target_event_type_id,
                    "hazard_sum_per_second": type_hazard,
                    "probability": type_hazard / total_hazard,
                },
                "entity_given_time_and_type": {
                    "selected_entity_ids": list(selected_entities),
                    "hazard_sum_per_second": entity_hazard,
                    "probability": entity_hazard / type_hazard,
                },
                "mechanism_given_time_type_and_entity": {
                    "selected_mechanism_id": selected.mechanism_id,
                    "hazard_per_second": selected_hazard,
                    "probability": selected_hazard / entity_hazard,
                },
                "selected_candidate_probability_given_time": (
                    selected_hazard / total_hazard
                ),
                "selected_joint_density_per_second": selected_hazard
                * joint_survival,
            }
        else:
            selection_mode = "sampled_clock_order_without_continuous_hazard"
            factorization = {
                "basis": "sampled_clock_order",
                "continuous_hazard_factorization_applicable": False,
                "selected_event_type_id": selected.target_event_type_id,
                "selected_entity_ids": selected_alternative.participant_entity_ids,
                "explanation": (
                    "The selected clock does not expose a continuous hazard; "
                    "its sampled due time remains auditable."
                ),
            }

    if not selected.parent_event_ids:
        parent_attribution = {
            "mode": "independent_background",
            "background_component": 1.0,
            "weight_semantics": "generator_attribution_not_causal_effect_size",
            "realized_parent_event_ids": [],
        }
    elif len(selected.parent_event_ids) == 1:
        parent_attribution = {
            "mode": "explicit_realized_parents",
            "combination": selected.combination,
            "realized_parent_event_ids": list(selected.parent_event_ids),
            "contribution_weights": [
                {"event_id": selected.parent_event_ids[0], "weight": 1.0}
            ],
            "weight_semantics": "generator_attribution_not_causal_effect_size",
        }
    else:
        parent_attribution = {
            "mode": "explicit_realized_parents",
            "combination": selected.combination,
            "realized_parent_event_ids": list(selected.parent_event_ids),
            "contribution_weights": None,
            "weight_semantics": "not_identified_for_explicit_combination_rule",
            "explanation": (
                "These parents jointly satisfied an explicit rule; no unsupported "
                "probabilistic contribution split is asserted."
            ),
        }

    return RiskSetRecord(
        risk_set_id=risk_set_id,
        episode_id=selected.episode_id,
        evaluated_at_offset_seconds=at_time,
        selected_candidate_id=selected.candidate_id,
        selected_event_id=event_id,
        selection_mode=selection_mode,
        alternatives=alternatives,
        factorization=factorization,
        parent_attribution=parent_attribution,
        tie_break={
            "phase_priority": selected.phase_priority,
            "domain_priority": selected.domain_priority,
            "due_atom_candidate_ids": [item.candidate_id for item in due_atoms],
        },
        provenance={
            "scheduler": "independent_candidate_clocks_with_equivalent_competing_risks",
            "evaluated_before_selected_event_state_update": True,
        },
    )


def _selected_factor_summary(factorization: Dict[str, object]) -> Dict[str, object]:
    if not factorization.get("continuous_hazard_factorization_applicable"):
        return {"continuous_hazard_factorization_applicable": False}
    type_component = factorization["type_given_time"]
    entity_component = factorization["entity_given_time_and_type"]
    mechanism_component = factorization["mechanism_given_time_type_and_entity"]
    assert isinstance(type_component, dict)
    assert isinstance(entity_component, dict)
    assert isinstance(mechanism_component, dict)
    return {
        "continuous_hazard_factorization_applicable": True,
        "candidate_probability_given_time": factorization[
            "selected_candidate_probability_given_time"
        ],
        "type_probability_given_time": type_component["probability"],
        "entity_probability_given_time_and_type": entity_component["probability"],
        "mechanism_probability_given_time_type_and_entity": mechanism_component[
            "probability"
        ],
    }


def _resolve_activation_time(
    spec: CandidateSpec, event_by_id: Dict[str, EventRecord]
) -> float:
    if spec.combination not in {"none", "single", "all_of", "any_of", "k_of_n", "ordered_sequence"}:
        raise ValueError(f"Unsupported parent combination: {spec.combination}")
    parent_times: List[float] = []
    for parent_id in spec.parent_event_ids:
        try:
            parent_times.append(
                event_by_id[parent_id].temporal.occurrence_start_offset_seconds
            )
        except KeyError as exc:
            raise ValueError(f"Candidate references missing parent event {parent_id}") from exc
    parent_count = len(parent_times)
    if spec.combination == "none" and parent_count != 0:
        raise ValueError("none candidates cannot record a realized parent")
    if spec.combination == "single" and parent_count != 1:
        raise ValueError("single candidates require exactly one realized parent")
    if spec.combination == "all_of" and parent_count < 2:
        raise ValueError("all_of candidates require at least two realized parents")
    if spec.combination == "any_of" and parent_count != 1:
        raise ValueError(
            "any_of candidates record exactly the parent that actually triggered them"
        )
    if spec.combination == "k_of_n":
        required = int(spec.attributes.get("required_parent_count", 0))
        pool_size = int(spec.attributes.get("candidate_parent_pool_size", 0))
        if required <= 0 or pool_size < required or parent_count != required:
            raise ValueError(
                "k_of_n candidates require a valid required_parent_count, "
                "candidate_parent_pool_size, and exactly k realized parents"
            )
    if spec.combination == "ordered_sequence":
        if parent_count < 2:
            raise ValueError("ordered_sequence candidates require at least two parents")
        if any(left > right for left, right in zip(parent_times, parent_times[1:])):
            raise ValueError("ordered_sequence parents are not chronologically ordered")
    derived = max(parent_times, default=0.0)
    if spec.activation_time is None:
        return derived
    explicit = float(spec.activation_time)
    if explicit + 1e-9 < derived:
        raise ValueError("Candidate activation precedes a required parent event")
    return explicit


def _terminate_candidate(
    candidate: Candidate,
    reason: str | None,
    superseded_by_event_id: str | None,
) -> None:
    if superseded_by_event_id:
        candidate.status = CandidateStatus.SUPERSEDED
        candidate.superseded_by_event_id = superseded_by_event_id
    else:
        candidate.status = CandidateStatus.CANCELLED
    candidate.terminal_reason = reason or "precondition_became_false"
    candidate.revision += 1


def _apply_observation_plan(
    event: EventRecord,
    candidate: Candidate,
    context: EpisodeContext,
    domain_package: DomainPackage,
    temporal_models: TemporalModelRegistry,
    rng: random.Random,
) -> None:
    plan = domain_package.observation_plan(event, candidate, context)
    occurrence = event.temporal.occurrence_start_offset_seconds
    observed = occurrence
    observation_provenance: Dict[str, object] = {}
    if plan.observation_model_ref:
        delay, observation_provenance = temporal_models.sample_delay(
            plan.observation_model_ref,
            occurrence,
            plan.observation_inputs,
            rng,
        )
        observed += delay
    if observed <= context.duration_seconds:
        event.temporal.observed_at_offset_seconds = observed
        observation_status = "observed"
    else:
        event.temporal.observed_at_offset_seconds = None
        observation_status = "right_censored"

    recorded = observed
    recording_provenance: Dict[str, object] = {}
    if plan.recording_model_ref:
        delay, recording_provenance = temporal_models.sample_delay(
            plan.recording_model_ref,
            observed,
            plan.recording_inputs,
            rng,
        )
        recorded += delay
    if observation_status == "observed" and recorded <= context.duration_seconds:
        event.temporal.recorded_at_offset_seconds = recorded
        recording_status = "recorded"
    else:
        event.temporal.recorded_at_offset_seconds = None
        recording_status = "right_censored"
    event.observation = {
        "observation_status": observation_status,
        "recording_status": recording_status,
        "source": plan.source,
        "observation_model_ref": plan.observation_model_ref,
        "recording_model_ref": plan.recording_model_ref,
        "observation_model_provenance": observation_provenance,
        "recording_model_provenance": recording_provenance,
    }


def _materialize_relation(
    relation_id: str,
    context: EpisodeContext,
    target: EventRecord,
    candidate: Candidate,
    spec: RelationSpec,
    event_by_id: Dict[str, EventRecord],
) -> EventRelation:
    try:
        source = event_by_id[spec.source_event_id]
    except KeyError as exc:
        raise ValueError(f"Relation references missing source {spec.source_event_id}") from exc
    source_time = source.temporal.anchor(spec.source_anchor)
    target_time = target.temporal.anchor(spec.target_anchor)
    lag = target_time - source_time
    return EventRelation(
        relation_id=relation_id,
        episode_id=context.episode_id,
        source_event_id=source.event_id,
        target_event_id=target.event_id,
        relation_class=spec.relation_class,
        relation_type_id=spec.relation_type_id,
        rule_id=spec.rule_id,
        temporal_link=TemporalLink(
            lag_seconds=round(lag, 9),
            source_anchor=spec.source_anchor,
            target_anchor=spec.target_anchor,
            temporal_model_ref=candidate.temporal_model_ref,
        ),
        status=spec.status,
        mechanism_group_id=spec.mechanism_group_id,
        context_evidence=copy.deepcopy(spec.context_evidence),
        attributes=copy.deepcopy(spec.attributes),
        provenance={
            "recorded_when_rule_executed": True,
            "candidate_id": candidate.candidate_id,
            "temporal_model": copy.deepcopy(candidate.temporal_provenance),
            **copy.deepcopy(spec.provenance),
        },
    )
