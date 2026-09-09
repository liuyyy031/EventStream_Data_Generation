"""Deterministic discrete-event scheduler with auditable candidate clocks."""

from __future__ import annotations

import copy
import heapq
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
        heap: List[Tuple[float, int, int, int, str, int]] = []
        sequence = count()
        candidate_counter = count(1)
        event_counter = count(1)
        relation_counter = count(1)
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

            decision = self.domain_package.revalidate_candidate(
                candidate, context, event_by_id, due
            )
            if decision.context_evidence is not None:
                candidate.context_evidence = copy.deepcopy(decision.context_evidence)
            if not decision.valid:
                _terminate_candidate(candidate, decision.reason, decision.superseded_by_event_id)
                continue

            event_id = f"{context.episode_id}_event_{next(event_counter):06d}"
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
            final_state=copy.deepcopy(context.state),
            termination_reason=termination_reason,
            context_attributes=copy.deepcopy(context.context_attributes),
            episode_attributes=copy.deepcopy(context.episode_attributes),
        )
        result.validation = validate_episode_result(
            result, self.domain_package.catalog()
        )
        return result


def _resolve_activation_time(
    spec: CandidateSpec, event_by_id: Dict[str, EventRecord]
) -> float:
    if spec.combination not in {"single", "all_of", "any_of", "k_of_n", "ordered_sequence"}:
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
    if spec.combination == "single" and parent_count > 1:
        raise ValueError("single candidates may record at most one realized parent")
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
