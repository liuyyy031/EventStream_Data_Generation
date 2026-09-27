"""Deterministic discrete-event scheduler with auditable candidate clocks."""

from __future__ import annotations

import copy
import heapq
import math
import random
from itertools import count
from typing import Dict, List, Tuple

from .domain import CandidateSpec, DomainPackage, EpisodeContext, RelationSpec
from .mechanism_reweighting import (
    MECHANISM_HAZARD_MULTIPLIER,
    LegalMechanismReweighter,
)
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
from .history import HistoryIndex
from .time_axis import TimeAxis, sample_duration
from .continuation import SimulationSnapshot, resample_surviving_clock


class SimulationEngine:
    """Run one episode without embedding any domain-specific semantics."""

    def __init__(
        self,
        domain_package: DomainPackage,
        temporal_models: TemporalModelRegistry,
        *,
        max_events: int = 10_000,
        mechanism_reweighter: LegalMechanismReweighter | None = None,
        time_axis: dict | None = None,
        observation_duration: dict | None = None,
    ) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.domain_package = domain_package
        self.temporal_models = temporal_models
        self.max_events = max_events
        self.mechanism_reweighter = mechanism_reweighter
        self.time_axis = TimeAxis(**(time_axis or {}))
        self.observation_duration = observation_duration

    def run(self, episode_index: int, seed: int, *, snapshot: SimulationSnapshot | None = None,
            stop_after_events: int | None = None, continuation_seed: int | None = None) -> EpisodeResult | SimulationSnapshot:
        if stop_after_events is not None and not 0 <= stop_after_events < self.max_events:
            raise ValueError("Snapshot event count must lie below the administrative event cap")
        rng = random.Random(seed)
        context = self.domain_package.create_episode_context(episode_index, seed)
        support_horizon = context.duration_seconds
        if self.observation_duration:
            policy = self.observation_duration
            maximum = policy.get("maximum", policy.get("value", max(policy.get("values", [0]))))
            if TimeAxis(kind="relative", unit=policy.get("unit", "second")).to_seconds(maximum) > support_horizon:
                raise ValueError("Declared observation-duration support exceeds context support horizon")
        context.duration_seconds = sample_duration(
            self.observation_duration, support_horizon, random.Random(seed ^ 0x484F5249)
        )
        if context.duration_seconds > support_horizon:
            raise ValueError("Observation duration exceeds context support horizon; increase domain duration_seconds")
        context.time_axis = self.time_axis.descriptor()
        if math.ulp(context.duration_seconds) > self.time_axis.to_seconds(self.time_axis.resolution):
            raise ValueError("Requested time resolution exceeds floating-point precision at this horizon")
        if self.time_axis.kind != "calendar":
            context.start_time = self.time_axis.origin_label
        context.episode_attributes["observation_duration_policy"] = self.observation_duration or {"family": "fixed", "unit": "second", "value": support_horizon}
        context.episode_attributes["context_support_horizon_seconds"] = support_horizon
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
        history = HistoryIndex()
        survival_baselines = {}
        conditioning_time = 0.0
        hazard_accounted = {}
        waiting_log_survival = 0.0
        if snapshot is not None:
            if snapshot.domain != context.domain or snapshot.model_descriptors != self.temporal_models.descriptors():
                raise ValueError("Snapshot domain/model configuration does not match this engine")
            if snapshot.state["engine_signature"] != (episode_index, seed, self.max_events, getattr(self.domain_package, "config", {}), self.time_axis.descriptor(), self.observation_duration):
                raise ValueError("Snapshot generation configuration does not match this engine")
            saved = copy.deepcopy(snapshot.state)
            context, candidates, events = saved["context"], saved["candidates"], saved["events"]
            relations, risk_sets, heap = saved["relations"], saved["risk_sets"], saved["heap"]
            end_time = context.duration_seconds
            last_processed_time = conditioning_time = saved["time"]
            survival_baselines = saved["survival_baselines"]
            hazard_accounted = saved["hazard_accounted"]
            event_by_id = {event.event_id: event for event in events}
            history = HistoryIndex(events)
            sequence = count(saved["next_sequence"])
            candidate_counter, event_counter = count(len(candidates)+1), count(len(events)+1)
            relation_counter, risk_set_counter = count(len(relations)+1), count(len(risk_sets)+1)
            rng.setstate(saved["rng_state"])
            if continuation_seed is not None:
                rng = random.Random(continuation_seed)
                for active in candidates.values():
                    if active.status is CandidateStatus.SCHEDULED:
                        resample_surviving_clock(active, self.temporal_models.get(active.temporal_model_ref), conditioning_time, end_time, rng)
                heap = [row for row in heap if row[4].startswith("__wake_")]
                heapq.heapify(heap)
                context.episode_attributes.update({"prefix_group_id": saved["prefix_group_id"], "continuation_seed": continuation_seed,
                    "continuation_semantics": "conditional_stochastic_not_causal_counterfactual", "prefix_event_ids": list(event_by_id)})

        def capture():
            return SimulationSnapshot(copy.deepcopy({"context": context, "candidates": candidates, "events": events,
                "relations": relations, "risk_sets": risk_sets, "heap": heap, "time": last_processed_time,
                "survival_baselines": survival_baselines, "rng_state": rng.getstate(), "next_sequence": next(sequence),
                "hazard_accounted": hazard_accounted,
                "engine_signature": (episode_index, seed, self.max_events, getattr(self.domain_package, "config", {}), self.time_axis.descriptor(), self.observation_duration),
                "prefix_group_id": f"{context.episode_id}:seed:{seed}:prefix:{len(events)}"}), self.temporal_models.descriptors(), context.domain)

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

        def wake_at(when):
            if when is not None and last_processed_time < when <= end_time:
                serial = next(sequence)
                heapq.heappush(heap, (float(when), -100, 0, serial, f"__wake_{serial}", 0))

        def accrue_survival(now):
            nonlocal waiting_log_survival
            for active in candidates.values():
                if active.status is not CandidateStatus.SCHEDULED or active.activation_time > now:
                    continue
                measure = self.temporal_models.measure(active, now)
                if measure.get("clock_kind") == "stochastic_continuous":
                    cumulative = float(measure["cumulative_hazard"])
                    waiting_log_survival -= max(0.0, cumulative - hazard_accounted.get(active.candidate_id, 0.0))
                    hazard_accounted[active.candidate_id] = cumulative

        def refresh(now, fired=None):
            for active in list(candidates.values()):
                if active.status is not CandidateStatus.SCHEDULED or active.activation_time > now:
                    continue
                group = fired.attributes.get("exclusive_group_id") if fired else None
                if group is not None and active.attributes.get("exclusive_group_id") == group:
                    _terminate_candidate(active, "explicit_mutual_exclusion", fired.fired_event_id)
                    continue
                update = self.domain_package.revalidate_candidate(active, context, event_by_id, now)
                if update.context_evidence is not None:
                    active.context_evidence = copy.deepcopy(update.context_evidence)
                if not update.valid:
                    _terminate_candidate(active, update.reason, update.superseded_by_event_id)
                    continue
                model = self.temporal_models.get(active.temporal_model_ref)
                if model.covariate_mode.value == "time_varying_hazard":
                    inputs = dict(update.temporal_inputs or active.temporal_inputs)
                    for key in (MECHANISM_HAZARD_MULTIPLIER, "history_spec"):
                        if key in active.temporal_inputs:
                            inputs[key] = active.temporal_inputs[key]
                    probe = copy.copy(active)
                    probe.temporal_inputs = inputs
                    self.temporal_models.update(active, now, history.inputs(probe, now))
                    active.revision += 1
                    push(active)

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
            if self.mechanism_reweighter is not None:
                model = self.temporal_models.get(candidate.temporal_model_ref)
                self.mechanism_reweighter.apply(
                    candidate,
                    event_by_id=event_by_id,
                    event_history=events,
                    temporal_family=model.family,
                )
            candidate.temporal_inputs = history.inputs(candidate, activation)
            model = self.temporal_models.get(candidate.temporal_model_ref)
            history_spec = candidate.temporal_inputs.get("history_spec", {})
            if history_spec and set(getattr(model, "coefficients", {})) & {"history_decayed_count", "time_since_last_event"}:
                raise ValueError("Continuously changing history features require an integrable temporal model; use self_exciting_hazard for exponential decay")
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
            wake_at(activation)
            if history_spec.get("lookback_seconds") is not None:
                lookback = float(history_spec["lookback_seconds"])
                if not math.isfinite(lookback) or lookback < 0:
                    raise ValueError("History lookback must be finite and non-negative")
                for offset in candidate.temporal_inputs.get("history_event_offsets", []):
                    wake_at(math.nextafter(offset + lookback, math.inf))
            return candidate

        if snapshot is None:
            for relation in context.context_relations:
                wake_at(relation.valid_from_offset_seconds)
                if relation.valid_to_offset_seconds is not None:
                    wake_at(math.nextafter(relation.valid_to_offset_seconds, math.inf))
            for spec in self.domain_package.seed_candidates(context, rng):
                activate(spec)
        elif continuation_seed is not None:
            for active in candidates.values():
                if active.status is CandidateStatus.SCHEDULED:
                    push(active)
        if stop_after_events == len(events):
            return capture()

        while heap and len(events) < self.max_events:
            due, _, _, _, candidate_id, revision = heapq.heappop(heap)
            if candidate_id.startswith("__wake_"):
                last_processed_time = due
                accrue_survival(due)
                refresh(due)
                # Keep survival conditioned on the previous realized event;
                # scheduled covariate changes do not reset the waiting interval.
                continue
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
            accrue_survival(due)

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
                survival_baselines=survival_baselines,
                conditioning_time=conditioning_time,
                interval_log_survival=waiting_log_survival,
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
            history.add(event)
            wake_at(event.temporal.observed_at_offset_seconds)
            wake_at(event.temporal.recorded_at_offset_seconds)
            for active in candidates.values():
                spec = active.temporal_inputs.get("history_spec", {})
                if active.status is CandidateStatus.SCHEDULED and spec.get("lookback_seconds") is not None:
                    wake_at(math.nextafter(due + float(spec["lookback_seconds"]), math.inf))

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
            refresh(due, candidate)

            for spec in self.domain_package.spawn_candidates(
                event, candidate, context, event_by_id, rng
            ):
                activate(spec)

            conditioning_time = due
            waiting_log_survival = 0.0
            survival_baselines = {
                item.candidate_id: self.temporal_models.measure(item, due).get("survival_probability", 1.0)
                for item in candidates.values()
                if item.status is CandidateStatus.SCHEDULED and item.activation_time <= due
            }
            if stop_after_events == len(events):
                return capture()

        if stop_after_events is not None:
            raise ValueError("Episode ended before the requested prefix length; use a shorter prefix")

        termination_reason = (
            "event_limit_reached" if len(events) >= self.max_events else
            "candidate_set_exhausted" if not any(c.status is CandidateStatus.SCHEDULED for c in candidates.values()) else
            "observation_window_ended"
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
            time_axis=context.time_axis,
        )
        depths = {}
        parents_by_event = {c.fired_event_id: c.parent_event_ids for c in candidates.values() if c.fired_event_id}
        for event in events:
            parents = parents_by_event[event.event_id]
            depths[event.event_id] = max((depths[p] + 1 for p in parents), default=0)
        result.episode_attributes["length_summary"] = {
            "context_entity_count": len(context.entities), "realized_event_count": len(events),
            "max_causal_depth": max(depths.values(), default=0),
            "observation_duration_seconds": end_time,
            "administrative_event_cap": self.max_events,
            "natural_candidate_exhaustion": not any(c.status is CandidateStatus.RIGHT_CENSORED for c in candidates.values()),
        }
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
    survival_baselines: dict | None = None,
    conditioning_time: float = 0.0,
    interval_log_survival: float | None = None,
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
                        min(1.0, float(item.clock_measure.get("survival_probability", 1.0)) / max(float((survival_baselines or {}).get(item.candidate_id, 1.0)), 1e-300)),
                        1e-300,
                    )
                )
                for item in continuous
            )
            joint_survival = math.exp(max(-745.0, joint_log_survival if interval_log_survival is None else interval_log_survival))
            factorization = {
                "basis": "one_shared_competing-risk_set",
                "continuous_hazard_factorization_applicable": True,
                "history_conditioning": "candidate_activation_parents_context_and_state",
                "time": {
                    "conditioned_at_offset_seconds": conditioning_time,
                    "survival_semantics": "conditional_since_previous_event_under_current_risk_set",
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
