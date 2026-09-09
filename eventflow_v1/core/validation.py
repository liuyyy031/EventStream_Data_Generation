"""Deterministic and dataset-level validation for EventFlow v1."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .models import EpisodeBundle


@dataclass
class ValidationResult:
    passed: bool
    errors: List[str]
    warnings: List[str]
    checks: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
        }


def validate_common_contract(bundle: EpisodeBundle, valid_entity_ids: set[str]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, bool] = {}

    event_ids = [event.event_id for event in bundle.events]
    relation_ids = [relation.relation_id for relation in bundle.relations]
    mechanism_ids = [mechanism.mechanism_group_id for mechanism in bundle.mechanisms]
    text_ids = [record.text_id for record in bundle.texts]
    checks["unique_ids"] = all(
        len(values) == len(set(values))
        for values in (event_ids, relation_ids, mechanism_ids, text_ids)
    )
    if not checks["unique_ids"]:
        errors.append("Duplicate IDs exist within the episode")

    event_by_id = {event.event_id: event for event in bundle.events}
    relation_by_id = {relation.relation_id: relation for relation in bundle.relations}
    mechanism_by_id = {
        mechanism.mechanism_group_id: mechanism for mechanism in bundle.mechanisms
    }
    checks["event_contract"] = True
    for event in bundle.events:
        if event.episode_id != bundle.episode_id:
            errors.append(f"Event {event.event_id} has the wrong episode_id")
            checks["event_contract"] = False
        if not 0.0 <= event.time_offset_seconds <= bundle.duration_seconds:
            errors.append(f"Event {event.event_id} is outside the episode time range")
            checks["event_contract"] = False
        participant_links = [
            (participant.entity_id, participant.role) for participant in event.participants
        ]
        if not participant_links or any(
            entity_id not in valid_entity_ids for entity_id in event.participant_ids
        ):
            errors.append(f"Event {event.event_id} references invalid participants")
            checks["event_contract"] = False
        if any(not role for _, role in participant_links):
            errors.append(f"Event {event.event_id} has an empty participant role")
            checks["event_contract"] = False
        if len(participant_links) != len(set(participant_links)):
            errors.append(f"Event {event.event_id} repeats a qualified participant")
            checks["event_contract"] = False

    checks["relation_references"] = True
    allowed_status = {"ground_truth", "observed_link"}
    for relation in bundle.relations:
        if relation.source_event_id not in event_by_id or relation.target_event_id not in event_by_id:
            errors.append(f"Relation {relation.relation_id} has dangling event references")
            checks["relation_references"] = False
            continue
        source = event_by_id[relation.source_event_id]
        target = event_by_id[relation.target_event_id]
        if source.time_offset_seconds >= target.time_offset_seconds:
            errors.append(
                f"Relation {relation.relation_id} does not point forward in exact time"
            )
            checks["relation_references"] = False
        if relation.status not in allowed_status:
            errors.append(f"Relation {relation.relation_id} has unsupported status")
            checks["relation_references"] = False
        if not relation.provenance.get("recorded_when_rule_executed"):
            errors.append(f"Relation {relation.relation_id} lacks executed-rule provenance")
            checks["relation_references"] = False

    checks["acyclic_event_relations"] = _is_acyclic(event_ids, bundle)
    if not checks["acyclic_event_relations"]:
        errors.append("The event relation graph contains a directed cycle")

    checks["mechanism_contract"] = True
    grouped_relation_ids: set[str] = set()
    for mechanism in bundle.mechanisms:
        if mechanism.combination not in {"all_of", "any_of"}:
            errors.append(f"Mechanism {mechanism.mechanism_group_id} has invalid combination")
            checks["mechanism_contract"] = False
        if mechanism.combination == "all_of" and len(mechanism.member_relation_ids) < 2:
            errors.append(f"all_of mechanism {mechanism.mechanism_group_id} needs >=2 relations")
            checks["mechanism_contract"] = False
        for relation_id in mechanism.member_relation_ids:
            relation = relation_by_id.get(relation_id)
            if relation is None:
                errors.append(f"Mechanism {mechanism.mechanism_group_id} has a missing relation")
                checks["mechanism_contract"] = False
                continue
            grouped_relation_ids.add(relation_id)
            if relation.mechanism_group_id != mechanism.mechanism_group_id:
                errors.append(f"Relation {relation_id} disagrees with its mechanism group")
                checks["mechanism_contract"] = False
            if relation.target_event_id != mechanism.target_event_id:
                errors.append(f"Mechanism {mechanism.mechanism_group_id} mixes target events")
                checks["mechanism_contract"] = False
    for relation in bundle.relations:
        if relation.mechanism_group_id and relation.mechanism_group_id not in mechanism_by_id:
            errors.append(f"Relation {relation.relation_id} refers to a missing mechanism")
            checks["mechanism_contract"] = False

    incoming_causal: Dict[str, List[Any]] = defaultdict(list)
    for relation in bundle.relations:
        if relation.relation_class == "causal":
            incoming_causal[relation.target_event_id].append(relation)
    for target_id, incoming_relations in incoming_causal.items():
        if len(incoming_relations) <= 1:
            continue
        group_ids = {relation.mechanism_group_id for relation in incoming_relations}
        if None in group_ids or len(group_ids) != 1:
            errors.append(
                f"Multi-parent target {target_id} is not represented by one explicit mechanism group"
            )
            checks["mechanism_contract"] = False

    termination = bundle.program.get("termination")
    checks["termination_contract"] = isinstance(termination, dict)
    if not isinstance(termination, dict):
        errors.append("EpisodeProgram lacks termination metadata")
    else:
        if termination.get("event_count") != len(bundle.events):
            errors.append("Termination metadata has the wrong event count")
            checks["termination_contract"] = False
        if termination.get("reason") not in {"completed", "event_limit_reached"}:
            errors.append("Termination metadata has an unsupported reason")
            checks["termination_contract"] = False
        if bool(termination.get("truncated")) != (
            termination.get("reason") == "event_limit_reached"
        ):
            errors.append("Termination reason and truncated flag disagree")
            checks["termination_contract"] = False

    root_event_ids = {
        event.event_id for event in bundle.events if event.event_role == "root"
    }
    primary_root_ids = set(bundle.program.get("primary_root_event_ids", []))
    independent_root_ids = set(bundle.program.get("independent_root_event_ids", []))
    checks["root_role_contract"] = True
    if primary_root_ids.intersection(independent_root_ids):
        errors.append("Primary and independent root sets overlap")
        checks["root_role_contract"] = False
    if primary_root_ids.union(independent_root_ids) != root_event_ids:
        errors.append("EpisodeProgram root classifications do not cover all root events")
        checks["root_role_contract"] = False
    for event_id in independent_root_ids:
        event = event_by_id.get(event_id)
        if event is None or not event.provenance.get("independent_from_primary_flow"):
            errors.append(f"Independent root {event_id} lacks independence provenance")
            checks["root_role_contract"] = False

    checks["text_alignment"] = _validate_text_alignment(
        bundle, event_by_id, relation_by_id, mechanism_by_id, errors
    )
    if not bundle.events:
        errors.append("Episode contains no events")
    if not bundle.relations:
        warnings.append("Episode contains no event relations")

    return ValidationResult(not errors, errors, warnings, checks)


def build_distribution_report(bundles: Iterable[EpisodeBundle]) -> Dict[str, Any]:
    accumulator = DistributionAccumulator()
    for bundle in bundles:
        accumulator.update(bundle)
    return accumulator.report()


class DistributionAccumulator:
    """Streaming statistics so large runs do not retain episode payloads."""

    def __init__(self, event_type_registry: Any | None = None) -> None:
        self.event_type_registry = event_type_registry
        self.episode_count = 0
        self.event_type_counts: Counter[str] = Counter()
        self.event_category_counts: Counter[str] = Counter()
        self.event_metric_tag_counts: Counter[str] = Counter()
        self.scenario_counts: Counter[str] = Counter()
        self.relation_type_counts: Counter[str] = Counter()
        self.event_counts: List[int] = []
        self.relation_counts: List[int] = []
        self.inter_event_gaps: List[float] = []
        self.multi_parent_targets = 0
        self.targets = 0
        self.relation_lags: Dict[str, List[float]] = defaultdict(list)
        self.max_causal_children = 0
        self.max_propagation_children = 0
        self.truncated_episode_count = 0
        self.used_entity_ids: set[str] = set()
        self.calibration_profile_counts: Counter[str] = Counter()
        self.calibration_status_counts: Counter[str] = Counter()
        self.calibrated_propagation_lag_count = 0
        self.calibrated_recovery_lag_count = 0
        self.propagation_to_free_flow_ratios: List[float] = []

    def update(self, bundle: EpisodeBundle) -> None:
        self.episode_count += 1
        self.scenario_counts[bundle.scenario_family] += 1
        self.event_type_counts.update(event.event_type_id for event in bundle.events)
        if self.event_type_registry is not None:
            for event in bundle.events:
                self.event_category_counts[
                    self.event_type_registry.category(event.event_type_id)
                ] += 1
                self.event_metric_tag_counts.update(
                    self.event_type_registry.metric_tags(event.event_type_id)
                )
        self.relation_type_counts.update(
            relation.relation_type for relation in bundle.relations
        )
        self.event_counts.append(len(bundle.events))
        self.relation_counts.append(len(bundle.relations))
        ordered = sorted(event.time_offset_seconds for event in bundle.events)
        self.inter_event_gaps.extend(
            right - left for left, right in zip(ordered, ordered[1:])
        )
        event_by_id = {event.event_id: event for event in bundle.events}
        incoming: Dict[str, int] = defaultdict(int)
        for relation in bundle.relations:
            incoming[relation.target_event_id] += 1
            source = event_by_id[relation.source_event_id]
            target = event_by_id[relation.target_event_id]
            self.relation_lags[relation.relation_type].append(
                target.time_offset_seconds - source.time_offset_seconds
            )
        self.targets += len(incoming)
        self.multi_parent_targets += sum(value > 1 for value in incoming.values())
        metrics = compute_episode_metrics(bundle)
        self.max_causal_children = max(
            self.max_causal_children, int(metrics["max_causal_children"])
        )
        self.max_propagation_children = max(
            self.max_propagation_children, int(metrics["max_propagation_children"])
        )
        self.truncated_episode_count += int(bool(metrics["truncated"]))
        self.used_entity_ids.update(bundle.involved_entity_ids)
        parameters = bundle.program.get("parameters", {})
        profile_id = parameters.get("calibration_profile_id")
        calibration_status = parameters.get("calibration_status")
        if profile_id:
            self.calibration_profile_counts[str(profile_id)] += 1
        if calibration_status:
            self.calibration_status_counts[str(calibration_status)] += 1
        for relation in bundle.relations:
            calibration = relation.attributes.get("lag_calibration")
            if not isinstance(calibration, dict):
                continue
            if relation.relation_type == "propagates_downstream":
                self.calibrated_propagation_lag_count += 1
                inputs = calibration.get("inputs", {})
                free_flow = float(inputs.get("free_flow_travel_seconds", 0.0))
                lag = float(relation.attributes.get("lag_seconds", 0.0))
                if free_flow > 0.0:
                    self.propagation_to_free_flow_ratios.append(lag / free_flow)
            elif relation.relation_type == "transitions_to_recovery":
                self.calibrated_recovery_lag_count += 1

    def report(self, total_entity_count: int | None = None) -> Dict[str, Any]:
        warnings: List[str] = []
        if len(self.scenario_counts) <= 1 and self.episode_count > 4:
            warnings.append("Only one scenario family is represented")
        if self.episode_count and not self.relation_type_counts.get("propagates_downstream"):
            warnings.append("No downstream propagation relation was generated")
        if self.truncated_episode_count:
            warnings.append(f"{self.truncated_episode_count} written episodes are truncated")
        return {
            "episode_count": self.episode_count,
            "scenario_counts": dict(self.scenario_counts),
            "event_type_counts": dict(self.event_type_counts),
            "event_category_counts": dict(self.event_category_counts),
            "event_metric_tag_counts": dict(self.event_metric_tag_counts),
            "relation_type_counts": dict(self.relation_type_counts),
            "events_per_episode": _numeric_summary(self.event_counts),
            "relations_per_episode": _numeric_summary(self.relation_counts),
            "inter_event_gap_seconds": _numeric_summary(self.inter_event_gaps),
            "relation_lag_seconds_by_type": {
                relation_type: _numeric_summary(values)
                for relation_type, values in sorted(self.relation_lags.items())
            },
            "max_causal_children_observed": self.max_causal_children,
            "max_propagation_children_observed": self.max_propagation_children,
            "truncated_episode_count": self.truncated_episode_count,
            "unique_episode_entity_count": len(self.used_entity_ids),
            "network_entity_utilization_fraction": (
                round(len(self.used_entity_ids) / total_entity_count, 6)
                if total_entity_count
                else None
            ),
            "multi_parent_target_fraction": (
                round(self.multi_parent_targets / self.targets, 6) if self.targets else 0.0
            ),
            "calibration_status": (
                next(iter(self.calibration_status_counts))
                if len(self.calibration_status_counts) == 1
                else "mixed_or_missing"
            ),
            "calibration_evidence": {
                "profile_counts": dict(self.calibration_profile_counts),
                "status_counts": dict(self.calibration_status_counts),
                "calibrated_propagation_lag_count": self.calibrated_propagation_lag_count,
                "calibrated_recovery_lag_count": self.calibrated_recovery_lag_count,
                "propagation_lag_to_free_flow_ratio": _numeric_summary(
                    self.propagation_to_free_flow_ratios
                ),
            },
            "warnings": warnings,
        }


def compute_episode_metrics(bundle: EpisodeBundle) -> Dict[str, Any]:
    causal_outgoing: Dict[str, int] = defaultdict(int)
    propagation_outgoing: Dict[str, int] = defaultdict(int)
    for relation in bundle.relations:
        if relation.relation_class == "causal":
            causal_outgoing[relation.source_event_id] += 1
        if relation.relation_type == "propagates_downstream":
            propagation_outgoing[relation.source_event_id] += 1
    termination = bundle.program.get("termination", {})
    return {
        "event_count": len(bundle.events),
        "relation_count": len(bundle.relations),
        "root_event_count": sum(event.event_role == "root" for event in bundle.events),
        "independent_root_count": len(bundle.program.get("independent_root_event_ids", [])),
        "max_causal_children": max(causal_outgoing.values(), default=0),
        "max_propagation_children": max(propagation_outgoing.values(), default=0),
        "recovery_role_event_count": sum(
            event.event_role == "recovery" for event in bundle.events
        ),
        "truncated": bool(termination.get("truncated", False)),
        "termination_reason": termination.get("reason", "missing"),
        "pending_propagation_frontier_count": int(
            termination.get("pending_propagation_frontier_count", 0)
        ),
        "skipped_recovery_candidate_count": int(
            termination.get("skipped_recovery_candidate_count", 0)
        ),
    }


def _validate_text_alignment(
    bundle: EpisodeBundle,
    event_by_id: Dict[str, Any],
    relation_by_id: Dict[str, Any],
    mechanism_by_id: Dict[str, Any],
    errors: List[str],
) -> bool:
    passed = True
    text_by_id = {record.text_id: record for record in bundle.texts}
    canonical_event_ids: set[str] = set()
    canonical_relation_ids: set[str] = set()
    canonical_episode_count = 0
    for record in bundle.texts:
        if any(event_id not in event_by_id for event_id in record.aligned_event_ids):
            errors.append(f"Text {record.text_id} has dangling event alignment")
            passed = False
        if any(relation_id not in relation_by_id for relation_id in record.aligned_relation_ids):
            errors.append(f"Text {record.text_id} has dangling relation alignment")
            passed = False
        if any(mechanism_id not in mechanism_by_id for mechanism_id in record.aligned_mechanism_ids):
            errors.append(f"Text {record.text_id} has dangling mechanism alignment")
            passed = False
        if any(event_id not in record.content for event_id in record.aligned_event_ids):
            errors.append(f"Text {record.text_id} does not mention every aligned event ID")
            passed = False
        if record.variant == "paraphrase":
            source = text_by_id.get(record.derived_from or "")
            if source is None or source.variant != "canonical":
                errors.append(f"Paraphrase {record.text_id} lacks a canonical source")
                passed = False
            elif record.grounded_facts != source.grounded_facts:
                errors.append(f"Paraphrase {record.text_id} changed grounded facts")
                passed = False
            elif record.verbalized_fact_keys != source.verbalized_fact_keys:
                errors.append(f"Paraphrase {record.text_id} changed fact coverage")
                passed = False
        if record.variant == "canonical" and record.text_role == "event":
            canonical_event_ids.update(record.aligned_event_ids)
            if len(record.aligned_event_ids) != 1:
                errors.append(f"Canonical event text {record.text_id} must align to one event")
                passed = False
            else:
                event = event_by_id[record.aligned_event_ids[0]]
                expected = {
                    "event_id": event.event_id,
                    "event_type_id": event.event_type_id,
                    "event_time": event.event_time,
                    "participants": [
                        participant.to_dict() for participant in event.participants
                    ],
                    "attributes": dict(event.attributes),
                }
                if record.grounded_facts != expected:
                    errors.append(f"Event text {record.text_id} has inconsistent grounded facts")
                    passed = False
                required_fact_keys = {
                    "event_id",
                    "event_type_id",
                    "event_time",
                    "participants",
                    *{f"attributes.{key}" for key in event.attributes},
                }
                if not required_fact_keys.issubset(set(record.verbalized_fact_keys)):
                    errors.append(f"Event text {record.text_id} does not verbalize all attributes")
                    passed = False
                required_phrases = [
                    event.event_type_id.rsplit(".", 1)[-1].replace("_", " "),
                    event.event_time,
                    *event.participant_ids,
                ]
                if any(phrase not in record.content for phrase in required_phrases):
                    errors.append(f"Event text {record.text_id} omits a required grounded phrase")
                    passed = False
        if record.variant == "canonical" and record.text_role == "relation":
            canonical_relation_ids.update(record.aligned_relation_ids)
            if len(record.aligned_relation_ids) != 1:
                errors.append(f"Canonical relation text {record.text_id} must align to one relation")
                passed = False
            else:
                relation = relation_by_id[record.aligned_relation_ids[0]]
                source = event_by_id[relation.source_event_id]
                target = event_by_id[relation.target_event_id]
                lag = round(target.time_offset_seconds - source.time_offset_seconds, 3)
                expected = {
                    "relation_id": relation.relation_id,
                    "source_event_id": relation.source_event_id,
                    "target_event_id": relation.target_event_id,
                    "relation_type": relation.relation_type,
                    "rule_id": relation.rule_id,
                    "status": relation.status,
                    "mechanism_group_id": relation.mechanism_group_id,
                    "time_lag_seconds": lag,
                }
                if record.grounded_facts != expected:
                    errors.append(f"Relation text {record.text_id} has inconsistent grounded facts")
                    passed = False
                if f"{lag:.3f}" not in record.content:
                    errors.append(f"Relation text {record.text_id} omits its exact lag")
                    passed = False
        if record.variant == "canonical" and record.text_role == "episode":
            canonical_episode_count += 1
            expected_order = [
                event.event_id
                for event in sorted(bundle.events, key=lambda event: event.time_offset_seconds)
            ]
            if record.grounded_facts.get("chronological_event_ids") != expected_order:
                errors.append(f"Episode text {record.text_id} has the wrong event order")
                passed = False
            independent_ids = sorted(bundle.program.get("independent_root_event_ids", []))
            if record.grounded_facts.get("independent_root_event_ids") != independent_ids:
                errors.append(f"Episode text {record.text_id} has wrong independent-root facts")
                passed = False
            if independent_ids and "independent" not in record.content.lower():
                errors.append(f"Episode text {record.text_id} does not explain independent roots")
                passed = False
    if canonical_event_ids != set(event_by_id):
        errors.append("Canonical event-text coverage is incomplete")
        passed = False
    if canonical_relation_ids != set(relation_by_id):
        errors.append("Canonical relation-text coverage is incomplete")
        passed = False
    if canonical_episode_count != 1:
        errors.append("Each episode needs exactly one canonical episode narrative")
        passed = False
    return passed


def _is_acyclic(event_ids: List[str], bundle: EpisodeBundle) -> bool:
    indegree = {event_id: 0 for event_id in event_ids}
    outgoing: Dict[str, List[str]] = defaultdict(list)
    for relation in bundle.relations:
        if relation.source_event_id in indegree and relation.target_event_id in indegree:
            outgoing[relation.source_event_id].append(relation.target_event_id)
            indegree[relation.target_event_id] += 1
    queue = deque(event_id for event_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in outgoing[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(event_ids)


def _numeric_summary(values: List[float | int]) -> Dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
        "mean": round(float(statistics.fmean(values)), 6),
        "median": round(float(statistics.median(values)), 6),
    }
