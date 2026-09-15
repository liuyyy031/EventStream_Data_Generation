"""Deterministic checks that are valid across domains."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .models import CandidateStatus, ContextEvidence, EpisodeResult
from .topology import relation_is_active


def validate_episode_result(
    result: EpisodeResult,
    domain_catalog: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, bool] = {}
    entity_ids = {entity.entity_id for entity in result.entities}
    event_by_id = {event.event_id: event for event in result.events}
    context_relation_by_id = {
        relation.relation_id: relation for relation in result.context_relations
    }
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in result.candidates
    }

    id_groups = [
        [entity.entity_id for entity in result.entities],
        [relation.relation_id for relation in result.context_relations],
        [event.event_id for event in result.events],
        [relation.relation_id for relation in result.event_relations],
        [candidate.candidate_id for candidate in result.candidates],
        [risk_set.risk_set_id for risk_set in result.risk_sets],
    ]
    checks["unique_ids"] = all(len(ids) == len(set(ids)) for ids in id_groups)
    if not checks["unique_ids"]:
        errors.append("Duplicate IDs exist")

    checks["context_relations"] = True
    for relation in result.context_relations:
        if relation.source_entity_id not in entity_ids or relation.target_entity_id not in entity_ids:
            errors.append(f"Context relation {relation.relation_id} has a missing entity")
            checks["context_relations"] = False
        if (
            relation.valid_from_offset_seconds is not None
            and relation.valid_to_offset_seconds is not None
            and relation.valid_from_offset_seconds > relation.valid_to_offset_seconds
        ):
            errors.append(f"Context relation {relation.relation_id} has an invalid interval")
            checks["context_relations"] = False

    checks["event_temporal_contract"] = True
    for event in result.events:
        temporal = event.temporal
        start = temporal.occurrence_start_offset_seconds
        end = temporal.occurrence_end_offset_seconds
        observed = temporal.observed_at_offset_seconds
        recorded = temporal.recorded_at_offset_seconds
        if start < 0.0 or start > result.duration_seconds:
            errors.append(f"Event {event.event_id} starts outside the observation window")
            checks["event_temporal_contract"] = False
        if end is not None and end < start:
            errors.append(f"Event {event.event_id} ends before it starts")
            checks["event_temporal_contract"] = False
        if observed is not None and observed < start:
            errors.append(f"Event {event.event_id} is observed before occurrence")
            checks["event_temporal_contract"] = False
        if recorded is not None and (observed is None or recorded < observed):
            errors.append(f"Event {event.event_id} has invalid recording time")
            checks["event_temporal_contract"] = False
        participant_links = {(item.entity_id, item.role) for item in event.participants}
        if not participant_links or any(entity_id not in entity_ids for entity_id, _ in participant_links):
            errors.append(f"Event {event.event_id} has invalid participants")
            checks["event_temporal_contract"] = False

    checks["event_relations"] = True
    relation_pairs = set()
    relations_by_target: Dict[str, List[Any]] = {}
    for relation in result.event_relations:
        source = event_by_id.get(relation.source_event_id)
        target = event_by_id.get(relation.target_event_id)
        if source is None or target is None:
            errors.append(f"Event relation {relation.relation_id} has a missing event")
            checks["event_relations"] = False
            continue
        relation_pairs.add((relation.source_event_id, relation.target_event_id))
        relations_by_target.setdefault(relation.target_event_id, []).append(relation)
        try:
            actual = target.temporal.anchor(
                relation.temporal_link.target_anchor
            ) - source.temporal.anchor(relation.temporal_link.source_anchor)
        except (KeyError, ValueError) as exc:
            errors.append(f"Event relation {relation.relation_id}: {exc}")
            checks["event_relations"] = False
            continue
        if abs(actual - relation.temporal_link.lag_seconds) > 1e-6:
            errors.append(f"Event relation {relation.relation_id} has inconsistent lag")
            checks["event_relations"] = False
        if actual < -1e-9:
            errors.append(f"Event relation {relation.relation_id} points backward in its anchors")
            checks["event_relations"] = False
        if not _validate_context_evidence(
            relation.context_evidence,
            record_label=f"Event relation {relation.relation_id}",
            entity_ids=entity_ids,
            context_relation_by_id=context_relation_by_id,
            duration_seconds=result.duration_seconds,
            errors=errors,
            require_passed=True,
        ):
            checks["event_relations"] = False
        generating_candidate_id = relation.provenance.get("candidate_id")
        generating_candidate = candidate_by_id.get(generating_candidate_id)
        if generating_candidate is None:
            errors.append(
                f"Event relation {relation.relation_id} lacks its generating candidate"
            )
            checks["event_relations"] = False
        elif (
            relation.context_evidence.to_dict()
            != generating_candidate.context_evidence.to_dict()
        ):
            errors.append(
                f"Event relation {relation.relation_id} does not preserve candidate context evidence"
            )
            checks["event_relations"] = False

    checks["candidate_lifecycle"] = True
    status_counts = Counter(candidate.status.value for candidate in result.candidates)
    for candidate in result.candidates:
        if candidate.status is CandidateStatus.SCHEDULED:
            errors.append(f"Candidate {candidate.candidate_id} was left scheduled")
            checks["candidate_lifecycle"] = False
        if candidate.status is CandidateStatus.FIRED:
            if not candidate.fired_event_id or candidate.fired_event_id not in event_by_id:
                errors.append(f"Fired candidate {candidate.candidate_id} lacks its event")
                checks["candidate_lifecycle"] = False
            elif any(
                (parent_id, candidate.fired_event_id) not in relation_pairs
                for parent_id in candidate.parent_event_ids
            ):
                errors.append(
                    f"Fired candidate {candidate.candidate_id} does not expose every parent as an event relation"
                )
                checks["candidate_lifecycle"] = False
            elif len(candidate.parent_event_ids) > 1:
                parent_relations = [
                    relation
                    for relation in relations_by_target.get(candidate.fired_event_id, [])
                    if relation.source_event_id in candidate.parent_event_ids
                ]
                group_ids = {
                    relation.mechanism_group_id for relation in parent_relations
                }
                if len(group_ids) != 1 or None in group_ids:
                    errors.append(
                        f"Multi-parent candidate {candidate.candidate_id} lacks one explicit mechanism group"
                    )
                    checks["candidate_lifecycle"] = False
        if candidate.status is CandidateStatus.RIGHT_CENSORED:
            if candidate.censored_at is None:
                errors.append(f"Censored candidate {candidate.candidate_id} lacks censoring time")
                checks["candidate_lifecycle"] = False
            elif not (
                candidate.activation_time - 1e-9
                <= candidate.censored_at
                <= result.duration_seconds + 1e-9
            ):
                errors.append(
                    f"Censored candidate {candidate.candidate_id} has an invalid censoring time"
                )
                checks["candidate_lifecycle"] = False
        for parent_id in candidate.parent_event_ids:
            if parent_id not in event_by_id:
                errors.append(f"Candidate {candidate.candidate_id} has a missing parent")
                checks["candidate_lifecycle"] = False
        if candidate.parent_event_ids:
            latest_parent = max(
                event_by_id[parent_id].temporal.occurrence_start_offset_seconds
                for parent_id in candidate.parent_event_ids
            )
            if candidate.activation_time + 1e-9 < latest_parent:
                errors.append(f"Candidate {candidate.candidate_id} activates before a parent")
                checks["candidate_lifecycle"] = False
        if not _validate_context_evidence(
            candidate.context_evidence,
            record_label=f"Candidate {candidate.candidate_id}",
            entity_ids=entity_ids,
            context_relation_by_id=context_relation_by_id,
            duration_seconds=result.duration_seconds,
            errors=errors,
            require_passed=candidate.status is CandidateStatus.FIRED,
        ):
            checks["candidate_lifecycle"] = False

    checks["risk_set_contract"] = True
    risk_set_by_event = {
        risk_set.selected_event_id: risk_set for risk_set in result.risk_sets
    }
    if len(risk_set_by_event) != len(result.risk_sets):
        errors.append("More than one risk set selects the same event")
        checks["risk_set_contract"] = False
    if set(risk_set_by_event) != set(event_by_id):
        errors.append("Fired events and risk-set selections are not one-to-one")
        checks["risk_set_contract"] = False
    for risk_set in result.risk_sets:
        selected_candidate = candidate_by_id.get(risk_set.selected_candidate_id)
        selected_event = event_by_id.get(risk_set.selected_event_id)
        alternative_by_id = {
            item.candidate_id: item for item in risk_set.alternatives
        }
        if selected_candidate is None or selected_event is None:
            errors.append(f"Risk set {risk_set.risk_set_id} selects a missing record")
            checks["risk_set_contract"] = False
            continue
        if selected_candidate.fired_event_id != selected_event.event_id:
            errors.append(f"Risk set {risk_set.risk_set_id} disagrees with candidate lifecycle")
            checks["risk_set_contract"] = False
        if selected_candidate.risk_set_id != risk_set.risk_set_id:
            errors.append(f"Risk set {risk_set.risk_set_id} is not linked from its candidate")
            checks["risk_set_contract"] = False
        if selected_event.provenance.get("risk_set_id") != risk_set.risk_set_id:
            errors.append(f"Risk set {risk_set.risk_set_id} is not linked from its event")
            checks["risk_set_contract"] = False
        if risk_set.selected_candidate_id not in alternative_by_id:
            errors.append(f"Risk set {risk_set.risk_set_id} omits its selected candidate")
            checks["risk_set_contract"] = False
        if abs(
            selected_event.temporal.occurrence_start_offset_seconds
            - risk_set.evaluated_at_offset_seconds
        ) > 1e-8:
            errors.append(f"Risk set {risk_set.risk_set_id} uses the wrong decision time")
            checks["risk_set_contract"] = False
        for alternative in risk_set.alternatives:
            if alternative.candidate_id not in candidate_by_id:
                errors.append(
                    f"Risk set {risk_set.risk_set_id} references an unknown candidate"
                )
                checks["risk_set_contract"] = False
            hazard = alternative.clock_measure.get("hazard_per_second")
            survival = alternative.clock_measure.get("survival_probability")
            if hazard is not None and hazard < 0.0:
                errors.append(f"Risk set {risk_set.risk_set_id} has a negative hazard")
                checks["risk_set_contract"] = False
            if survival is not None and not 0.0 <= survival <= 1.0:
                errors.append(f"Risk set {risk_set.risk_set_id} has invalid survival")
                checks["risk_set_contract"] = False
        if risk_set.selection_mode == "cause_specific_competing_hazards":
            probabilities = [
                item.candidate_probability_given_time
                for item in risk_set.alternatives
                if item.candidate_probability_given_time is not None
            ]
            if not probabilities or abs(sum(probabilities) - 1.0) > 1e-8:
                errors.append(
                    f"Risk set {risk_set.risk_set_id} has unnormalized candidate probabilities"
                )
                checks["risk_set_contract"] = False
            for component in (
                "type_given_time",
                "entity_given_time_and_type",
                "mechanism_given_time_type_and_entity",
            ):
                probability = risk_set.factorization.get(component, {}).get(
                    "probability"
                )
                if probability is None or not 0.0 <= probability <= 1.0:
                    errors.append(
                        f"Risk set {risk_set.risk_set_id} has an invalid {component} probability"
                    )
                    checks["risk_set_contract"] = False
        attributed_parents = set(
            risk_set.parent_attribution.get("realized_parent_event_ids", [])
        )
        if attributed_parents != set(selected_candidate.parent_event_ids):
            errors.append(f"Risk set {risk_set.risk_set_id} misstates realized parents")
            checks["risk_set_contract"] = False

    checks["domain_catalog_usage"] = True
    if domain_catalog is not None:
        catalog_error_count_before = len(errors)
        if domain_catalog.get("domain") != result.domain:
            errors.append("Domain catalog does not match the episode domain")
            checks["domain_catalog_usage"] = False
        _validate_typed_records(
            result.entities,
            domain_catalog.get("entity_types", []),
            "entity_type_id",
            "entity_id",
            errors,
        )
        _validate_typed_records(
            result.events,
            domain_catalog.get("event_types", []),
            "event_type_id",
            "event_id",
            errors,
        )
        registered_context_relations = {
            item["relation_type_id"]
            for item in domain_catalog.get("context_relation_types", [])
        }
        registered_event_relations = {
            item["relation_type_id"]
            for item in domain_catalog.get("event_relation_types", [])
        }
        event_relation_definition_by_id = {
            item["relation_type_id"]: item
            for item in domain_catalog.get("event_relation_types", [])
        }
        registered_mechanisms = {
            item["mechanism_id"]
            for item in domain_catalog.get("mechanisms", [])
        }
        mechanism_definition_by_id = {
            item["mechanism_id"]: item
            for item in domain_catalog.get("mechanisms", [])
        }
        for relation in result.context_relations:
            if relation.relation_type_id not in registered_context_relations:
                errors.append(
                    f"Context relation {relation.relation_id} uses an unregistered type"
                )
                checks["domain_catalog_usage"] = False
        for relation in result.event_relations:
            if relation.relation_type_id not in registered_event_relations:
                errors.append(
                    f"Event relation {relation.relation_id} uses an unregistered type"
                )
                checks["domain_catalog_usage"] = False
            else:
                expected_class = event_relation_definition_by_id[
                    relation.relation_type_id
                ].get("relation_class")
                if expected_class and relation.relation_class != expected_class:
                    errors.append(
                        f"Event relation {relation.relation_id} has class "
                        f"{relation.relation_class}, expected {expected_class}"
                    )
                    checks["domain_catalog_usage"] = False
        for candidate in result.candidates:
            if candidate.mechanism_id not in registered_mechanisms:
                errors.append(
                    f"Candidate {candidate.candidate_id} uses an unregistered mechanism"
                )
                checks["domain_catalog_usage"] = False
            else:
                mechanism_definition = mechanism_definition_by_id[
                    candidate.mechanism_id
                ]
                expected_model_ref = mechanism_definition.get(
                    "generation_temporal_model_ref"
                )
                if (
                    expected_model_ref
                    and candidate.temporal_model_ref != expected_model_ref
                ):
                    errors.append(
                        f"Candidate {candidate.candidate_id} uses temporal model "
                        f"{candidate.temporal_model_ref}, expected {expected_model_ref}"
                    )
                    checks["domain_catalog_usage"] = False
                expected_combination = mechanism_definition.get(
                    "parent_combination"
                )
                if (
                    expected_combination
                    and candidate.combination != expected_combination
                ):
                    errors.append(
                        f"Candidate {candidate.candidate_id} uses parent combination "
                        f"{candidate.combination}, expected {expected_combination}"
                    )
                    checks["domain_catalog_usage"] = False
        for risk_set in result.risk_sets:
            selected_candidate = candidate_by_id.get(risk_set.selected_candidate_id)
            if selected_candidate is None:
                continue
            mechanism_definition = mechanism_definition_by_id.get(
                selected_candidate.mechanism_id, {}
            )
            expected_attribution = mechanism_definition.get(
                "parent_attribution_mode"
            )
            if (
                expected_attribution
                and risk_set.parent_attribution.get("mode")
                != expected_attribution
            ):
                errors.append(
                    f"Risk set {risk_set.risk_set_id} uses parent attribution "
                    f"{risk_set.parent_attribution.get('mode')}, expected "
                    f"{expected_attribution}"
                )
                checks["domain_catalog_usage"] = False
        if len(errors) > catalog_error_count_before and checks["domain_catalog_usage"]:
            # Attribute/type errors from the helper are catalog failures.
            checks["domain_catalog_usage"] = False

    if not result.events:
        warnings.append("Episode contains no fired events")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "candidate_status_counts": dict(status_counts),
    }


def _validate_typed_records(
    records: List[Any],
    definitions: List[Dict[str, Any]],
    type_field: str,
    id_field: str,
    errors: List[str],
) -> None:
    definition_by_id = {item[type_field]: item for item in definitions}
    for record in records:
        record_type = getattr(record, type_field)
        record_id = getattr(record, id_field)
        definition = definition_by_id.get(record_type)
        if definition is None:
            errors.append(f"Record {record_id} uses unregistered type {record_type}")
            continue
        missing = set(definition.get("required_attributes", [])) - set(
            record.attributes
        )
        if missing:
            errors.append(
                f"Record {record_id} lacks required attributes: {sorted(missing)}"
            )


def _validate_context_evidence(
    evidence: ContextEvidence,
    *,
    record_label: str,
    entity_ids: set[str],
    context_relation_by_id: Dict[str, Any],
    duration_seconds: float,
    errors: List[str],
    require_passed: bool,
) -> bool:
    valid = True
    if len(evidence.context_relation_ids) != len(set(evidence.context_relation_ids)):
        errors.append(f"{record_label} repeats a context relation in its evidence")
        valid = False
    if len(evidence.entity_ids) != len(set(evidence.entity_ids)):
        errors.append(f"{record_label} repeats an entity in its context evidence")
        valid = False
    unknown_entities = set(evidence.entity_ids) - entity_ids
    if unknown_entities:
        errors.append(
            f"{record_label} context evidence references unknown entities: {sorted(unknown_entities)}"
        )
        valid = False
    evaluated_at = evidence.evaluated_at_offset_seconds
    if evidence.context_relation_ids and evaluated_at is None:
        errors.append(f"{record_label} relation evidence lacks an evaluation time")
        valid = False
    if evaluated_at is not None and not 0.0 <= evaluated_at <= duration_seconds:
        errors.append(f"{record_label} has an invalid context evaluation time")
        valid = False
    for relation_id in evidence.context_relation_ids:
        relation = context_relation_by_id.get(relation_id)
        if relation is None:
            errors.append(f"{record_label} references unknown context relation {relation_id}")
            valid = False
            continue
        endpoints = {relation.source_entity_id, relation.target_entity_id}
        if not endpoints <= set(evidence.entity_ids):
            errors.append(
                f"{record_label} does not expose both endpoints of context relation {relation_id}"
            )
            valid = False
        if (
            require_passed
            and evaluated_at is not None
            and not relation_is_active(relation, evaluated_at)
        ):
            errors.append(
                f"{record_label} cites inactive context relation {relation_id}"
            )
            valid = False
    for predicate in evidence.state_predicates:
        if not predicate.state_path or not predicate.operator:
            errors.append(f"{record_label} contains an incomplete state predicate")
            valid = False
        if (
            predicate.subject_entity_id is not None
            and predicate.subject_entity_id not in entity_ids
        ):
            errors.append(f"{record_label} state predicate references an unknown entity")
            valid = False
        if require_passed and not predicate.passed:
            errors.append(f"{record_label} fired despite failed context predicate")
            valid = False
        if predicate.operator == "equals" and predicate.passed != (
            predicate.actual == predicate.expected
        ):
            errors.append(f"{record_label} has inconsistent equals-predicate evidence")
            valid = False
        if predicate.operator == "not_equals" and predicate.passed != (
            predicate.actual != predicate.expected
        ):
            errors.append(f"{record_label} has inconsistent not-equals predicate evidence")
            valid = False
    return valid


def validate_text_alignment(
    result: EpisodeResult,
    payload: Dict[str, Any],
    domain_catalog: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    if payload.get("episode_id") != result.episode_id:
        errors.append("Text alignment uses the wrong episode_id")
    sentences = payload.get("sentences")
    claims = payload.get("claims")
    if not isinstance(sentences, list) or not all(
        isinstance(sentence, str) and sentence.strip() for sentence in sentences
    ):
        errors.append("Text alignment must contain non-empty sentences")
        sentences = []
    if not isinstance(claims, list):
        errors.append("Text alignment claims must be a list")
        claims = []
    event_ids = {event.event_id for event in result.events}
    relation_ids = {relation.relation_id for relation in result.event_relations}
    relation_by_id = {
        relation.relation_id: relation for relation in result.event_relations
    }
    catalog_relation_by_type = {
        item["relation_type_id"]: item
        for item in (domain_catalog or {}).get("event_relation_types", [])
    }
    covered_events: set[str] = set()
    covered_relations: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("Text alignment contains a non-object claim")
            continue
        sentence_index = claim.get("sentence_index")
        sentence = None
        if not isinstance(sentence_index, int) or not 0 <= sentence_index < len(sentences):
            errors.append("Text alignment claim has an invalid sentence_index")
        else:
            sentence = sentences[sentence_index]
        claim_events = set(claim.get("event_ids", []))
        claim_relations = set(claim.get("relation_ids", []))
        if not claim_events <= event_ids:
            errors.append("Text alignment claim references an unknown event")
        if not claim_relations <= relation_ids:
            errors.append("Text alignment claim references an unknown relation")
        if not claim_events and not claim_relations:
            errors.append("Text alignment claim has no structured evidence target")
        assertions = claim.get("relation_assertions", [])
        if not isinstance(assertions, list) or any(
            not isinstance(assertion, dict) for assertion in assertions
        ):
            errors.append("Text alignment relation_assertions must be a list of objects")
            assertions = []
        assertion_by_relation_id = {
            assertion.get("relation_id"): assertion
            for assertion in assertions
            if assertion.get("relation_id")
        }
        if set(assertion_by_relation_id) != claim_relations:
            errors.append(
                "Text alignment relation assertions do not match the claimed relations"
            )
        for relation_id in claim_relations:
            relation = relation_by_id.get(relation_id)
            assertion = assertion_by_relation_id.get(relation_id)
            if relation is None or assertion is None:
                continue
            asserted_class = assertion.get("asserted_relation_class")
            surface_predicate = assertion.get("surface_predicate")
            if asserted_class != relation.relation_class:
                errors.append(
                    f"Text claim for relation {relation_id} asserts class "
                    f"{asserted_class}, expected {relation.relation_class}"
                )
            if not isinstance(surface_predicate, str) or not surface_predicate.strip():
                errors.append(
                    f"Text claim for relation {relation_id} lacks a surface predicate"
                )
            elif sentence is not None and surface_predicate not in sentence:
                errors.append(
                    f"Text claim for relation {relation_id} does not use its declared "
                    "surface predicate"
                )
            if domain_catalog is not None:
                relation_definition = catalog_relation_by_type.get(
                    relation.relation_type_id, {}
                )
                expected_rendering = relation_definition.get("text_rendering")
                if not isinstance(expected_rendering, dict):
                    errors.append(
                        f"Catalog relation type {relation.relation_type_id} lacks "
                        "a deterministic text rendering"
                    )
                    continue
                expected_class = expected_rendering.get(
                    "asserted_relation_class"
                )
                expected_predicate = expected_rendering.get("surface_predicate")
                if expected_class != relation.relation_class:
                    errors.append(
                        f"Catalog text rendering for {relation.relation_type_id} "
                        f"asserts class {expected_class}, expected "
                        f"{relation.relation_class}"
                    )
                if asserted_class != expected_class:
                    errors.append(
                        f"Text claim for relation {relation_id} does not preserve "
                        "the catalog relation class"
                    )
                if surface_predicate != expected_predicate:
                    errors.append(
                        f"Text claim for relation {relation_id} does not preserve "
                        "the catalog surface predicate"
                    )
        covered_events.update(claim_events)
        covered_relations.update(claim_relations)
    if covered_events != event_ids:
        errors.append("Text alignment does not cover every emitted event")
    if covered_relations != relation_ids:
        errors.append("Text alignment does not cover every emitted event relation")
    if sentences and payload.get("text") != " ".join(sentences):
        errors.append("Text alignment text does not equal its joined sentences")
    return {"passed": not errors, "errors": errors}
