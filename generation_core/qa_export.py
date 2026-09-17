"""Ground-truth-derived QA and LLM training views for accepted episodes.

The exporter is deliberately downstream of generation and semantic correction:
answers come only from immutable structured records, while visible evidence uses
the final (possibly corrected) grounded text.  No LLM is asked to invent labels.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .models import EpisodeResult, EventRecord, EventRelation


QA_SCHEMA_VERSION = "event-sequence-qa-v1"
TRAINING_SCHEMA_VERSION = "event-sequence-training-v1"
SYSTEM_PROMPT = (
    "Answer the event-sequence question using only the supplied evidence. "
    "Preserve the declared relation semantics: a transition is not automatically "
    "a cause, statistical influence is not proven causality, and context adjacency "
    "is not event parenthood."
)


def generate_episode_qa(
    result: EpisodeResult,
    text_alignment: Dict[str, Any],
    *,
    history_window_events: int = 32,
) -> List[Dict[str, Any]]:
    """Create a bounded set of QA labels from one accepted episode."""

    ordered_events = _ordered_events(result.events)
    if not ordered_events:
        return []
    event_by_id = {event.event_id: event for event in result.events}
    relation_by_id = {
        relation.relation_id: relation for relation in result.event_relations
    }
    incoming = _incoming_relations(result.event_relations)
    sentence_index = _GroundedSentenceIndex(text_alignment)
    qas: List[Dict[str, Any]] = []

    temporal_ids = [event.event_id for event in ordered_events]
    qas.append(
        _qa_record(
            result,
            qa_type="temporal_order",
            suffix="all",
            question="List all event IDs in ascending occurrence-time order.",
            answer=temporal_ids,
            answer_source="events.temporal.occurrence_start_offset_seconds",
            input_view="all_event_occurrences",
            input_text=_render_evidence(
                result,
                sentence_index,
                event_ids=temporal_ids,
                relation_ids=[],
                view_name="all_event_occurrences",
            ),
            evidence_record_ids=temporal_ids,
        )
    )

    if len(ordered_events) >= 2:
        target_index = max(1, len(ordered_events) // 2)
        target = ordered_events[target_index]
        window = max(1, int(history_window_events))
        visible_events = ordered_events[max(0, target_index - window) : target_index]
        visible_ids = [event.event_id for event in visible_events]
        qas.append(
            _qa_record(
                result,
                qa_type="next_event_type",
                suffix=target.event_id,
                question=(
                    "What event type occurs next after the final visible event in "
                    "this generated sequence prefix?"
                ),
                answer=target.event_type_id,
                answer_source="events.event_type_id",
                input_view="observed_event_prefix",
                input_text=_render_evidence(
                    result,
                    sentence_index,
                    event_ids=visible_ids,
                    relation_ids=[],
                    view_name="observed_event_prefix",
                    extra_lines=[
                        f"visible_event_count={len(visible_ids)}",
                        "The event immediately after this prefix is hidden.",
                    ],
                ),
                evidence_record_ids=[*visible_ids, target.event_id],
                target_event_id=target.event_id,
            )
        )

    if incoming:
        target_id = _select_relation_target(
            incoming,
            event_by_id,
        )
        direct_relations = _sorted_relations(incoming[target_id], event_by_id)
        parent_answer = [_relation_answer(relation) for relation in direct_relations]
        direct_event_ids = _ordered_unique(
            [
                *(relation.source_event_id for relation in direct_relations),
                target_id,
            ]
        )
        direct_relation_ids = [relation.relation_id for relation in direct_relations]
        direct_evidence = [*direct_event_ids, *direct_relation_ids]
        qas.append(
            _qa_record(
                result,
                qa_type="direct_predecessors",
                suffix=target_id,
                question=(
                    f"Which directly linked predecessor events point to event "
                    f"{target_id}, and what is each declared relationship?"
                ),
                answer=parent_answer,
                answer_source="event_relations.incoming",
                input_view="direct_relation_evidence",
                input_text=_render_evidence(
                    result,
                    sentence_index,
                    event_ids=direct_event_ids,
                    relation_ids=direct_relation_ids,
                    view_name="direct_relation_evidence",
                ),
                evidence_record_ids=direct_evidence,
                target_event_id=target_id,
            )
        )

        root_ids, ancestor_event_ids, ancestor_relation_ids = _upstream_roots(
            target_id,
            incoming,
            event_by_id,
        )
        qas.append(
            _qa_record(
                result,
                qa_type="upstream_roots",
                suffix=target_id,
                question=(
                    f"Which parentless root events are upstream of event {target_id} "
                    "in the generated event-relation graph?"
                ),
                answer=root_ids,
                answer_source="event_relations.upstream_roots",
                input_view="upstream_relation_subgraph",
                input_text=_render_evidence(
                    result,
                    sentence_index,
                    event_ids=ancestor_event_ids,
                    relation_ids=ancestor_relation_ids,
                    view_name="upstream_relation_subgraph",
                ),
                evidence_record_ids=[
                    *ancestor_event_ids,
                    *ancestor_relation_ids,
                ],
                target_event_id=target_id,
            )
        )

        selected_relation = direct_relations[0]
        relation_event_ids = [
            selected_relation.source_event_id,
            selected_relation.target_event_id,
        ]
        relation_evidence = [
            *relation_event_ids,
            selected_relation.relation_id,
        ]
        relation_input = _render_evidence(
            result,
            sentence_index,
            event_ids=relation_event_ids,
            relation_ids=[selected_relation.relation_id],
            view_name="single_relation_evidence",
        )
        qas.append(
            _qa_record(
                result,
                qa_type="relation_semantics",
                suffix=selected_relation.relation_id,
                question=(
                    f"What declared relationship connects event "
                    f"{selected_relation.source_event_id} to event "
                    f"{selected_relation.target_event_id}?"
                ),
                answer=_relation_answer(selected_relation),
                answer_source="event_relations",
                input_view="single_relation_evidence",
                input_text=relation_input,
                evidence_record_ids=relation_evidence,
                target_event_id=selected_relation.target_event_id,
                source_event_id=selected_relation.source_event_id,
                relation_id=selected_relation.relation_id,
            )
        )
        qas.append(
            _qa_record(
                result,
                qa_type="relation_lag",
                suffix=selected_relation.relation_id,
                question=(
                    f"What is the declared occurrence-time lag in seconds from "
                    f"event {selected_relation.source_event_id} to event "
                    f"{selected_relation.target_event_id}?"
                ),
                answer=selected_relation.temporal_link.lag_seconds,
                answer_source="event_relations.temporal_link.lag_seconds",
                input_view="single_relation_evidence",
                input_text=relation_input,
                evidence_record_ids=relation_evidence,
                target_event_id=selected_relation.target_event_id,
                source_event_id=selected_relation.source_event_id,
                relation_id=selected_relation.relation_id,
            )
        )

    for qa in qas:
        validation = validate_qa_pair(result, qa)
        qa["validation"] = validation
    return qas


def validate_qa_pair(
    result: EpisodeResult,
    qa: Dict[str, Any],
) -> Dict[str, Any]:
    """Recompute one answer from structured truth and compare it exactly."""

    try:
        expected = _expected_answer(result, qa)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        return {"passed": False, "errors": [str(exc)]}
    if qa.get("answer") != expected:
        return {
            "passed": False,
            "errors": [
                "QA answer mismatch: "
                f"expected={expected!r}; actual={qa.get('answer')!r}"
            ],
        }
    if not str(qa.get("input", "")).strip():
        return {"passed": False, "errors": ["QA input evidence is empty"]}
    return {"passed": True, "errors": []}


def instruction_training_example(qa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "id": f"{qa['episode_id']}:{qa['qa_id']}",
        "source_episode_id": qa["episode_id"],
        "qa_id": qa["qa_id"],
        "task_type": qa["qa_type"],
        "instruction": qa["question"],
        "input": qa["input"],
        "output": _answer_text(qa["answer"]),
        "answer": qa["answer"],
        "answer_source": qa["answer_source"],
        "metadata": _training_metadata(qa),
    }


def chat_training_example(qa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "id": f"{qa['episode_id']}:{qa['qa_id']}",
        "source_episode_id": qa["episode_id"],
        "qa_id": qa["qa_id"],
        "task_type": qa["qa_type"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{qa['input']}\n\nQuestion: {qa['question']}",
            },
            {"role": "assistant", "content": _answer_text(qa["answer"])},
        ],
        "answer": qa["answer"],
        "answer_source": qa["answer_source"],
        "metadata": _training_metadata(qa),
    }


def training_export_report(
    qas: Iterable[Dict[str, Any]],
    *,
    instruction_path: str,
    chat_path: str,
) -> Dict[str, Any]:
    records = list(qas)
    task_counts = Counter(str(record["qa_type"]) for record in records)
    episode_ids = {str(record["episode_id"]) for record in records}
    validation_failures = sum(
        not bool(record.get("validation", {}).get("passed")) for record in records
    )
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "qa_schema_version": QA_SCHEMA_VERSION,
        "num_source_episodes": len(episode_ids),
        "num_qa_pairs": len(records),
        "num_instruction_examples": len(records),
        "num_chat_examples": len(records),
        "qa_validation_failure_count": validation_failures,
        "task_type_counts": dict(sorted(task_counts.items())),
        "outputs": {
            "instruction_jsonl": instruction_path,
            "chat_jsonl": chat_path,
        },
        "label_policy": "answers_are_recomputed_from_immutable_structured_truth",
        "semantic_text_policy": (
            "inputs_use_final_grounded_text_after_any_accepted_semantic_correction"
        ),
    }


def _qa_record(
    result: EpisodeResult,
    *,
    qa_type: str,
    suffix: str,
    question: str,
    answer: Any,
    answer_source: str,
    input_view: str,
    input_text: str,
    evidence_record_ids: Sequence[str],
    target_event_id: str | None = None,
    source_event_id: str | None = None,
    relation_id: str | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": QA_SCHEMA_VERSION,
        "qa_id": f"{result.episode_id}_qa_{qa_type}_{suffix}",
        "episode_id": result.episode_id,
        "domain": result.domain,
        "context_id": result.context_id,
        "qa_type": qa_type,
        "question": question,
        "answer": answer,
        "answer_source": answer_source,
        "input_view": input_view,
        "input": input_text,
        "evidence_record_ids": list(evidence_record_ids),
        "target_event_id": target_event_id,
        "source_event_id": source_event_id,
        "relation_id": relation_id,
        "metadata": {
            "scenario_family": result.episode_attributes.get("scenario_family"),
            "answers_generated_by_llm": False,
            "structured_truth_frozen": True,
        },
    }


def _expected_answer(result: EpisodeResult, qa: Dict[str, Any]) -> Any:
    qa_type = qa.get("qa_type")
    event_by_id = {event.event_id: event for event in result.events}
    relation_by_id = {
        relation.relation_id: relation for relation in result.event_relations
    }
    incoming = _incoming_relations(result.event_relations)
    if qa_type == "temporal_order":
        return [event.event_id for event in _ordered_events(result.events)]
    if qa_type == "next_event_type":
        return event_by_id[str(qa["target_event_id"])].event_type_id
    if qa_type == "direct_predecessors":
        relations = _sorted_relations(
            incoming[str(qa["target_event_id"])],
            event_by_id,
        )
        return [_relation_answer(relation) for relation in relations]
    if qa_type == "upstream_roots":
        roots, _, _ = _upstream_roots(
            str(qa["target_event_id"]),
            incoming,
            event_by_id,
        )
        return roots
    if qa_type == "relation_semantics":
        return _relation_answer(relation_by_id[str(qa["relation_id"])])
    if qa_type == "relation_lag":
        return relation_by_id[str(qa["relation_id"])].temporal_link.lag_seconds
    raise ValueError(f"Unsupported QA type: {qa_type}")


def _relation_answer(relation: EventRelation) -> Dict[str, Any]:
    return {
        "source_event_id": relation.source_event_id,
        "target_event_id": relation.target_event_id,
        "relation_id": relation.relation_id,
        "relation_class": relation.relation_class,
        "relation_type_id": relation.relation_type_id,
    }


def _ordered_events(events: Iterable[EventRecord]) -> List[EventRecord]:
    return sorted(
        events,
        key=lambda event: (
            event.temporal.occurrence_start_offset_seconds,
            event.event_id,
        ),
    )


def _incoming_relations(
    relations: Iterable[EventRelation],
) -> Dict[str, List[EventRelation]]:
    output: Dict[str, List[EventRelation]] = {}
    for relation in relations:
        output.setdefault(relation.target_event_id, []).append(relation)
    return output


def _sorted_relations(
    relations: Iterable[EventRelation],
    event_by_id: Dict[str, EventRecord],
) -> List[EventRelation]:
    return sorted(
        relations,
        key=lambda relation: (
            event_by_id[relation.source_event_id]
            .temporal.occurrence_start_offset_seconds,
            relation.source_event_id,
            relation.relation_id,
        ),
    )


def _select_relation_target(
    incoming: Dict[str, List[EventRelation]],
    event_by_id: Dict[str, EventRecord],
) -> str:
    memo: Dict[str, int] = {}

    def depth(event_id: str, visiting: set[str]) -> int:
        if event_id in memo:
            return memo[event_id]
        if event_id in visiting:
            return 0
        parents = incoming.get(event_id, [])
        if not parents:
            memo[event_id] = 0
            return 0
        next_visiting = set(visiting)
        next_visiting.add(event_id)
        value = 1 + max(
            depth(relation.source_event_id, next_visiting) for relation in parents
        )
        memo[event_id] = value
        return value

    return max(
        incoming,
        key=lambda event_id: (
            len(incoming[event_id]) > 1,
            depth(event_id, set()),
            event_by_id[event_id].temporal.occurrence_start_offset_seconds,
            event_id,
        ),
    )


def _upstream_roots(
    target_id: str,
    incoming: Dict[str, List[EventRelation]],
    event_by_id: Dict[str, EventRecord],
) -> Tuple[List[str], List[str], List[str]]:
    roots: set[str] = set()
    event_ids: set[str] = {target_id}
    relation_ids: set[str] = set()
    stack = [target_id]
    visited: set[str] = set()
    while stack:
        event_id = stack.pop()
        if event_id in visited:
            continue
        visited.add(event_id)
        parents = incoming.get(event_id, [])
        if not parents:
            roots.add(event_id)
            continue
        for relation in parents:
            relation_ids.add(relation.relation_id)
            event_ids.add(relation.source_event_id)
            stack.append(relation.source_event_id)
    ordered_event_ids = [
        event.event_id
        for event in _ordered_events(event_by_id[event_id] for event_id in event_ids)
    ]
    ordered_relation_ids = sorted(
        relation_ids,
        key=lambda relation_id: relation_id,
    )
    ordered_roots = [event_id for event_id in ordered_event_ids if event_id in roots]
    return ordered_roots, ordered_event_ids, ordered_relation_ids


class _GroundedSentenceIndex:
    def __init__(self, text_alignment: Dict[str, Any]) -> None:
        sentences = list(text_alignment.get("sentences", []))
        self.projection_mode = text_alignment.get("projection_mode")
        self.event_sentences: Dict[str, str] = {}
        self.relation_sentences: Dict[str, str] = {}
        for claim in text_alignment.get("claims", []):
            if not isinstance(claim, dict):
                continue
            index = claim.get("sentence_index")
            if not isinstance(index, int) or not 0 <= index < len(sentences):
                continue
            sentence = str(sentences[index])
            relation_ids = [str(item) for item in claim.get("relation_ids", [])]
            event_ids = [str(item) for item in claim.get("event_ids", [])]
            if relation_ids:
                for relation_id in relation_ids:
                    self.relation_sentences.setdefault(relation_id, sentence)
            else:
                for event_id in event_ids:
                    self.event_sentences.setdefault(event_id, sentence)


def _render_evidence(
    result: EpisodeResult,
    sentence_index: _GroundedSentenceIndex,
    *,
    event_ids: Sequence[str],
    relation_ids: Sequence[str],
    view_name: str,
    extra_lines: Sequence[str] = (),
) -> str:
    event_by_id = {event.event_id: event for event in result.events}
    relation_by_id = {
        relation.relation_id: relation for relation in result.event_relations
    }
    lines = [
        "Synthetic event-sequence evidence:",
        f"episode_id={result.episode_id}",
        f"domain={result.domain}",
        f"view={view_name}",
        f"text_projection_mode={sentence_index.projection_mode}",
        *extra_lines,
        "Events:",
    ]
    selected_events = [
        event_by_id[event_id] for event_id in _ordered_unique(event_ids)
    ]
    for event in _ordered_events(selected_events):
        sentence = sentence_index.event_sentences.get(event.event_id)
        if sentence is None:
            sentence = _fallback_event_sentence(event)
        lines.append(f"- event_id={event.event_id}; {sentence}")
    if relation_ids:
        lines.append("Declared event relations:")
        for relation_id in _ordered_unique(relation_ids):
            relation = relation_by_id[relation_id]
            sentence = sentence_index.relation_sentences.get(relation_id)
            if sentence is None:
                sentence = _fallback_relation_sentence(relation)
            lines.append(f"- relation_id={relation_id}; {sentence}")
    return "\n".join(lines)


def _fallback_event_sentence(event: EventRecord) -> str:
    return (
        f"occurrence_offset_seconds="
        f"{event.temporal.occurrence_start_offset_seconds:.3f}; "
        f"event_type_id={event.event_type_id}; "
        f"participants={json.dumps([item.to_dict() for item in event.participants], ensure_ascii=False)}; "
        f"attributes={json.dumps(event.attributes, ensure_ascii=False, sort_keys=True)}"
    )


def _fallback_relation_sentence(relation: EventRelation) -> str:
    return (
        f"source_event_id={relation.source_event_id}; "
        f"target_event_id={relation.target_event_id}; "
        f"relation_class={relation.relation_class}; "
        f"relation_type_id={relation.relation_type_id}; "
        f"lag_seconds={relation.temporal_link.lag_seconds}"
    )


def _training_metadata(qa: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(qa.get("metadata", {}))
    return {
        "domain": qa.get("domain"),
        "context_id": qa.get("context_id"),
        "scenario_family": metadata.get("scenario_family"),
        "input_view": qa.get("input_view"),
        "evidence_record_ids": list(qa.get("evidence_record_ids", [])),
        "structured_truth_frozen": True,
        "answers_generated_by_llm": False,
    }


def _answer_text(answer: Any) -> str:
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False, sort_keys=True)
    return str(answer)


def _ordered_unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(item) for item in items))
