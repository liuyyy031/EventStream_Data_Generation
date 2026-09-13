"""Normalized reference-data contract used before any parameter fitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Protocol

from .models import ContextRelation, Entity, EventParticipant


REFERENCE_SPLITS = {"train", "validation", "holdout"}


@dataclass(frozen=True)
class ReferenceWindow:
    window_id: str
    context_id: str
    domain: str
    start_time: str
    duration_seconds: float
    split: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceEvent:
    source_record_id: str
    window_id: str
    context_id: str
    domain: str
    event_type_id: str
    occurrence_offset_seconds: float
    participants: List[EventParticipant]
    observed_at_offset_seconds: float | None = None
    recorded_at_offset_seconds: float | None = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    explicit_parent_record_ids: List[str] = field(default_factory=list)
    relation_or_trace_ids: List[str] = field(default_factory=list)
    context_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceDataset:
    domain: str
    windows: List[ReferenceWindow]
    events: List[ReferenceEvent]
    entities: List[Entity] = field(default_factory=list)
    context_relations: List[ContextRelation] = field(default_factory=list)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = {
            "domain": self.domain,
            "windows": [item.to_dict() for item in sorted(self.windows, key=lambda x: x.window_id)],
            "events": [item.to_dict() for item in sorted(self.events, key=lambda x: x.source_record_id)],
            "entities": [item.to_dict() for item in sorted(self.entities, key=lambda x: x.entity_id)],
            "context_relations": [
                item.to_dict()
                for item in sorted(self.context_relations, key=lambda x: x.relation_id)
            ],
            "source_metadata": self.source_metadata,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class ReferenceDataAdapter(Protocol):
    def load(self, source: str | Path) -> ReferenceDataset: ...


class NormalizedJsonlReferenceAdapter:
    """Read the common contract; raw-source mapping stays in a domain adapter."""

    def load(self, source: str | Path) -> ReferenceDataset:
        root = Path(source)
        metadata_path = root / "reference_manifest.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        domain = str(metadata["domain"])
        windows = [ReferenceWindow(**item) for item in _read_jsonl(root / "reference_windows.jsonl")]
        events = []
        for item in _read_jsonl(root / "reference_events.jsonl"):
            copied = dict(item)
            copied["participants"] = [
                EventParticipant(**participant)
                for participant in copied.get("participants", [])
            ]
            events.append(ReferenceEvent(**copied))
        entities = [Entity(**item) for item in _read_optional_jsonl(root / "reference_entities.jsonl")]
        relations = [
            ContextRelation(**item)
            for item in _read_optional_jsonl(root / "reference_context_relations.jsonl")
        ]
        dataset = ReferenceDataset(
            domain=domain,
            windows=windows,
            events=events,
            entities=entities,
            context_relations=relations,
            source_metadata=metadata,
        )
        report = validate_reference_dataset(dataset)
        if not report["passed"]:
            raise ValueError("Invalid normalized reference dataset: " + "; ".join(report["errors"]))
        return dataset


def validate_reference_dataset(dataset: ReferenceDataset) -> Dict[str, Any]:
    errors: list[str] = []
    window_by_id = {item.window_id: item for item in dataset.windows}
    event_by_id = {item.source_record_id: item for item in dataset.events}
    entity_ids = {item.entity_id for item in dataset.entities}
    if not dataset.domain:
        errors.append("Reference dataset domain is empty")
    if dataset.source_metadata.get("domain") != dataset.domain:
        errors.append("Reference manifest domain does not match the dataset")
    for metadata_field in ("source_dataset_id", "source_version"):
        value = dataset.source_metadata.get(metadata_field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"Reference manifest lacks {metadata_field}")
    if not dataset.windows:
        errors.append("Reference dataset has no observation windows")
    if len(window_by_id) != len(dataset.windows):
        errors.append("Reference window IDs are not unique")
    if len(event_by_id) != len(dataset.events):
        errors.append("Reference event source_record_ids are not unique")
    split_policy = dataset.source_metadata.get("split_policy")
    split_unit = None
    allow_overlap = False
    if not isinstance(split_policy, dict):
        errors.append("Reference manifest lacks an explicit split_policy")
    else:
        split_unit = split_policy.get("unit")
        if split_unit not in {"context", "time_block"}:
            errors.append("split_policy.unit must be context or time_block")
        if not isinstance(split_policy.get("assignment_method"), str) or not split_policy.get(
            "assignment_method", ""
        ).strip():
            errors.append("split_policy lacks an assignment_method")
        allow_overlap = bool(split_policy.get("allow_overlapping_windows", False))
    parsed_intervals: Dict[str, list[tuple[datetime, datetime, str]]] = {}
    context_splits: Dict[str, set[str]] = {}
    for window in dataset.windows:
        if window.domain != dataset.domain:
            errors.append(f"Window {window.window_id} has the wrong domain")
        if window.duration_seconds <= 0.0:
            errors.append(f"Window {window.window_id} has non-positive duration")
        if window.split not in REFERENCE_SPLITS:
            errors.append(f"Window {window.window_id} has unsupported split {window.split}")
        context_splits.setdefault(window.context_id, set()).add(window.split)
        try:
            start = datetime.fromisoformat(window.start_time)
            if start.tzinfo is None:
                raise ValueError("timezone is missing")
            parsed_intervals.setdefault(window.context_id, []).append(
                (
                    start,
                    start + timedelta(seconds=window.duration_seconds),
                    window.window_id,
                )
            )
        except ValueError:
            errors.append(f"Window {window.window_id} has an invalid timezone-aware start_time")
    if split_unit == "context":
        for context_id, splits in context_splits.items():
            if len(splits) > 1:
                errors.append(
                    f"Context {context_id} leaks across reference-data splits"
                )
    if not allow_overlap:
        for context_id, intervals in parsed_intervals.items():
            ordered = sorted(intervals)
            for left, right in zip(ordered, ordered[1:]):
                if right[0] < left[1]:
                    errors.append(
                        f"Reference windows {left[2]} and {right[2]} overlap in context {context_id}"
                    )
    for event in dataset.events:
        window = window_by_id.get(event.window_id)
        if window is None:
            errors.append(f"Event {event.source_record_id} references an unknown window")
            continue
        if event.domain != dataset.domain or event.context_id != window.context_id:
            errors.append(f"Event {event.source_record_id} has inconsistent domain/context")
        if not 0.0 <= event.occurrence_offset_seconds <= window.duration_seconds:
            errors.append(f"Event {event.source_record_id} occurs outside its window")
        if event.observed_at_offset_seconds is not None and (
            event.observed_at_offset_seconds < event.occurrence_offset_seconds
        ):
            errors.append(f"Event {event.source_record_id} is observed before occurrence")
        if event.recorded_at_offset_seconds is not None and (
            event.observed_at_offset_seconds is None
            or event.recorded_at_offset_seconds < event.observed_at_offset_seconds
        ):
            errors.append(f"Event {event.source_record_id} has invalid recording time")
        if not event.participants:
            errors.append(f"Event {event.source_record_id} has no participants")
        if entity_ids and any(
            participant.entity_id not in entity_ids for participant in event.participants
        ):
            errors.append(f"Event {event.source_record_id} references an unknown entity")
        if len(event.explicit_parent_record_ids) != len(set(event.explicit_parent_record_ids)):
            errors.append(f"Event {event.source_record_id} repeats an explicit parent")
        for parent_id in event.explicit_parent_record_ids:
            parent = event_by_id.get(parent_id)
            if parent is None:
                errors.append(f"Event {event.source_record_id} references unknown parent {parent_id}")
            elif parent.window_id != event.window_id:
                errors.append(f"Event {event.source_record_id} has a parent in another window")
            elif parent.occurrence_offset_seconds > event.occurrence_offset_seconds:
                errors.append(f"Event {event.source_record_id} has a later explicit parent")
    for relation in dataset.context_relations:
        if relation.domain != dataset.domain:
            errors.append(f"Context relation {relation.relation_id} has the wrong domain")
        if entity_ids and (
            relation.source_entity_id not in entity_ids
            or relation.target_entity_id not in entity_ids
        ):
            errors.append(f"Context relation {relation.relation_id} references an unknown entity")
    return {
        "passed": not errors,
        "errors": errors,
        "counts": {
            "windows": len(dataset.windows),
            "events": len(dataset.events),
            "entities": len(dataset.entities),
            "context_relations": len(dataset.context_relations),
        },
        "fingerprint": dataset.fingerprint(),
        "split_policy": split_policy,
    }


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reference stream is missing: {path}")
    return _read_optional_jsonl(path)


def _read_optional_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records
