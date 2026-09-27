"""Normalize a raw event table into the reference-data calibration contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import EventParticipant
from .reference_data import (
    REFERENCE_SPLITS,
    ReferenceDataset,
    ReferenceEvent,
    ReferenceWindow,
    validate_reference_dataset,
)


class GenericTabularReferenceAdapter:
    """Read CSV, JSONL, or Parquet records using an explicit column mapping.

    One source stream becomes one observation window.  Context-level split
    assignment keeps every stream belonging to the same subject/context in the
    same split.  Sequential adjacency is deliberately not converted into an
    explicit parent relation.
    """

    def __init__(self, mapping: Mapping[str, Any]):
        self.mapping = dict(mapping)
        self.last_report: Dict[str, Any] | None = None
        _validate_mapping(self.mapping)

    @classmethod
    def from_json(cls, path: str | Path) -> "GenericTabularReferenceAdapter":
        mapping_path = Path(path)
        with mapping_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("Reference mapping config must be a JSON object")
        return cls(value)

    def load(self, source: str | Path) -> ReferenceDataset:
        source_path = Path(source)
        records = _read_records(source_path, self.mapping)
        if not records and not self.mapping.get("window", {}).get("records"):
            raise ValueError(f"Raw reference file has no records: {source_path}")

        columns = self.mapping["columns"]
        domain = str(self.mapping["domain"])
        timezone_name = self.mapping.get("timezone")
        normalized_rows: list[Dict[str, Any]] = []
        event_ids: set[str] = set()
        raw_event_type_counts: Dict[str, int] = defaultdict(int)
        normalized_event_type_counts: Dict[str, int] = defaultdict(int)

        for row_index, row in enumerate(records, 1):
            stream_id = _required_text(row, columns["stream_id"], row_index)
            context_id = _optional_text(row, columns.get("context_id")) or stream_id
            occurrence = _parse_timestamp(
                _required_value(row, columns["timestamp"], row_index),
                timezone_name=timezone_name,
                timestamp_format=self.mapping.get("timestamp_format"),
                timestamp_unit=self.mapping.get("timestamp_unit"),
                row_index=row_index,
                column_name=columns["timestamp"],
            )
            observed = _optional_timestamp(
                row,
                columns.get("observed_timestamp"),
                timezone_name,
                self.mapping,
                row_index,
            )
            recorded = _optional_timestamp(
                row,
                columns.get("recorded_timestamp"),
                timezone_name,
                self.mapping,
                row_index,
            )
            if observed is not None and observed < occurrence:
                raise ValueError(f"Row {row_index}: observed time precedes occurrence time")
            if recorded is not None and (observed is None or recorded < observed):
                raise ValueError(
                    f"Row {row_index}: recorded time requires and must follow observed time"
                )

            raw_event_type = _required_text(row, columns["event_type"], row_index)
            event_type_id = _map_event_type(raw_event_type, self.mapping, row_index)
            raw_event_id = _optional_text(row, columns.get("event_id"))
            source_record_id = _event_record_id(stream_id, raw_event_id, row_index)
            if source_record_id in event_ids:
                raise ValueError(
                    f"Row {row_index}: duplicate event identifier {source_record_id!r}"
                )
            event_ids.add(source_record_id)

            parent_values = _parse_multi_value(
                _optional_value(row, columns.get("parent_event_id")),
                delimiter=str(self.mapping.get("multi_value_delimiter", "|")),
            )
            trace_values = _parse_multi_value(
                _optional_value(row, columns.get("trace_id")),
                delimiter=str(self.mapping.get("multi_value_delimiter", "|")),
            )
            participants = _participants(row, columns, self.mapping, context_id, row_index)
            split_value = _optional_text(row, columns.get("split"))

            normalized_rows.append(
                {
                    "row_index": row_index,
                    "stream_id": stream_id,
                    "context_id": context_id,
                    "occurrence": occurrence,
                    "observed": observed,
                    "recorded": recorded,
                    "source_record_id": source_record_id,
                    "mechanism_id": _optional_text(row, columns.get("mechanism_id")),
                    "raw_event_type": raw_event_type,
                    "event_type_id": event_type_id,
                    "participants": participants,
                    "parent_record_ids": [
                        _event_record_id(stream_id, value, row_index)
                        for value in parent_values
                    ],
                    "trace_ids": trace_values,
                    "split_value": split_value,
                    "attributes": _project_fields(row, self.mapping.get("attribute_columns")),
                    "context_state": _project_fields(
                        row, self.mapping.get("context_state_columns")
                    ),
                    "scenario_family": _optional_text(
                        row, columns.get("scenario_family")
                    ),
                }
            )
            raw_event_type_counts[raw_event_type] += 1
            normalized_event_type_counts[event_type_id] += 1

        stream_rows: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
        stream_context: Dict[str, str] = {}
        for row in normalized_rows:
            stream_id = row["stream_id"]
            previous_context = stream_context.setdefault(stream_id, row["context_id"])
            if previous_context != row["context_id"]:
                raise ValueError(
                    f"Stream {stream_id!r} is associated with multiple context IDs"
                )
            stream_rows[stream_id].append(row)

        context_splits = _resolve_context_splits(normalized_rows, self.mapping)
        window_config = self.mapping.get("window", {})
        declared_windows = {str(item["stream_id"]): item for item in window_config.get("records", [])}
        declared_splits = {}
        if len(declared_windows) != len(window_config.get("records", [])):
            raise ValueError("Duplicate declared window stream_id")
        for stream_id, item in declared_windows.items():
            context_id = str(item["context_id"])
            if stream_id in stream_context and stream_context[stream_id] != context_id:
                raise ValueError("Declared observation window context does not match events")
            if (columns.get("split") and context_id in context_splits and context_splits[context_id] != item["split"]) or (context_id in declared_splits and declared_splits[context_id] != item["split"]):
                raise ValueError("Declared observation window split conflicts with event split")
            declared_splits[context_id] = item["split"]
            stream_context[stream_id] = context_id
            context_splits[context_id] = item["split"]
            stream_rows.setdefault(stream_id, [])
        if declared_windows and set(stream_rows) - set(declared_windows):
            raise ValueError("Every event stream must have a declared window when window.records is supplied")
        padding_seconds = float(window_config.get("padding_seconds", 1.0))
        minimum_duration = float(window_config.get("minimum_duration_seconds", 1.0))
        allow_overlapping_windows = bool(
            window_config.get("allow_overlapping_windows", False)
        )
        if padding_seconds < 0.0 or minimum_duration <= 0.0:
            raise ValueError("Window padding must be non-negative and duration must be positive")

        windows: list[ReferenceWindow] = []
        events: list[ReferenceEvent] = []
        for stream_id in sorted(stream_rows):
            rows = stream_rows[stream_id]
            context_id = stream_context[stream_id]
            declared = declared_windows.get(stream_id)
            if declared:
                start = datetime.fromisoformat(str(declared["start_time"]).replace("Z", "+00:00"))
                if start.tzinfo is None:
                    raise ValueError("Declared window start_time requires a timezone")
                duration = float(declared["duration_seconds"])
            else:
                start = min(item["occurrence"] for item in rows)
                latest = max(item["recorded"] or item["observed"] or item["occurrence"] for item in rows)
                duration = max(minimum_duration, (latest - start).total_seconds() + padding_seconds)
            window_id = f"window::{stream_id}"
            scenario_values = {
                item["scenario_family"] for item in rows if item["scenario_family"]
            }
            if len(scenario_values) > 1:
                raise ValueError(
                    f"Stream {stream_id!r} has multiple scenario_family values"
                )
            window_attributes: Dict[str, Any] = {}
            if declared:
                window_attributes.update(declared.get("attributes", {}))
            if scenario_values:
                window_attributes["scenario_family"] = next(iter(scenario_values))
            windows.append(
                ReferenceWindow(
                    window_id=window_id,
                    context_id=context_id,
                    domain=domain,
                    start_time=start.isoformat(),
                    duration_seconds=duration,
                    split=context_splits[context_id],
                    attributes=window_attributes,
                    provenance={
                        "adapter": "generic_tabular_reference_v1",
                        "source_stream_id": stream_id,
                        "window_source": "declared_observation_window" if declared else "inferred_event_span",
                        "observation_complete": bool(declared and declared.get("observation_complete") is True),
                        "eligible_for_exposure_rate_fit": bool(declared and declared.get("observation_complete") is True),
                    },
                )
            )
            for item in sorted(
                rows, key=lambda value: (value["occurrence"], value["source_record_id"])
            ):
                events.append(
                    ReferenceEvent(
                        source_record_id=item["source_record_id"],
                        window_id=window_id,
                        context_id=context_id,
                        domain=domain,
                        event_type_id=item["event_type_id"],
                        occurrence_offset_seconds=(item["occurrence"] - start).total_seconds(),
                        observed_at_offset_seconds=(
                            None
                            if item["observed"] is None
                            else (item["observed"] - start).total_seconds()
                        ),
                        recorded_at_offset_seconds=(
                            None
                            if item["recorded"] is None
                            else (item["recorded"] - start).total_seconds()
                        ),
                        participants=item["participants"],
                        attributes=item["attributes"],
                        explicit_parent_record_ids=item["parent_record_ids"],
                        relation_or_trace_ids=item["trace_ids"],
                        context_state_snapshot=item["context_state"],
                        provenance={
                            "adapter": "generic_tabular_reference_v1",
                            "source_row_number": item["row_index"],
                            "raw_event_type": item["raw_event_type"],
                            "mechanism_id": item["mechanism_id"],
                        },
                    )
                )

        mapping_fingerprint = _json_fingerprint(self.mapping)
        source_metadata = {
            "contract_version": "reference-data-v1",
            "domain": domain,
            "source_dataset_id": str(self.mapping["source_dataset_id"]),
            "source_version": str(self.mapping["source_version"]),
            "source_format": _resolve_format(source_path, self.mapping),
            "source_file_name": source_path.name,
            "source_file_sha256": _file_sha256(source_path),
            "adapter": "generic_tabular_reference_v1",
            "mapping_fingerprint": mapping_fingerprint,
            "sequence_scope": self.mapping.get("sequence_scope", "window"),
            "split_policy": {
                "unit": "context",
                "assignment_method": (
                    "source_column"
                    if columns.get("split")
                    else "seeded_stable_hash_partition"
                ),
                "allow_overlapping_windows": allow_overlapping_windows,
                "seed": int(self.mapping.get("split", {}).get("seed", 0)),
            },
        }
        dataset = ReferenceDataset(
            domain=domain,
            windows=windows,
            events=events,
            source_metadata=source_metadata,
        )
        validation = validate_reference_dataset(dataset)
        if not validation["passed"]:
            raise ValueError(
                "Raw reference normalization produced an invalid dataset: "
                + "; ".join(validation["errors"])
            )
        split_counts: Dict[str, int] = defaultdict(int)
        for window in windows:
            split_counts[window.split] += 1
        self.last_report = {
            "passed": True,
            "source_file": str(source_path),
            "source_record_count": len(records),
            "window_count": len(windows),
            "event_count": len(events),
            "context_count": len({window.context_id for window in windows}),
            "split_window_counts": dict(sorted(split_counts.items())),
            "raw_event_type_counts": dict(sorted(raw_event_type_counts.items())),
            "normalized_event_type_counts": dict(
                sorted(normalized_event_type_counts.items())
            ),
            "explicit_parent_link_count": sum(
                len(item.explicit_parent_record_ids) for item in events
            ),
            "adjacent_events_promoted_to_parents": 0,
            "dataset_fingerprint": dataset.fingerprint(),
            "validation": validation,
        }
        return dataset


def write_normalized_reference_dataset(
    dataset: ReferenceDataset,
    output_dir: str | Path,
    *,
    normalization_report: Mapping[str, Any] | None = None,
) -> None:
    """Write the canonical audit snapshot accepted by the normalized adapter."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "reference_manifest.json", dataset.source_metadata)
    _write_jsonl(target / "reference_windows.jsonl", dataset.windows)
    _write_jsonl(target / "reference_events.jsonl", dataset.events)
    if dataset.entities:
        _write_jsonl(target / "reference_entities.jsonl", dataset.entities)
    if dataset.context_relations:
        _write_jsonl(
            target / "reference_context_relations.jsonl", dataset.context_relations
        )
    if normalization_report is not None:
        _write_json(target / "normalization_report.json", dict(normalization_report))


def _validate_mapping(mapping: Mapping[str, Any]) -> None:
    if mapping.get("schema_version") != "generic-tabular-reference-mapping-v1":
        raise ValueError(
            "mapping.schema_version must be generic-tabular-reference-mapping-v1"
        )
    for name in ("domain", "source_dataset_id", "source_version"):
        if not isinstance(mapping.get(name), str) or not mapping[name].strip():
            raise ValueError(f"mapping.{name} must be a non-empty string")
    columns = mapping.get("columns")
    if not isinstance(columns, dict):
        raise ValueError("mapping.columns must be an object")
    for name in ("stream_id", "timestamp", "event_type"):
        if not isinstance(columns.get(name), str) or not columns[name].strip():
            raise ValueError(f"mapping.columns.{name} must name a source column")
    fractions = mapping.get("split", {}).get(
        "fractions", {"train": 0.8, "validation": 0.1, "holdout": 0.1}
    )
    if not isinstance(fractions, dict):
        raise ValueError("mapping.split.fractions must be an object")
    if any(float(fractions.get(name, 0.0)) < 0.0 for name in REFERENCE_SPLITS):
        raise ValueError("Split fractions cannot be negative")
    if sum(float(fractions.get(name, 0.0)) for name in REFERENCE_SPLITS) <= 0.0:
        raise ValueError("At least one split fraction must be positive")


def _read_records(path: Path, mapping: Mapping[str, Any]) -> list[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Raw reference file is missing: {path}")
    source_format = _resolve_format(path, mapping)
    if source_format == "csv":
        with path.open("r", encoding=str(mapping.get("encoding", "utf-8-sig")), newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if source_format == "jsonl":
        records: list[Dict[str, Any]] = []
        with path.open("r", encoding=str(mapping.get("encoding", "utf-8"))) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(value)
        return records
    if source_format == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas is in project requirements
            raise RuntimeError("Reading Parquet requires pandas") from exc
        try:
            frame = pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet also requires a pandas Parquet engine such as pyarrow"
            ) from exc
        return [
            {key: _clean_scalar(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]
    raise ValueError(f"Unsupported raw reference format: {source_format}")


def _resolve_format(path: Path, mapping: Mapping[str, Any]) -> str:
    configured = mapping.get("format")
    if configured:
        value = str(configured).lower()
    else:
        value = path.suffix.lower().lstrip(".")
    aliases = {"ndjson": "jsonl", "jsonlines": "jsonl", "pq": "parquet"}
    return aliases.get(value, value)


def _parse_timestamp(
    value: Any,
    *,
    timezone_name: str | None,
    timestamp_format: str | None,
    timestamp_unit: str | None,
    row_index: int,
    column_name: str,
) -> datetime:
    try:
        if isinstance(value, datetime):
            parsed = value
        elif timestamp_unit:
            numeric = float(value)
            divisor = {"seconds": 1.0, "milliseconds": 1_000.0}.get(timestamp_unit)
            if divisor is None:
                raise ValueError("timestamp_unit must be seconds or milliseconds")
            parsed = datetime.fromtimestamp(numeric / divisor, tz=timezone.utc)
        elif timestamp_format:
            parsed = datetime.strptime(str(value), timestamp_format)
        else:
            text = str(value).strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            if not timezone_name:
                raise ValueError(
                    "naive timestamps require mapping.timezone (for example Asia/Shanghai)"
                )
            parsed = parsed.replace(tzinfo=ZoneInfo(str(timezone_name)))
        return parsed
    except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(
            f"Row {row_index}: invalid timestamp in column {column_name!r}: {value!r} ({exc})"
        ) from exc


def _optional_timestamp(
    row: Mapping[str, Any],
    column: str | None,
    timezone_name: str | None,
    mapping: Mapping[str, Any],
    row_index: int,
) -> datetime | None:
    value = _optional_value(row, column)
    if _is_blank(value):
        return None
    return _parse_timestamp(
        value,
        timezone_name=timezone_name,
        timestamp_format=mapping.get("timestamp_format"),
        timestamp_unit=mapping.get("timestamp_unit"),
        row_index=row_index,
        column_name=str(column),
    )


def _map_event_type(raw_value: str, mapping: Mapping[str, Any], row_index: int) -> str:
    event_type_mapping = mapping.get("event_type_mapping", {})
    if not event_type_mapping:
        return raw_value
    if raw_value in event_type_mapping:
        mapped = str(event_type_mapping[raw_value]).strip()
        if mapped:
            return mapped
    policy = mapping.get("unknown_event_type_policy", "error")
    if policy == "keep":
        return raw_value
    raise ValueError(
        f"Row {row_index}: event type {raw_value!r} is absent from event_type_mapping"
    )


def _participants(
    row: Mapping[str, Any],
    columns: Mapping[str, Any],
    mapping: Mapping[str, Any],
    context_id: str,
    row_index: int,
) -> list[EventParticipant]:
    specs = mapping.get("participants")
    participants: list[EventParticipant] = []
    if specs is not None:
        if not isinstance(specs, list) or not specs:
            raise ValueError("mapping.participants must be a non-empty array")
        for spec in specs:
            if not isinstance(spec, dict) or not spec.get("column") or not spec.get("role"):
                raise ValueError("Each participant mapping needs column and role")
            value = _optional_text(row, str(spec["column"]))
            if value:
                participants.append(EventParticipant(value, str(spec["role"])))
    else:
        entity_id = _optional_text(row, columns.get("entity_id")) or context_id
        participants.append(
            EventParticipant(entity_id, str(mapping.get("participant_role", "subject")))
        )
    if not participants:
        raise ValueError(f"Row {row_index}: no mapped participant is populated")
    unique: Dict[tuple[str, str], EventParticipant] = {}
    for participant in participants:
        unique[(participant.entity_id, participant.role)] = participant
    return list(unique.values())


def _resolve_context_splits(
    rows: Iterable[Mapping[str, Any]], mapping: Mapping[str, Any]
) -> Dict[str, str]:
    rows = list(rows)
    split_column = mapping["columns"].get("split")
    context_ids = sorted({str(item["context_id"]) for item in rows})
    if split_column:
        aliases = {
            "train": "train",
            "training": "train",
            "val": "validation",
            "valid": "validation",
            "validation": "validation",
            "test": "holdout",
            "holdout": "holdout",
        }
        aliases.update(
            {
                str(key).lower(): str(value)
                for key, value in mapping.get("split", {}).get("value_mapping", {}).items()
            }
        )
        by_context: Dict[str, set[str]] = defaultdict(set)
        for row in rows:
            raw = row["split_value"]
            if not raw:
                raise ValueError(
                    f"Context {row['context_id']!r} has a blank source split value"
                )
            normalized = aliases.get(str(raw).lower(), str(raw).lower())
            if normalized not in REFERENCE_SPLITS:
                raise ValueError(f"Unsupported source split value: {raw!r}")
            by_context[str(row["context_id"])].add(normalized)
        result: Dict[str, str] = {}
        for context_id, values in by_context.items():
            if len(values) != 1:
                raise ValueError(f"Context {context_id!r} leaks across source splits")
            result[context_id] = next(iter(values))
        return result

    split_config = mapping.get("split", {})
    fractions = split_config.get(
        "fractions", {"train": 0.8, "validation": 0.1, "holdout": 0.1}
    )
    seed = int(split_config.get("seed", 0))
    names = ["train", "validation", "holdout"]
    weights = {name: float(fractions.get(name, 0.0)) for name in names}
    positive = [name for name in names if weights[name] > 0.0]
    counts = {name: 0 for name in names}
    remaining = len(context_ids)
    if len(context_ids) >= len(positive):
        for name in positive:
            counts[name] = 1
            remaining -= 1
    total_weight = sum(weights.values())
    quotas = {name: remaining * weights[name] / total_weight for name in names}
    for name in names:
        addition = math.floor(quotas[name])
        counts[name] += addition
        remaining -= addition
    for name in sorted(names, key=lambda item: (-(quotas[item] % 1), names.index(item))):
        if remaining <= 0:
            break
        if weights[name] > 0.0:
            counts[name] += 1
            remaining -= 1

    ordered = sorted(
        context_ids,
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest(),
    )
    result: Dict[str, str] = {}
    cursor = 0
    for name in names:
        for context_id in ordered[cursor : cursor + counts[name]]:
            result[context_id] = name
        cursor += counts[name]
    return result


def _project_fields(
    row: Mapping[str, Any], specification: Any
) -> Dict[str, Any]:
    if not specification:
        return {}
    if isinstance(specification, list):
        field_mapping = {str(name): str(name) for name in specification}
    elif isinstance(specification, dict):
        field_mapping = {str(source): str(target) for source, target in specification.items()}
    else:
        raise ValueError("Projected column configuration must be an array or object")
    projected: Dict[str, Any] = {}
    for source, target in field_mapping.items():
        value = _optional_value(row, source)
        if not _is_blank(value):
            projected[target] = _clean_scalar(value)
    return projected


def _event_record_id(stream_id: str, raw_event_id: str | None, row_index: int) -> str:
    suffix = raw_event_id if raw_event_id is not None else f"row_{row_index}"
    return f"{stream_id}::{suffix}"


def _parse_multi_value(value: Any, *, delimiter: str) -> list[str]:
    if _is_blank(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError("A JSON multi-value field must decode to an array")
            raw_values = parsed
        else:
            raw_values = stripped.split(delimiter)
    else:
        raw_values = [value]
    result: list[str] = []
    for item in raw_values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _required_value(row: Mapping[str, Any], column: str, row_index: int) -> Any:
    if column not in row:
        raise ValueError(f"Row {row_index}: required source column {column!r} is missing")
    value = row[column]
    if _is_blank(value):
        raise ValueError(f"Row {row_index}: required source column {column!r} is blank")
    return value


def _required_text(row: Mapping[str, Any], column: str, row_index: int) -> str:
    return str(_required_value(row, column, row_index)).strip()


def _optional_value(row: Mapping[str, Any], column: str | None) -> Any:
    if not column:
        return None
    return row.get(column)


def _optional_text(row: Mapping[str, Any], column: str | None) -> str | None:
    value = _optional_value(row, column)
    if _is_blank(value):
        return None
    return str(value).strip()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _clean_scalar(value: Any) -> Any:
    if _is_blank(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _json_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_dict() if hasattr(record, "to_dict") else record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
