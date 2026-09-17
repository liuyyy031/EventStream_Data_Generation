"""Versioned event-type registries shared by all EventFlow domains."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


class EventTypeRegistry:
    """Read-only semantic registry and lightweight runtime validator."""

    def __init__(self, payload: Dict[str, Any], source_path: Path):
        self.payload = copy.deepcopy(payload)
        self.source_path = source_path
        self.registry_id = str(payload["registry_id"])
        self.registry_version = str(payload["registry_version"])
        self.domain = str(payload["domain"])
        self.subdomain = str(payload["subdomain"])
        self.categories: Dict[str, str] = copy.deepcopy(payload["categories"])
        self.event_types: Dict[str, Dict[str, Any]] = copy.deepcopy(
            payload["event_types"]
        )
        self._validate_registry_contract()

    @classmethod
    def load(cls, path: str | Path) -> "EventTypeRegistry":
        source_path = Path(path)
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Event-type registry must be a JSON object")
        return cls(payload, source_path)

    def type_spec(self, event_type_id: str) -> Dict[str, Any]:
        try:
            return self.event_types[event_type_id]
        except KeyError as exc:
            raise KeyError(f"Unknown event_type_id: {event_type_id}") from exc

    def display_name(self, event_type_id: str) -> str:
        return str(self.type_spec(event_type_id).get("display_name", event_type_id))

    def category(self, event_type_id: str) -> str:
        return str(self.type_spec(event_type_id)["category"])

    def metric_tags(self, event_type_id: str) -> List[str]:
        return [str(tag) for tag in self.type_spec(event_type_id).get("metric_tags", [])]

    def manifest_record(self, output_filename: str) -> Dict[str, Any]:
        # EventFlowWriter writes JSON with this exact formatting, including the
        # trailing newline, so the manifest hash verifies the released file.
        encoded = (json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        return {
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "file": output_filename,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def validate_event(
        self,
        event: Any,
        entity_by_id: Mapping[str, Any],
    ) -> List[str]:
        errors: List[str] = []
        if event.domain != self.domain:
            errors.append(
                f"Event {event.event_id} domain {event.domain} disagrees with registry domain {self.domain}"
            )
            return errors
        spec = self.event_types.get(event.event_type_id)
        if spec is None:
            return [f"Unknown {self.domain} event_type_id: {event.event_type_id}"]

        allowed_roles: Dict[str, Dict[str, Any]] = spec["participant_roles"]
        role_counts: Dict[str, int] = {}
        seen_links: set[tuple[str, str]] = set()
        for participant in event.participants:
            link = (participant.entity_id, participant.role)
            if link in seen_links:
                errors.append(
                    f"Event {event.event_id} repeats participant link {participant.entity_id}/{participant.role}"
                )
                continue
            seen_links.add(link)
            role_counts[participant.role] = role_counts.get(participant.role, 0) + 1
            role_spec = allowed_roles.get(participant.role)
            if role_spec is None:
                errors.append(
                    f"Event {event.event_id} type {event.event_type_id} rejects participant role {participant.role}"
                )
                continue
            entity = entity_by_id.get(participant.entity_id)
            if entity is None:
                errors.append(
                    f"Event {event.event_id} references missing participant {participant.entity_id}"
                )
                continue
            allowed_entity_types = set(role_spec.get("entity_types", []))
            if entity.entity_type not in allowed_entity_types:
                errors.append(
                    f"Event {event.event_id} role {participant.role} rejects entity type {entity.entity_type}"
                )
        for role, role_spec in allowed_roles.items():
            count = role_counts.get(role, 0)
            minimum = int(role_spec.get("min_count", 0))
            maximum = int(role_spec.get("max_count", 2**31 - 1))
            if count < minimum or count > maximum:
                errors.append(
                    f"Event {event.event_id} role {role} count {count} is outside [{minimum}, {maximum}]"
                )

        attribute_contract = spec["attributes"]
        required: Dict[str, Dict[str, Any]] = attribute_contract.get("required", {})
        optional: Dict[str, Dict[str, Any]] = attribute_contract.get("optional", {})
        allowed_names = set(required).union(optional)
        missing = sorted(set(required).difference(event.attributes))
        if missing:
            errors.append(
                f"Event {event.event_id} lacks required attributes: {', '.join(missing)}"
            )
        unexpected = sorted(set(event.attributes).difference(allowed_names))
        if unexpected and not bool(attribute_contract.get("additional_properties", False)):
            errors.append(
                f"Event {event.event_id} has unregistered attributes: {', '.join(unexpected)}"
            )
        for name, value in event.attributes.items():
            value_spec = required.get(name) or optional.get(name)
            if value_spec is None:
                continue
            error = _validate_value(value, value_spec)
            if error:
                errors.append(f"Event {event.event_id} attribute {name} {error}")
        return errors

    def _validate_registry_contract(self) -> None:
        if not self.event_types:
            raise ValueError("Event-type registry contains no event types")
        if not self.categories:
            raise ValueError("Event-type registry contains no categories")
        prefix = f"{self.domain}.{self.subdomain}."
        for event_type_id, spec in self.event_types.items():
            if not event_type_id.startswith(prefix):
                raise ValueError(
                    f"Registry event type {event_type_id} must start with {prefix}"
                )
            required = {
                "display_name",
                "category",
                "definition",
                "participant_roles",
                "attributes",
            }
            missing = sorted(required.difference(spec))
            if missing:
                raise ValueError(
                    f"Registry event type {event_type_id} is missing: {', '.join(missing)}"
                )
            if not spec["participant_roles"]:
                raise ValueError(
                    f"Registry event type {event_type_id} needs participant roles"
                )
            if spec["category"] not in self.categories:
                raise ValueError(
                    f"Registry event type {event_type_id} uses undeclared category {spec['category']}"
                )
            attributes = spec["attributes"]
            required_attributes = set(attributes.get("required", {}))
            optional_attributes = set(attributes.get("optional", {}))
            overlap = sorted(required_attributes.intersection(optional_attributes))
            if overlap:
                raise ValueError(
                    f"Registry event type {event_type_id} repeats attributes in required and optional: {', '.join(overlap)}"
                )


def _validate_value(value: Any, spec: Mapping[str, Any]) -> str | None:
    expected = spec.get("type")
    if expected == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected == "boolean":
        valid = isinstance(value, bool)
    elif expected == "string":
        valid = isinstance(value, str)
    elif expected == "array":
        valid = isinstance(value, list)
    else:
        return f"uses unsupported registry value type {expected}"
    if not valid:
        return f"must be {expected}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return "must be finite"
        if "minimum" in spec and float(value) < float(spec["minimum"]):
            return f"must be >= {spec['minimum']}"
        if "maximum" in spec and float(value) > float(spec["maximum"]):
            return f"must be <= {spec['maximum']}"
    if "enum" in spec and value not in spec["enum"]:
        return f"must be one of {spec['enum']}"
    return None
