"""Domain-neutral, declarative event-mechanism specifications."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable


PARENT_COMBINATIONS = {
    "none",
    "single",
    "all_of",
    "any_of",
    "k_of_n",
    "ordered_sequence",
}
RELATION_SEMANTICS = {
    "exogenous",
    "causal",
    "transition",
    "statistical_influence",
    "scheduled",
    "observation",
    "workflow",
    "communication",
    "derivation",
}
EVIDENCE_POLICIES = {
    "independent_root",
    "explicit_link",
    "rule_supported",
    "statistically_inferred",
    "configured_schedule",
}
PARAMETER_STATUSES = {
    "synthetic_prior_pending_empirical_fit",
    "policy_defined",
    "pending_reference_data",
    "empirically_fitted",
}


@dataclass(frozen=True)
class MechanismSpec:
    mechanism_id: str
    domain: str
    source_event_type_ids: tuple[str, ...]
    target_event_type_id: str
    parent_combination: str
    required_context_relation_type_ids: tuple[str, ...]
    temporal_model_family: str
    generation_temporal_model_ref: str
    covariate_fields: tuple[str, ...]
    relation_semantics: str
    evidence_policy: str
    calibration_fitter_id: str | None
    parameter_status: str
    notes: str | None = None

    @property
    def is_root(self) -> bool:
        return not self.source_event_type_ids

    def descriptor(self) -> Dict[str, Any]:
        return asdict(self)


class MechanismRegistry:
    def __init__(
        self,
        domain: str,
        mechanisms: Iterable[MechanismSpec],
        *,
        source_path: Path | None = None,
        implementation_status: str | None = None,
    ) -> None:
        self.domain = domain
        self.source_path = source_path
        self.implementation_status = implementation_status
        self._by_id: Dict[str, MechanismSpec] = {}
        for mechanism in mechanisms:
            if mechanism.domain != domain:
                raise ValueError(
                    f"Mechanism {mechanism.mechanism_id} belongs to another domain"
                )
            if mechanism.mechanism_id in self._by_id:
                raise ValueError(f"Duplicate mechanism ID: {mechanism.mechanism_id}")
            self._by_id[mechanism.mechanism_id] = mechanism

    def get(self, mechanism_id: str) -> MechanismSpec:
        try:
            return self._by_id[mechanism_id]
        except KeyError as exc:
            raise KeyError(f"Unknown mechanism: {mechanism_id}") from exc

    def descriptors(self) -> list[Dict[str, Any]]:
        return [self._by_id[key].descriptor() for key in sorted(self._by_id)]

    def __len__(self) -> int:
        return len(self._by_id)


def load_mechanism_registry(
    path: str | Path,
    *,
    known_event_type_ids: Iterable[str] | None = None,
    known_context_relation_type_ids: Iterable[str] | None = None,
) -> MechanismRegistry:
    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("A mechanism specification file must be a JSON object")
    if payload.get("schema_version") != "mechanism-spec-v1":
        raise ValueError("Unsupported mechanism specification version")
    domain = _nonempty_string(payload.get("domain"), "domain")
    raw_mechanisms = payload.get("mechanisms")
    if not isinstance(raw_mechanisms, list):
        raise ValueError("mechanisms must be a list")

    validate_event_types = known_event_type_ids is not None
    validate_context_relation_types = known_context_relation_type_ids is not None
    known_events = set(known_event_type_ids or ())
    known_context_relations = set(known_context_relation_type_ids or ())
    mechanisms: list[MechanismSpec] = []
    for item in raw_mechanisms:
        mechanism = _parse_mechanism(item, domain)
        referenced_events = set(mechanism.source_event_type_ids) | {
            mechanism.target_event_type_id
        }
        if validate_event_types and not referenced_events <= known_events:
            raise ValueError(
                f"Mechanism {mechanism.mechanism_id} references unknown event types: "
                f"{sorted(referenced_events - known_events)}"
            )
        unknown_relations = (
            set(mechanism.required_context_relation_type_ids)
            - known_context_relations
        )
        if validate_context_relation_types and unknown_relations:
            raise ValueError(
                f"Mechanism {mechanism.mechanism_id} references unknown context relations: "
                f"{sorted(unknown_relations)}"
            )
        mechanisms.append(mechanism)
    return MechanismRegistry(
        domain,
        mechanisms,
        source_path=source_path,
        implementation_status=payload.get("implementation_status"),
    )


def _parse_mechanism(item: Any, domain: str) -> MechanismSpec:
    if not isinstance(item, dict):
        raise ValueError("Every mechanism declaration must be an object")
    mechanism_id = _nonempty_string(item.get("mechanism_id"), "mechanism_id")
    if not mechanism_id.startswith(f"{domain}."):
        raise ValueError(f"Mechanism {mechanism_id} is not namespaced by {domain}")
    source_types = _string_tuple(item.get("source_event_type_ids", []), "source_event_type_ids")
    target_type = _nonempty_string(item.get("target_event_type_id"), "target_event_type_id")
    parent_combination = _nonempty_string(item.get("parent_combination"), "parent_combination")
    if parent_combination not in PARENT_COMBINATIONS:
        raise ValueError(f"Unsupported parent combination: {parent_combination}")
    if bool(source_types) == (parent_combination == "none"):
        raise ValueError(
            f"Mechanism {mechanism_id} has inconsistent source types and parent combination"
        )
    relation_semantics = _nonempty_string(item.get("relation_semantics"), "relation_semantics")
    if relation_semantics not in RELATION_SEMANTICS:
        raise ValueError(f"Unsupported relation semantics: {relation_semantics}")
    evidence_policy = _nonempty_string(item.get("evidence_policy"), "evidence_policy")
    if evidence_policy not in EVIDENCE_POLICIES:
        raise ValueError(f"Unsupported evidence policy: {evidence_policy}")
    parameter_status = _nonempty_string(item.get("parameter_status"), "parameter_status")
    if parameter_status not in PARAMETER_STATUSES:
        raise ValueError(f"Unsupported parameter status: {parameter_status}")
    fitter_id = item.get("calibration_fitter_id")
    if fitter_id is not None:
        fitter_id = _nonempty_string(fitter_id, "calibration_fitter_id")
    if parameter_status == "policy_defined" and fitter_id is not None:
        raise ValueError(
            f"Policy-defined mechanism {mechanism_id} must not declare an empirical fitter"
        )
    return MechanismSpec(
        mechanism_id=mechanism_id,
        domain=domain,
        source_event_type_ids=source_types,
        target_event_type_id=target_type,
        parent_combination=parent_combination,
        required_context_relation_type_ids=_string_tuple(
            item.get("required_context_relation_type_ids", []),
            "required_context_relation_type_ids",
        ),
        temporal_model_family=_nonempty_string(
            item.get("temporal_model_family"), "temporal_model_family"
        ),
        generation_temporal_model_ref=_nonempty_string(
            item.get("generation_temporal_model_ref"),
            "generation_temporal_model_ref",
        ),
        covariate_fields=_string_tuple(item.get("covariate_fields", []), "covariate_fields"),
        relation_semantics=relation_semantics,
        evidence_policy=evidence_policy,
        calibration_fitter_id=fitter_id,
        parameter_status=parameter_status,
        notes=(str(item["notes"]) if item.get("notes") is not None else None),
    )


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicates")
    return tuple(value)
