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
SELECTION_SEMANTICS = {
    "cause_specific_competing_hazard",
    "deterministic_atom",
    "empirical_discrete_atom",
    "policy_clock",
}
PARENT_ATTRIBUTION_MODES = {
    "independent_background",
    "explicit_realized_parents",
    "contribution_weighted_history",
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
    selection_semantics: str = "cause_specific_competing_hazard"
    parent_attribution_mode: str = "explicit_realized_parents"

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
    temporal_family = _nonempty_string(
        item.get("temporal_model_family"), "temporal_model_family"
    )
    if temporal_family in {"scheduled_time", "deterministic_delay"}:
        default_selection = "deterministic_atom"
    elif temporal_family == "empirical_delay":
        default_selection = "empirical_discrete_atom"
    elif temporal_family == "exponential_backoff":
        default_selection = "policy_clock"
    else:
        default_selection = "cause_specific_competing_hazard"
    selection_semantics = str(
        item.get("selection_semantics", default_selection)
    )
    if selection_semantics not in SELECTION_SEMANTICS:
        raise ValueError(f"Unsupported selection semantics: {selection_semantics}")
    expected_selection_by_family = {
        "scheduled_time": "deterministic_atom",
        "deterministic_delay": "deterministic_atom",
        "empirical_delay": "empirical_discrete_atom",
        "exponential_backoff": "policy_clock",
        "conditional_lognormal": "cause_specific_competing_hazard",
        "conditional_gamma": "cause_specific_competing_hazard",
        "conditional_weibull": "cause_specific_competing_hazard",
        "exponential_arrival": "cause_specific_competing_hazard",
        "piecewise_exponential_hazard": "cause_specific_competing_hazard",
    }
    expected_selection = expected_selection_by_family.get(temporal_family)
    if expected_selection and selection_semantics != expected_selection:
        raise ValueError(
            f"Mechanism {mechanism_id} uses {selection_semantics} for "
            f"{temporal_family}, expected {expected_selection}"
        )
    default_parent_attribution = (
        "independent_background" if not source_types else "explicit_realized_parents"
    )
    parent_attribution_mode = str(
        item.get("parent_attribution_mode", default_parent_attribution)
    )
    if parent_attribution_mode not in PARENT_ATTRIBUTION_MODES:
        raise ValueError(
            f"Unsupported parent attribution mode: {parent_attribution_mode}"
        )
    if not source_types and parent_attribution_mode != "independent_background":
        raise ValueError(
            f"Root mechanism {mechanism_id} must use independent_background attribution"
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
        temporal_model_family=temporal_family,
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
        selection_semantics=selection_semantics,
        parent_attribution_mode=parent_attribution_mode,
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
