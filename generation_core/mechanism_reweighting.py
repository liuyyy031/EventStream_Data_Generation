"""Safe transition-informed reweighting of already-legal mechanisms.

The reweighter never creates a candidate, event type, mechanism, or parent
edge.  It only applies a proportional-hazards multiplier to a candidate that
the domain package has already emitted and whose declared mechanism matches
the realized parent and target event types.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

from .models import Candidate, EventRecord


MECHANISM_HAZARD_MULTIPLIER = "_mechanism_hazard_multiplier"
SUPPORTED_CONTINUOUS_FAMILIES = {
    "conditional_lognormal",
    "conditional_gamma",
    "conditional_weibull",
    "exponential_arrival",
    "piecewise_exponential_hazard",
    "periodic_piecewise_hazard",
    "self_exciting_hazard",
    "refractory_hazard",
    "conditional_lognormal_mixture",
}


class LegalMechanismReweighter:
    """Reweight a legal continuous-time candidate from train transitions."""

    def __init__(
        self,
        *,
        first_order_transitions: Mapping[str, Any],
        second_order_transitions: Mapping[str, Any],
        mechanism_descriptors: Sequence[Mapping[str, Any]],
        mode: str = "first_second_order",
        strength: float = 1.0,
        min_history_count: int = 5,
        minimum_multiplier: float = 0.25,
        maximum_multiplier: float = 4.0,
        sequence_scope: str = "window",
    ) -> None:
        if mode not in {"first_order", "first_second_order"}:
            raise ValueError("Unsupported mechanism reweighting mode")
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("Reweighting strength must be finite and non-negative")
        if min_history_count < 1:
            raise ValueError("min_history_count must be positive")
        if not 0.0 < minimum_multiplier <= 1.0:
            raise ValueError("minimum_multiplier must be in (0, 1]")
        if maximum_multiplier < 1.0:
            raise ValueError("maximum_multiplier must be at least 1")
        self.first_order_transitions = deepcopy(dict(first_order_transitions))
        self.second_order_transitions = deepcopy(dict(second_order_transitions))
        self.mechanisms = {
            str(item["mechanism_id"]): deepcopy(dict(item))
            for item in mechanism_descriptors
        }
        self.mode = mode
        self.strength = float(strength)
        self.min_history_count = int(min_history_count)
        self.minimum_multiplier = float(minimum_multiplier)
        self.maximum_multiplier = float(maximum_multiplier)
        if sequence_scope not in {"window", "entity", "process"}:
            raise ValueError("Unsupported sequence scope")
        self.sequence_scope = sequence_scope

    @classmethod
    def from_bundle(
        cls,
        bundle: Mapping[str, Any],
        *,
        mechanism_descriptors: Sequence[Mapping[str, Any]],
        mode: str = "first_second_order",
        strength: float = 1.0,
        min_history_count: int = 5,
        minimum_multiplier: float = 0.25,
        maximum_multiplier: float = 4.0,
    ) -> "LegalMechanismReweighter":
        safety = dict(bundle.get("safety_contract", {}))
        if not safety.get("sequential_transitions_are_descriptive_not_causal"):
            raise ValueError("Calibration bundle lacks the non-causal transition contract")
        train = dict(bundle.get("sequence_statistics", {}).get("train", {}))
        return cls(
            first_order_transitions=train.get("first_order_transitions", {}),
            second_order_transitions=train.get("second_order_transitions", {}),
            mechanism_descriptors=mechanism_descriptors,
            mode=mode,
            strength=strength,
            min_history_count=min_history_count,
            minimum_multiplier=minimum_multiplier,
            maximum_multiplier=maximum_multiplier,
            sequence_scope=str(train.get("sequence_scope", "window")),
        )

    def apply(
        self,
        candidate: Candidate,
        *,
        event_by_id: Mapping[str, EventRecord],
        event_history: Sequence[EventRecord],
        temporal_family: str,
    ) -> Dict[str, Any]:
        audit: Dict[str, Any] = {
            "status": "not_applied",
            "mechanism_id": candidate.mechanism_id,
            "target_event_type_id": candidate.target_event_type_id,
            "mode": self.mode,
            "strength": self.strength,
            "safety_scope": "already_emitted_legal_candidate_only",
        }
        if len(candidate.parent_event_ids) != 1:
            return self._record(candidate, audit, "requires_one_realized_parent")
        parent = event_by_id.get(candidate.parent_event_ids[0])
        if parent is None:
            return self._record(candidate, audit, "parent_record_not_available")
        mechanism = self.mechanisms.get(candidate.mechanism_id)
        if mechanism is None:
            return self._record(candidate, audit, "mechanism_not_declared")
        if mechanism.get("target_event_type_id") != candidate.target_event_type_id:
            return self._record(candidate, audit, "declared_target_mismatch")
        if parent.event_type_id not in mechanism.get("source_event_type_ids", []):
            return self._record(candidate, audit, "declared_source_mismatch")
        if temporal_family not in SUPPORTED_CONTINUOUS_FAMILIES:
            return self._record(candidate, audit, "temporal_family_not_reweightable")

        if self.sequence_scope == "entity":
            entities = {p.entity_id for p in parent.participants}
            if len(entities) != 1:
                return self._record(candidate, audit, "ambiguous_entity_sequence_scope")
            event_history = [e for e in event_history if entities & {p.entity_id for p in e.participants}]
        elif self.sequence_scope == "process":
            process = parent.attributes.get("process_id")
            if process is None:
                return self._record(candidate, audit, "missing_process_scope")
            event_history = [e for e in event_history if e.attributes.get("process_id") == process]
        audit["sequence_scope"] = self.sequence_scope

        history_order = "first_order"
        history_key = parent.event_type_id
        outgoing = self.first_order_transitions.get(history_key)
        if (
            self.mode == "first_second_order"
            and len(event_history) >= 2
            and event_history[-1].event_id == parent.event_id
            and event_history[-2].temporal.occurrence_start_offset_seconds < parent.temporal.occurrence_start_offset_seconds
            and sum(e.temporal.occurrence_start_offset_seconds == event_history[-2].temporal.occurrence_start_offset_seconds for e in event_history) == 1
        ):
            second_key = (
                f"{event_history[-2].event_type_id}|{parent.event_type_id}"
            )
            second_outgoing = self.second_order_transitions.get(second_key)
            if second_outgoing:
                history_order = "second_order"
                history_key = second_key
                outgoing = second_outgoing
        if not outgoing:
            return self._record(candidate, audit, "history_not_observed")

        positive_records = {
            str(target): record
            for target, record in dict(outgoing).items()
            if float(record.get("count", 0.0)) > 0.0
        }
        total_count = sum(
            float(record.get("count", 0.0))
            for record in positive_records.values()
        )
        audit.update(
            {
                "history_order": history_order,
                "history_key": history_key,
                "history_sample_count": int(total_count),
            }
        )
        if total_count < self.min_history_count:
            return self._record(candidate, audit, "insufficient_history_samples")
        target_record = positive_records.get(candidate.target_event_type_id)
        if target_record is None:
            # Absence is not treated as impossibility.  This is essential for
            # finite reference datasets and for legal but rare mechanisms.
            return self._record(candidate, audit, "legal_transition_unobserved")

        probability = float(target_record.get("count", 0.0)) / total_count
        neutral_probability = 1.0 / len(positive_records)
        raw_multiplier = (probability / neutral_probability) ** self.strength
        multiplier = min(
            self.maximum_multiplier,
            max(self.minimum_multiplier, raw_multiplier),
        )
        candidate.temporal_inputs[MECHANISM_HAZARD_MULTIPLIER] = multiplier
        audit.update(
            {
                "status": "applied",
                "reason": "observed_transition_reweights_declared_mechanism",
                "transition_count": int(target_record.get("count", 0)),
                "transition_probability": probability,
                "neutral_probability": neutral_probability,
                "raw_multiplier": raw_multiplier,
                "applied_hazard_multiplier": multiplier,
                "formula": "clip((p_transition / uniform_observed_target_probability)^strength)",
                "creates_candidate_or_parent_edge": False,
            }
        )
        candidate.provenance["mechanism_reweighting"] = audit
        return audit

    def descriptor(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "sequence_scope": self.sequence_scope,
            "mode": self.mode,
            "strength": self.strength,
            "min_history_count": self.min_history_count,
            "minimum_multiplier": self.minimum_multiplier,
            "maximum_multiplier": self.maximum_multiplier,
            "supported_temporal_families": sorted(SUPPORTED_CONTINUOUS_FAMILIES),
            "unobserved_legal_transition_policy": "leave_prior_unchanged",
            "multi_parent_policy": "leave_prior_unchanged",
            "causal_edge_creation_allowed": False,
        }

    @staticmethod
    def _record(
        candidate: Candidate, audit: Dict[str, Any], reason: str
    ) -> Dict[str, Any]:
        audit["reason"] = reason
        audit["applied_hazard_multiplier"] = 1.0
        candidate.provenance["mechanism_reweighting"] = audit
        return audit
