"""Common records shared by every domain package.

The records deliberately separate context topology, event occurrence,
observation, event-to-event relations, and candidate lifecycle state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CandidateStatus(str, Enum):
    SCHEDULED = "scheduled"
    FIRED = "fired"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    RIGHT_CENSORED = "right_censored"


class CovariateMode(str, Enum):
    STATIC_AT_ACTIVATION = "static_at_activation"
    TIME_VARYING_HAZARD = "time_varying_hazard"
    DETERMINISTIC_SCHEDULE = "deterministic_schedule"
    POLICY_DRIVEN = "policy_driven"
    EMPIRICAL_RESAMPLE = "empirical_resample"


@dataclass(frozen=True)
class Entity:
    entity_id: str
    domain: str
    entity_type_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    context_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ContextRelation:
    relation_id: str
    domain: str
    relation_type_id: str
    source_entity_id: str
    target_entity_id: str
    directed: bool = True
    valid_from_offset_seconds: Optional[float] = None
    valid_to_offset_seconds: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    context_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class EventParticipant:
    entity_id: str
    role: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalExtent:
    occurrence_start_offset_seconds: float
    occurrence_end_offset_seconds: Optional[float] = None
    observed_at_offset_seconds: Optional[float] = None
    recorded_at_offset_seconds: Optional[float] = None
    precision: str = "millisecond"

    def anchor(self, name: str) -> float:
        anchors = {
            "occurrence_start": self.occurrence_start_offset_seconds,
            "occurrence_end": self.occurrence_end_offset_seconds,
            "observed_at": self.observed_at_offset_seconds,
            "recorded_at": self.recorded_at_offset_seconds,
        }
        if name not in anchors:
            raise KeyError(f"Unsupported temporal anchor: {name}")
        value = anchors[name]
        if value is None:
            raise ValueError(f"Temporal anchor {name} is not populated")
        return float(value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventRecord:
    event_id: str
    episode_id: str
    domain: str
    event_type_id: str
    temporal: TemporalExtent
    participants: List[EventParticipant]
    attributes: Dict[str, Any] = field(default_factory=dict)
    event_role: Optional[str] = None
    observation: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class TemporalLink:
    lag_seconds: float
    source_anchor: str
    target_anchor: str
    temporal_model_ref: str
    censoring_status: str = "observed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventRelation:
    relation_id: str
    episode_id: str
    source_event_id: str
    target_event_id: str
    relation_class: str
    relation_type_id: str
    rule_id: str
    temporal_link: TemporalLink
    status: str = "generated_ground_truth"
    mechanism_group_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class Candidate:
    candidate_id: str
    episode_id: str
    mechanism_id: str
    target_event_type_id: str
    parent_event_ids: List[str]
    participants: List[EventParticipant]
    activation_time: float
    temporal_model_ref: str
    temporal_inputs: Dict[str, Any]
    combination: str = "single"
    phase_priority: int = 20
    domain_priority: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    status: CandidateStatus = CandidateStatus.SCHEDULED
    scheduled_time: Optional[float] = None
    revision: int = 0
    random_threshold: Optional[float] = None
    accumulated_hazard: float = 0.0
    current_hazard_rate: Optional[float] = None
    hazard_active_from: Optional[float] = None
    last_updated_at: Optional[float] = None
    temporal_provenance: Dict[str, Any] = field(default_factory=dict)
    terminal_reason: Optional[str] = None
    superseded_by_event_id: Optional[str] = None
    censored_at: Optional[float] = None
    fired_event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class EpisodeResult:
    episode_id: str
    domain: str
    context_id: str
    start_time: str
    duration_seconds: float
    entities: List[Entity]
    context_relations: List[ContextRelation]
    events: List[EventRecord]
    event_relations: List[EventRelation]
    candidates: List[Candidate]
    final_state: Dict[str, Any]
    termination_reason: str
    episode_attributes: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)

    def episode_record(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "context_id": self.context_id,
            "start_time": self.start_time,
            "duration_seconds": self.duration_seconds,
            "termination_reason": self.termination_reason,
            "attributes": _jsonable(self.episode_attributes),
            "counts": {
                "entities": len(self.entities),
                "context_relations": len(self.context_relations),
                "events": len(self.events),
                "event_relations": len(self.event_relations),
                "candidates": len(self.candidates),
            },
            "validation": self.validation,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
