"""Typed records used by the EventFlow v1 generation contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Entity:
    entity_id: str
    network_id: str
    entity_type: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NetworkEdge:
    edge_id: str
    network_id: str
    source_entity_id: str
    target_entity_id: str
    edge_type: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventParticipant:
    """A qualified event-to-entity link.

    A bare entity ID is insufficient across domains: the same event can refer
    to an affected road, an actor, a resource, or an execution context.  The
    role is validated by the selected domain event-type registry.
    """

    entity_id: str
    role: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Event:
    event_id: str
    episode_id: str
    domain: str
    event_type_id: str
    event_time: str
    time_offset_seconds: float
    participants: List[EventParticipant]
    attributes: Dict[str, Any]
    event_role: Optional[str] = None
    observation: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def participant_ids(self) -> List[str]:
        """Convenience view for graph joins; not serialized as a second source."""

        return [participant.entity_id for participant in self.participants]

    def participant_ids_for_role(self, role: str) -> List[str]:
        return [
            participant.entity_id
            for participant in self.participants
            if participant.role == role
        ]


@dataclass(frozen=True)
class Relation:
    relation_id: str
    episode_id: str
    source_event_id: str
    target_event_id: str
    relation_class: str
    relation_type: str
    status: str
    rule_id: str
    mechanism_group_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Mechanism:
    mechanism_group_id: str
    episode_id: str
    target_event_id: str
    combination: str
    member_relation_ids: List[str]
    rule_id: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TextRecord:
    text_id: str
    episode_id: str
    text_role: str
    variant: str
    content: str
    aligned_event_ids: List[str]
    aligned_relation_ids: List[str]
    aligned_mechanism_ids: List[str]
    grounded_facts: Dict[str, Any]
    verbalized_fact_keys: List[str] = field(default_factory=list)
    derived_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeBundle:
    episode_id: str
    network_id: str
    domain: str
    scenario_family: str
    start_time: str
    duration_seconds: float
    involved_entity_ids: List[str]
    program: Dict[str, Any]
    events: List[Event]
    relations: List[Relation]
    mechanisms: List[Mechanism]
    texts: List[TextRecord] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)

    def episode_record(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "network_id": self.network_id,
            "domain": self.domain,
            "scenario_family": self.scenario_family,
            "start_time": self.start_time,
            "duration_seconds": self.duration_seconds,
            "involved_entity_ids": self.involved_entity_ids,
            "program": self.program,
            "counts": {
                "events": len(self.events),
                "relations": len(self.relations),
                "mechanisms": len(self.mechanisms),
                "texts": len(self.texts),
            },
            "validation": self.validation,
        }
