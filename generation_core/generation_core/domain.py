"""Interface between the domain-neutral scheduler and a domain package."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .models import (
    Candidate,
    ContextEvidence,
    ContextRelation,
    Entity,
    EpisodeResult,
    EventParticipant,
    EventRecord,
)
from .topology import ContextRelationIndex, RelationNeighbor


@dataclass
class EpisodeContext:
    episode_id: str
    context_id: str
    domain: str
    start_time: str
    duration_seconds: float
    entities: List[Entity]
    context_relations: List[ContextRelation]
    state: Dict[str, Any] = field(default_factory=dict)
    context_attributes: Dict[str, Any] = field(default_factory=dict)
    episode_attributes: Dict[str, Any] = field(default_factory=dict)
    relation_index: ContextRelationIndex = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.relation_index = ContextRelationIndex(self.context_relations)

    @property
    def entity_by_id(self) -> Dict[str, Entity]:
        return {entity.entity_id: entity for entity in self.entities}

    def neighbors(
        self,
        entity_id: str,
        relation_type_id: str | None = None,
        *,
        at_time: float | None = None,
        direction: str = "outgoing",
    ) -> List[RelationNeighbor]:
        return self.relation_index.neighbors(
            entity_id,
            relation_type_id,
            at_time=at_time,
            direction=direction,
        )


@dataclass
class CandidateSpec:
    mechanism_id: str
    target_event_type_id: str
    parent_event_ids: List[str]
    participants: List[EventParticipant]
    temporal_model_ref: str
    temporal_inputs: Dict[str, Any]
    activation_time: Optional[float] = None
    combination: str = "single"
    phase_priority: int = 20
    domain_priority: int = 0
    context_evidence: ContextEvidence = field(default_factory=ContextEvidence)
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateUpdate:
    valid: bool
    reason: Optional[str] = None
    temporal_inputs: Optional[Dict[str, Any]] = None
    superseded_by_event_id: Optional[str] = None
    context_evidence: Optional[ContextEvidence] = None


@dataclass
class RelationSpec:
    source_event_id: str
    relation_class: str
    relation_type_id: str
    rule_id: str
    source_anchor: str = "occurrence_start"
    target_anchor: str = "occurrence_start"
    status: str = "generated_ground_truth"
    mechanism_group_id: Optional[str] = None
    context_evidence: ContextEvidence = field(default_factory=ContextEvidence)
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationPlan:
    observation_model_ref: Optional[str] = None
    observation_inputs: Dict[str, Any] = field(default_factory=dict)
    recording_model_ref: Optional[str] = None
    recording_inputs: Dict[str, Any] = field(default_factory=dict)
    source: str = "synthetic_generator"


class DomainPackage(Protocol):
    domain_id: str

    def catalog(self) -> Dict[str, Any]: ...

    def render_episode_text(self, result: EpisodeResult) -> Dict[str, Any]: ...

    def create_episode_context(
        self, episode_index: int, seed: int
    ) -> EpisodeContext: ...

    def seed_candidates(
        self, context: EpisodeContext, rng: random.Random
    ) -> List[CandidateSpec]: ...

    def revalidate_candidate(
        self,
        candidate: Candidate,
        context: EpisodeContext,
        event_by_id: Dict[str, EventRecord],
        now: float,
    ) -> CandidateUpdate: ...

    def materialize_event(
        self,
        event_id: str,
        candidate: Candidate,
        context: EpisodeContext,
        rng: random.Random,
    ) -> EventRecord: ...

    def apply_event(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
    ) -> None: ...

    def spawn_candidates(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
        event_by_id: Dict[str, EventRecord],
        rng: random.Random,
    ) -> List[CandidateSpec]: ...

    def relation_specs(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
        event_by_id: Dict[str, EventRecord],
    ) -> List[RelationSpec]: ...

    def observation_plan(
        self,
        event: EventRecord,
        candidate: Candidate,
        context: EpisodeContext,
    ) -> ObservationPlan: ...
