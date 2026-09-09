"""Domain-neutral building blocks for synthetic event-stream generation."""

from .domain import CandidateSpec, DomainPackage, EpisodeContext, ObservationPlan, RelationSpec
from .models import (
    Candidate,
    CandidateStatus,
    ContextRelation,
    Entity,
    EventParticipant,
    EventRecord,
    EventRelation,
    EpisodeResult,
    TemporalExtent,
)
from .scheduler import SimulationEngine
from .temporal import TemporalModelRegistry, build_standard_temporal_registry

__all__ = [
    "Candidate",
    "CandidateSpec",
    "CandidateStatus",
    "ContextRelation",
    "DomainPackage",
    "Entity",
    "EpisodeContext",
    "EpisodeResult",
    "EventParticipant",
    "EventRecord",
    "EventRelation",
    "ObservationPlan",
    "RelationSpec",
    "SimulationEngine",
    "TemporalExtent",
    "TemporalModelRegistry",
    "build_standard_temporal_registry",
]
