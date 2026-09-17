"""Domain-neutral building blocks for synthetic event-stream generation."""

from .calibration import (
    CalibrationBundle,
    CalibrationEngine,
    CalibrationObservation,
    ExponentialRateMLEFitter,
    build_root_event_observations,
    build_standard_calibration_fitters,
)
from .domain import CandidateSpec, DomainPackage, EpisodeContext, ObservationPlan, RelationSpec
from .domain_spec import LoadedDomainSpec, load_domain_spec
from .mechanisms import MechanismRegistry, MechanismSpec, load_mechanism_registry
from .models import (
    Candidate,
    CandidateStatus,
    ContextEvidence,
    ContextRelation,
    Entity,
    EventParticipant,
    EventRecord,
    EventRelation,
    EpisodeResult,
    RiskAlternativeEvidence,
    RiskSetRecord,
    StatePredicateEvidence,
    TemporalExtent,
)
from .scheduler import SimulationEngine
from .semantic_correction import (
    LLMSemanticCorrector,
    SemanticCorrectionResult,
    SemanticCorrector,
)
from .semantic_judge import LLMSemanticJudge, SemanticJudge, SemanticJudgeResult
from .qa_export import (
    chat_training_example,
    generate_episode_qa,
    instruction_training_example,
    validate_qa_pair,
)
from .reference_data import (
    NormalizedJsonlReferenceAdapter,
    ReferenceDataset,
    ReferenceEvent,
    ReferenceWindow,
    validate_reference_dataset,
)
from .temporal import (
    ConditionalGammaModel,
    ConditionalWeibullModel,
    TemporalModelRegistry,
    build_standard_temporal_registry,
)
from .topology import (
    ContextRelationIndex,
    GeneratedTopology,
    RelationLayerSpec,
    SparseHeterogeneousTopologyGenerator,
    RelationNeighbor,
    relation_is_active,
)

__all__ = [
    "Candidate",
    "CandidateSpec",
    "CandidateStatus",
    "CalibrationBundle",
    "CalibrationEngine",
    "CalibrationObservation",
    "ContextEvidence",
    "ContextRelation",
    "DomainPackage",
    "LoadedDomainSpec",
    "MechanismRegistry",
    "MechanismSpec",
    "NormalizedJsonlReferenceAdapter",
    "Entity",
    "EpisodeContext",
    "EpisodeResult",
    "RiskAlternativeEvidence",
    "RiskSetRecord",
    "EventParticipant",
    "EventRecord",
    "EventRelation",
    "ObservationPlan",
    "RelationSpec",
    "ReferenceDataset",
    "ReferenceEvent",
    "ReferenceWindow",
    "RelationLayerSpec",
    "RelationNeighbor",
    "SimulationEngine",
    "SemanticJudge",
    "SemanticJudgeResult",
    "LLMSemanticJudge",
    "SemanticCorrector",
    "SemanticCorrectionResult",
    "LLMSemanticCorrector",
    "generate_episode_qa",
    "validate_qa_pair",
    "instruction_training_example",
    "chat_training_example",
    "SparseHeterogeneousTopologyGenerator",
    "StatePredicateEvidence",
    "TemporalExtent",
    "TemporalModelRegistry",
    "ConditionalGammaModel",
    "ConditionalWeibullModel",
    "ExponentialRateMLEFitter",
    "GeneratedTopology",
    "ContextRelationIndex",
    "build_standard_temporal_registry",
    "build_root_event_observations",
    "build_standard_calibration_fitters",
    "load_domain_spec",
    "load_mechanism_registry",
    "relation_is_active",
    "validate_reference_dataset",
]
