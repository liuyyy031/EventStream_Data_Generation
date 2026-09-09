"""Dataset-level orchestration over a domain package and temporal registry."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .domain import DomainPackage
from .scheduler import SimulationEngine
from .temporal import TemporalModelRegistry
from .validation import validate_text_alignment
from .writer import DatasetWriter


class GenerationPipeline:
    def __init__(
        self,
        domain_package: DomainPackage,
        temporal_models: TemporalModelRegistry,
        config: Dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        self.domain_package = domain_package
        self.temporal_models = temporal_models
        self.config = dict(config)
        self.output_dir = _unique_output_directory(Path(output_dir))

    def run(self) -> Dict[str, Any]:
        episode_count = int(self.config.get("episode_count", 1))
        base_seed = int(self.config.get("seed", 1))
        max_events = int(self.config.get("max_events_per_episode", 10_000))
        if episode_count <= 0:
            raise ValueError("episode_count must be positive")
        engine = SimulationEngine(
            self.domain_package, self.temporal_models, max_events=max_events
        )
        manifest = {
            "schema_version": "event-stream-contract-v2",
            "generator_version": "multidomain-generation-core-v0.1.0",
            "domain": self.domain_package.domain_id,
            "seed": base_seed,
            "created_at": datetime.now().astimezone().isoformat(),
            "configuration": self.config,
            "contract_notes": {
                "context_topology_is_not_event_parenthood": True,
                "occurrence_observation_recording_times_are_separate": True,
                "candidate_terminal_states_are_explicit": True,
                "right_censoring_is_not_treated_as_non_occurrence": True,
                "time_varying_hazards_preserve_random_thresholds": True,
                "statistical_realism_requires_domain_fit_and_holdout_validation": True,
            },
        }
        validation_failures = 0
        text_alignment_failures = 0
        event_type_counts: Counter[str] = Counter()
        relation_type_counts: Counter[str] = Counter()
        candidate_status_counts: Counter[str] = Counter()
        scenario_counts: Counter[str] = Counter()
        temporal_model_usage_counts: Counter[str] = Counter()
        termination_reason_counts: Counter[str] = Counter()
        total_events = 0
        with DatasetWriter(self.output_dir) as writer:
            writer.write_manifest(
                manifest,
                self.temporal_models,
                self.domain_package.catalog(),
            )
            for episode_index in range(episode_count):
                result = engine.run(
                    episode_index, base_seed + episode_index * 104729
                )
                text_alignment = self.domain_package.render_episode_text(result)
                text_validation = validate_text_alignment(result, text_alignment)
                result.validation["checks"]["text_alignment"] = text_validation[
                    "passed"
                ]
                if text_validation["errors"]:
                    result.validation["errors"].extend(text_validation["errors"])
                    result.validation["passed"] = False
                text_alignment_failures += int(not text_validation["passed"])
                writer.write_episode(result)
                writer.write_episode_text(text_alignment)
                validation_failures += int(not result.validation["passed"])
                total_events += len(result.events)
                event_type_counts.update(event.event_type_id for event in result.events)
                relation_type_counts.update(
                    relation.relation_type_id for relation in result.event_relations
                )
                candidate_status_counts.update(
                    candidate.status.value for candidate in result.candidates
                )
                scenario = result.episode_attributes.get("scenario_family")
                if scenario:
                    scenario_counts[str(scenario)] += 1
                temporal_model_usage_counts.update(
                    candidate.temporal_model_ref for candidate in result.candidates
                )
                termination_reason_counts[result.termination_reason] += 1
            administratively_truncated = termination_reason_counts.get(
                "event_limit_reached", 0
            )
            quality = {
                "passed": (
                    validation_failures == 0
                    and administratively_truncated == 0
                ),
                "episode_count": episode_count,
                "validation_failure_count": validation_failures,
                "text_alignment_failure_count": text_alignment_failures,
                "total_event_count": total_events,
                "event_type_counts": dict(event_type_counts),
                "relation_type_counts": dict(relation_type_counts),
                "candidate_status_counts": dict(candidate_status_counts),
                "scenario_counts": dict(scenario_counts),
                "temporal_model_usage_counts": dict(temporal_model_usage_counts),
                "termination_reason_counts": dict(termination_reason_counts),
                "administratively_truncated_episode_count": administratively_truncated,
                "statistical_realism": "not_claimed_without_domain_fit_and_holdout_validation",
            }
            writer.write_json("quality_report.json", quality)
        return {"output_dir": str(self.output_dir.resolve()), "quality_report": quality}


def _unique_output_directory(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return path.with_name(f"{path.name}_{timestamp}")
