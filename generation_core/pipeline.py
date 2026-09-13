"""Dataset-level orchestration over a domain package and temporal registry."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .domain import DomainPackage
from .models import EpisodeResult
from .scheduler import SimulationEngine
from .semantic_judge import SemanticJudge, SemanticJudgeResult
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
        semantic_judge: SemanticJudge | None = None,
    ) -> None:
        self.domain_package = domain_package
        self.temporal_models = temporal_models
        self.config = dict(config)
        self.output_dir = _unique_output_directory(Path(output_dir))
        self.semantic_judge = semantic_judge

    def run(self) -> Dict[str, Any]:
        episode_count = int(self.config.get("episode_count", 1))
        base_seed = int(self.config.get("seed", 1))
        max_events = int(self.config.get("max_events_per_episode", 10_000))
        judge_config = dict(self.config.get("semantic_judge", {}))
        judge_mode = str(judge_config.get("mode", "none"))
        judge_workers = max(1, int(judge_config.get("workers", 1)))
        max_generation_attempts = max(
            1, int(judge_config.get("max_generation_attempts", 3))
        )
        if episode_count <= 0:
            raise ValueError("episode_count must be positive")
        if judge_mode not in {"none", "llm"}:
            raise ValueError("semantic_judge.mode must be 'none' or 'llm'")
        if judge_mode == "llm" and self.semantic_judge is None:
            raise ValueError("LLM judge mode requires an injected semantic judge")
        engine = SimulationEngine(
            self.domain_package, self.temporal_models, max_events=max_events
        )
        domain_catalog = self.domain_package.catalog()
        manifest = {
            "schema_version": "event-stream-contract-v2",
            "generator_version": "multidomain-generation-core-v0.4.0",
            "domain": self.domain_package.domain_id,
            "seed": base_seed,
            "created_at": datetime.now().astimezone().isoformat(),
            "configuration": self.config,
            "contract_notes": {
                "primary_validity_target": "mechanism_coherence_and_auditable_event_chains",
                "context_topology_is_not_event_parenthood": True,
                "domain_topology_spec_is_executable_and_type_checked": True,
                "context_evidence_is_time_scoped_and_rule_specific": True,
                "occurrence_observation_recording_times_are_separate": True,
                "candidate_terminal_states_are_explicit": True,
                "right_censoring_is_not_treated_as_non_occurrence": True,
                "time_varying_hazards_preserve_random_thresholds": True,
                "time_type_entity_choices_share_one_risk_set": True,
                "deterministic_atoms_are_not_given_fictitious_densities": True,
                "realized_parent_attribution_is_explicit": True,
                "llm_judge_is_semantic_advisory_not_statistical_proof": True,
                "judge_protocol_failure_is_not_semantic_rejection": True,
                "statistical_realism_requires_domain_fit_and_holdout_validation": True,
            },
            "semantic_judge": {
                "mode": judge_mode,
                "model": judge_config.get("model"),
                "base_url": judge_config.get("base_url"),
                "workers": judge_workers,
                "max_generation_attempts": max_generation_attempts,
            },
        }

        accepted: Dict[int, Tuple[EpisodeResult, Dict[str, Any]]] = {}
        judge_records: List[Dict[str, Any]] = []
        semantic_judged_episode_count = 0
        semantic_judged_indices: set[int] = set()
        semantic_pass_count = 0
        semantic_rejection_count = 0
        judge_protocol_retry_count = 0
        judge_protocol_failure_count = 0
        deterministic_rejection_count = 0
        generation_retry_count = 0
        failed_episode_indices: set[int] = set()
        rejection_reason_counts: Counter[str] = Counter()
        judge_protocol_error_counts: Counter[str] = Counter()

        if judge_mode == "none":
            for episode_index in range(episode_count):
                episode_seed = base_seed + episode_index * 104729
                result, text_alignment = self._prepare_episode(
                    engine, episode_index, episode_seed
                )
                result.validation["semantic_judge"] = {
                    "outcome": "not_run",
                    "passed": None,
                    "mode": "none",
                    "reasons": ["No semantic judge was requested"],
                }
                accepted[episode_index] = (result, text_alignment)
                judge_records.append(
                    {
                        "episode_index": episode_index,
                        "episode_id": result.episode_id,
                        "generation_attempt": 1,
                        "seed": episode_seed,
                        **result.validation["semantic_judge"],
                    }
                )
        else:
            pending = set(range(episode_count))
            protocol_abort = False
            for generation_attempt in range(1, max_generation_attempts + 1):
                if not pending or protocol_abort:
                    break
                ready: Dict[int, Tuple[int, EpisodeResult, Dict[str, Any]]] = {}
                for episode_index in sorted(pending):
                    if generation_attempt > 1:
                        generation_retry_count += 1
                    episode_seed = (
                        base_seed
                        + episode_index * 104729
                        + (generation_attempt - 1) * 15485863
                    )
                    result, text_alignment = self._prepare_episode(
                        engine, episode_index, episode_seed
                    )
                    if not result.validation["passed"]:
                        deterministic_rejection_count += 1
                        rejection_reason_counts.update(result.validation["errors"])
                        judge_records.append(
                            {
                                "episode_index": episode_index,
                                "episode_id": result.episode_id,
                                "generation_attempt": generation_attempt,
                                "seed": episode_seed,
                                "outcome": "deterministic_rejected",
                                "passed": False,
                                "mode": "skipped",
                                "reasons": list(result.validation["errors"]),
                                "flagged_record_ids": [],
                                "protocol_attempt_count": 0,
                            }
                        )
                        continue
                    ready[episode_index] = (
                        episode_seed,
                        result,
                        text_alignment,
                    )

                assert self.semantic_judge is not None
                with ThreadPoolExecutor(max_workers=judge_workers) as pool:
                    futures = {
                        pool.submit(
                            self.semantic_judge.evaluate,
                            result,
                            text_alignment,
                            domain_catalog,
                        ): (episode_index, episode_seed, result, text_alignment)
                        for episode_index, (
                            episode_seed,
                            result,
                            text_alignment,
                        ) in ready.items()
                    }
                    for future in as_completed(futures):
                        (
                            episode_index,
                            episode_seed,
                            result,
                            text_alignment,
                        ) = futures[future]
                        try:
                            judge_result = future.result()
                        except Exception as exc:  # noqa: BLE001 - API boundary
                            judge_result = SemanticJudgeResult(
                                outcome="protocol_error",
                                passed=False,
                                reasons=[f"LLM judge call failed: {exc}"],
                                flagged_record_ids=[],
                                mode="llm",
                                protocol_attempt_count=1,
                                model_id=judge_config.get("model"),
                                provider_base_url=judge_config.get("base_url"),
                                protocol_errors=[{"attempt": 1, "error": str(exc)}],
                            )
                        semantic_judged_episode_count += 1
                        semantic_judged_indices.add(episode_index)
                        judge_protocol_retry_count += max(
                            0, judge_result.protocol_attempt_count - 1
                        )
                        record = {
                            "episode_index": episode_index,
                            "episode_id": result.episode_id,
                            "generation_attempt": generation_attempt,
                            "seed": episode_seed,
                            **judge_result.to_dict(),
                        }
                        judge_records.append(record)
                        if judge_result.outcome == "protocol_error":
                            judge_protocol_failure_count += 1
                            judge_protocol_error_counts.update(
                                str(item.get("error", "unknown protocol error"))
                                for item in judge_result.protocol_errors
                            )
                            failed_episode_indices.add(episode_index)
                            protocol_abort = True
                            continue
                        if judge_result.passed:
                            semantic_pass_count += 1
                            result.validation["semantic_judge"] = judge_result.to_dict()
                            accepted[episode_index] = (result, text_alignment)
                            pending.discard(episode_index)
                        else:
                            semantic_rejection_count += 1
                            rejection_reason_counts.update(judge_result.reasons)
                if protocol_abort:
                    break
            failed_episode_indices.update(set(range(episode_count)) - set(accepted))

        validation_failures = 0
        text_alignment_failures = 0
        event_type_counts: Counter[str] = Counter()
        relation_type_counts: Counter[str] = Counter()
        candidate_status_counts: Counter[str] = Counter()
        scenario_counts: Counter[str] = Counter()
        temporal_model_usage_counts: Counter[str] = Counter()
        termination_reason_counts: Counter[str] = Counter()
        topology_contexts: Dict[str, Any] = {}
        total_events = 0
        context_grounded_event_relation_count = 0
        evaluated_state_predicate_count = 0
        risk_set_selection_mode_counts: Counter[str] = Counter()
        continuous_factorization_count = 0
        total_risk_alternative_count = 0
        with DatasetWriter(self.output_dir) as writer:
            writer.write_manifest(
                manifest,
                self.temporal_models,
                domain_catalog,
            )
            for record in sorted(
                judge_records,
                key=lambda item: (
                    int(item["episode_index"]),
                    int(item["generation_attempt"]),
                ),
            ):
                writer.write_judge_result(record)
            for episode_index in sorted(accepted):
                result, text_alignment = accepted[episode_index]
                writer.write_episode(result)
                writer.write_episode_text(text_alignment)
                validation_failures += int(not result.validation["passed"])
                text_alignment_failures += int(
                    not result.validation["checks"].get("text_alignment", False)
                )
                total_events += len(result.events)
                event_type_counts.update(event.event_type_id for event in result.events)
                relation_type_counts.update(
                    relation.relation_type_id for relation in result.event_relations
                )
                context_grounded_event_relation_count += sum(
                    bool(relation.context_evidence.context_relation_ids)
                    for relation in result.event_relations
                )
                evaluated_state_predicate_count += sum(
                    len(candidate.context_evidence.state_predicates)
                    for candidate in result.candidates
                )
                candidate_status_counts.update(
                    candidate.status.value for candidate in result.candidates
                )
                risk_set_selection_mode_counts.update(
                    risk_set.selection_mode for risk_set in result.risk_sets
                )
                continuous_factorization_count += sum(
                    bool(
                        risk_set.factorization.get(
                            "continuous_hazard_factorization_applicable"
                        )
                    )
                    for risk_set in result.risk_sets
                )
                total_risk_alternative_count += sum(
                    len(risk_set.alternatives) for risk_set in result.risk_sets
                )
                scenario = result.episode_attributes.get("scenario_family")
                if scenario:
                    scenario_counts[str(scenario)] += 1
                temporal_model_usage_counts.update(
                    candidate.temporal_model_ref for candidate in result.candidates
                )
                termination_reason_counts[result.termination_reason] += 1
                if result.context_id not in topology_contexts:
                    topology_contexts[result.context_id] = result.context_attributes.get(
                        "topology_profile", {}
                    )
            administratively_truncated = termination_reason_counts.get(
                "event_limit_reached", 0
            )
            quality = {
                "passed": (
                    len(accepted) == episode_count
                    and validation_failures == 0
                    and administratively_truncated == 0
                    and judge_protocol_failure_count == 0
                ),
                "episode_count": episode_count,
                "accepted_episode_count": len(accepted),
                "failed_episode_indices": sorted(failed_episode_indices),
                "validation_failure_count": validation_failures,
                "text_alignment_failure_count": text_alignment_failures,
                "total_event_count": total_events,
                "event_type_counts": dict(event_type_counts),
                "relation_type_counts": dict(relation_type_counts),
                "context_grounded_event_relation_count": context_grounded_event_relation_count,
                "evaluated_state_predicate_count": evaluated_state_predicate_count,
                "candidate_status_counts": dict(candidate_status_counts),
                "risk_set_selection_mode_counts": dict(
                    risk_set_selection_mode_counts
                ),
                "continuous_hazard_factorization_count": continuous_factorization_count,
                "mean_risk_set_size": (
                    total_risk_alternative_count / total_events
                    if total_events
                    else 0.0
                ),
                "scenario_counts": dict(scenario_counts),
                "temporal_model_usage_counts": dict(temporal_model_usage_counts),
                "termination_reason_counts": dict(termination_reason_counts),
                "administratively_truncated_episode_count": administratively_truncated,
                "semantic_judge_mode": judge_mode,
                "semantic_judge_model": judge_config.get("model"),
                "semantic_judged_episode_count": len(semantic_judged_indices),
                "semantic_judge_call_count": semantic_judged_episode_count,
                "semantic_pass_count": semantic_pass_count,
                "semantic_rejection_count": semantic_rejection_count,
                "deterministic_rejection_count": deterministic_rejection_count,
                "generation_retry_count": generation_retry_count,
                "judge_protocol_retry_count": judge_protocol_retry_count,
                "judge_protocol_failure_count": judge_protocol_failure_count,
                "rejection_reason_counts": dict(rejection_reason_counts),
                "judge_protocol_error_counts": dict(
                    judge_protocol_error_counts
                ),
                "topology_contexts": [
                    {"context_id": context_id, **profile}
                    for context_id, profile in sorted(topology_contexts.items())
                ],
                "statistical_realism": "not_claimed_without_domain_fit_and_holdout_validation",
                "validity_scope": "mechanism_coherent_synthetic_generation_not_reference_dataset_replication",
            }
            writer.write_json("quality_report.json", quality)
        return {"output_dir": str(self.output_dir.resolve()), "quality_report": quality}

    def _prepare_episode(
        self,
        engine: SimulationEngine,
        episode_index: int,
        episode_seed: int,
    ) -> Tuple[EpisodeResult, Dict[str, Any]]:
        result = engine.run(episode_index, episode_seed)
        text_alignment = self.domain_package.render_episode_text(result)
        text_validation = validate_text_alignment(result, text_alignment)
        result.validation["checks"]["text_alignment"] = text_validation["passed"]
        if text_validation["errors"]:
            result.validation["errors"].extend(text_validation["errors"])
            result.validation["passed"] = False
        return result, text_alignment


def _unique_output_directory(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return path.with_name(f"{path.name}_{timestamp}")
