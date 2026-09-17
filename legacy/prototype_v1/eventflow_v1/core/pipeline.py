"""End-to-end EventFlow v1 orchestration."""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .judge import build_judge
from .validation import DistributionAccumulator, validate_common_contract
from .writer import EventFlowWriter
from ..domains.transportation.generator import TransportationEpisodeGenerator
from ..domains.transportation.calibration import load_calibration_profile
from ..domains.transportation.event_types import load_event_type_registry
from ..domains.transportation.network import build_network
from ..domains.transportation.text import TransportationTextGenerator
from ..domains.transportation.validation import TransportationValidator


class EventFlowPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str | Path | None = None,
        judge: Any | None = None,
    ):
        self.config = copy.deepcopy(config)
        self.judge_override = judge
        configured_output = self.config.get("output", {}).get(
            "directory", "data_generation/output_eventflow_v1"
        )
        self.output_dir = _unique_output_directory(Path(output_dir or configured_output))

    def run(self) -> Dict[str, Any]:
        seed = int(self.config["seed"])
        calibration_profile = load_calibration_profile(
            self.config.get("calibration", {})
        )
        event_type_registry = load_event_type_registry()
        networks = _build_networks(
            self.config["network"], seed, calibration_profile.get("network_demand", {})
        )
        validation_config = self.config["validation"]
        judge = self.judge_override or build_judge(
            str(validation_config.get("judge_mode", "heuristic")),
            int(validation_config.get("llm_judge_max_tokens", 1200)),
            int(validation_config.get("llm_judge_protocol_max_attempts", 3)),
            int(validation_config.get("llm_judge_reason_limit", 4)),
            int(validation_config.get("llm_judge_response_preview_chars", 2000)),
        )
        max_attempts = int(validation_config.get("max_episode_attempts", 8))
        episode_count = int(self.config["episode_count"])
        stats = DistributionAccumulator(event_type_registry)
        rejected_attempts = 0
        generation_retry_count = 0
        semantic_rejection_count = 0
        judge_protocol_retry_count = 0
        judge_protocol_failure_count = 0
        rejection_reason_counts: Counter[str] = Counter()
        judge_protocol_error_counts: Counter[str] = Counter()
        accepted = 0

        manifest = {
            "schema_version": self.config.get("schema_version", "eventflow-v0"),
            "generator_version": self.config.get("generator_version", "eventflow-v1"),
            "domain": self.config["domain"],
            "seed": seed,
            "created_at": datetime.now().astimezone().isoformat(),
            "configuration": self.config,
            "contract_notes": {
                "relation_source_of_truth": "relations.jsonl",
                "topology_is_not_parenthood": True,
                "time_representation": "exact_timestamp_and_float_offset",
                "matrix_representation": "not_used",
                "node_count_semantics": "road_segments_per_generated_network",
                "judge_protocol_errors_regenerate_episode": False,
                "calibration_profile_id": calibration_profile["profile_id"],
                "calibration_status": calibration_profile["status"],
                "statistical_realism": "not_claimed_until_empirical_fit_and_holdout_validation",
                "event_type_policy": "namespaced_ids_resolved_by_versioned_registry",
                "participant_policy": "qualified_event_to_entity_links",
            },
            "event_type_registry": event_type_registry.manifest_record(
                "event_type_registry.json"
            ),
        }

        with EventFlowWriter(self.output_dir) as writer:
            writer.write_json("manifest.json", manifest)
            writer.write_json("calibration_profile.json", calibration_profile)
            writer.write_json("event_type_registry.json", event_type_registry.payload)
            for network in networks:
                writer.write_network(network)
            for episode_index in range(episode_count):
                network = networks[episode_index % len(networks)]
                generator = TransportationEpisodeGenerator(
                    network,
                    self.config["simulation"],
                    calibration_profile,
                    event_type_registry,
                )
                text_generator = TransportationTextGenerator(network, self.config["text"])
                domain_validator = TransportationValidator(
                    network,
                    self.config["simulation"],
                    self.config["network"],
                    calibration_profile,
                    event_type_registry,
                )
                accepted_bundle = None
                attempt_reports = []
                for attempt in range(max_attempts):
                    episode_seed = seed + episode_index * 1009 + attempt * 104729
                    bundle = generator.generate(episode_index, episode_seed)
                    bundle.texts = text_generator.generate(bundle, episode_seed + 17)
                    common = validate_common_contract(
                        bundle, set(network.entity_by_id)
                    )
                    domain = domain_validator.validate(bundle)
                    judge_result = (
                        judge.evaluate(bundle) if common.passed and domain.passed else None
                    )
                    if judge_result:
                        judge_protocol_retry_count += max(
                            0, judge_result.protocol_attempt_count - 1
                        )
                        protocol_errors = judge_result.raw.get(
                            "protocol_errors_before_success",
                            judge_result.raw.get("protocol_errors", []),
                        )
                        for protocol_error in protocol_errors:
                            judge_protocol_error_counts[
                                str(protocol_error.get("error", "unknown protocol error"))
                            ] += 1
                    attempt_report = {
                        "attempt": attempt + 1,
                        "seed": episode_seed,
                        "common": common.to_dict(),
                        "domain": domain.to_dict(),
                        "judge": judge_result.to_dict() if judge_result else {
                            "outcome": "skipped",
                            "passed": False,
                            "reasons": ["Skipped because deterministic validation failed"],
                            "mode": "skipped",
                            "protocol_attempt_count": 0,
                        },
                    }
                    attempt_reports.append(attempt_report)
                    if judge_result and judge_result.outcome == "protocol_error":
                        judge_protocol_failure_count += 1
                        failure_payload = {
                            "failure_type": "judge_protocol_error",
                            "episode_id": bundle.episode_id,
                            "episode_seed": episode_seed,
                            "generation_attempt": attempt + 1,
                            "judge": judge_result.to_dict(),
                            "routing": "pipeline_stopped_without_regenerating_episode",
                        }
                        writer.write_json(
                            f"judge_protocol_failure_{episode_index:07d}.json",
                            failure_payload,
                        )
                        raise RuntimeError(
                            f"Judge protocol failed for episode {episode_index} after "
                            f"{judge_result.protocol_attempt_count} same-episode attempts; "
                            "the episode was not regenerated"
                        )
                    if common.passed and domain.passed and judge_result and judge_result.passed:
                        bundle.validation = {
                            "passed": True,
                            "accepted_attempt": attempt + 1,
                            "common": common.to_dict(),
                            "domain": domain.to_dict(),
                            "judge": judge_result.to_dict(),
                        }
                        accepted_bundle = bundle
                        break
                    if not common.passed or not domain.passed:
                        generation_retry_count += 1
                        rejected_attempts += 1
                        for reason in common.errors + domain.errors:
                            rejection_reason_counts[reason] += 1
                    elif judge_result and judge_result.outcome == "semantic_rejected":
                        semantic_rejection_count += 1
                        rejected_attempts += 1
                        for reason in judge_result.reasons:
                            rejection_reason_counts[f"semantic_judge: {reason}"] += 1
                if accepted_bundle is None:
                    failure_path = self.output_dir / f"failed_episode_{episode_index:07d}.json"
                    with failure_path.open("w", encoding="utf-8") as handle:
                        json.dump(attempt_reports, handle, ensure_ascii=False, indent=2)
                    raise RuntimeError(
                        f"Episode {episode_index} failed after {max_attempts} attempts; "
                        f"see {failure_path}"
                    )
                writer.write_episode(accepted_bundle)
                stats.update(accepted_bundle)
                accepted += 1

            total_entity_count = sum(len(network.entities) for network in networks)
            distribution = stats.report(total_entity_count=total_entity_count)
            congestion_count = int(
                distribution.get("event_metric_tag_counts", {}).get("congestion", 0)
            )
            recovery_count = int(
                distribution.get("event_metric_tag_counts", {}).get("recovery", 0)
            )
            distribution["recovery_to_congestion_ratio"] = (
                round(recovery_count / congestion_count, 6)
                if congestion_count
                else 0.0
            )
            topology_report = []
            for network in networks:
                degrees = [len(network.outgoing.get(road_id, [])) for road_id in network.road_ids]
                topology_report.append(
                    {
                        "network_id": network.network_id,
                        "structure_family": network.structure_family,
                        "road_segment_count": len(network.road_ids),
                        "edge_count": len(network.edges),
                        "max_out_degree": max(degrees, default=0),
                        "mean_out_degree": (
                            round(sum(degrees) / len(degrees), 6) if degrees else 0.0
                        ),
                    }
                )
            hard_fail = bool(validation_config.get("distribution_hard_fail", False))
            distribution_passed = not (hard_fail and distribution["warnings"])
            quality_report = {
                "passed": distribution_passed and accepted == episode_count,
                "accepted_episode_count": accepted,
                "network_count": len(networks),
                "rejected_attempt_count": rejected_attempts,
                "generation_retry_count": generation_retry_count,
                "semantic_rejection_count": semantic_rejection_count,
                "judge_protocol_retry_count": judge_protocol_retry_count,
                "judge_protocol_failure_count": judge_protocol_failure_count,
                "rejection_reason_counts": dict(rejection_reason_counts),
                "judge_protocol_error_counts": dict(judge_protocol_error_counts),
                "deterministic_validation": "passed_for_all_written_episodes",
                "semantic_judge_mode": validation_config.get("judge_mode", "heuristic"),
                "distribution": distribution,
                "topology": topology_report,
                "calibration": {
                    "profile_id": calibration_profile["profile_id"],
                    "profile_version": calibration_profile.get("profile_version"),
                    "status": calibration_profile["status"],
                    "fit_requirements": calibration_profile.get("fit_requirements", {}),
                    "sources": {
                        "propagation_lag": calibration_profile["propagation_lag"].get(
                            "source", {}
                        ),
                        "recovery_lag": calibration_profile["recovery_lag"].get(
                            "source", {}
                        ),
                    },
                },
                "limitations": [
                    "Propagation uses an FHWA-documented reference formula, but local coefficients and generated demand inputs are not empirically fitted.",
                    "Recovery coefficients are transparent synthetic priors pending fit to clearance and return-to-normal-flow observations.",
                    "Root-event capacity reductions and initial congestion severity are transparent synthetic priors pending empirical fit.",
                    "The semantic judge cannot establish statistical realism.",
                    "All events are currently emitted as directly observed with zero observation delay.",
                ],
            }
            writer.write_json("quality_report.json", quality_report)

        return {
            "output_dir": str(self.output_dir.resolve()),
            "accepted_episode_count": accepted,
            "rejected_attempt_count": rejected_attempts,
            "quality_report": quality_report,
        }


def _unique_output_directory(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return path.with_name(f"{path.name}_{timestamp}")


def _build_networks(
    config: Dict[str, Any],
    seed: int,
    demand_profile: Dict[str, Any] | None = None,
) -> list[Any]:
    if config.get("external_network_file"):
        return [build_network(config, seed, demand_profile)]
    count = int(config.get("network_count", 1))
    families = list(config.get("structure_families", ["random_connected"]))
    base_id = str(config.get("network_id", "transport_network"))
    networks = []
    for index in range(count):
        network_config = copy.deepcopy(config)
        network_config["network_id"] = f"{base_id}_{index:03d}"
        network_config["structure_family"] = families[index % len(families)]
        networks.append(
            build_network(network_config, seed + index * 7919, demand_profile)
        )
    return networks
