"""Layered counterexample challenge for deterministic, domain and LLM judges."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from ..core.config import load_config
from ..core.judge import build_judge
from ..core.models import EpisodeBundle
from ..core.validation import validate_common_contract
from ..domains.transportation.calibration import load_calibration_profile
from ..domains.transportation.generator import TransportationEpisodeGenerator
from ..domains.transportation.network import TransportNetwork, generate_network
from ..domains.transportation.text import TransportationTextGenerator
from ..domains.transportation.validation import TransportationValidator


Mutation = Callable[[EpisodeBundle, TransportNetwork, Dict[str, Any]], EpisodeBundle]


def run_judge_challenge(
    config: Dict[str, Any],
    judge_mode: str,
    output_dir: str | Path,
    judge_override: Any | None = None,
) -> Dict[str, Any]:
    output = _unique_directory(Path(output_dir))
    output.mkdir(parents=True, exist_ok=False)
    calibration = load_calibration_profile(config.get("calibration", {}))
    network = generate_network(
        "judge_challenge_network",
        "grid",
        64,
        0.1,
        int(config["network"]["max_topology_out_degree"]),
        int(config["seed"]),
    )
    simulation = copy.deepcopy(config["simulation"])
    simulation["scenario_families"] = ["road_closure"]
    simulation["duration_seconds"] = max(14400.0, float(simulation["duration_seconds"]))
    simulation["recovery_probability"] = 1.0
    simulation["background_root_rate_per_hour"] = 0.0
    simulation["max_events_per_episode"] = max(
        32, int(simulation["max_events_per_episode"])
    )
    text_config = copy.deepcopy(config["text"])
    text_config["paraphrase_rate"] = 0.0
    base = _build_base_bundle(network, simulation, text_config, calibration, int(config["seed"]))
    validator = TransportationValidator(
        network, simulation, config["network"], calibration
    )
    validation_config = config["validation"]
    judge = judge_override or build_judge(
        judge_mode,
        int(validation_config.get("llm_judge_max_tokens", 1200)),
        int(validation_config.get("llm_judge_protocol_max_attempts", 3)),
        int(validation_config.get("llm_judge_reason_limit", 4)),
        int(validation_config.get("llm_judge_response_preview_chars", 2000)),
    )

    cases = [
        ("valid_control", "none", None),
        ("child_before_parent", "common", _child_before_parent),
        ("text_grounded_fact_mismatch", "common", _text_grounded_fact_mismatch),
        ("nonexistent_network_edge", "domain", _nonexistent_network_edge),
        ("calibration_lag_not_reproducible", "domain", _calibration_lag_not_reproducible),
        ("orphan_recovery_without_relation", "domain", _orphan_recovery),
        ("unsupported_direct_cause_in_text", "semantic", _unsupported_direct_cause_text),
    ]
    results = []
    for name, expected_layer, mutation in cases:
        bundle = copy.deepcopy(base)
        if mutation is not None:
            bundle = mutation(bundle, network, text_config)
        common = validate_common_contract(bundle, set(network.entity_by_id))
        domain = validator.validate(bundle)
        judge_result = (
            judge.evaluate(bundle) if common.passed and domain.passed else None
        )
        actual_layer = _actual_detection_layer(common, domain, judge_result)
        assessable = not (expected_layer == "semantic" and judge_mode != "llm")
        expected_met = assessable and actual_layer == expected_layer
        if expected_layer == "none":
            expected_met = common.passed and domain.passed and bool(
                judge_result and judge_result.passed
            )
        results.append(
            {
                "case_id": name,
                "expected_detection_layer": expected_layer,
                "actual_detection_layer": actual_layer,
                "assessable_in_mode": assessable,
                "expected_met": expected_met,
                "common": common.to_dict(),
                "domain": domain.to_dict(),
                "judge": judge_result.to_dict() if judge_result else None,
            }
        )

    assessed = [case for case in results if case["assessable_in_mode"]]
    report = {
        "experiment": "eventflow_layered_judge_challenge_v1.1",
        "created_at": datetime.now().astimezone().isoformat(),
        "judge_mode": judge_mode,
        "calibration_profile_id": calibration["profile_id"],
        "passed": all(case["expected_met"] for case in assessed),
        "assessed_case_count": len(assessed),
        "unassessed_case_count": len(results) - len(assessed),
        "detected_as_expected_count": sum(case["expected_met"] for case in assessed),
        "cases": results,
        "interpretation": {
            "common": "schema, exact time, references and text grounding",
            "domain": "transport rules, topology agreement and reproducible calibrated lag",
            "semantic": "unsupported but structurally valid claims; requires --judge-mode llm",
        },
    }
    (output / "judge_challenge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    default_config = Path(__file__).resolve().parents[1] / "config" / "transportation_default.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--judge-mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-profile")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.calibration_profile:
        config.setdefault("calibration", {})["profile_file"] = args.calibration_profile
    report = run_judge_challenge(
        config, args.judge_mode, args.output_dir
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def _build_base_bundle(
    network: TransportNetwork,
    simulation: Dict[str, Any],
    text_config: Dict[str, Any],
    calibration: Dict[str, Any],
    seed: int,
) -> EpisodeBundle:
    for offset in range(100):
        episode_seed = seed + offset * 7919
        bundle = TransportationEpisodeGenerator(
            network, simulation, calibration
        ).generate(0, episode_seed)
        relation_types = {relation.relation_type for relation in bundle.relations}
        if {"propagates_downstream", "transitions_to_recovery"}.issubset(relation_types):
            bundle.texts = TransportationTextGenerator(network, text_config).generate(
                bundle, episode_seed + 17
            )
            return bundle
    raise RuntimeError("Could not generate a challenge control with propagation and recovery")


def _regenerate_texts(
    bundle: EpisodeBundle, network: TransportNetwork, text_config: Dict[str, Any]
) -> EpisodeBundle:
    bundle.texts = TransportationTextGenerator(network, text_config).generate(bundle, 99017)
    bundle.involved_entity_ids = sorted(
        {entity for event in bundle.events for entity in event.participant_ids}
    )
    return bundle


def _child_before_parent(
    bundle: EpisodeBundle, network: TransportNetwork, text_config: Dict[str, Any]
) -> EpisodeBundle:
    relation = bundle.relations[0]
    source = next(event for event in bundle.events if event.event_id == relation.source_event_id)
    target_index = next(
        index for index, event in enumerate(bundle.events) if event.event_id == relation.target_event_id
    )
    target = bundle.events[target_index]
    new_offset = max(0.0, source.time_offset_seconds - 1.0)
    start = datetime.fromisoformat(bundle.start_time)
    bundle.events[target_index] = replace(
        target,
        time_offset_seconds=round(new_offset, 3),
        event_time=(start + timedelta(seconds=new_offset)).isoformat(),
    )
    bundle.events.sort(key=lambda event: (event.time_offset_seconds, event.event_id))
    return _regenerate_texts(bundle, network, text_config)


def _text_grounded_fact_mismatch(
    bundle: EpisodeBundle, _network: TransportNetwork, _text_config: Dict[str, Any]
) -> EpisodeBundle:
    index = next(
        index
        for index, text in enumerate(bundle.texts)
        if text.text_role == "event" and text.variant == "canonical"
    )
    record = bundle.texts[index]
    grounded = copy.deepcopy(record.grounded_facts)
    grounded["event_type_id"] = "transportation.road.fabricated.event_type"
    bundle.texts[index] = replace(record, grounded_facts=grounded)
    return bundle


def _nonexistent_network_edge(
    bundle: EpisodeBundle, _network: TransportNetwork, _text_config: Dict[str, Any]
) -> EpisodeBundle:
    index = next(
        index
        for index, relation in enumerate(bundle.relations)
        if relation.relation_type == "propagates_downstream"
    )
    relation = bundle.relations[index]
    attributes = copy.deepcopy(relation.attributes)
    attributes["network_edge_id"] = "missing_edge_for_challenge"
    bundle.relations[index] = replace(relation, attributes=attributes)
    return bundle


def _calibration_lag_not_reproducible(
    bundle: EpisodeBundle, network: TransportNetwork, text_config: Dict[str, Any]
) -> EpisodeBundle:
    relation = next(
        relation
        for relation in bundle.relations
        if relation.relation_type == "propagates_downstream"
    )
    source = next(event for event in bundle.events if event.event_id == relation.source_event_id)
    target_index = next(
        index for index, event in enumerate(bundle.events) if event.event_id == relation.target_event_id
    )
    target = bundle.events[target_index]
    new_offset = source.time_offset_seconds + 1.0
    start = datetime.fromisoformat(bundle.start_time)
    bundle.events[target_index] = replace(
        target,
        time_offset_seconds=round(new_offset, 3),
        event_time=(start + timedelta(seconds=new_offset)).isoformat(),
    )
    bundle.events.sort(key=lambda event: (event.time_offset_seconds, event.event_id))
    return _regenerate_texts(bundle, network, text_config)


def _orphan_recovery(
    bundle: EpisodeBundle, network: TransportNetwork, text_config: Dict[str, Any]
) -> EpisodeBundle:
    relation = next(
        relation
        for relation in bundle.relations
        if relation.relation_type == "transitions_to_recovery"
    )
    bundle.relations = [
        item for item in bundle.relations if item.relation_id != relation.relation_id
    ]
    return _regenerate_texts(bundle, network, text_config)


def _unsupported_direct_cause_text(
    bundle: EpisodeBundle, _network: TransportNetwork, _text_config: Dict[str, Any]
) -> EpisodeBundle:
    index = next(
        index
        for index, text in enumerate(bundle.texts)
        if text.text_role == "episode" and text.variant == "canonical"
    )
    record = bundle.texts[index]
    source_id = bundle.events[0].event_id
    target_id = bundle.events[-1].event_id
    unsupported = (
        f" The data establishes that {source_id} directly caused {target_id}."
    )
    bundle.texts[index] = replace(record, content=record.content + unsupported)
    return bundle


def _actual_detection_layer(common: Any, domain: Any, judge_result: Any) -> str:
    if not common.passed:
        return "common"
    if not domain.passed:
        return "domain"
    if judge_result is None:
        return "none"
    if judge_result.outcome == "semantic_rejected":
        return "semantic"
    if judge_result.outcome == "protocol_error":
        return "protocol_error"
    return "none"


def _unique_directory(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return path.with_name(f"{path.name}_{timestamp}")


if __name__ == "__main__":
    raise SystemExit(main())
