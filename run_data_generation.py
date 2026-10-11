"""Command-line entry point for the neutral multi-domain generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_generation.domain_packages.registry import DOMAIN_BUILDERS
from data_generation.domain_packages.coverage_profiles import get_coverage_profile
from data_generation.generation_core.pipeline import GenerationPipeline
from data_generation.generation_core.distribution_calibration import (
    VALID_CALIBRATION_COMPONENTS,
    apply_sequence_calibration_bundle,
)
from data_generation.generation_core.mechanism_reweighting import (
    LegalMechanismReweighter,
)
from data_generation.generation_core.semantic_correction import LLMSemanticCorrector
from data_generation.generation_core.semantic_judge import LLMSemanticJudge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=tuple(DOMAIN_BUILDERS),
        default="transportation",
        help="Domain package to run when --config is omitted.",
    )
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--episode-count", type=int)
    parser.add_argument(
        "--nodes-per-context",
        "--nodes-per-network",
        dest="nodes_per_context",
        type=int,
    )
    parser.add_argument("--max-events-per-episode", type=int)
    parser.add_argument("--scenario-family")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--time-unit", choices=("second", "millisecond", "minute", "hour", "day", "year", "kyr", "Myr"))
    parser.add_argument("--time-axis-kind", choices=("calendar", "relative", "deep_time"))
    parser.add_argument("--duration-min", type=float, help="Uniform observation horizon lower bound, in --time-unit")
    parser.add_argument("--duration-max", type=float, help="Uniform observation horizon upper bound, in --time-unit")
    parser.add_argument(
        "--calibration-bundle",
        help="Apply a sequence-calibration-bundle-v1 before generation.",
    )
    parser.add_argument(
        "--allow-pending-calibration",
        action="store_true",
        help=(
            "Allow an empirically fitted bundle that is still pending domain "
            "acceptance. Intended for controlled evaluation runs only."
        ),
    )
    parser.add_argument(
        "--calibration-components",
        default="scenario,temporal,mechanism_reweighting",
        help=(
            "Comma-separated calibration components: scenario, temporal, "
            "mechanism_reweighting. This supports component-level ablations."
        ),
    )
    parser.add_argument(
        "--mechanism-reweighting-mode",
        choices=("first_order", "first_second_order"),
        default="first_second_order",
    )
    parser.add_argument("--mechanism-reweighting-strength", type=float, default=1.0)
    parser.add_argument(
        "--mechanism-reweighting-min-history-count", type=int, default=5
    )
    parser.add_argument(
        "--mechanism-reweighting-min-multiplier", type=float, default=0.25
    )
    parser.add_argument(
        "--mechanism-reweighting-max-multiplier", type=float, default=4.0
    )
    parser.add_argument(
        "--judge-mode",
        choices=("none", "llm"),
        help="Run no semantic judge or a server-side LLM semantic judge.",
    )
    parser.add_argument(
        "--judge-model",
        help="Explicit model ID passed to the OpenAI-compatible client.",
    )
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-workers", type=int)
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument("--judge-protocol-max-attempts", type=int)
    parser.add_argument("--judge-max-generation-attempts", type=int)
    parser.add_argument(
        "--semantic-correction-mode",
        choices=("none", "llm"),
        help=(
            "For text-scoped Judge rejections, disable correction or rewrite "
            "grounded text while keeping structured truth frozen."
        ),
    )
    parser.add_argument("--semantic-correction-max-rounds", type=int)
    parser.add_argument("--semantic-correction-max-tokens", type=int)
    parser.add_argument("--semantic-correction-protocol-max-attempts", type=int)
    parser.add_argument(
        "--qa-export-mode",
        choices=("none", "ground_truth_template"),
        help="Disable QA/training export or derive labels from structured truth.",
    )
    parser.add_argument("--qa-history-window-events", type=int)
    args = parser.parse_args()

    config_path = (
        Path(args.config)
        if args.config
        else (
            Path(__file__).resolve().parent
            / "domain_packages"
            / args.domain
            / "config"
            / "default.json"
        )
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    domain = str(config.get("domain", ""))
    if args.config and args.domain != "transportation" and args.domain != domain:
        parser.error("--domain and the configured domain disagree")
    if domain not in DOMAIN_BUILDERS:
        raise ValueError(f"No executable domain package is registered for {domain!r}")
    config.setdefault("coverage_profile", get_coverage_profile(domain))
    if args.episode_count is not None:
        config["episode_count"] = args.episode_count
    if args.nodes_per_context is not None:
        config["domain_config"]["nodes_per_context"] = args.nodes_per_context
    if args.seed is not None:
        config["seed"] = args.seed
        config["domain_config"]["base_seed"] = args.seed
    if args.max_events_per_episode is not None:
        config["max_events_per_episode"] = args.max_events_per_episode
    axis = dict(config.get("time_axis", {}))
    if args.time_unit:
        axis["unit"] = args.time_unit
    if args.time_axis_kind:
        axis["kind"] = args.time_axis_kind
    config["time_axis"] = axis
    if args.duration_min is not None or args.duration_max is not None:
        if args.duration_min is None or args.duration_max is None:
            parser.error("Both --duration-min and --duration-max are required")
        config["observation_duration"] = {"family": "uniform", "unit": axis.get("unit", "second"), "minimum": args.duration_min, "maximum": args.duration_max}
    judge_config = dict(config.get("semantic_judge", {}))
    if args.judge_mode is not None:
        judge_config["mode"] = args.judge_mode
    judge_config.setdefault("mode", "none")
    if args.judge_model is not None:
        judge_config["model"] = args.judge_model
    if args.judge_base_url is not None:
        judge_config["base_url"] = args.judge_base_url
    if args.judge_workers is not None:
        judge_config["workers"] = args.judge_workers
    judge_config.setdefault("workers", 1)
    if args.judge_max_tokens is not None:
        judge_config["max_tokens"] = args.judge_max_tokens
    judge_config.setdefault("max_tokens", 1000)
    if args.judge_protocol_max_attempts is not None:
        judge_config["protocol_max_attempts"] = args.judge_protocol_max_attempts
    judge_config.setdefault("protocol_max_attempts", 3)
    if args.judge_max_generation_attempts is not None:
        judge_config["max_generation_attempts"] = (
            args.judge_max_generation_attempts
        )
    judge_config.setdefault("max_generation_attempts", 3)
    correction_config = dict(config.get("semantic_correction", {}))
    if args.semantic_correction_mode is not None:
        correction_config["mode"] = args.semantic_correction_mode
    correction_config.setdefault("mode", "none")
    if args.semantic_correction_max_rounds is not None:
        correction_config["max_rounds"] = args.semantic_correction_max_rounds
    correction_config.setdefault("max_rounds", 2)
    if args.semantic_correction_max_tokens is not None:
        correction_config["max_tokens"] = args.semantic_correction_max_tokens
    correction_config.setdefault("max_tokens", 1800)
    if args.semantic_correction_protocol_max_attempts is not None:
        correction_config["protocol_max_attempts"] = (
            args.semantic_correction_protocol_max_attempts
        )
    correction_config.setdefault("protocol_max_attempts", 3)
    qa_export_config = dict(config.get("qa_export", {}))
    if args.qa_export_mode is not None:
        qa_export_config["mode"] = args.qa_export_mode
    qa_export_config.setdefault("mode", "ground_truth_template")
    if args.qa_history_window_events is not None:
        qa_export_config["history_window_events"] = args.qa_history_window_events
    qa_export_config.setdefault("history_window_events", 32)
    semantic_judge = None
    semantic_corrector = None
    if judge_config["mode"] == "llm":
        if not judge_config.get("model"):
            parser.error("--judge-model is required when --judge-mode llm")
        from data_generation.llm_client import LLMClient

        client = LLMClient(
            base_url=judge_config.get("base_url"),
            model=str(judge_config["model"]),
        )
        semantic_judge = LLMSemanticJudge(
            client,
            max_tokens=int(judge_config["max_tokens"]),
            protocol_max_attempts=int(
                judge_config["protocol_max_attempts"]
            ),
        )
        if correction_config["mode"] == "llm":
            semantic_corrector = LLMSemanticCorrector(
                client,
                max_tokens=int(correction_config["max_tokens"]),
                protocol_max_attempts=int(
                    correction_config["protocol_max_attempts"]
                ),
            )
    config["semantic_judge"] = judge_config
    config["semantic_correction"] = correction_config
    config["qa_export"] = qa_export_config
    output_dir = args.output_dir or config["output_directory"]
    calibration_bundle = None
    calibration_components = {
        item.strip()
        for item in str(args.calibration_components).split(",")
        if item.strip()
    }
    unknown_calibration_components = (
        calibration_components - VALID_CALIBRATION_COMPONENTS
    )
    if unknown_calibration_components:
        parser.error(
            "Unknown --calibration-components values: "
            + ", ".join(sorted(unknown_calibration_components))
        )
    if not calibration_components:
        parser.error("--calibration-components must select at least one component")
    if args.calibration_bundle:
        with Path(args.calibration_bundle).open("r", encoding="utf-8") as handle:
            calibration_bundle = json.load(handle)
    package_type, build_models = DOMAIN_BUILDERS[domain]
    temporal_models = build_models()
    if calibration_bundle is not None:
        calibration_application = apply_sequence_calibration_bundle(
            calibration_bundle,
            domain=domain,
            domain_config=config["domain_config"],
            temporal_models=temporal_models,
            allow_pending=args.allow_pending_calibration,
            components=calibration_components,
        )
        config["distribution_calibration"] = {
            "mode": "sequence_calibration_bundle",
            "bundle_path": str(Path(args.calibration_bundle).resolve()),
            **calibration_application,
        }
    else:
        config["distribution_calibration"] = {"mode": "generic_prior"}
    if args.scenario_family is not None:
        config["domain_config"]["scenario_weights"] = {
            args.scenario_family: 1.0
        }
    domain_package = package_type(config.get("domain_config", {}))
    mechanism_reweighter = None
    if (
        calibration_bundle is not None
        and "mechanism_reweighting" in calibration_components
    ):
        mechanism_reweighter = LegalMechanismReweighter.from_bundle(
            calibration_bundle,
            mechanism_descriptors=domain_package.catalog().get("mechanisms", []),
            mode=args.mechanism_reweighting_mode,
            strength=args.mechanism_reweighting_strength,
            min_history_count=args.mechanism_reweighting_min_history_count,
            minimum_multiplier=args.mechanism_reweighting_min_multiplier,
            maximum_multiplier=args.mechanism_reweighting_max_multiplier,
        )
    result = GenerationPipeline(
        domain_package,
        temporal_models,
        config,
        output_dir,
        semantic_judge=semantic_judge,
        semantic_corrector=semantic_corrector,
        mechanism_reweighter=mechanism_reweighter,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
