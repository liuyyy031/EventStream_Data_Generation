"""Command-line entry point for the neutral multi-domain generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_generation.domain_packages.healthcare import (
    HealthcarePackage,
    build_healthcare_temporal_models,
)
from data_generation.domain_packages.transportation import (
    TransportationPackage,
    build_transportation_temporal_models,
)
from data_generation.generation_core.pipeline import GenerationPipeline
from data_generation.generation_core.semantic_correction import LLMSemanticCorrector
from data_generation.generation_core.semantic_judge import LLMSemanticJudge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=("transportation", "healthcare"),
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
    if domain not in {"transportation", "healthcare"}:
        raise ValueError(f"No executable domain package is registered for {domain!r}")
    if args.episode_count is not None:
        config["episode_count"] = args.episode_count
    if args.nodes_per_context is not None:
        config["domain_config"]["nodes_per_context"] = args.nodes_per_context
    if args.seed is not None:
        config["seed"] = args.seed
        config["domain_config"]["base_seed"] = args.seed
    if args.max_events_per_episode is not None:
        config["max_events_per_episode"] = args.max_events_per_episode
    if args.scenario_family is not None:
        config["domain_config"]["scenario_weights"] = {
            args.scenario_family: 1.0
        }
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
    if domain == "transportation":
        domain_package = TransportationPackage(config.get("domain_config", {}))
        temporal_models = build_transportation_temporal_models()
    else:
        domain_package = HealthcarePackage(config.get("domain_config", {}))
        temporal_models = build_healthcare_temporal_models()
    result = GenerationPipeline(
        domain_package,
        temporal_models,
        config,
        output_dir,
        semantic_judge=semantic_judge,
        semantic_corrector=semantic_corrector,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
