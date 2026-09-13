"""Command-line entry point for the neutral multi-domain generator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from domain_packages.transportation import (
    TransportationPackage,
    build_transportation_temporal_models,
)
from generation_core.pipeline import GenerationPipeline
from generation_core.semantic_judge import LLMSemanticJudge


def main() -> None:
    parser = argparse.ArgumentParser()
    default_config = (
        Path(__file__).resolve().parent
        / "domain_packages"
        / "transportation"
        / "config"
        / "default.json"
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir")
    parser.add_argument("--episode-count", type=int)
    parser.add_argument(
        "--nodes-per-context",
        "--nodes-per-network",
        dest="nodes_per_context",
        type=int,
    )
    parser.add_argument("--max-events-per-episode", type=int)
    parser.add_argument(
        "--scenario-family",
        choices=(
            "accident_propagation",
            "weather_disruption",
            "planned_closure",
            "compound_weather_accident",
        ),
    )
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
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("domain") != "transportation":
        raise ValueError(
            "The first implemented domain package is transportation; other domain packages are structural stubs."
        )
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
    semantic_judge = None
    if judge_config["mode"] == "llm":
        if not judge_config.get("model"):
            parser.error("--judge-model is required when --judge-mode llm")
        from llm_client import LLMClient

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
    config["semantic_judge"] = judge_config
    output_dir = args.output_dir or config["output_directory"]
    result = GenerationPipeline(
        TransportationPackage(config.get("domain_config", {})),
        build_transportation_temporal_models(),
        config,
        output_dir,
        semantic_judge=semantic_judge,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
