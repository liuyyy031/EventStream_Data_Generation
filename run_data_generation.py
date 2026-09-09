"""Command-line entry point for the neutral multi-domain generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domain_packages.transportation import (
    TransportationPackage,
    build_transportation_temporal_models,
)
from generation_core.pipeline import GenerationPipeline


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
    output_dir = args.output_dir or config["output_directory"]
    result = GenerationPipeline(
        TransportationPackage(config.get("domain_config", {})),
        build_transportation_temporal_models(),
        config,
        output_dir,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
