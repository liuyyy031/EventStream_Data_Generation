"""Command-line interface for the Transportation EventFlow v1 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .core.config import load_config, validate_config
from .core.pipeline import EventFlowPipeline


def main(argv: Sequence[str] | None = None) -> int:
    default_config = Path(__file__).resolve().parent / "config" / "transportation_default.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir")
    parser.add_argument("--episode-count", type=int)
    node_group = parser.add_mutually_exclusive_group()
    node_group.add_argument(
        "--nodes-per-network",
        type=int,
        help="Road-segment count for each generated network.",
    )
    node_group.add_argument(
        "--node-count",
        type=int,
        help="Deprecated alias for --nodes-per-network.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--judge-mode", choices=["heuristic", "llm"])
    parser.add_argument("--network-file")
    parser.add_argument(
        "--calibration-profile",
        help="Fitted or reference calibration profile JSON.",
    )
    parser.add_argument("--paraphrase-rate", type=float)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.episode_count is not None:
        config["episode_count"] = args.episode_count
    requested_nodes = args.nodes_per_network
    if args.node_count is not None:
        print(
            "WARNING: --node-count is deprecated; use --nodes-per-network. "
            "The value applies to each generated network.",
            file=sys.stderr,
        )
        requested_nodes = args.node_count
    if requested_nodes is not None:
        config["network"]["node_count"] = requested_nodes
    if args.seed is not None:
        config["seed"] = args.seed
    if args.judge_mode is not None:
        config["validation"]["judge_mode"] = args.judge_mode
    if args.network_file is not None:
        config["network"]["external_network_file"] = args.network_file
    if args.calibration_profile is not None:
        config.setdefault("calibration", {})["profile_file"] = args.calibration_profile
    if args.paraphrase_rate is not None:
        config["text"]["paraphrase_rate"] = args.paraphrase_rate
    validate_config(config)

    result = EventFlowPipeline(config, output_dir=args.output_dir).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["quality_report"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
