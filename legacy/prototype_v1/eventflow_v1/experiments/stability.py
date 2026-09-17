"""Multi-seed stability experiment for EventFlow generation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from ..core.config import load_config, validate_config
from ..core.pipeline import EventFlowPipeline


def run_stability_experiment(
    config: Dict[str, Any], seeds: Sequence[int], output_dir: str | Path
) -> Dict[str, Any]:
    root = _unique_directory(Path(output_dir))
    root.mkdir(parents=True, exist_ok=False)
    runs = []
    scenario_counts: Counter[str] = Counter()
    for seed in seeds:
        run_config = copy.deepcopy(config)
        run_config["seed"] = int(seed)
        run_dir = root / f"seed_{seed}"
        started = time.perf_counter()
        result = EventFlowPipeline(run_config, run_dir).run()
        elapsed = time.perf_counter() - started
        quality = result["quality_report"]
        distribution = quality["distribution"]
        scenario_counts.update(distribution["scenario_counts"])
        runs.append(
            {
                "seed": int(seed),
                "output_dir": str(run_dir),
                "elapsed_seconds": round(elapsed, 6),
                "passed": bool(quality["passed"]),
                "accepted_episode_count": int(quality["accepted_episode_count"]),
                "rejected_attempt_count": int(quality["rejected_attempt_count"]),
                "generation_retry_count": int(quality["generation_retry_count"]),
                "semantic_rejection_count": int(quality["semantic_rejection_count"]),
                "judge_protocol_retry_count": int(quality["judge_protocol_retry_count"]),
                "judge_protocol_failure_count": int(quality["judge_protocol_failure_count"]),
                "truncated_episode_count": int(
                    distribution["truncated_episode_count"]
                ),
                "events_per_episode_mean": distribution["events_per_episode"]["mean"],
                "relations_per_episode_mean": distribution["relations_per_episode"]["mean"],
                "recovery_to_congestion_ratio": distribution[
                    "recovery_to_congestion_ratio"
                ],
                "multi_parent_target_fraction": distribution[
                    "multi_parent_target_fraction"
                ],
                "network_entity_utilization_fraction": distribution[
                    "network_entity_utilization_fraction"
                ],
                "max_propagation_children_observed": int(
                    distribution["max_propagation_children_observed"]
                ),
                "max_topology_out_degree": max(
                    (int(item["max_out_degree"]) for item in quality["topology"]),
                    default=0,
                ),
                "scenario_counts": distribution["scenario_counts"],
                "calibration_status": distribution["calibration_status"],
            }
        )

    max_children = int(config["simulation"]["max_children_per_event"])
    max_degree = int(config["network"]["max_topology_out_degree"])
    expected_scenarios = set(config["simulation"]["scenario_families"])
    checks = {
        "all_runs_passed": all(run["passed"] for run in runs),
        "all_requested_episodes_written": all(
            run["accepted_episode_count"] == int(config["episode_count"])
            for run in runs
        ),
        "no_written_episode_truncated": all(
            run["truncated_episode_count"] == 0 for run in runs
        ),
        "event_branching_within_limit": all(
            run["max_propagation_children_observed"] <= max_children for run in runs
        ),
        "topology_degree_within_limit": all(
            run["max_topology_out_degree"] <= max_degree for run in runs
        ),
        "no_judge_protocol_failure": all(
            run["judge_protocol_failure_count"] == 0 for run in runs
        ),
        "combined_scenario_coverage": expected_scenarios.issubset(scenario_counts),
    }
    total_generation_attempts = sum(
        run["accepted_episode_count"] + run["generation_retry_count"] for run in runs
    )
    generation_retry_fraction = (
        sum(run["generation_retry_count"] for run in runs) / total_generation_attempts
        if total_generation_attempts
        else 0.0
    )
    checks["generation_retry_fraction_below_5_percent"] = (
        generation_retry_fraction <= 0.05
    )
    metric_names = [
        "elapsed_seconds",
        "events_per_episode_mean",
        "relations_per_episode_mean",
        "recovery_to_congestion_ratio",
        "multi_parent_target_fraction",
        "network_entity_utilization_fraction",
    ]
    report = {
        "experiment": "eventflow_multi_seed_stability_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(root),
        "configuration": config,
        "seeds": list(map(int, seeds)),
        "run_count": len(runs),
        "total_episode_count": sum(run["accepted_episode_count"] for run in runs),
        "generation_retry_fraction": round(generation_retry_fraction, 6),
        "passed": all(checks.values()),
        "checks": checks,
        "combined_scenario_counts": dict(scenario_counts),
        "across_seed_metrics": {
            name: _summary(float(run[name]) for run in runs) for name in metric_names
        },
        "runs": runs,
        "interpretation": {
            "engineering_stability": "supported only when all hard checks pass",
            "statistical_realism": "not established by multi-seed stability",
            "recommended_use": "use heuristic mode for large runs and a stratified LLM sample for semantic review",
        },
    }
    (root / "stability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(root / "stability_runs.csv", runs)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    default_config = Path(__file__).resolve().parents[1] / "config" / "transportation_default.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-seed", type=int, default=20260716)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--seeds", help="Comma-separated explicit seeds; overrides --base-seed/--seed-count")
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--nodes-per-network", type=int, default=1000)
    parser.add_argument("--judge-mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--calibration-profile")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config["episode_count"] = args.episodes_per_seed
    config["network"]["node_count"] = args.nodes_per_network
    config["validation"]["judge_mode"] = args.judge_mode
    if args.calibration_profile:
        config.setdefault("calibration", {})["profile_file"] = args.calibration_profile
    validate_config(config)
    seeds = (
        [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [args.base_seed + index * 1000003 for index in range(args.seed_count)]
    )
    if not seeds:
        parser.error("At least one seed is required")
    report = run_stability_experiment(config, seeds, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def _summary(values: Iterable[float]) -> Dict[str, float]:
    items = list(values)
    mean = statistics.fmean(items) if items else 0.0
    standard_deviation = statistics.pstdev(items) if len(items) > 1 else 0.0
    return {
        "count": len(items),
        "min": round(min(items), 6) if items else 0.0,
        "max": round(max(items), 6) if items else 0.0,
        "mean": round(mean, 6),
        "population_standard_deviation": round(standard_deviation, 6),
        "coefficient_of_variation": (
            round(standard_deviation / abs(mean), 6) if mean else 0.0
        ),
    }


def _write_csv(path: Path, runs: Sequence[Dict[str, Any]]) -> None:
    scalar_keys = [key for key, value in runs[0].items() if not isinstance(value, dict)]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run[key] for key in scalar_keys})


def _unique_directory(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return path.with_name(f"{path.name}_{timestamp}")


if __name__ == "__main__":
    raise SystemExit(main())
