#!/usr/bin/env python3
"""Parallel batch driver for root/propagation-separated STPP v6."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple


DEFAULT_DOMAINS = [
    "Transportation",
    "Energy",
    "Environment&Pollution",
    "Ecology",
    "Public Health",
    "Hydrology",
    "Oceanography",
    "Agriculture",
    "Mobility",
    "Climate",
]


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _generate_one(
    idx: int,
    domain: str,
    num_nodes: int,
    out_dir: str,
    enabled_judges: List[int],
    aggregation: str,
    base_seed: int | None,
    edge_branching_ratio: float,
    target_immigrant_rate: float,
) -> Tuple[int, str, str]:
    from demo_sts_stpp_v6 import demo_stpp_generation_v6
    from stpp_adapter_v6 import STPPV6Config

    seed = None if base_seed is None else base_seed + idx
    result = demo_stpp_generation_v6(
        enabled_judges=enabled_judges,
        enable_logging=False,
        num_nodes=num_nodes,
        domain=domain,
        generate_viz=False,
        stpp_config=STPPV6Config(
            aggregation=aggregation,
            seed=seed,
            edge_branching_ratio=edge_branching_ratio,
            target_immigrant_rate=target_immigrant_rate,
        ),
    )
    if not result["generation_info"]["quality_report"]["passed"]:
        raise RuntimeError("v6 quality gates did not pass")
    src_pkl = result["data_files"]["pickle"]
    dst_pkl = os.path.join(
        out_dir, f"task_{idx:04d}_{_slug(domain)}_n{num_nodes}_stpp_v6.pkl"
    )
    os.makedirs(out_dir, exist_ok=True)
    shutil.move(src_pkl, dst_pkl)
    return idx, domain, dst_pkl


def _safe_call(args):
    try:
        return _generate_one(*args)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return args[0], args[1], f"ERROR: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_tasks", type=int, default=100)
    parser.add_argument("--node_counts", default="3,5,10")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument(
        "--out_dir",
        default=os.path.join("data_generation", "batch_output_stpp_v6"),
    )
    parser.add_argument("--judges", default="1")
    parser.add_argument(
        "--aggregation",
        choices=["count", "magnitude", "rolling_count", "rolling_magnitude"],
        default="magnitude",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--edge_branching_ratio", type=float, default=0.35)
    parser.add_argument("--target_immigrant_rate", type=float, default=0.30)
    args = parser.parse_args()

    node_counts = [
        int(item.strip()) for item in args.node_counts.split(",") if item.strip()
    ]
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    if not node_counts or not domains:
        print("ERROR: node_counts and domains cannot be empty", file=sys.stderr)
        return 2
    judges = (
        []
        if args.judges.lower() == "none"
        else [int(item.strip()) for item in args.judges.split(",") if item.strip()]
    )
    combos = [(domain, count) for count in node_counts for domain in domains]
    out_dir = os.path.abspath(args.out_dir)
    jobs = []
    for idx in range(args.num_tasks):
        domain, num_nodes = combos[idx % len(combos)]
        jobs.append(
            (
                idx,
                domain,
                num_nodes,
                out_dir,
                judges,
                args.aggregation,
                args.seed,
                args.edge_branching_ratio,
                args.target_immigrant_rate,
            )
        )
    successes = failures = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        for idx, domain, result in pool.map(_safe_call, jobs):
            if result.startswith("ERROR:"):
                failures += 1
                print(f"[task {idx:04d} | {domain}] {result}", file=sys.stderr)
            else:
                successes += 1
                print(f"[task {idx:04d} | {domain}] -> {result}")
    print(f"Done. {successes} succeeded, {failures} failed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
