#!/usr/bin/env python3
"""
Stage 1 — Batch driver for the 6-Agent + 2-Judge STS pipeline.

Repeatedly invokes ``demo_sts_sde.demo_network_sde_generation`` across
(domain, num_nodes) combinations and collects the resulting ``.pkl`` files
under ``data_generation/batch_output/`` so Stage 2 can pick them up.

Concurrency is achieved with ``ProcessPoolExecutor`` (each task is an
independent process) which side-steps matplotlib thread-safety issues and
gives true parallelism for the SDE simulation.

Configure the LLM through the environment, then run e.g.::

    export LLM_API_KEY=<your_api_key>
    python data_generation/run_pipeline.py --num_tasks 100 --node_counts 3,5,10
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _generate_one(idx: int, domain: str, num_nodes: int, out_dir: str,
                  enabled_judges: List[int]) -> Tuple[int, str, str]:
    """Worker that runs the pipeline once and moves the .pkl into ``out_dir``."""
    # Import inside the worker to avoid heavy imports in the parent process.
    from demo_sts_sde import demo_network_sde_generation

    result = demo_network_sde_generation(
        enabled_judges=enabled_judges,
        enable_logging=False,
        num_nodes=num_nodes,
        domain=domain,
        generate_viz=False,
    )
    src_pkl = result["data_files"]["pickle"]
    dst_pkl = os.path.join(out_dir, f"task_{idx:04d}_{_slug(domain)}_n{num_nodes}.pkl")
    os.makedirs(out_dir, exist_ok=True)
    shutil.move(src_pkl, dst_pkl)
    return idx, domain, dst_pkl


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _safe_call(args):
    idx, domain, num_nodes, out_dir, enabled_judges = args
    try:
        return _generate_one(idx, domain, num_nodes, out_dir, enabled_judges)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return idx, domain, f"ERROR: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_tasks", type=int, default=100,
                        help="Total number of independent scenarios to generate (default: 100).")
    parser.add_argument("--node_counts", type=str, default="3,5,10",
                        help="Comma-separated list of node counts to cycle through (default: 3,5,10).")
    parser.add_argument("--domains", type=str, default=",".join(DEFAULT_DOMAINS),
                        help="Comma-separated list of domain names to cycle through.")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Number of concurrent worker processes (default: 8).")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join("data_generation", "batch_output"),
                        help="Directory to write task_*.pkl files into.")
    parser.add_argument("--judges", type=str, default="1,2",
                        help='Judge agents to enable (e.g., "1,2"). Use "none" to disable.')
    args = parser.parse_args()

    node_counts = [int(x.strip()) for x in args.node_counts.split(",") if x.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not node_counts or not domains:
        print("ERROR: --node_counts and --domains must each contain at least one value.",
              file=sys.stderr)
        return 2

    if args.judges.lower() == "none":
        enabled_judges: List[int] = []
    else:
        enabled_judges = [int(j.strip()) for j in args.judges.split(",") if j.strip()]

    combos = [(d, n) for n in node_counts for d in domains]
    out_dir = os.path.abspath(args.out_dir)

    print(f"Generating {args.num_tasks} tasks across {len(combos)} (domain, num_nodes) "
          f"combinations using {args.max_workers} workers.")
    print(f"Output directory: {out_dir}")

    jobs = []
    for i in range(args.num_tasks):
        domain, n = combos[i % len(combos)]
        jobs.append((i, domain, n, out_dir, enabled_judges))

    successes = 0
    failures = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        for idx, domain, result in pool.map(_safe_call, jobs):
            if isinstance(result, str) and result.startswith("ERROR:"):
                failures += 1
                print(f"  [task {idx:04d} | {domain}] {result}", file=sys.stderr)
            else:
                successes += 1
                print(f"  [task {idx:04d} | {domain}] -> {result}")

    print(f"\nDone. {successes} succeeded, {failures} failed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
