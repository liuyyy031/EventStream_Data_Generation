#!/usr/bin/env python3
"""Readable standalone launcher for the validated, unmodified v27 source."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BUNDLE_DIR / "runtime"
DATA_GENERATION_DIR = BUNDLE_DIR.parent
ENV_PATH = DATA_GENERATION_DIR / ".env"
MANIFEST_PATH = BUNDLE_DIR / "manifest.json"


def _check_source_bundle() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures = []
    for module, metadata in manifest["modules"].items():
        path = RUNTIME_DIR / metadata["relative_path"]
        if not path.is_file():
            failures.append(f"{module}: missing {path}")
            continue
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != metadata["sha256"]:
            failures.append(f"{module}: SHA-256 mismatch")
            continue
        compile(content, str(path), "exec", dont_inherit=True)
    if failures:
        raise RuntimeError("Source-bundle check failed:\n- " + "\n- ".join(failures))
    print(
        "STPP v27 release source check passed: "
        f"{len(manifest['modules'])} modules, "
        f"manifest={manifest['bundle_sha256']}"
    )
    return 0


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required") from exc
    if not ENV_PATH.is_file():
        raise RuntimeError(f"Expected .env file was not found: {ENV_PATH}")
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError(f"LLM_API_KEY is missing after loading {ENV_PATH}")


def _load_entrypoint():
    _load_env()
    sys.path.insert(0, str(RUNTIME_DIR))
    from demo_sts_stpp_v27 import main as v27_main
    return v27_main


def _check_route_guard() -> int:
    sys.path.insert(0, str(RUNTIME_DIR))
    from stpp_adapter_v19 import audit_structured_contract_v19

    malformed = {
        "time_axis": {
            "mode": "cyclic_local",
            "repeat": True,
            "repeat_period": 24,
            "interval_semantics": "inclusive",
            "week_start": "Sunday",
        },
        "nodes": [
            {"id": 0, "type": "demand_source"},
            {"id": 1, "type": "propagation"},
        ],
        "edges": [
            {"source": 0, "target": 1, "time_lag": 1},
            {"source": 1, "target": 0, "time_lag": 1},
        ],
        "drift_patterns": {
            "repeat": True,
            "repeat_period": 24,
            "nodes": [
                {
                    "id": 0,
                    "patterns": [],
                    "propagated_variations": [
                        {
                            "event_id": "EVT-001",
                            "schedule_id": "EVT-001-weekday",
                            "days": [1, 2, 3, 4, 5],
                            "time": 3,
                        }
                    ],
                }
            ],
        },
        "adjacency_modulation": {
            "patterns": [
                {
                    "event_id": "EVT-001",
                    "schedule_id": "EVT-001-weekday",
                    "path": [0, 1, 0],
                    "path_stage": 0,
                    "time_period": "1-2",
                    "destination_arrival_period": "2-3",
                    "days": [1, 2, 3, 4, 5],
                    "effect": "strong",
                    "applies_to": "0->1",
                },
                {
                    "event_id": "EVT-001",
                    "schedule_id": "EVT-001-weekday",
                    "path": [0, 1, 0],
                    "path_stage": 1,
                    "time_period": "2-3",
                    "destination_arrival_period": "3-4",
                    "days": [1, 2, 3, 4, 5],
                    "effect": "strong",
                    "applies_to": "1->0",
                },
            ]
        },
        "spatial_layout": {
            "0": {"x": 0.1, "y": 0.1},
            "1": {"x": 0.9, "y": 0.9},
        },
    }
    audit = audit_structured_contract_v19(malformed, 168)
    route_issues = audit.get("route_contract_issues", [])
    guarded = any(
        "simple path without repeated nodes" in str(issue.get("problem", ""))
        for issue in route_issues
    )
    if audit.get("passed") or not guarded:
        raise RuntimeError(
            "Repeated-node route guard check failed: "
            + json.dumps(audit, ensure_ascii=False, default=str)
        )
    print(
        "STPP v27 release route guard check passed: repeated-node path was "
        "rejected without an exception"
    )
    return 0


def _check_lag_scope() -> int:
    sys.path.insert(0, str(RUNTIME_DIR))
    from stpp_adapter_v4 import _lag_report
    from stpp_adapter_v16 import STPPV16Config

    structured = {
        "edges": [
            {"source": 0, "target": 1, "time_lag": 2},
            {"source": 1, "target": 0, "time_lag": 3},
        ],
        "adjacency_modulation": {
            "patterns": [
                {
                    "event_id": "EVT-001",
                    "schedule_id": "EVT-001-weekday",
                    "path": [0, 1],
                    "path_stage": 0,
                }
            ]
        },
    }
    events = [
        {"edge": "0->1", "realized_delay": 2.0},
        {"edge": "0->1", "realized_delay": 2.1},
    ]
    config = STPPV16Config()
    report = _lag_report(events, structured, config)
    expected_pass = (
        report.get("passed") is True
        and report.get("evaluation_scope") == "declared_event_route_edges"
        and report.get("ignored_configured_edges") == ["1->0"]
    )
    if not expected_pass:
        raise RuntimeError(
            "Declared-route lag scope check failed: "
            + json.dumps(report, ensure_ascii=False, default=str)
        )

    structured["adjacency_modulation"]["patterns"].append(
        {
            "event_id": "EVT-002",
            "schedule_id": "EVT-002-weekday",
            "path": [1, 0],
            "path_stage": 0,
        }
    )
    missing_declared_edge = _lag_report(events, structured, config)
    if missing_declared_edge.get("passed") is not False:
        raise RuntimeError(
            "Missing declared-route edge was not rejected: "
            + json.dumps(
                missing_declared_edge, ensure_ascii=False, default=str
            )
        )
    print(
        "STPP v27 release lag scope check passed: unused graph edges were "
        "ignored, while missing declared-route edges were rejected"
    )
    return 0


def _check_raw_section_parser() -> int:
    import ast
    import re
    from typing import List, Set

    module_path = RUNTIME_DIR / "demo_sts_stpp_v22.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), str(module_path))
    required_assignments = {"_SECTION_HEADING", "_EVENT_ID"}
    required_functions = {"_normalised_heading", "_section", "_event_ids"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            selected.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names.intersection(required_assignments):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in required_functions:
            selected.append(node)
    namespace = {"re": re, "List": List, "Set": Set}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(module_path), "exec"),
        namespace,
    )
    _event_ids = namespace["_event_ids"]
    _section = namespace["_section"]

    scenario = """
- PROPAGATED ARRIVALS:
  - (EVENT ID: EVT-001, RECEIVING NODE: 1)
- PROPAGATED ARRIVALS:
  - EVENT ID: EVT-001, RECEIVING NODE: 2
### Propagated Arrivals:
  - EVENT ID: EVT-002, RECEIVING NODE: 1

## EDGE MODULATION:
- EVENT ID: EVT-001, PATH: 0->1
- EVENT ID: EVT-002, PATH: 2->1

## SPATIAL LAYOUT:
- NODE 0: (0.1, 0.1)
"""
    arrivals = _event_ids(_section(scenario, "PROPAGATED ARRIVALS"))
    scheduled = _event_ids(_section(scenario, "EDGE MODULATION"))
    expected = {"EVT-001", "EVT-002"}
    if arrivals != expected or scheduled != expected:
        raise RuntimeError(
            "Raw repeated-section parser check failed: "
            + json.dumps(
                {
                    "arrival_event_ids": sorted(arrivals),
                    "scheduled_event_ids": sorted(scheduled),
                },
                ensure_ascii=False,
            )
        )
    print(
        "STPP v27 release raw-section parser check passed: bullet, mixed-case "
        "and repeated propagated-arrival headings were combined"
    )
    return 0


def main() -> int:
    if "--check-source-bundle" in sys.argv:
        return _check_source_bundle()
    if "--check-route-guard" in sys.argv:
        return _check_route_guard()
    if "--check-lag-scope" in sys.argv:
        return _check_lag_scope()
    if "--check-raw-section-parser" in sys.argv:
        return _check_raw_section_parser()
    if "--check-env" in sys.argv:
        _load_env()
        print(f"Loaded environment file: {ENV_PATH}")
        print("LLM_API_KEY: set")
        print(
            "LLM_BASE_URL: "
            + (os.environ.get("LLM_BASE_URL") or "using llm_client default")
        )
        return 0
    if "--check-imports" in sys.argv:
        entrypoint = _load_entrypoint()
        print(
            f"STPP v27 release import check passed: "
            f"{entrypoint.__module__}.main"
        )
        return 0
    return int(_load_entrypoint()())


if __name__ == "__main__":
    raise SystemExit(main())
