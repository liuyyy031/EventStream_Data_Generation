#!/usr/bin/env python3
"""Re-audit an existing STPP JSON with v11 envelope and semantic gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import numpy as np

from stpp_adapter_v11 import (
    STPPV11Config,
    _conditional_simple_path_envelope,
    _semantic_peak_audit,
)


def audit_existing_json(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    generation = data["generation_info"]
    raw_config = generation.get("point_process_parameters", {})
    allowed = {field.name for field in fields(STPPV11Config)}
    config = STPPV11Config(
        **{key: value for key, value in raw_config.items() if key in allowed}
    )
    config.validate()
    structured = data["agent2_structured_scenario"]
    matrix = np.asarray(data["agent5_simulation_data"], dtype=float)
    target_profile = np.asarray(generation["target_magnitude_profile"], dtype=float)
    envelope, envelope_audit = _conditional_simple_path_envelope(
        structured, target_profile, config
    )
    ratios = np.divide(
        matrix, envelope, out=np.zeros_like(matrix), where=envelope > 0
    )
    flat_index = int(np.argmax(ratios))
    node_id, time_index = [
        int(value) for value in np.unravel_index(flat_index, ratios.shape)
    ]
    conditional_ratio = float(ratios[node_id, time_index])
    semantic = _semantic_peak_audit(structured)
    old_ratio = float(
        generation.get(
            "expected_one_hop_max_matrix_envelope_ratio",
            generation.get("max_matrix_envelope_ratio", 0.0),
        )
    )
    checks = {
        "conditional_simple_path_envelope_within_limit": (
            conditional_ratio <= config.max_matrix_target_ratio
        ),
        "exactly_one_self_pattern_per_demand_source": not semantic[
            "invalid_self_pattern_counts"
        ],
        "no_self_generated_propagated_time_conflicts": not semantic[
            "self_generated_propagated_time_conflicts"
        ],
        "no_positive_patterns_on_propagation_nodes": not semantic[
            "positive_patterns_on_propagation_nodes"
        ],
    }
    return {
        "input_json": str(input_path),
        "audit_version": "stpp_v11",
        "expected_one_hop_max_matrix_envelope_ratio": old_ratio,
        "conditional_simple_path_max_matrix_envelope_ratio": conditional_ratio,
        "max_ratio_cell": {
            "node_id": node_id,
            "time_index": time_index,
            "matrix_value": float(matrix[node_id, time_index]),
            "conditional_envelope": float(envelope[node_id, time_index]),
        },
        "checks": checks,
        "passed_v11_reaudit": all(checks.values()),
        "semantic_peak_audit": semantic,
        "conditional_envelope_audit": envelope_audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    input_path = Path(args.input_json)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_v11_audit.json")
    )
    audit = audit_existing_json(input_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(
        "Expected one-hop max ratio: "
        f"{audit['expected_one_hop_max_matrix_envelope_ratio']:.6f}"
    )
    print(
        "Conditional simple-path max ratio: "
        f"{audit['conditional_simple_path_max_matrix_envelope_ratio']:.6f}"
    )
    print(f"v11 re-audit passed: {audit['passed_v11_reaudit']}")
    print(f"Audit JSON: {output_path}")
    return 0 if audit["passed_v11_reaudit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
