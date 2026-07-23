#!/usr/bin/env python3
"""Re-audit an existing STPP JSON against the v12 temporal contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import numpy as np

from stpp_adapter_v11 import _conditional_simple_path_envelope, _semantic_peak_audit
from stpp_adapter_v12 import (
    STPPV12Config,
    canonicalise_temporal_contract_v12,
)


def audit_existing_json(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    generation = data["generation_info"]
    raw_config = generation.get("point_process_parameters", {})
    allowed = {field.name for field in fields(STPPV12Config)}
    config = STPPV12Config(
        **{key: value for key, value in raw_config.items() if key in allowed}
    )
    config.validate()
    raw_structured = data["agent2_structured_scenario"]
    matrix = np.asarray(data["agent5_simulation_data"], dtype=float)
    seq_len = int(data.get("seq_len", matrix.shape[1]))
    structured, temporal = canonicalise_temporal_contract_v12(
        raw_structured, seq_len
    )
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
    judgment = (generation.get("judge1_validation", {}) or {}).get(
        "last_judgment"
    ) or {}
    approved = judgment.get("approved") is True
    judge_consistent = (
        judgment.get("error_source") is None and not judgment.get("issues")
        if approved
        else judgment.get("error_source") in {"agent1", "agent2"}
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
        "temporal_contract_valid": temporal["passed"],
        "temporal_contract_repair_free": temporal["repair_free"],
        "no_causal_window_violations": not temporal[
            "causal_window_violations"
        ],
        "judge_response_contract_consistent": judge_consistent,
    }
    return {
        "input_json": str(input_path),
        "audit_version": "stpp_v12",
        "conditional_simple_path_max_matrix_envelope_ratio": conditional_ratio,
        "max_ratio_cell": {
            "node_id": node_id,
            "time_index": time_index,
            "matrix_value": float(matrix[node_id, time_index]),
            "conditional_envelope": float(envelope[node_id, time_index]),
        },
        "checks": checks,
        "passed_v12_reaudit": all(checks.values()),
        "temporal_contract_audit": temporal,
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
        else input_path.with_name(input_path.stem + "_v12_audit.json")
    )
    audit = audit_existing_json(input_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(
        "Conditional simple-path max ratio: "
        f"{audit['conditional_simple_path_max_matrix_envelope_ratio']:.6f}"
    )
    print(f"v12 re-audit passed: {audit['passed_v12_reaudit']}")
    print(f"Audit JSON: {output_path}")
    return 0 if audit["passed_v12_reaudit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
