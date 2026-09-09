"""Fit a transportation calibration profile from reference observations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..domains.transportation.calibration import load_calibration_profile


def fit_profile(
    rows: Sequence[Dict[str, str]], base_profile: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    propagation_rows = [row for row in rows if row.get("record_type") == "propagation"]
    recovery_rows = [row for row in rows if row.get("record_type") == "recovery"]
    if len(propagation_rows) < 3:
        raise ValueError("At least 3 propagation observations are required")
    if len(recovery_rows) < 3:
        raise ValueError("At least 3 recovery observations are required")

    profile = copy.deepcopy(base_profile)
    propagation_train, propagation_test = _split(propagation_rows)
    recovery_train, recovery_test = _split(recovery_rows)
    alpha, beta = _fit_bpr(propagation_train, profile["propagation_lag"])
    severity_weight, scenario_medians = _fit_recovery(recovery_train)
    profile["propagation_lag"]["alpha"] = round(alpha, 8)
    profile["propagation_lag"]["beta"] = round(beta, 8)
    profile["recovery_lag"]["severity_weight"] = round(severity_weight, 8)
    profile["recovery_lag"]["scenario_median_seconds"].update(
        {key: round(value, 6) for key, value in scenario_medians.items()}
    )
    has_holdout = bool(propagation_test and recovery_test)
    profile["status"] = (
        "empirically_fitted_with_holdout"
        if has_holdout
        else "empirically_fitted_without_holdout"
    )
    profile["profile_id"] = f"transportation-fitted-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    profile["fit_requirements"]["status"] = "completed"

    report = {
        "experiment": "transportation_calibration_fit_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "profile_id": profile["profile_id"],
        "status": profile["status"],
        "row_counts": {
            "propagation_train": len(propagation_train),
            "propagation_test": len(propagation_test),
            "recovery_train": len(recovery_train),
            "recovery_test": len(recovery_test),
        },
        "fitted_parameters": {
            "bpr_alpha": profile["propagation_lag"]["alpha"],
            "bpr_beta": profile["propagation_lag"]["beta"],
            "recovery_severity_weight": profile["recovery_lag"]["severity_weight"],
            "recovery_scenario_median_seconds": profile["recovery_lag"][
                "scenario_median_seconds"
            ],
        },
        "train_metrics": {
            "propagation": _lag_metrics(
                propagation_train,
                lambda row: _predict_propagation(row, profile["propagation_lag"]),
                "observed_propagation_lag_seconds",
            ),
            "recovery": _lag_metrics(
                recovery_train,
                lambda row: _predict_recovery(row, profile["recovery_lag"]),
                "observed_return_to_normal_seconds",
            ),
        },
        "holdout_metrics": {
            "propagation": _lag_metrics(
                propagation_test,
                lambda row: _predict_propagation(row, profile["propagation_lag"]),
                "observed_propagation_lag_seconds",
            ),
            "recovery": _lag_metrics(
                recovery_test,
                lambda row: _predict_recovery(row, profile["recovery_lag"]),
                "observed_return_to_normal_seconds",
            ),
        },
        "warning": (
            None
            if has_holdout
            else "No complete holdout split was supplied; fitted coefficients must not be presented as validated."
        ),
    }
    profile["fit_metadata"] = copy.deepcopy(report)
    return profile, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--report-file")
    parser.add_argument("--base-profile")
    args = parser.parse_args(argv)
    base = load_calibration_profile(
        {"profile_file": args.base_profile} if args.base_profile else {}
    )
    with Path(args.input_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    profile, report = fit_profile(rows, base)
    output_profile = Path(args.output_profile)
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    output_profile.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = Path(args.report_file) if args.report_file else output_profile.with_name(
        f"{output_profile.stem}_fit_report.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"profile": str(output_profile), "report": str(report_path), **report}, ensure_ascii=False, indent=2))
    return 0


def _split(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    train = [row for row in rows if row.get("split", "train").lower() != "test"]
    test = [row for row in rows if row.get("split", "train").lower() == "test"]
    return train, test


def _fit_bpr(rows: Sequence[Dict[str, str]], spec: Dict[str, Any]) -> Tuple[float, float]:
    points = []
    for row in rows:
        free_flow = _positive(row, "free_flow_travel_seconds")
        volume = _positive(row, "traffic_volume_vehicles_per_hour")
        capacity = _positive(row, "capacity_vehicles_per_hour")
        observed = _positive(row, "observed_propagation_lag_seconds")
        severity = float(row.get("congestion_severity") or 0.0)
        vc_ratio = volume / capacity * (
            1.0 + float(spec["incident_vc_multiplier_per_severity"]) * severity
        )
        vc_ratio = max(
            float(spec["volume_capacity_ratio_min"]),
            min(float(spec["volume_capacity_ratio_max"]), vc_ratio),
        )
        weather = float(spec["rain_weather_factor"]) if _bool(row.get("rain_active")) else 1.0
        effective = (
            observed - float(spec["response_delay_seconds"])
        ) / (free_flow * weather)
        delay_ratio = effective - 1.0
        if delay_ratio > 0.0:
            points.append((math.log(vc_ratio), math.log(delay_ratio)))
    if len(points) < 2:
        raise ValueError("Propagation observations do not contain enough positive BPR delay ratios")
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0.0:
        raise ValueError("Propagation observations need varying volume/capacity ratios")
    beta = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    alpha = math.exp(mean_y - beta * mean_x)
    if not (alpha > 0.0 and beta > 0.0):
        raise ValueError("Fitted BPR alpha and beta must be positive")
    return alpha, beta


def _fit_recovery(rows: Sequence[Dict[str, str]]) -> Tuple[float, Dict[str, float]]:
    best = None
    for step in range(21):
        weight = step * 0.05
        normalized_by_scenario: Dict[str, List[float]] = {}
        for row in rows:
            scenario = row["scenario_family"]
            severity = float(row.get("congestion_severity") or 0.0)
            observed = _positive(row, "observed_return_to_normal_seconds")
            normalized_by_scenario.setdefault(scenario, []).append(
                observed / (1.0 + weight * severity)
            )
        medians = {
            scenario: statistics.median(values)
            for scenario, values in normalized_by_scenario.items()
        }
        squared_log_errors = []
        for row in rows:
            predicted = medians[row["scenario_family"]] * (
                1.0 + weight * float(row.get("congestion_severity") or 0.0)
            )
            observed = _positive(row, "observed_return_to_normal_seconds")
            squared_log_errors.append((math.log(predicted) - math.log(observed)) ** 2)
        score = statistics.fmean(squared_log_errors)
        if best is None or score < best[0]:
            best = (score, weight, medians)
    assert best is not None
    return best[1], best[2]


def _predict_propagation(row: Dict[str, str], spec: Dict[str, Any]) -> float:
    free_flow = _positive(row, "free_flow_travel_seconds")
    vc_ratio = _positive(row, "traffic_volume_vehicles_per_hour") / _positive(
        row, "capacity_vehicles_per_hour"
    )
    vc_ratio *= 1.0 + float(spec["incident_vc_multiplier_per_severity"]) * float(
        row.get("congestion_severity") or 0.0
    )
    vc_ratio = max(
        float(spec["volume_capacity_ratio_min"]),
        min(float(spec["volume_capacity_ratio_max"]), vc_ratio),
    )
    weather = float(spec["rain_weather_factor"]) if _bool(row.get("rain_active")) else 1.0
    return float(spec["response_delay_seconds"]) + free_flow * (
        1.0 + float(spec["alpha"]) * vc_ratio ** float(spec["beta"])
    ) * weather


def _predict_recovery(row: Dict[str, str], spec: Dict[str, Any]) -> float:
    median = float(spec["scenario_median_seconds"][row["scenario_family"]])
    return median * (
        1.0
        + float(spec["severity_weight"])
        * float(row.get("congestion_severity") or 0.0)
    )


def _lag_metrics(
    rows: Sequence[Dict[str, str]], predictor: Any, observed_key: str
) -> Dict[str, Any] | None:
    if not rows:
        return None
    errors = [predictor(row) - _positive(row, observed_key) for row in rows]
    return {
        "count": len(errors),
        "mae_seconds": round(statistics.fmean(abs(error) for error in errors), 6),
        "rmse_seconds": round(math.sqrt(statistics.fmean(error**2 for error in errors)), 6),
        "mean_error_seconds": round(statistics.fmean(errors), 6),
    }


def _positive(row: Dict[str, str], key: str) -> float:
    value = float(row[key])
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())
