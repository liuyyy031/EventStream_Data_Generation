"""Transparent transportation parameter formulas and provenance.

This module deliberately distinguishes a reference formula from empirical
calibration.  The bundled profile uses an FHWA-documented BPR travel-time
function, while synthetic demand and recovery coefficients remain visibly
marked as pending empirical fitting.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_PROFILE_PATH = Path(__file__).with_name("calibration_profile_v1.json")


def load_calibration_profile(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    calibration_config = config or {}
    configured_path = calibration_config.get("profile_file")
    path = Path(str(configured_path)) if configured_path else DEFAULT_PROFILE_PATH
    with path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    required = {"profile_id", "status", "propagation_lag", "recovery_lag"}
    missing = sorted(required.difference(profile))
    if missing:
        raise ValueError(f"Calibration profile is missing: {', '.join(missing)}")
    return copy.deepcopy(profile)


def propagation_lag(
    *,
    free_flow_seconds: float,
    baseline_volume_capacity_ratio: float,
    congestion_severity: float,
    rain_active: bool,
    rng: random.Random,
    profile: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    spec = profile["propagation_lag"]
    vc_ratio = baseline_volume_capacity_ratio * (
        1.0 + float(spec["incident_vc_multiplier_per_severity"]) * congestion_severity
    )
    vc_ratio = _clamp(
        vc_ratio,
        float(spec["volume_capacity_ratio_min"]),
        float(spec["volume_capacity_ratio_max"]),
    )
    alpha = float(spec["alpha"])
    beta = float(spec["beta"])
    bpr_seconds = free_flow_seconds * (1.0 + alpha * vc_ratio**beta)
    weather_factor = float(spec["rain_weather_factor"]) if rain_active else 1.0
    stochastic_factor = rng.uniform(
        float(spec["stochastic_factor_min"]),
        float(spec["stochastic_factor_max"]),
    )
    response_delay = float(spec["response_delay_seconds"])
    lag = response_delay + bpr_seconds * weather_factor * stochastic_factor
    inputs = {
        "free_flow_travel_seconds": round(free_flow_seconds, 6),
        "baseline_volume_capacity_ratio": round(baseline_volume_capacity_ratio, 6),
        "incident_adjusted_volume_capacity_ratio": round(vc_ratio, 6),
        "congestion_severity": round(congestion_severity, 6),
        "rain_active": rain_active,
        "alpha": alpha,
        "beta": beta,
        "weather_factor": weather_factor,
        "stochastic_factor": round(stochastic_factor, 9),
        "response_delay_seconds": response_delay,
        "bpr_effective_travel_seconds": round(bpr_seconds, 6),
    }
    return round(lag, 3), _provenance(profile, spec, inputs)


def recovery_lag(
    *,
    scenario_family: str,
    congestion_severity: float,
    rng: random.Random,
    profile: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    spec = profile["recovery_lag"]
    medians = spec["scenario_median_seconds"]
    median = float(medians.get(scenario_family, medians["accident_propagation"]))
    severity_factor = 1.0 + float(spec["severity_weight"]) * congestion_severity
    stochastic_factor = rng.lognormvariate(0.0, float(spec["lognormal_sigma"]))
    stochastic_factor = _clamp(
        stochastic_factor,
        float(spec["stochastic_factor_min"]),
        float(spec["stochastic_factor_max"]),
    )
    lag = median * severity_factor * stochastic_factor
    inputs = {
        "scenario_family": scenario_family,
        "scenario_median_seconds": median,
        "congestion_severity": round(congestion_severity, 6),
        "severity_weight": float(spec["severity_weight"]),
        "severity_factor": round(severity_factor, 9),
        "stochastic_factor": round(stochastic_factor, 9),
        "lognormal_sigma": float(spec["lognormal_sigma"]),
    }
    return round(lag, 3), _provenance(profile, spec, inputs)


def recompute_lag(provenance: Dict[str, Any]) -> float:
    inputs = provenance["inputs"]
    model = provenance["model"]
    if model == "bpr_conditioned_effective_link_time":
        bpr_seconds = float(inputs["free_flow_travel_seconds"]) * (
            1.0
            + float(inputs["alpha"])
            * float(inputs["incident_adjusted_volume_capacity_ratio"])
            ** float(inputs["beta"])
        )
        return round(
            float(inputs["response_delay_seconds"])
            + bpr_seconds
            * float(inputs["weather_factor"])
            * float(inputs["stochastic_factor"]),
            3,
        )
    if model == "scenario_conditioned_lognormal_reference_prior":
        return round(
            float(inputs["scenario_median_seconds"])
            * float(inputs["severity_factor"])
            * float(inputs["stochastic_factor"]),
            3,
        )
    raise ValueError(f"Unsupported lag calibration model: {model}")


def _provenance(
    profile: Dict[str, Any], spec: Dict[str, Any], inputs: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "profile_id": profile["profile_id"],
        "profile_status": profile["status"],
        "model": spec["model"],
        "formula": spec["formula"],
        "inputs": inputs,
        "source": copy.deepcopy(spec.get("source", {})),
        "limitation": spec.get("limitation", ""),
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Calibration input must be finite")
    return max(lower, min(upper, value))
