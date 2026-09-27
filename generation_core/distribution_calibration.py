"""Reference-sequence calibration and held-out distribution comparison.

Sequential transition statistics are descriptive and never become causal
parents. Temporal model overrides are fitted only from explicit parent links
that can be resolved to one declared mechanism.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .mechanisms import MechanismRegistry
from .reference_data import ReferenceDataset, ReferenceEvent
from .temporal import (
    ConditionalGammaModel,
    ConditionalLogNormalModel,
    ConditionalWeibullModel,
    DeterministicDelayModel,
    ExponentialArrivalModel,
    TemporalModelRegistry,
)


SEQUENCE_CALIBRATION_SCHEMA = "sequence-calibration-bundle-v1"
VALID_CALIBRATION_COMPONENTS = {
    "scenario",
    "temporal",
    "mechanism_reweighting",
}


def fit_sequence_calibration_bundle(
    dataset: ReferenceDataset,
    mechanism_registry: MechanismRegistry,
    *,
    min_transition_samples: int = 5,
) -> Dict[str, Any]:
    """Fit descriptive sequence priors and safe explicit-parent time overrides."""

    if dataset.domain != mechanism_registry.domain:
        raise ValueError("Reference dataset and mechanism registry use different domains")
    if min_transition_samples < 2:
        raise ValueError("min_transition_samples must be at least 2")
    statistics = {
        split: summarize_reference_sequences(dataset, split=split)
        for split in ("train", "validation", "holdout")
    }
    overrides, unresolved = _fit_temporal_overrides(
        dataset,
        mechanism_registry,
        min_samples=min_transition_samples,
    )
    scenario_weights = _scenario_weights(dataset, split="train")
    diagnostics = compare_sequence_summaries(
        statistics["train"], statistics["holdout"]
    )
    payload = {
        "schema_version": SEQUENCE_CALIBRATION_SCHEMA,
        "domain": dataset.domain,
        "created_at": datetime.now().astimezone().isoformat(),
        "reference_data_fingerprint": dataset.fingerprint(),
        "parameter_status": "empirically_estimated_pending_domain_acceptance",
        "fit_scope": {
            "training_split": "train",
            "validation_split": "validation",
            "holdout_split": "holdout",
            "min_transition_samples": min_transition_samples,
        },
        "sequence_statistics": statistics,
        "temporal_model_overrides": overrides,
        "domain_config_overrides": (
            {"scenario_weights": scenario_weights} if scenario_weights else {}
        ),
        "mechanism_reweighting_prior": {
            "source_split": "train",
            "supported_history_orders": ["first_order", "second_order"],
            "default_mode": "first_second_order",
            "weight_semantics": "relative_proportional_hazards_multiplier",
            "normalization_reference": "uniform_over_observed_targets_for_history",
            "only_declared_legal_mechanisms_are_reweightable": True,
            "unobserved_legal_transition_policy": "leave_prior_unchanged",
            "multi_parent_policy": "leave_prior_unchanged",
        },
        "holdout_diagnostics": diagnostics,
        "unresolved_explicit_parent_links": unresolved,
        "safety_contract": {
            "sequential_transitions_are_descriptive_not_causal": True,
            "transition_statistics_may_only_reweight_declared_legal_mechanisms": True,
            "only_explicit_parent_links_can_fit_generation_clocks": True,
            "llm_does_not_estimate_parameters": True,
            "holdout_is_not_used_for_parameter_estimation": True,
        },
        "limitations": [
            "Calibration improves only measured reference-distribution dimensions.",
            "An empirically fitted transition is not evidence of causal direction.",
            "Automatic use requires explicit acceptance of the pending bundle.",
            "Marginal delay fits do not recover unobserved covariate effects.",
            "Transition reweighting does not infer new mechanisms or causal edges.",
            "Unobserved and multi-parent legal mechanisms retain their original priors.",
        ],
    }
    payload["bundle_fingerprint"] = calibration_bundle_fingerprint(payload)
    return payload


def summarize_reference_sequences(
    dataset: ReferenceDataset,
    *,
    split: str,
) -> Dict[str, Any]:
    window_ids = {item.window_id for item in dataset.windows if item.split == split}
    # Keep zero-event windows in the sequence-length distribution.  Omitting
    # them would condition the fitted prior on "at least one event" and make
    # sparse reference datasets appear artificially dense.
    streams: Dict[str, list[tuple[float, str]]] = {
        window_id: [] for window_id in window_ids
    }
    scope = dataset.source_metadata.get("sequence_scope", "window")
    if scope not in {"window", "entity", "process"}:
        raise ValueError("sequence_scope must be window, entity, or process")
    if scope != "window":
        streams = {}
    for event in dataset.events:
        if event.window_id in window_ids:
            keys = [event.window_id]
            if scope == "entity":
                keys = [f"{event.window_id}::{p.entity_id}" for p in event.participants]
            elif scope == "process":
                if "process_id" not in event.attributes:
                    raise ValueError("Process-scoped statistics require process_id on every event")
                keys = [f"{event.window_id}::{event.attributes['process_id']}"]
            for key in set(keys):
                streams.setdefault(key, []).append((event.occurrence_offset_seconds, event.event_type_id))
    result = summarize_streams(streams.values())
    result["sequence_scope"] = scope
    result["transition_semantics"] = "descriptive_order_not_causal_parenthood"
    result["zero_event_window_count"] = sum(not any(e.window_id == w for e in dataset.events) for w in window_ids)
    return result


def summarize_generated_output(output_dir: str | Path) -> Dict[str, Any]:
    root = Path(output_dir)
    streams: Dict[str, list[tuple[float, str]]] = defaultdict(list)
    with (root / "events.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            temporal = event.get("temporal", {})
            streams[str(event["episode_id"])].append(
                (
                    float(temporal["occurrence_start_offset_seconds"]),
                    str(event["event_type_id"]),
                )
            )
    return summarize_streams(streams.values())


def summarize_streams(
    streams: Iterable[Sequence[tuple[float, str]]],
) -> Dict[str, Any]:
    event_counts: Counter[str] = Counter()
    first_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    second_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    transition_counts: Counter[str] = Counter()
    sequence_lengths: Counter[str] = Counter()
    inter_event_times: list[float] = []
    event_type_sequences: list[list[str]] = []
    stream_count = 0
    for raw_stream in streams:
        ordered = sorted(raw_stream, key=lambda item: (float(item[0]), str(item[1])))
        stream_count += 1
        types = [str(item[1]) for item in ordered]
        times = [float(item[0]) for item in ordered]
        time_counts = Counter(times)
        event_type_sequences.append(types)
        sequence_lengths[str(len(types))] += 1
        event_counts.update(types)
        for left_index in range(max(0, len(types) - 1)):
            if any(time_counts[times[i]] != 1 for i in (left_index, left_index + 1)):
                continue
            source = types[left_index]
            target = types[left_index + 1]
            first_counts[source][target] += 1
            transition_counts[f"{source}->{target}"] += 1
            delta = times[left_index + 1] - times[left_index]
            if delta > 0.0:
                inter_event_times.append(delta)
        for index in range(max(0, len(types) - 2)):
            if any(time_counts[times[i]] != 1 for i in range(index, index + 3)):
                continue
            second_counts[f"{types[index]}|{types[index + 1]}"][types[index + 2]] += 1
    return {
        "stream_count": stream_count,
        "simultaneous_event_policy": "exclude_ambiguous_transition_windows",
        "event_count": sum(event_counts.values()),
        "event_type_counts": dict(sorted(event_counts.items())),
        "event_type_probabilities": _normalize(event_counts),
        "first_order_transitions": _nested_probabilities(first_counts),
        "second_order_transitions": _nested_probabilities(second_counts),
        "transition_counts": dict(sorted(transition_counts.items())),
        "sequence_length_distribution": dict(sorted(sequence_lengths.items())),
        "inter_event_time_summary": _numeric_summary(inter_event_times),
        "inter_event_times": inter_event_times,
        "repeated_event_ratio": _repeated_event_ratio(event_type_sequences),
        "burstiness": _burstiness(inter_event_times),
    }


def compare_reference_to_generation(
    dataset: ReferenceDataset,
    *,
    generic_output_dir: str | Path,
    calibrated_output_dir: str | Path,
    split: str = "holdout",
) -> Dict[str, Any]:
    real = summarize_reference_sequences(dataset, split=split)
    generic = summarize_generated_output(generic_output_dir)
    calibrated = summarize_generated_output(calibrated_output_dir)
    comparison = {
        "schema_version": "sequence-calibration-comparison-v1",
        "domain": dataset.domain,
        "reference_split": split,
        "reference_data_fingerprint": dataset.fingerprint(),
        "generic_output_dir": str(Path(generic_output_dir).resolve()),
        "calibrated_output_dir": str(Path(calibrated_output_dir).resolve()),
        "metrics": {
            "event_type_frequency_l1": _three_way_frequency_metric(
                real["event_type_counts"],
                generic["event_type_counts"],
                calibrated["event_type_counts"],
            ),
            "transition_frequency_l1": _three_way_frequency_metric(
                real["transition_counts"],
                generic["transition_counts"],
                calibrated["transition_counts"],
            ),
            "sequence_length_ks": _three_way_continuous_metric(
                _expand_integer_distribution(real["sequence_length_distribution"]),
                _expand_integer_distribution(generic["sequence_length_distribution"]),
                _expand_integer_distribution(calibrated["sequence_length_distribution"]),
                ks_statistic,
            ),
            "sequence_length_wasserstein": _three_way_continuous_metric(
                _expand_integer_distribution(real["sequence_length_distribution"]),
                _expand_integer_distribution(generic["sequence_length_distribution"]),
                _expand_integer_distribution(calibrated["sequence_length_distribution"]),
                wasserstein_distance,
            ),
            "inter_event_time_ks": _three_way_continuous_metric(
                real["inter_event_times"],
                generic["inter_event_times"],
                calibrated["inter_event_times"],
                ks_statistic,
            ),
            "inter_event_time_wasserstein": _three_way_continuous_metric(
                real["inter_event_times"],
                generic["inter_event_times"],
                calibrated["inter_event_times"],
                wasserstein_distance,
            ),
        },
        "descriptive": {
            "real": _compact_summary(real),
            "generic": _compact_summary(generic),
            "calibrated": _compact_summary(calibrated),
        },
        "interpretation": (
            "A lower calibrated distance is evidence of improved agreement only "
            "for the reported held-out statistics, not proof of causal truth."
        ),
    }
    comparison["metric_improvement_count"] = sum(
        bool(item["improved"])
        for item in comparison["metrics"].values()
    )
    comparison["metric_count"] = len(comparison["metrics"])
    return comparison


def compare_sequence_summaries(
    fitted: Mapping[str, Any], held_out: Mapping[str, Any]
) -> Dict[str, Any]:
    if not held_out.get("stream_count"):
        return {"status": "holdout_not_provided"}
    return {
        "status": "diagnostic_only_no_universal_acceptance_threshold",
        "event_type_frequency_l1": l1_frequency_distance(
            fitted.get("event_type_counts", {}), held_out.get("event_type_counts", {})
        ),
        "transition_frequency_l1": l1_frequency_distance(
            fitted.get("transition_counts", {}), held_out.get("transition_counts", {})
        ),
        "sequence_length_ks": ks_statistic(
            _expand_integer_distribution(fitted.get("sequence_length_distribution", {})),
            _expand_integer_distribution(held_out.get("sequence_length_distribution", {})),
        ),
        "inter_event_time_ks": ks_statistic(
            fitted.get("inter_event_times", []), held_out.get("inter_event_times", [])
        ),
    }


def apply_sequence_calibration_bundle(
    bundle: Mapping[str, Any],
    *,
    domain: str,
    domain_config: Dict[str, Any],
    temporal_models: TemporalModelRegistry,
    allow_pending: bool = False,
    components: Iterable[str] | None = None,
) -> Dict[str, Any]:
    if bundle.get("schema_version") != SEQUENCE_CALIBRATION_SCHEMA:
        raise ValueError("Unsupported sequence calibration bundle")
    expected_fingerprint = calibration_bundle_fingerprint(bundle)
    if bundle.get("bundle_fingerprint") != expected_fingerprint:
        raise ValueError("Calibration bundle fingerprint mismatch")
    if bundle.get("domain") != domain:
        raise ValueError("Calibration bundle belongs to another domain")
    status = str(bundle.get("parameter_status", ""))
    if status != "accepted_for_generation" and not allow_pending:
        raise ValueError(
            "Calibration bundle is pending domain acceptance; pass the explicit "
            "allow-pending option only for evaluation runs"
        )
    selected_components = set(
        components
        if components is not None
        else VALID_CALIBRATION_COMPONENTS
    )
    unknown_components = selected_components - VALID_CALIBRATION_COMPONENTS
    if unknown_components:
        raise ValueError(
            f"Unknown calibration components: {sorted(unknown_components)}"
        )
    if not selected_components:
        raise ValueError("At least one calibration component must be selected")
    domain_overrides = dict(bundle.get("domain_config_overrides", {}))
    if (
        "scenario" in selected_components
        and domain_overrides.get("scenario_weights")
    ):
        domain_config["scenario_weights"] = dict(
            domain_overrides["scenario_weights"]
        )
    applied_models = []
    skipped_models = {}
    if "temporal" in selected_components:
        for model_id, record in sorted(
            dict(bundle.get("temporal_model_overrides", {})).items()
        ):
            if record.get("eligible_for_generation_override") is not True:
                skipped_models[model_id] = "Missing censoring/risk-set evidence; descriptive completed-link fit only"
                continue
            model = temporal_model_from_override(model_id, record)
            if model is None:
                continue
            temporal_models.replace(model, parameter_status=status)
            applied_models.append(model_id)
    return {
        "schema_version": bundle["schema_version"],
        "bundle_fingerprint": bundle.get("bundle_fingerprint"),
        "reference_data_fingerprint": bundle.get("reference_data_fingerprint"),
        "parameter_status": status,
        "selected_components": sorted(selected_components),
        "scenario_weights_applied": bool(
            "scenario" in selected_components
            and domain_overrides.get("scenario_weights")
        ),
        "temporal_model_ids_applied": applied_models,
        "temporal_models_skipped": skipped_models,
        "mechanism_reweighting_enabled": (
            "mechanism_reweighting" in selected_components
        ),
        "pending_bundle_explicitly_allowed": bool(
            allow_pending and status != "accepted_for_generation"
        ),
        "sequence_length_statistics_usage": "held_out_evaluation_only",
        "transition_statistics_usage": (
            "descriptive_evaluation_only_not_promoted_to_causal_edges"
        ),
    }


def temporal_model_from_override(model_id: str, record: Mapping[str, Any]):
    regime = str(record.get("regime", ""))
    params = dict(record.get("params", {}))
    if regime == "deterministic":
        return DeterministicDelayModel(model_id, float(params["value_seconds"]))
    if regime == "exponential":
        return ExponentialArrivalModel(model_id, float(params["rate_per_second"]))
    if regime == "gamma":
        return ConditionalGammaModel(
            model_id,
            shape=float(params["shape"]),
            log_rate_intercept=math.log(1.0 / float(params["scale_seconds"])),
            coefficients={},
        )
    if regime == "lognormal":
        return ConditionalLogNormalModel(
            model_id,
            intercept=float(params["mu"]),
            sigma=float(params["sigma"]),
            coefficients={},
        )
    if regime == "weibull":
        return ConditionalWeibullModel(
            model_id,
            shape=float(params["shape"]),
            log_scale_intercept=math.log(float(params["scale_seconds"])),
            coefficients={},
        )
    return None


def fit_best_regime(
    samples: Iterable[float], *, min_samples: int = 5
) -> Dict[str, Any]:
    values = sorted(
        float(item)
        for item in samples
        if math.isfinite(float(item)) and float(item) > 0.0
    )
    if len(values) < min_samples:
        return {
            "regime": "insufficient_samples",
            "params": {},
            "sample_count": len(values),
            "fit": {"status": "insufficient_samples", "min_samples": min_samples},
        }
    if values[-1] - values[0] <= 1e-12:
        return {
            "regime": "deterministic",
            "params": {"value_seconds": values[0]},
            "sample_count": len(values),
            "fit": {"status": "zero_variance_samples", "bic": 0.0},
        }
    candidates = [
        _fit_exponential(values),
        _fit_gamma(values),
        _fit_lognormal(values),
        _fit_weibull(values),
    ]
    best = min(candidates, key=lambda item: item["fit"]["bic"])
    best["sample_count"] = len(values)
    best["sample_summary"] = _numeric_summary(values)
    return best


def calibration_bundle_fingerprint(payload: Mapping[str, Any]) -> str:
    copied = dict(payload)
    copied.pop("bundle_fingerprint", None)
    encoded = json.dumps(
        copied, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _fit_temporal_overrides(
    dataset: ReferenceDataset,
    registry: MechanismRegistry,
    *,
    min_samples: int,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    window_by_id = {item.window_id: item for item in dataset.windows}
    event_by_id = {item.source_record_id: item for item in dataset.events}
    mechanism_by_id = {
        item["mechanism_id"]: item for item in registry.descriptors()
    }
    grouped: Dict[str, list[float]] = defaultdict(list)
    mechanism_ids_by_model: Dict[str, set[str]] = defaultdict(set)
    unresolved = []
    for child in dataset.events:
        window = window_by_id.get(child.window_id)
        if window is None or window.split != "train" or not child.explicit_parent_record_ids:
            continue
        parents = [event_by_id[parent_id] for parent_id in child.explicit_parent_record_ids]
        mechanism = _resolve_mechanism(child, parents, mechanism_by_id)
        if mechanism is None:
            unresolved.append(
                {
                    "child_record_id": child.source_record_id,
                    "parent_record_ids": list(child.explicit_parent_record_ids),
                    "reason": "no_unique_declared_mechanism_match",
                }
            )
            continue
        model_id = str(mechanism["generation_temporal_model_ref"])
        if model_id in {"core.immediate", "core.scheduled"} or mechanism.get(
            "parameter_status"
        ) == "policy_defined":
            continue
        activation_time = max(parent.occurrence_offset_seconds for parent in parents)
        delay = child.occurrence_offset_seconds - activation_time
        if delay <= 0.0:
            unresolved.append(
                {
                    "child_record_id": child.source_record_id,
                    "parent_record_ids": list(child.explicit_parent_record_ids),
                    "reason": "non_positive_parent_child_delay",
                }
            )
            continue
        grouped[model_id].append(delay)
        mechanism_ids_by_model[model_id].add(str(mechanism["mechanism_id"]))
    overrides = {}
    for model_id, samples in sorted(grouped.items()):
        fit = fit_best_regime(samples, min_samples=min_samples)
        if fit["regime"] == "insufficient_samples":
            continue
        overrides[model_id] = {
            **fit,
            "source": "explicit_parent_child_delays_only",
            "source_mechanism_ids": sorted(mechanism_ids_by_model[model_id]),
            "conditioning_policy": "marginal_fit_covariates_not_recovered",
            "fit_scope": "observed_completed_links_only",
            "eligible_for_generation_override": False,
            "limitation": "Selection and right-censoring are not modeled; descriptive fit only",
        }
    return overrides, unresolved


def _resolve_mechanism(
    child: ReferenceEvent,
    parents: Sequence[ReferenceEvent],
    mechanism_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    declared_id = child.provenance.get("mechanism_id")
    if declared_id in mechanism_by_id:
        return mechanism_by_id[str(declared_id)]
    parent_types = [parent.event_type_id for parent in parents]
    candidates = []
    for mechanism in mechanism_by_id.values():
        if mechanism["target_event_type_id"] != child.event_type_id:
            continue
        source_types = list(mechanism.get("source_event_type_ids", []))
        combination = mechanism.get("parent_combination")
        if combination == "single" and len(parent_types) == 1 and parent_types[0] in source_types:
            candidates.append(mechanism)
        elif combination in {"all_of", "ordered_sequence"} and sorted(parent_types) == sorted(source_types):
            candidates.append(mechanism)
        elif combination in {"any_of", "k_of_n"} and set(parent_types) <= set(source_types):
            candidates.append(mechanism)
    return candidates[0] if len(candidates) == 1 else None


def _scenario_weights(dataset: ReferenceDataset, *, split: str) -> Dict[str, float]:
    counts = Counter(
        str(window.attributes["scenario_family"])
        for window in dataset.windows
        if window.split == split and window.attributes.get("scenario_family")
    )
    total = sum(counts.values())
    return {
        key: count / total for key, count in sorted(counts.items())
    } if total else {}


def _fit_exponential(values: list[float]) -> Dict[str, Any]:
    rate = 1.0 / (sum(values) / len(values))
    log_likelihood = len(values) * math.log(rate) - rate * sum(values)
    return _fit_record("exponential", {"rate_per_second": rate}, log_likelihood, 1, len(values))


def _fit_gamma(values: list[float]) -> Dict[str, Any]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    shape = max(1e-6, mean * mean / max(variance, 1e-12))
    scale = max(1e-12, variance / mean)
    log_likelihood = sum(
        (shape - 1.0) * math.log(value)
        - value / scale
        - math.lgamma(shape)
        - shape * math.log(scale)
        for value in values
    )
    return _fit_record("gamma", {"shape": shape, "scale_seconds": scale}, log_likelihood, 2, len(values))


def _fit_lognormal(values: list[float]) -> Dict[str, Any]:
    logs = [math.log(value) for value in values]
    mu = sum(logs) / len(logs)
    sigma = max(1e-9, (sum((value - mu) ** 2 for value in logs) / len(logs)) ** 0.5)
    log_likelihood = sum(
        -math.log(value * sigma * math.sqrt(2.0 * math.pi))
        - ((math.log(value) - mu) ** 2) / (2.0 * sigma * sigma)
        for value in values
    )
    return _fit_record("lognormal", {"mu": mu, "sigma": sigma}, log_likelihood, 2, len(values))


def _fit_weibull(values: list[float]) -> Dict[str, Any]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    target_cv2 = variance / max(mean * mean, 1e-12)
    lower, upper = 0.1, 100.0
    for _ in range(100):
        shape = (lower + upper) / 2.0
        cv2 = math.gamma(1.0 + 2.0 / shape) / math.gamma(1.0 + 1.0 / shape) ** 2 - 1.0
        if cv2 > target_cv2:
            lower = shape
        else:
            upper = shape
    shape = (lower + upper) / 2.0
    scale = mean / math.gamma(1.0 + 1.0 / shape)
    log_likelihood = sum(
        math.log(shape / scale)
        + (shape - 1.0) * math.log(value / scale)
        - (value / scale) ** shape
        for value in values
    )
    return _fit_record("weibull", {"shape": shape, "scale_seconds": scale}, log_likelihood, 2, len(values))


def _fit_record(
    regime: str,
    params: Dict[str, float],
    log_likelihood: float,
    parameter_count: int,
    sample_count: int,
) -> Dict[str, Any]:
    return {
        "regime": regime,
        "params": {key: round(float(value), 12) for key, value in params.items()},
        "fit": {
            "status": "fitted",
            "selection_metric": "bic",
            "log_likelihood": round(log_likelihood, 6),
            "aic": round(2 * parameter_count - 2 * log_likelihood, 6),
            "bic": round(math.log(sample_count) * parameter_count - 2 * log_likelihood, 6),
        },
    }


def _nested_probabilities(counts: Mapping[str, Counter[str]]) -> Dict[str, Any]:
    output = {}
    for source, targets in sorted(counts.items()):
        total = sum(targets.values())
        output[source] = {
            target: {"count": count, "probability": count / total}
            for target, count in sorted(targets.items())
        }
    return output


def _normalize(values: Mapping[str, int | float]) -> Dict[str, float]:
    total = sum(float(value) for value in values.values())
    return {
        str(key): float(value) / total for key, value in sorted(values.items())
    } if total else {}


def l1_frequency_distance(
    left: Mapping[str, int | float], right: Mapping[str, int | float]
) -> float:
    left_probability = _normalize(left)
    right_probability = _normalize(right)
    return round(
        sum(
            abs(left_probability.get(key, 0.0) - right_probability.get(key, 0.0))
            for key in set(left_probability) | set(right_probability)
        ),
        9,
    )


def ks_statistic(left: Iterable[float], right: Iterable[float]) -> float:
    a = sorted(float(value) for value in left)
    b = sorted(float(value) for value in right)
    if not a or not b:
        return 0.0
    points = sorted(set(a + b))
    ai = bi = 0
    maximum = 0.0
    for point in points:
        while ai < len(a) and a[ai] <= point:
            ai += 1
        while bi < len(b) and b[bi] <= point:
            bi += 1
        maximum = max(maximum, abs(ai / len(a) - bi / len(b)))
    return round(maximum, 9)


def wasserstein_distance(left: Iterable[float], right: Iterable[float]) -> float:
    a = sorted(float(value) for value in left)
    b = sorted(float(value) for value in right)
    if not a or not b:
        return 0.0
    points = sorted(set(a + b))
    ai = bi = 0
    previous = points[0]
    distance = 0.0
    for point in points:
        distance += (point - previous) * abs(ai / len(a) - bi / len(b))
        while ai < len(a) and a[ai] <= point:
            ai += 1
        while bi < len(b) and b[bi] <= point:
            bi += 1
        previous = point
    return round(distance, 9)


def _numeric_summary(values: Sequence[float]) -> Dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "max": ordered[-1],
    }


def _repeated_event_ratio(streams: Sequence[Sequence[str]]) -> float:
    ratios = []
    for stream in streams:
        if stream:
            counts = Counter(stream)
            ratios.append(sum(count for count in counts.values() if count > 1) / len(stream))
    return round(sum(ratios) / len(ratios), 9) if ratios else 0.0


def _burstiness(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    standard_deviation = (
        sum((value - mean) ** 2 for value in values) / len(values)
    ) ** 0.5
    return round((standard_deviation - mean) / (standard_deviation + mean), 9) if standard_deviation + mean else 0.0


def _expand_integer_distribution(distribution: Mapping[str, int]) -> list[float]:
    return [
        float(value)
        for value, count in distribution.items()
        for _ in range(int(count))
    ]


def _three_way_frequency_metric(real, generic, calibrated) -> Dict[str, Any]:
    generic_distance = l1_frequency_distance(generic, real)
    calibrated_distance = l1_frequency_distance(calibrated, real)
    return _improvement_record(generic_distance, calibrated_distance)


def _three_way_continuous_metric(real, generic, calibrated, metric) -> Dict[str, Any]:
    return _improvement_record(metric(generic, real), metric(calibrated, real))


def _improvement_record(generic: float, calibrated: float) -> Dict[str, Any]:
    return {
        "generic_vs_real": generic,
        "calibrated_vs_real": calibrated,
        "absolute_improvement": round(generic - calibrated, 9),
        "relative_improvement": round((generic - calibrated) / generic, 9) if generic else 0.0,
        "improved": calibrated < generic,
    }


def _compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stream_count": summary.get("stream_count", 0),
        "event_count": summary.get("event_count", 0),
        "event_type_counts": summary.get("event_type_counts", {}),
        "sequence_length_distribution": summary.get("sequence_length_distribution", {}),
        "inter_event_time_summary": summary.get("inter_event_time_summary", {}),
        "burstiness": summary.get("burstiness", 0.0),
        "repeated_event_ratio": summary.get("repeated_event_ratio", 0.0),
    }
