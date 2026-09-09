"""Generic calibration observations, fitters, holdout evaluation, and artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Protocol

from .mechanisms import MechanismSpec
from .reference_data import ReferenceDataset


@dataclass(frozen=True)
class CalibrationObservation:
    observation_id: str
    mechanism_id: str
    split: str
    exposure_seconds: float
    event_count: int
    covariates: Dict[str, float] = field(default_factory=dict)
    context_id: str | None = None
    window_id: str | None = None
    censoring_status: str = "observed_or_window_censored"
    source_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.exposure_seconds <= 0.0:
            raise ValueError("Calibration exposure must be positive")
        if self.event_count < 0:
            raise ValueError("Calibration event_count must be non-negative")
        if self.split not in {"train", "validation", "holdout"}:
            raise ValueError(f"Unsupported calibration split: {self.split}")
        if any(not math.isfinite(float(value)) for value in self.covariates.values()):
            raise ValueError("Calibration covariates must be finite numbers")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FitterResult:
    fitter_id: str
    model_family: str
    parameter_estimates: Dict[str, Any]
    uncertainty: Dict[str, Any]
    fit_metrics: Dict[str, Any]
    training_scope: Dict[str, Any]


class CalibrationFitter(Protocol):
    fitter_id: str

    def fit(
        self,
        mechanism: MechanismSpec,
        observations: list[CalibrationObservation],
    ) -> FitterResult: ...

    def evaluate(
        self,
        mechanism: MechanismSpec,
        parameters: Mapping[str, Any],
        observations: list[CalibrationObservation],
    ) -> Dict[str, Any]: ...


class ExponentialRateMLEFitter:
    """MLE for a constant event intensity using counts and exposure time.

    The likelihood is sum(n_i log(lambda) - lambda * exposure_i), so windows
    with zero events still contribute their full exposure instead of being
    discarded as negative examples.
    """

    fitter_id = "core.exponential_rate_mle"

    def fit(
        self,
        mechanism: MechanismSpec,
        observations: list[CalibrationObservation],
    ) -> FitterResult:
        if mechanism.temporal_model_family not in {
            "exponential_arrival",
            "piecewise_exponential_hazard",
        }:
            raise ValueError(
                f"{self.fitter_id} cannot fit {mechanism.temporal_model_family}"
            )
        if mechanism.covariate_fields:
            raise ValueError(
                f"{self.fitter_id} only fits a constant rate without covariates"
            )
        _validate_observations(mechanism, observations, required_split="train")
        event_count = sum(item.event_count for item in observations)
        exposure = sum(item.exposure_seconds for item in observations)
        if event_count <= 0:
            raise ValueError(
                "A positive rate cannot be estimated from training data with zero events"
            )
        rate = event_count / exposure
        standard_error = math.sqrt(event_count) / exposure
        lower = max(0.0, rate - 1.959963984540054 * standard_error)
        upper = rate + 1.959963984540054 * standard_error
        log_likelihood = _poisson_process_log_likelihood(rate, observations)
        return FitterResult(
            fitter_id=self.fitter_id,
            model_family=mechanism.temporal_model_family,
            parameter_estimates={"rate_per_second": rate},
            uncertainty={
                "method": "wald_large_sample_approximation",
                "standard_error": standard_error,
                "confidence_interval_95": [lower, upper],
            },
            fit_metrics={
                "log_likelihood": log_likelihood,
                "observed_event_count": event_count,
                "expected_event_count_at_mle": rate * exposure,
                "total_exposure_seconds": exposure,
                "small_sample_warning": event_count < 30,
            },
            training_scope=_observation_scope(observations),
        )

    def evaluate(
        self,
        mechanism: MechanismSpec,
        parameters: Mapping[str, Any],
        observations: list[CalibrationObservation],
    ) -> Dict[str, Any]:
        del mechanism
        if not observations:
            return {"status": "not_provided"}
        rate = float(parameters["rate_per_second"])
        exposure = sum(item.exposure_seconds for item in observations)
        observed = sum(item.event_count for item in observations)
        expected = rate * exposure
        return {
            "status": "diagnostic_only_no_universal_acceptance_threshold",
            "log_likelihood": _poisson_process_log_likelihood(rate, observations),
            "observed_event_count": observed,
            "expected_event_count": expected,
            "observed_to_expected_ratio": (
                observed / expected if expected > 0.0 else None
            ),
            "total_exposure_seconds": exposure,
            "scope": _observation_scope(observations),
        }


class CalibrationFitterRegistry:
    def __init__(self, fitters: Iterable[CalibrationFitter] = ()) -> None:
        self._fitters: Dict[str, CalibrationFitter] = {}
        for fitter in fitters:
            self.register(fitter)

    def register(self, fitter: CalibrationFitter) -> None:
        if fitter.fitter_id in self._fitters:
            raise ValueError(f"Duplicate calibration fitter: {fitter.fitter_id}")
        self._fitters[fitter.fitter_id] = fitter

    def get(self, fitter_id: str) -> CalibrationFitter:
        try:
            return self._fitters[fitter_id]
        except KeyError as exc:
            raise KeyError(f"Unsupported calibration fitter: {fitter_id}") from exc


@dataclass(frozen=True)
class CalibrationBundle:
    schema_version: str
    domain: str
    mechanism_id: str
    temporal_model_family: str
    fitter_id: str
    parameter_estimates: Dict[str, Any]
    uncertainty: Dict[str, Any]
    reference_data_fingerprint: str
    observation_fingerprint: str
    training_scope: Dict[str, Any]
    fit_metrics: Dict[str, Any]
    holdout_metrics: Dict[str, Any]
    parameter_status: str
    created_at: str
    limitations: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")


class CalibrationEngine:
    def __init__(self, fitters: CalibrationFitterRegistry | None = None) -> None:
        self.fitters = fitters or build_standard_calibration_fitters()

    def fit(
        self,
        mechanism: MechanismSpec,
        train_observations: list[CalibrationObservation],
        *,
        reference_data_fingerprint: str,
        holdout_observations: list[CalibrationObservation] | None = None,
    ) -> CalibrationBundle:
        if mechanism.parameter_status == "policy_defined":
            raise ValueError("Policy-defined mechanisms are not empirically fitted")
        if not mechanism.calibration_fitter_id:
            raise ValueError(
                f"Mechanism {mechanism.mechanism_id} has no selected calibration fitter"
            )
        fitter = self.fitters.get(mechanism.calibration_fitter_id)
        fitted = fitter.fit(mechanism, train_observations)
        holdout = list(holdout_observations or [])
        if holdout:
            _validate_observations(mechanism, holdout, required_split="holdout")
        holdout_metrics = fitter.evaluate(
            mechanism, fitted.parameter_estimates, holdout
        )
        return CalibrationBundle(
            schema_version="calibration-bundle-v1",
            domain=mechanism.domain,
            mechanism_id=mechanism.mechanism_id,
            temporal_model_family=mechanism.temporal_model_family,
            fitter_id=fitted.fitter_id,
            parameter_estimates=fitted.parameter_estimates,
            uncertainty=fitted.uncertainty,
            reference_data_fingerprint=reference_data_fingerprint,
            observation_fingerprint=_observation_fingerprint(
                [*train_observations, *holdout]
            ),
            training_scope=fitted.training_scope,
            fit_metrics=fitted.fit_metrics,
            holdout_metrics=holdout_metrics,
            parameter_status="empirically_estimated_pending_domain_acceptance",
            created_at=datetime.now().astimezone().isoformat(),
            limitations=(
                "Parameter estimation does not by itself establish causal semantics.",
                "Domain acceptance requires predeclared holdout checks and reference-data review.",
                "Wald uncertainty can be inaccurate for small event counts; inspect small_sample_warning.",
            ),
        )


def build_root_event_observations(
    dataset: ReferenceDataset,
    mechanism: MechanismSpec,
) -> list[CalibrationObservation]:
    """Build exposure/count rows for an exogenous root-event mechanism."""

    if not mechanism.is_root or mechanism.parent_combination != "none":
        raise ValueError("Root observation construction requires an exogenous mechanism")
    events_by_window: Dict[str, list[str]] = {}
    for event in dataset.events:
        if event.event_type_id == mechanism.target_event_type_id:
            events_by_window.setdefault(event.window_id, []).append(event.source_record_id)
    return [
        CalibrationObservation(
            observation_id=f"{mechanism.mechanism_id}:{window.window_id}",
            mechanism_id=mechanism.mechanism_id,
            split=window.split,
            exposure_seconds=window.duration_seconds,
            event_count=len(events_by_window.get(window.window_id, ())),
            context_id=window.context_id,
            window_id=window.window_id,
            censoring_status="observation_window_complete",
            source_record_ids=tuple(sorted(events_by_window.get(window.window_id, ()))),
        )
        for window in dataset.windows
    ]


def build_standard_calibration_fitters() -> CalibrationFitterRegistry:
    return CalibrationFitterRegistry([ExponentialRateMLEFitter()])


def _validate_observations(
    mechanism: MechanismSpec,
    observations: list[CalibrationObservation],
    *,
    required_split: str,
) -> None:
    if not observations:
        raise ValueError(f"No {required_split} calibration observations were supplied")
    for observation in observations:
        if observation.mechanism_id != mechanism.mechanism_id:
            raise ValueError("Calibration observation uses another mechanism")
        if observation.split != required_split:
            raise ValueError(
                f"Expected {required_split} observation, got {observation.split}"
            )


def _poisson_process_log_likelihood(
    rate: float, observations: list[CalibrationObservation]
) -> float:
    if rate <= 0.0:
        return float("-inf")
    return sum(
        item.event_count * math.log(rate)
        - rate * item.exposure_seconds
        - math.lgamma(item.event_count + 1.0)
        for item in observations
    )


def _observation_scope(observations: list[CalibrationObservation]) -> Dict[str, Any]:
    return {
        "observation_count": len(observations),
        "context_count": len({item.context_id for item in observations}),
        "window_count": len({item.window_id for item in observations}),
        "split_counts": {
            split: sum(item.split == split for item in observations)
            for split in sorted({item.split for item in observations})
        },
    }


def _observation_fingerprint(observations: list[CalibrationObservation]) -> str:
    payload = [
        item.to_dict()
        for item in sorted(observations, key=lambda value: value.observation_id)
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
