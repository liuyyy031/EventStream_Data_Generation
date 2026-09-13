"""Pluggable temporal-model families and auditable candidate clocks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Dict, Iterable, Mapping, Protocol

from .models import Candidate, CovariateMode


class TemporalModel(Protocol):
    model_id: str
    family: str
    covariate_mode: CovariateMode

    def initialize(
        self, candidate: Candidate, rng: random.Random
    ) -> None: ...

    def update(
        self, candidate: Candidate, now: float, inputs: Mapping[str, Any]
    ) -> None: ...

    def descriptor(self) -> Dict[str, Any]: ...

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class DeterministicDelayModel:
    model_id: str
    default_delay_seconds: float = 0.0
    family: str = "deterministic_delay"
    covariate_mode: CovariateMode = CovariateMode.DETERMINISTIC_SCHEDULE

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        del rng
        delay = float(candidate.temporal_inputs.get("delay_seconds", self.default_delay_seconds))
        if delay < 0.0:
            raise ValueError(f"{self.model_id} produced a negative delay")
        candidate.scheduled_time = candidate.activation_time + delay
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(self, {"delay_seconds": delay})

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(self, {"default_delay_seconds": self.default_delay_seconds})

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        return _atom_measure("deterministic_atom", candidate, at_time)


@dataclass(frozen=True)
class ScheduledTimeModel:
    model_id: str
    family: str = "scheduled_time"
    covariate_mode: CovariateMode = CovariateMode.DETERMINISTIC_SCHEDULE

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        del rng
        due = float(candidate.temporal_inputs["scheduled_offset_seconds"])
        if due < candidate.activation_time:
            raise ValueError(f"{self.model_id} scheduled a candidate before activation")
        candidate.scheduled_time = due
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self, {"scheduled_offset_seconds": due}
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(self, {})

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        return _atom_measure("deterministic_atom", candidate, at_time)


@dataclass(frozen=True)
class ConditionalLogNormalModel:
    model_id: str
    intercept: float
    sigma: float
    coefficients: Mapping[str, float]
    minimum_seconds: float = 0.0
    maximum_seconds: float | None = None
    family: str = "conditional_lognormal"
    covariate_mode: CovariateMode = CovariateMode.STATIC_AT_ACTIVATION

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        linear = self.intercept + sum(
            float(weight) * float(candidate.temporal_inputs.get(name, 0.0))
            for name, weight in self.coefficients.items()
        )
        if self.sigma <= 0.0:
            raise ValueError("Log-normal sigma must be positive")
        normal = NormalDist(mu=linear, sigma=self.sigma)
        lower_probability = (
            normal.cdf(math.log(self.minimum_seconds))
            if self.minimum_seconds > 0.0
            else 0.0
        )
        upper_probability = (
            normal.cdf(math.log(self.maximum_seconds))
            if self.maximum_seconds is not None
            else 1.0
        )
        if not 0.0 <= lower_probability < upper_probability <= 1.0:
            raise ValueError("Invalid truncated log-normal support")
        probability = rng.uniform(lower_probability, upper_probability)
        probability = min(1.0 - 1e-15, max(1e-15, probability))
        delay = math.exp(normal.inv_cdf(probability))
        candidate.scheduled_time = candidate.activation_time + delay
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "linear_predictor": linear,
                "delay_seconds": delay,
                "truncated_probability": probability,
                "probability_interval": [lower_probability, upper_probability],
                "conditioning_inputs": dict(candidate.temporal_inputs),
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "intercept": self.intercept,
                "sigma": self.sigma,
                "coefficients": dict(self.coefficients),
                "minimum_seconds": self.minimum_seconds,
                "maximum_seconds": self.maximum_seconds,
            },
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        elapsed = max(0.0, at_time - candidate.activation_time)
        linear = self.intercept + sum(
            float(weight) * float(candidate.temporal_inputs.get(name, 0.0))
            for name, weight in self.coefficients.items()
        )
        normal = NormalDist(mu=linear, sigma=self.sigma)
        lower_cdf = (
            normal.cdf(math.log(self.minimum_seconds))
            if self.minimum_seconds > 0.0
            else 0.0
        )
        upper_cdf = (
            normal.cdf(math.log(self.maximum_seconds))
            if self.maximum_seconds is not None
            else 1.0
        )
        normalizer = upper_cdf - lower_cdf
        if elapsed < self.minimum_seconds or elapsed <= 0.0:
            density = 0.0
            survival = 1.0
        elif self.maximum_seconds is not None and elapsed >= self.maximum_seconds:
            density = 0.0
            survival = 0.0
        else:
            z = (math.log(elapsed) - linear) / self.sigma
            density = (
                math.exp(-0.5 * z * z)
                / (elapsed * self.sigma * math.sqrt(2.0 * math.pi) * normalizer)
            )
            survival = max(
                0.0,
                (upper_cdf - normal.cdf(math.log(elapsed))) / normalizer,
            )
        return _continuous_measure(
            elapsed,
            density,
            survival,
            support=[self.minimum_seconds, self.maximum_seconds],
        )


@dataclass(frozen=True)
class ConditionalGammaModel:
    """Gamma waiting time whose rate is conditioned on candidate covariates."""

    model_id: str
    shape: float
    log_rate_intercept: float
    coefficients: Mapping[str, float]
    minimum_seconds: float = 0.0
    maximum_seconds: float | None = None
    family: str = "conditional_gamma"
    covariate_mode: CovariateMode = CovariateMode.STATIC_AT_ACTIVATION

    def _rate(self, inputs: Mapping[str, Any]) -> float:
        linear = self.log_rate_intercept + sum(
            float(weight) * float(inputs.get(name, 0.0))
            for name, weight in self.coefficients.items()
        )
        return math.exp(linear)

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        if self.shape <= 0.0:
            raise ValueError("Gamma shape must be positive")
        rate = self._rate(candidate.temporal_inputs)
        delay = _sample_truncated_gamma(
            rng,
            self.shape,
            rate,
            self.minimum_seconds,
            self.maximum_seconds,
        )
        candidate.scheduled_time = candidate.activation_time + delay
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "shape": self.shape,
                "rate_per_second": rate,
                "delay_seconds": delay,
                "conditioning_inputs": dict(candidate.temporal_inputs),
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "shape": self.shape,
                "log_rate_intercept": self.log_rate_intercept,
                "coefficients": dict(self.coefficients),
                "minimum_seconds": self.minimum_seconds,
                "maximum_seconds": self.maximum_seconds,
            },
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        elapsed = max(0.0, at_time - candidate.activation_time)
        rate = self._rate(candidate.temporal_inputs)
        return _truncated_gamma_measure(
            elapsed,
            self.shape,
            rate,
            self.minimum_seconds,
            self.maximum_seconds,
        )


@dataclass(frozen=True)
class ConditionalWeibullModel:
    """Weibull waiting time with a covariate-conditioned scale."""

    model_id: str
    shape: float
    log_scale_intercept: float
    coefficients: Mapping[str, float]
    minimum_seconds: float = 0.0
    maximum_seconds: float | None = None
    family: str = "conditional_weibull"
    covariate_mode: CovariateMode = CovariateMode.STATIC_AT_ACTIVATION

    def _scale(self, inputs: Mapping[str, Any]) -> float:
        linear = self.log_scale_intercept + sum(
            float(weight) * float(inputs.get(name, 0.0))
            for name, weight in self.coefficients.items()
        )
        return math.exp(linear)

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        if self.shape <= 0.0:
            raise ValueError("Weibull shape must be positive")
        scale = self._scale(candidate.temporal_inputs)
        lower_cdf = _weibull_cdf(self.minimum_seconds, self.shape, scale)
        upper_cdf = (
            _weibull_cdf(self.maximum_seconds, self.shape, scale)
            if self.maximum_seconds is not None
            else 1.0
        )
        probability = rng.uniform(lower_cdf, upper_cdf)
        probability = min(1.0 - 1e-15, max(0.0, probability))
        delay = scale * (-math.log1p(-probability)) ** (1.0 / self.shape)
        candidate.scheduled_time = candidate.activation_time + delay
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "shape": self.shape,
                "scale_seconds": scale,
                "delay_seconds": delay,
                "truncated_probability": probability,
                "probability_interval": [lower_cdf, upper_cdf],
                "conditioning_inputs": dict(candidate.temporal_inputs),
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "shape": self.shape,
                "log_scale_intercept": self.log_scale_intercept,
                "coefficients": dict(self.coefficients),
                "minimum_seconds": self.minimum_seconds,
                "maximum_seconds": self.maximum_seconds,
            },
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        elapsed = max(0.0, at_time - candidate.activation_time)
        scale = self._scale(candidate.temporal_inputs)
        lower_cdf = _weibull_cdf(self.minimum_seconds, self.shape, scale)
        upper_cdf = (
            _weibull_cdf(self.maximum_seconds, self.shape, scale)
            if self.maximum_seconds is not None
            else 1.0
        )
        normalizer = upper_cdf - lower_cdf
        if elapsed < self.minimum_seconds or elapsed <= 0.0:
            density = 0.0
            survival = 1.0
        elif self.maximum_seconds is not None and elapsed >= self.maximum_seconds:
            density = 0.0
            survival = 0.0
        else:
            base_survival = math.exp(-((elapsed / scale) ** self.shape))
            base_density = (
                self.shape
                / scale
                * (elapsed / scale) ** (self.shape - 1.0)
                * base_survival
            )
            density = base_density / normalizer
            survival = max(0.0, (upper_cdf - _weibull_cdf(elapsed, self.shape, scale)) / normalizer)
        return _continuous_measure(
            elapsed,
            density,
            survival,
            support=[self.minimum_seconds, self.maximum_seconds],
        )


@dataclass(frozen=True)
class EmpiricalDelayModel:
    model_id: str
    samples_seconds: tuple[float, ...]
    family: str = "empirical_delay"
    covariate_mode: CovariateMode = CovariateMode.EMPIRICAL_RESAMPLE

    def __post_init__(self) -> None:
        if not self.samples_seconds or any(value < 0.0 for value in self.samples_seconds):
            raise ValueError("Empirical delays must be a non-empty non-negative sample")

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        delay = float(rng.choice(self.samples_seconds))
        candidate.scheduled_time = candidate.activation_time + delay
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {"delay_seconds": delay, "sample_count": len(self.samples_seconds)},
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(self, {"sample_count": len(self.samples_seconds)})

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        elapsed = max(0.0, at_time - candidate.activation_time)
        surviving = sum(value >= elapsed - 1e-9 for value in self.samples_seconds)
        at_atom = sum(abs(value - elapsed) <= 1e-9 for value in self.samples_seconds)
        conditional_mass = at_atom / surviving if surviving else None
        return {
            "clock_kind": "empirical_discrete_atom",
            "elapsed_seconds": elapsed,
            "survival_probability": surviving / len(self.samples_seconds),
            "conditional_atom_probability": conditional_mass,
            "hazard_per_second": None,
            "density_per_second": None,
        }


@dataclass(frozen=True)
class ExponentialArrivalModel:
    model_id: str
    default_rate_per_second: float
    family: str = "exponential_arrival"
    covariate_mode: CovariateMode = CovariateMode.STATIC_AT_ACTIVATION

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        rate = float(candidate.temporal_inputs.get("rate_per_second", self.default_rate_per_second))
        if rate <= 0.0:
            candidate.scheduled_time = None
            delay = None
        else:
            delay = rng.expovariate(rate)
            candidate.scheduled_time = candidate.activation_time + delay
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self, {"rate_per_second": rate, "delay_seconds": delay}
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(self, {"default_rate_per_second": self.default_rate_per_second})

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        elapsed = max(0.0, at_time - candidate.activation_time)
        rate = float(candidate.temporal_inputs.get("rate_per_second", self.default_rate_per_second))
        rate = max(0.0, rate)
        survival = math.exp(-rate * elapsed)
        return _continuous_measure(elapsed, rate * survival, survival)


@dataclass(frozen=True)
class PiecewiseExponentialHazardModel:
    """Piecewise-constant hazard with a preserved exponential threshold.

    When covariates change, accumulated hazard is retained and only the future
    rate is updated.  This prevents the clock-reset bias caused by redrawing a
    complete delay after every state change.
    """

    model_id: str
    baseline_rate_per_second: float
    coefficients: Mapping[str, float]
    family: str = "piecewise_exponential_hazard"
    covariate_mode: CovariateMode = CovariateMode.TIME_VARYING_HAZARD

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        uniform = max(rng.random(), 1e-15)
        candidate.random_threshold = -math.log(uniform)
        candidate.accumulated_hazard = 0.0
        candidate.last_updated_at = candidate.activation_time
        onset_delay = max(
            0.0,
            float(candidate.temporal_inputs.get("hazard_start_delay_seconds", 0.0)),
        )
        candidate.hazard_active_from = candidate.activation_time + onset_delay
        rate = self._rate(candidate.temporal_inputs)
        candidate.current_hazard_rate = rate
        candidate.scheduled_time = _hazard_due_time(
            candidate, rate, candidate.hazard_active_from
        )
        candidate.temporal_provenance = _provenance(
            self,
            {
                "random_threshold": candidate.random_threshold,
                "initial_rate_per_second": rate,
                "hazard_start_delay_seconds": onset_delay,
                "conditioning_inputs": dict(candidate.temporal_inputs),
                "resampling_policy": "preserve_threshold",
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        if candidate.last_updated_at is None or candidate.random_threshold is None:
            raise ValueError("Hazard candidate was not initialized")
        hazard_active_from = float(
            candidate.hazard_active_from
            if candidate.hazard_active_from is not None
            else candidate.activation_time
        )
        accrual_start = max(candidate.last_updated_at, hazard_active_from)
        elapsed = max(0.0, now - accrual_start)
        old_rate = float(candidate.current_hazard_rate or 0.0)
        candidate.accumulated_hazard += old_rate * elapsed
        candidate.last_updated_at = now
        candidate.temporal_inputs = dict(inputs)
        new_rate = self._rate(inputs)
        candidate.current_hazard_rate = new_rate
        candidate.scheduled_time = _hazard_due_time(
            candidate, new_rate, max(now, hazard_active_from)
        )
        candidate.temporal_provenance["last_update"] = {
            "at_offset_seconds": now,
            "accumulated_hazard": candidate.accumulated_hazard,
            "new_rate_per_second": new_rate,
            "conditioning_inputs": dict(inputs),
        }

    def _rate(self, inputs: Mapping[str, Any]) -> float:
        if self.baseline_rate_per_second < 0.0:
            raise ValueError("Baseline hazard must be non-negative")
        linear = sum(
            float(weight) * float(inputs.get(name, 0.0))
            for name, weight in self.coefficients.items()
        )
        return self.baseline_rate_per_second * math.exp(linear)

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "baseline_rate_per_second": self.baseline_rate_per_second,
                "coefficients": dict(self.coefficients),
            },
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        active_from = float(
            candidate.hazard_active_from
            if candidate.hazard_active_from is not None
            else candidate.activation_time
        )
        last_updated = float(
            candidate.last_updated_at
            if candidate.last_updated_at is not None
            else candidate.activation_time
        )
        rate = float(candidate.current_hazard_rate or 0.0)
        cumulative = float(candidate.accumulated_hazard)
        cumulative += rate * max(0.0, at_time - max(last_updated, active_from))
        hazard = rate if at_time + 1e-12 >= active_from else 0.0
        survival = math.exp(-cumulative)
        result = _continuous_measure(
            max(0.0, at_time - candidate.activation_time),
            hazard * survival,
            survival,
        )
        result["cumulative_hazard"] = cumulative
        result["hazard_active_from_offset_seconds"] = active_from
        return result


@dataclass(frozen=True)
class ExponentialBackoffModel:
    model_id: str
    base_seconds: float
    multiplier: float = 2.0
    maximum_seconds: float = 300.0
    jitter_fraction: float = 0.1
    family: str = "exponential_backoff"
    covariate_mode: CovariateMode = CovariateMode.POLICY_DRIVEN

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        attempt = int(candidate.temporal_inputs.get("attempt", 0))
        deterministic = min(self.maximum_seconds, self.base_seconds * self.multiplier**attempt)
        jitter = deterministic * self.jitter_fraction * rng.uniform(-1.0, 1.0)
        delay = max(0.0, deterministic + jitter)
        candidate.scheduled_time = candidate.activation_time + delay
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "attempt": attempt,
                "deterministic_seconds": deterministic,
                "jitter_seconds": jitter,
                "delay_seconds": delay,
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "base_seconds": self.base_seconds,
                "multiplier": self.multiplier,
                "maximum_seconds": self.maximum_seconds,
                "jitter_fraction": self.jitter_fraction,
            },
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        return _atom_measure("policy_clock", candidate, at_time)


class TemporalModelRegistry:
    def __init__(self, models: Iterable[TemporalModel] = ()) -> None:
        self._models: Dict[str, TemporalModel] = {}
        for model in models:
            self.register(model)

    def register(self, model: TemporalModel) -> None:
        if model.model_id in self._models:
            raise ValueError(f"Duplicate temporal model ID: {model.model_id}")
        self._models[model.model_id] = model

    def get(self, model_id: str) -> TemporalModel:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown temporal model: {model_id}") from exc

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        self.get(candidate.temporal_model_ref).initialize(candidate, rng)

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        self.get(candidate.temporal_model_ref).update(candidate, now, inputs)

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        return self.get(candidate.temporal_model_ref).measure(candidate, at_time)

    def sample_delay(
        self,
        model_id: str,
        activation_time: float,
        inputs: Mapping[str, Any],
        rng: random.Random,
    ) -> tuple[float, Dict[str, Any]]:
        temporary = Candidate(
            candidate_id="temporary_delay_sample",
            episode_id="temporary",
            mechanism_id="temporary",
            target_event_type_id="temporary.delay",
            parent_event_ids=[],
            participants=[],
            activation_time=activation_time,
            temporal_model_ref=model_id,
            temporal_inputs=dict(inputs),
        )
        self.initialize(temporary, rng)
        if temporary.scheduled_time is None:
            raise ValueError(f"Temporal model {model_id} did not produce a finite delay")
        return temporary.scheduled_time - activation_time, temporary.temporal_provenance

    def descriptors(self) -> list[Dict[str, Any]]:
        return [self._models[key].descriptor() for key in sorted(self._models)]


def build_standard_temporal_registry() -> TemporalModelRegistry:
    """Return a registry with only neutral baseline models.

    Domain packages should register fitted or explicitly labelled prior
    profiles with namespaced IDs.
    """

    return TemporalModelRegistry(
        [
            DeterministicDelayModel("core.immediate", 0.0),
            ScheduledTimeModel("core.scheduled"),
        ]
    )


def _hazard_due_time(candidate: Candidate, rate: float, now: float) -> float | None:
    if candidate.random_threshold is None:
        raise ValueError("Hazard threshold is missing")
    remaining = candidate.random_threshold - candidate.accumulated_hazard
    if remaining <= 1e-12:
        return now
    if rate <= 0.0:
        return None
    return now + remaining / rate


def _continuous_measure(
    elapsed: float,
    density: float,
    survival: float,
    *,
    support: list[float | None] | None = None,
) -> Dict[str, Any]:
    survival = min(1.0, max(0.0, float(survival)))
    density = max(0.0, float(density))
    hazard = density / survival if survival > 1e-300 else None
    result: Dict[str, Any] = {
        "clock_kind": "stochastic_continuous",
        "elapsed_seconds": float(elapsed),
        "density_per_second": density,
        "survival_probability": survival,
        "hazard_per_second": hazard,
        "cumulative_hazard": -math.log(max(survival, 1e-300)),
    }
    if support is not None:
        result["support_seconds"] = support
    return result


def _atom_measure(
    clock_kind: str, candidate: Candidate, at_time: float
) -> Dict[str, Any]:
    atom_time = candidate.scheduled_time
    is_due = atom_time is not None and abs(float(atom_time) - at_time) <= 1e-8
    return {
        "clock_kind": clock_kind,
        "elapsed_seconds": max(0.0, at_time - candidate.activation_time),
        "atom_at_offset_seconds": atom_time,
        "is_due_atom": is_due,
        "density_per_second": None,
        "survival_probability": 1.0 if atom_time is None or at_time < atom_time else 0.0,
        "hazard_per_second": None,
        "cumulative_hazard": None,
    }


def _weibull_cdf(value: float | None, shape: float, scale: float) -> float:
    if value is None:
        return 1.0
    if value <= 0.0:
        return 0.0
    return 1.0 - math.exp(-((value / scale) ** shape))


def _sample_truncated_gamma(
    rng: random.Random,
    shape: float,
    rate: float,
    minimum: float,
    maximum: float | None,
) -> float:
    if rate <= 0.0 or minimum < 0.0 or (
        maximum is not None and maximum <= minimum
    ):
        raise ValueError("Invalid truncated gamma parameters")
    for _ in range(100_000):
        value = rng.gammavariate(shape, 1.0 / rate)
        if value >= minimum and (maximum is None or value <= maximum):
            return value
    raise ValueError("Gamma truncation interval has negligible probability mass")


def _truncated_gamma_measure(
    elapsed: float,
    shape: float,
    rate: float,
    minimum: float,
    maximum: float | None,
) -> Dict[str, Any]:
    lower_cdf = _regularized_gamma_p(shape, rate * minimum)
    upper_cdf = (
        _regularized_gamma_p(shape, rate * maximum)
        if maximum is not None
        else 1.0
    )
    normalizer = upper_cdf - lower_cdf
    if normalizer <= 0.0:
        raise ValueError("Invalid truncated gamma support")
    if elapsed < minimum or elapsed <= 0.0:
        density = 0.0
        survival = 1.0
    elif maximum is not None and elapsed >= maximum:
        density = 0.0
        survival = 0.0
    else:
        log_density = (
            shape * math.log(rate)
            + (shape - 1.0) * math.log(elapsed)
            - rate * elapsed
            - math.lgamma(shape)
        )
        density = math.exp(log_density) / normalizer
        survival = max(
            0.0,
            (upper_cdf - _regularized_gamma_p(shape, rate * elapsed)) / normalizer,
        )
    return _continuous_measure(
        elapsed,
        density,
        survival,
        support=[minimum, maximum],
    )


def _regularized_gamma_p(shape: float, value: float) -> float:
    """Regularized lower incomplete gamma using standard series/continued fractions."""

    if shape <= 0.0:
        raise ValueError("Gamma shape must be positive")
    if value <= 0.0:
        return 0.0
    epsilon = 3e-14
    tiny = 1e-300
    max_iterations = 10_000
    log_factor = -value + shape * math.log(value) - math.lgamma(shape)
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        denominator = shape
        for _ in range(max_iterations):
            denominator += 1.0
            term *= value / denominator
            total += term
            if abs(term) <= abs(total) * epsilon:
                break
        return min(1.0, max(0.0, total * math.exp(log_factor)))

    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / max(b, tiny)
    fraction = d
    for index in range(1, max_iterations + 1):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            break
    upper = math.exp(log_factor) * fraction
    return min(1.0, max(0.0, 1.0 - upper))


def _descriptor(model: TemporalModel, parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": model.model_id,
        "family": model.family,
        "covariate_mode": model.covariate_mode.value,
        "model_basis": "standard_mathematical_model_family",
        "mathematical_form": _family_definition(model.family),
        "parameter_status": (
            "transparent_synthetic_prior_pending_empirical_fit"
            if ".prior_" in model.model_id
            else "configuration_or_policy_defined"
        ),
        "parameters": parameters,
    }


def _provenance(model: TemporalModel, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": model.model_id,
        "family": model.family,
        "covariate_mode": model.covariate_mode.value,
        "details": details,
    }


def _family_definition(family: str) -> str:
    return {
        "deterministic_delay": "delta_t = configured_delay",
        "scheduled_time": "event_time = configured_schedule_time",
        "conditional_lognormal": "log(delta_t) = beta_0 + beta^T x + sigma * epsilon",
        "conditional_gamma": "delta_t ~ Gamma(shape, rate=exp(beta_0 + beta^T x))",
        "conditional_weibull": "S(delta_t|x) = exp(-(delta_t/exp(beta_0 + beta^T x))^shape)",
        "empirical_delay": "delta_t is resampled from an empirical delay set",
        "exponential_arrival": "delta_t ~ Exponential(rate)",
        "piecewise_exponential_hazard": "lambda(t|x) = lambda_0 * exp(beta^T x) within each state interval",
        "exponential_backoff": "delta_t = min(maximum, base * multiplier^attempt) + jitter",
    }.get(family, "domain_registered_temporal_model")
