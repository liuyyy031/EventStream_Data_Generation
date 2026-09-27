"""Additional domain-neutral clocks for recurrent, calendar, and resource time."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Dict, Iterable, Mapping, Sequence

from .models import Candidate, CovariateMode
from .temporal import _regularized_gamma_p


_HAZARD_MULTIPLIER = "_mechanism_hazard_multiplier"


@dataclass(frozen=True)
class PeriodicPiecewiseHazardModel:
    """A non-homogeneous arrival clock with repeating constant-rate bins."""

    model_id: str
    period_seconds: float
    rates_per_second: tuple[float, ...]
    phase_offset_seconds: float = 0.0
    family: str = "periodic_piecewise_hazard"
    covariate_mode: CovariateMode = CovariateMode.STATIC_AT_ACTIVATION

    def __post_init__(self) -> None:
        if self.period_seconds <= 0.0:
            raise ValueError("Periodic hazard period must be positive")
        if not self.rates_per_second or any(value < 0.0 for value in self.rates_per_second):
            raise ValueError("Periodic hazard rates must be non-empty and non-negative")

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        multiplier = _hazard_multiplier(candidate.temporal_inputs)
        threshold = -math.log(max(rng.random(), 1e-15))
        cycle_hazard = sum(self.rates_per_second) * (
            self.period_seconds / len(self.rates_per_second)
        )
        if cycle_hazard <= 0.0:
            due = None
        else:
            due = _solve_monotone_threshold(
                candidate.activation_time,
                threshold,
                lambda value: multiplier
                * self._integrated_hazard(candidate.activation_time, value),
                initial_span=max(
                    self.period_seconds,
                    threshold / max(multiplier * cycle_hazard / self.period_seconds, 1e-15),
                ),
            )
        candidate.scheduled_time = due
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "random_threshold": threshold,
                "period_seconds": self.period_seconds,
                "rate_bin_count": len(self.rates_per_second),
                "mechanism_hazard_multiplier": multiplier,
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "period_seconds": self.period_seconds,
                "rates_per_second": list(self.rates_per_second),
                "phase_offset_seconds": self.phase_offset_seconds,
            },
            "lambda(t) repeats a declared piecewise-constant rate profile",
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        elapsed = max(0.0, at_time - candidate.activation_time)
        multiplier = _hazard_multiplier(candidate.temporal_inputs)
        cumulative = multiplier * self._integrated_hazard(
            candidate.activation_time, at_time
        )
        hazard = multiplier * self._rate_at(at_time)
        survival = math.exp(-cumulative)
        result = _continuous_measure(elapsed, hazard * survival, survival)
        result["cumulative_hazard"] = cumulative
        result["mechanism_hazard_multiplier"] = multiplier
        return result

    def _rate_at(self, absolute_offset: float) -> float:
        bin_seconds = self.period_seconds / len(self.rates_per_second)
        phase = (absolute_offset + self.phase_offset_seconds) % self.period_seconds
        index = min(len(self.rates_per_second) - 1, int(phase / bin_seconds))
        return self.rates_per_second[index]

    def _primitive(self, absolute_offset: float) -> float:
        bin_seconds = self.period_seconds / len(self.rates_per_second)
        shifted = absolute_offset + self.phase_offset_seconds
        cycles = math.floor(shifted / self.period_seconds)
        remainder = shifted - cycles * self.period_seconds
        value = cycles * sum(self.rates_per_second) * bin_seconds
        full_bins = min(len(self.rates_per_second), int(remainder / bin_seconds))
        value += sum(self.rates_per_second[:full_bins]) * bin_seconds
        if full_bins < len(self.rates_per_second):
            value += self.rates_per_second[full_bins] * (
                remainder - full_bins * bin_seconds
            )
        return value

    def _integrated_hazard(self, start: float, end: float) -> float:
        if end <= start:
            return 0.0
        return max(0.0, self._primitive(end) - self._primitive(start))


@dataclass(frozen=True)
class SelfExcitingHazardModel:
    """Exponential-kernel Hawkes clock conditional on supplied event history."""

    model_id: str
    baseline_rate_per_second: float
    excitation_per_second: float
    decay_per_second: float
    history_input_key: str = "history_event_offsets"
    family: str = "self_exciting_hazard"
    covariate_mode: CovariateMode = CovariateMode.TIME_VARYING_HAZARD

    def __post_init__(self) -> None:
        if self.baseline_rate_per_second < 0.0 or self.excitation_per_second < 0.0:
            raise ValueError("Hawkes baseline and excitation must be non-negative")
        if self.decay_per_second <= 0.0:
            raise ValueError("Hawkes decay must be positive")

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        history = self._history(candidate)
        multiplier = _hazard_multiplier(candidate.temporal_inputs)
        threshold = -math.log(max(rng.random(), 1e-15))
        candidate.random_threshold = threshold
        candidate.accumulated_hazard = 0.0
        maximum_excitation = multiplier * sum(
            self.excitation_per_second
            / self.decay_per_second
            * math.exp(
                -self.decay_per_second * (candidate.activation_time - event_time)
            )
            for event_time in history
        )
        if self.baseline_rate_per_second <= 0.0 and threshold > maximum_excitation:
            due = None
        else:
            initial_span = (
                threshold / max(multiplier * self.baseline_rate_per_second, 1e-15)
                if self.baseline_rate_per_second > 0.0
                else 1.0 / self.decay_per_second
            )
            due = _solve_monotone_threshold(
                candidate.activation_time,
                threshold,
                lambda value: multiplier
                * self._integrated_hazard(
                    candidate.activation_time, value, history
                ),
                initial_span=initial_span,
                maximum_value=(
                    None
                    if self.baseline_rate_per_second > 0.0
                    else candidate.activation_time + 100.0 / self.decay_per_second
                ),
            )
        candidate.scheduled_time = due
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "random_threshold": threshold,
                "history_event_count": len(history),
                "mechanism_hazard_multiplier": multiplier,
                "history_is_fixed_at_activation": False,
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        previous = candidate.last_updated_at if candidate.last_updated_at is not None else candidate.activation_time
        if now < previous:
            raise ValueError("Hazard update cannot move backwards")
        candidate.accumulated_hazard += _hazard_multiplier(candidate.temporal_inputs) * self._integrated_hazard(previous, now, self._history(candidate))
        candidate.last_updated_at = now
        candidate.temporal_inputs = dict(inputs)
        history = self._history(candidate)
        multiplier = _hazard_multiplier(inputs)
        remaining = max(0.0, candidate.random_threshold - candidate.accumulated_hazard)
        total_tail = multiplier * sum(self.excitation_per_second / self.decay_per_second * math.exp(-self.decay_per_second * (now - t)) for t in history)
        if self.baseline_rate_per_second == 0 and remaining > total_tail:
            candidate.scheduled_time = None
        else:
            candidate.scheduled_time = _solve_monotone_threshold(now, remaining, lambda end: multiplier * self._integrated_hazard(now, end, history), initial_span=max(1.0, 1.0 / self.decay_per_second), maximum_value=None if self.baseline_rate_per_second > 0 else now + 100 / self.decay_per_second)

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "baseline_rate_per_second": self.baseline_rate_per_second,
                "excitation_per_second": self.excitation_per_second,
                "decay_per_second": self.decay_per_second,
                "history_input_key": self.history_input_key,
            },
            "lambda(t)=mu+sum(alpha*exp(-beta*(t-t_i))) for supplied t_i<t",
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        history = self._history(candidate)
        multiplier = _hazard_multiplier(candidate.temporal_inputs)
        cumulative = candidate.accumulated_hazard + multiplier * self._integrated_hazard(
            candidate.last_updated_at if candidate.last_updated_at is not None else candidate.activation_time, at_time, history
        )
        hazard = multiplier * self._rate_at(at_time, history)
        survival = math.exp(-cumulative)
        result = _continuous_measure(
            max(0.0, at_time - candidate.activation_time),
            hazard * survival,
            survival,
        )
        result["cumulative_hazard"] = cumulative
        result["history_event_count"] = len(history)
        result["mechanism_hazard_multiplier"] = multiplier
        return result

    def _history(self, candidate: Candidate) -> tuple[float, ...]:
        raw = candidate.temporal_inputs.get(self.history_input_key, ())
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"{self.history_input_key} must be an array of offsets")
        values = tuple(float(value) for value in raw)
        if any(value > max(candidate.activation_time, candidate.last_updated_at or 0.0) + 1e-9 for value in values):
            raise ValueError("Self-exciting history cannot contain future events")
        return values

    def _rate_at(self, at_time: float, history: Sequence[float]) -> float:
        return self.baseline_rate_per_second + sum(
            self.excitation_per_second
            * math.exp(-self.decay_per_second * (at_time - event_time))
            for event_time in history
            if event_time <= at_time
        )

    def _integrated_hazard(
        self, start: float, end: float, history: Sequence[float]
    ) -> float:
        if end <= start:
            return 0.0
        total = self.baseline_rate_per_second * (end - start)
        for event_time in history:
            lower = max(start, event_time)
            if lower >= end:
                continue
            total += self.excitation_per_second / self.decay_per_second * (
                math.exp(-self.decay_per_second * (lower - event_time))
                - math.exp(-self.decay_per_second * (end - event_time))
            )
        return total


@dataclass(frozen=True)
class RefractoryHazardModel:
    """A hard cooldown followed by an exponential event clock."""

    model_id: str
    cooldown_seconds: float
    baseline_rate_per_second: float
    last_event_input_key: str = "last_event_offset_seconds"
    family: str = "refractory_hazard"
    covariate_mode: CovariateMode = CovariateMode.TIME_VARYING_HAZARD

    def __post_init__(self) -> None:
        if self.cooldown_seconds < 0.0 or self.baseline_rate_per_second < 0.0:
            raise ValueError("Cooldown and refractory hazard rate must be non-negative")

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        active_from = self._active_from(candidate)
        rate = self.baseline_rate_per_second * _hazard_multiplier(
            candidate.temporal_inputs
        )
        candidate.random_threshold = -math.log(max(1.0-rng.random(), 1e-15))
        candidate.accumulated_hazard = 0.0
        delay = candidate.random_threshold / rate if rate > 0.0 else None
        candidate.hazard_active_from = active_from
        candidate.current_hazard_rate = rate
        candidate.scheduled_time = None if delay is None else active_from + delay
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "hazard_active_from_offset_seconds": active_from,
                "cooldown_seconds": self.cooldown_seconds,
                "rate_per_second": rate,
                "delay_after_cooldown_seconds": delay,
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        previous = candidate.last_updated_at if candidate.last_updated_at is not None else candidate.activation_time
        if now < previous:
            raise ValueError("Refractory hazard update cannot move backwards")
        candidate.accumulated_hazard += float(candidate.current_hazard_rate or 0) * max(0., now-max(previous, candidate.hazard_active_from))
        candidate.temporal_inputs = dict(inputs)
        candidate.last_updated_at = now
        candidate.hazard_active_from = self._active_from(candidate)
        rate = self.baseline_rate_per_second * _hazard_multiplier(inputs)
        candidate.current_hazard_rate = rate
        remaining = max(0., candidate.random_threshold-candidate.accumulated_hazard)
        candidate.scheduled_time = max(now, candidate.hazard_active_from) + remaining/rate if rate > 0 else None

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "cooldown_seconds": self.cooldown_seconds,
                "baseline_rate_per_second": self.baseline_rate_per_second,
                "last_event_input_key": self.last_event_input_key,
            },
            "lambda(t)=0 during cooldown, then lambda_0",
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        active_from = self._active_from(candidate)
        rate = self.baseline_rate_per_second * _hazard_multiplier(
            candidate.temporal_inputs
        )
        exposure = max(0.0, at_time - max(active_from, candidate.last_updated_at or candidate.activation_time))
        cumulative = candidate.accumulated_hazard + rate * exposure
        survival = math.exp(-cumulative)
        hazard = rate if at_time + 1e-12 >= active_from else 0.0
        result = _continuous_measure(
            max(0.0, at_time - candidate.activation_time),
            hazard * survival,
            survival,
            support=[active_from - candidate.activation_time, None],
        )
        result["hazard_active_from_offset_seconds"] = active_from
        result["cumulative_hazard"] = cumulative
        return result

    def _active_from(self, candidate: Candidate) -> float:
        last_event = float(
            candidate.temporal_inputs.get(
                self.last_event_input_key, candidate.activation_time
            )
        )
        return max(candidate.activation_time, last_event + self.cooldown_seconds)


@dataclass(frozen=True)
class LogNormalMixtureComponent:
    weight: float
    intercept: float
    sigma: float
    coefficients: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionalLogNormalMixtureModel:
    """A finite conditional log-normal mixture for multimodal delays."""

    model_id: str
    components: tuple[LogNormalMixtureComponent, ...]
    minimum_seconds: float = 0.0
    maximum_seconds: float | None = None
    family: str = "conditional_lognormal_mixture"
    covariate_mode: CovariateMode = CovariateMode.STATIC_AT_ACTIVATION

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("A log-normal mixture needs at least one component")
        if any(item.weight <= 0.0 or item.sigma <= 0.0 for item in self.components):
            raise ValueError("Mixture weights and component sigmas must be positive")
        if self.minimum_seconds < 0.0 or (
            self.maximum_seconds is not None
            and self.maximum_seconds <= self.minimum_seconds
        ):
            raise ValueError("Invalid mixture support")

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        lower = self._raw_cdf(self.minimum_seconds, candidate.temporal_inputs)
        upper = (
            self._raw_cdf(self.maximum_seconds, candidate.temporal_inputs)
            if self.maximum_seconds is not None
            else 1.0
        )
        if upper <= lower:
            raise ValueError("Log-normal mixture truncation has no probability mass")
        multiplier = _hazard_multiplier(candidate.temporal_inputs)
        probability = 1.0 - (1.0 - rng.random()) ** (1.0 / multiplier)
        target = lower + probability * (upper - lower)
        low = max(self.minimum_seconds, 1e-12)
        high = self.maximum_seconds or max(1.0, low * 2.0)
        while self._raw_cdf(high, candidate.temporal_inputs) < target:
            high *= 2.0
            if not math.isfinite(high):
                raise ValueError("Could not bracket log-normal mixture quantile")
        for _ in range(120):
            midpoint = (low + high) / 2.0
            if self._raw_cdf(midpoint, candidate.temporal_inputs) < target:
                low = midpoint
            else:
                high = midpoint
        delay = (low + high) / 2.0
        candidate.scheduled_time = candidate.activation_time + delay
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "delay_seconds": delay,
                "component_count": len(self.components),
                "mechanism_hazard_multiplier": multiplier,
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "components": [
                    {
                        "weight": item.weight,
                        "intercept": item.intercept,
                        "sigma": item.sigma,
                        "coefficients": dict(item.coefficients),
                    }
                    for item in self.components
                ],
                "minimum_seconds": self.minimum_seconds,
                "maximum_seconds": self.maximum_seconds,
            },
            "Delta t is drawn from a covariate-conditioned finite log-normal mixture",
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        elapsed = max(0.0, at_time - candidate.activation_time)
        lower = self._raw_cdf(self.minimum_seconds, candidate.temporal_inputs)
        upper = (
            self._raw_cdf(self.maximum_seconds, candidate.temporal_inputs)
            if self.maximum_seconds is not None
            else 1.0
        )
        normalizer = upper - lower
        if elapsed < self.minimum_seconds or elapsed <= 0.0:
            base_density, base_survival = 0.0, 1.0
        elif self.maximum_seconds is not None and elapsed >= self.maximum_seconds:
            base_density, base_survival = 0.0, 0.0
        else:
            base_density = self._raw_density(elapsed, candidate.temporal_inputs) / normalizer
            base_survival = max(
                0.0,
                (upper - self._raw_cdf(elapsed, candidate.temporal_inputs))
                / normalizer,
            )
        multiplier = _hazard_multiplier(candidate.temporal_inputs)
        if base_survival <= 0.0:
            density, survival = 0.0, 0.0
        else:
            survival = base_survival**multiplier
            density = (
                multiplier * base_density / base_survival * survival
            )
        result = _continuous_measure(
            elapsed,
            density,
            survival,
            support=[self.minimum_seconds, self.maximum_seconds],
        )
        result["component_count"] = len(self.components)
        result["mechanism_hazard_multiplier"] = multiplier
        return result

    def _weights(self) -> tuple[float, ...]:
        total = sum(item.weight for item in self.components)
        return tuple(item.weight / total for item in self.components)

    def _location(
        self, component: LogNormalMixtureComponent, inputs: Mapping[str, Any]
    ) -> float:
        return component.intercept + sum(
            float(weight) * float(inputs.get(name, 0.0))
            for name, weight in component.coefficients.items()
        )

    def _raw_cdf(self, value: float | None, inputs: Mapping[str, Any]) -> float:
        if value is None:
            return 1.0
        if value <= 0.0:
            return 0.0
        return sum(
            weight
            * NormalDist(
                self._location(component, inputs), component.sigma
            ).cdf(math.log(value))
            for weight, component in zip(self._weights(), self.components)
        )

    def _raw_density(self, value: float, inputs: Mapping[str, Any]) -> float:
        if value <= 0.0:
            return 0.0
        density = 0.0
        for weight, component in zip(self._weights(), self.components):
            location = self._location(component, inputs)
            z = (math.log(value) - location) / component.sigma
            density += weight * math.exp(-0.5 * z * z) / (
                value * component.sigma * math.sqrt(2.0 * math.pi)
            )
        return density


@dataclass(frozen=True)
class ResourceQueueDelayModel:
    """FCFS resource-availability delay plus a Gamma service clock."""

    model_id: str
    default_mean_service_seconds: float
    service_shape: float = 1.0
    family: str = "resource_queue_delay"
    covariate_mode: CovariateMode = CovariateMode.POLICY_DRIVEN

    def __post_init__(self) -> None:
        if self.default_mean_service_seconds <= 0.0 or self.service_shape <= 0.0:
            raise ValueError("Queue service mean and shape must be positive")

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        available = max(
            candidate.activation_time,
            float(
                candidate.temporal_inputs.get(
                    "resource_available_offset_seconds", candidate.activation_time
                )
            ),
        )
        queue_delay = max(
            0.0, float(candidate.temporal_inputs.get("queue_delay_seconds", 0.0))
        )
        service_start = available + queue_delay
        explicit_service = candidate.temporal_inputs.get("service_seconds")
        if explicit_service is None:
            mean = float(
                candidate.temporal_inputs.get(
                    "mean_service_seconds", self.default_mean_service_seconds
                )
            )
            if mean <= 0.0:
                raise ValueError("Queue mean service time must be positive")
            service_seconds = rng.gammavariate(
                self.service_shape, mean / self.service_shape
            )
            stochastic_service = True
        else:
            service_seconds = float(explicit_service)
            if service_seconds < 0.0:
                raise ValueError("Queue service time cannot be negative")
            mean = service_seconds
            stochastic_service = False
        candidate.scheduled_time = service_start + service_seconds
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "resource_available_offset_seconds": available,
                "queue_delay_seconds": queue_delay,
                "service_start_offset_seconds": service_start,
                "service_seconds": service_seconds,
                "mean_service_seconds": mean,
                "service_shape": self.service_shape,
                "stochastic_service": stochastic_service,
                "resource_state_is_domain_supplied": True,
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "default_mean_service_seconds": self.default_mean_service_seconds,
                "service_shape": self.service_shape,
                "required_temporal_inputs": [
                    "resource_available_offset_seconds or current activation",
                    "optional queue_delay_seconds",
                ],
            },
            "completion=max(activation,resource_available)+queue_delay+Gamma service time",
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        details = candidate.temporal_provenance.get("details", {})
        if not details.get("stochastic_service", False):
            result = _atom_measure("resource_policy_atom", candidate, at_time)
            result["resource_state_is_domain_supplied"] = True
            return result
        service_start = float(details["service_start_offset_seconds"])
        mean = float(details["mean_service_seconds"])
        service_elapsed = max(0.0, at_time - service_start)
        rate = self.service_shape / mean
        if at_time < service_start:
            density, survival = 0.0, 1.0
        elif service_elapsed <= 0.0:
            density, survival = 0.0, 1.0
        else:
            cdf = _regularized_gamma_p(self.service_shape, rate * service_elapsed)
            survival = max(0.0, 1.0 - cdf)
            density = math.exp(
                self.service_shape * math.log(rate)
                + (self.service_shape - 1.0) * math.log(service_elapsed)
                - rate * service_elapsed
                - math.lgamma(self.service_shape)
            )
        result = _continuous_measure(
            max(0.0, at_time - candidate.activation_time), density, survival
        )
        result["service_start_offset_seconds"] = service_start
        result["resource_state_is_domain_supplied"] = True
        return result


@dataclass(frozen=True)
class AvailabilityWindowDelayModel:
    """Consume required working time only inside declared availability windows."""

    model_id: str
    default_available_intervals: tuple[tuple[float, float], ...] = ()
    family: str = "availability_window_delay"
    covariate_mode: CovariateMode = CovariateMode.POLICY_DRIVEN

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        del rng
        required = float(candidate.temporal_inputs.get("required_work_seconds", 0.0))
        if required < 0.0:
            raise ValueError("Required working time cannot be negative")
        raw_intervals = candidate.temporal_inputs.get(
            "available_intervals", self.default_available_intervals
        )
        intervals = _normalize_intervals(raw_intervals)
        due = _consume_available_time(candidate.activation_time, required, intervals)
        candidate.scheduled_time = due
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "required_work_seconds": required,
                "available_intervals": [list(item) for item in intervals],
                "insufficient_calendar_horizon": due is None,
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {
                "default_available_intervals": [
                    list(item) for item in self.default_available_intervals
                ],
                "required_temporal_inputs": ["required_work_seconds"],
            },
            "working time accumulates only inside domain-supplied availability intervals",
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        return _atom_measure("calendar_policy_atom", candidate, at_time)


@dataclass(frozen=True)
class IntervalTimeModel:
    """Sample a latent occurrence inside an explicitly uncertain time interval."""

    model_id: str
    selection_policy: str = "uniform"
    family: str = "interval_time"
    covariate_mode: CovariateMode = CovariateMode.POLICY_DRIVEN

    def __post_init__(self) -> None:
        if self.selection_policy not in {"uniform", "midpoint"}:
            raise ValueError("Interval selection policy must be uniform or midpoint")

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        earliest = max(
            candidate.activation_time,
            float(candidate.temporal_inputs["earliest_offset_seconds"]),
        )
        latest = float(candidate.temporal_inputs["latest_offset_seconds"])
        if latest < earliest:
            raise ValueError("Interval latest time precedes earliest time")
        if self.selection_policy == "uniform" and latest > earliest:
            due = rng.uniform(earliest, latest)
        else:
            due = (earliest + latest) / 2.0
        candidate.scheduled_time = due
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {
                "earliest_offset_seconds": earliest,
                "latest_offset_seconds": latest,
                "selection_policy": self.selection_policy,
                "latent_occurrence_sampled_within_reported_interval": True,
            },
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {},
            "latent occurrence is uniform or midpoint within [earliest,latest]",
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        details = candidate.temporal_provenance.get("details", {})
        earliest = float(details["earliest_offset_seconds"])
        latest = float(details["latest_offset_seconds"])
        if self.selection_policy == "midpoint" or latest == earliest:
            return _atom_measure("interval_midpoint_atom", candidate, at_time)
        if at_time < earliest:
            density, survival = 0.0, 1.0
        elif at_time >= latest:
            density, survival = 0.0, 0.0
        else:
            width = latest - earliest
            density = 1.0 / width
            survival = (latest - at_time) / width
        result = _continuous_measure(
            max(0.0, at_time - candidate.activation_time),
            density,
            survival,
            support=[
                earliest - candidate.activation_time,
                latest - candidate.activation_time,
            ],
        )
        result["reported_interval_offsets"] = [earliest, latest]
        return result


@dataclass(frozen=True)
class BatchScheduledTimeModel:
    """A point-mass clock for events declared to share one batch timestamp."""

    model_id: str
    family: str = "batch_scheduled_time"
    covariate_mode: CovariateMode = CovariateMode.DETERMINISTIC_SCHEDULE

    def initialize(self, candidate: Candidate, rng: random.Random) -> None:
        del rng
        due = float(
            candidate.temporal_inputs.get(
                "batch_offset_seconds",
                candidate.temporal_inputs.get("scheduled_offset_seconds"),
            )
        )
        if due < candidate.activation_time:
            raise ValueError("Batch time precedes candidate activation")
        batch_id = candidate.temporal_inputs.get("batch_id")
        if batch_id is None or not str(batch_id).strip():
            raise ValueError("Batch clock requires a non-empty batch_id")
        candidate.scheduled_time = due
        candidate.last_updated_at = candidate.activation_time
        candidate.temporal_provenance = _provenance(
            self,
            {"batch_offset_seconds": due, "batch_id": str(batch_id)},
        )

    def update(self, candidate: Candidate, now: float, inputs: Mapping[str, Any]) -> None:
        del candidate, now, inputs

    def descriptor(self) -> Dict[str, Any]:
        return _descriptor(
            self,
            {"required_temporal_inputs": ["batch_id", "batch_offset_seconds"]},
            "all candidates sharing a batch ID and offset have the same point-mass time",
        )

    def measure(self, candidate: Candidate, at_time: float) -> Dict[str, Any]:
        result = _atom_measure("batch_atom", candidate, at_time)
        result["batch_id"] = candidate.temporal_provenance.get("details", {}).get(
            "batch_id"
        )
        return result


def _solve_monotone_threshold(
    start: float,
    threshold: float,
    cumulative_hazard,
    *,
    initial_span: float,
    maximum_value: float | None = None,
) -> float | None:
    low = start
    high = start + max(initial_span, 1e-9)
    if maximum_value is not None:
        high = min(high, maximum_value)
    while cumulative_hazard(high) < threshold:
        if maximum_value is not None and high >= maximum_value:
            return None
        high = start + 2.0 * (high - start)
        if maximum_value is not None:
            high = min(high, maximum_value)
        if not math.isfinite(high):
            return None
    for _ in range(120):
        midpoint = (low + high) / 2.0
        if cumulative_hazard(midpoint) < threshold:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _hazard_multiplier(inputs: Mapping[str, Any]) -> float:
    multiplier = float(inputs.get(_HAZARD_MULTIPLIER, 1.0))
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("Mechanism hazard multiplier must be finite and positive")
    return multiplier


def _normalize_intervals(raw: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("available_intervals must be an array of [start,end] pairs")
    intervals: list[tuple[float, float]] = []
    for value in raw:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Each availability interval must contain start and end")
        start, end = float(value[0]), float(value[1])
        if end <= start:
            raise ValueError("Availability interval end must follow start")
        intervals.append((start, end))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _consume_available_time(
    activation: float,
    required: float,
    intervals: Sequence[tuple[float, float]],
) -> float | None:
    if required <= 0.0:
        return activation
    remaining = required
    for start, end in intervals:
        cursor = max(activation, start)
        if cursor >= end:
            continue
        usable = end - cursor
        if usable >= remaining:
            return cursor + remaining
        remaining -= usable
    return None


def _continuous_measure(
    elapsed: float,
    density: float,
    survival: float,
    *,
    support: list[float | None] | None = None,
) -> Dict[str, Any]:
    survival = min(1.0, max(0.0, float(survival)))
    density = max(0.0, float(density))
    return {
        "clock_kind": "stochastic_continuous",
        "elapsed_seconds": elapsed,
        "density_per_second": density,
        "survival_probability": survival,
        "hazard_per_second": density / survival if survival > 1e-300 else None,
        "support_seconds": support,
    }


def _atom_measure(
    clock_kind: str, candidate: Candidate, at_time: float
) -> Dict[str, Any]:
    atom_time = candidate.scheduled_time
    return {
        "clock_kind": clock_kind,
        "elapsed_seconds": max(0.0, at_time - candidate.activation_time),
        "atom_at_offset_seconds": atom_time,
        "is_due_atom": atom_time is not None
        and abs(float(atom_time) - at_time) <= 1e-8,
        "density_per_second": None,
        "survival_probability": (
            1.0 if atom_time is None or at_time < atom_time else 0.0
        ),
        "hazard_per_second": None,
        "cumulative_hazard": None,
    }


def _descriptor(model: Any, parameters: Dict[str, Any], mathematical_form: str) -> Dict[str, Any]:
    return {
        "model_id": model.model_id,
        "family": model.family,
        "covariate_mode": model.covariate_mode.value,
        "model_basis": "standard_mathematical_model_family",
        "mathematical_form": mathematical_form,
        "parameter_status": (
            "transparent_synthetic_prior_pending_empirical_fit"
            if ".prior_" in model.model_id
            else "configuration_or_policy_defined"
        ),
        "parameters": parameters,
    }


def _provenance(model: Any, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": model.model_id,
        "family": model.family,
        "covariate_mode": model.covariate_mode.value,
        "details": details,
    }
