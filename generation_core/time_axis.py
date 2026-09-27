"""Explicit scientific time axes with a seconds-compatible runtime bridge.

Calendar years are not durations. The fixed year below is the Julian duration
365.25 days, explicitly declared for relative scientific axes only.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TimeAxis:
    kind: str = "calendar"
    unit: str = "second"
    origin_label: str = "episode start"
    resolution: float = 0.001
    year_definition: str = "julian_365.25_days"

    def __post_init__(self):
        if self.kind not in {"calendar", "relative", "deep_time"}:
            raise ValueError("Unsupported time axis kind")
        if self.unit not in self.scales():
            raise ValueError("Unsupported time unit")
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("Time resolution must be finite and positive")
        if self.kind == "calendar" and self.unit in {"year", "kyr", "Myr"}:
            raise ValueError("Calendar years need a calendar policy, not a fixed conversion")
        if self.year_definition != "julian_365.25_days":
            raise ValueError("Scientific year duration must be explicitly Julian")

    @staticmethod
    def scales():
        return {"second": 1., "millisecond": .001, "minute": 60.,
                "hour": 3600., "day": 86400., "year": 31557600.,
                "kyr": 31557600000., "Myr": 31557600000000.}

    @property
    def scale(self):
        return self.scales()[self.unit]

    def to_seconds(self, value):
        result = float(value) * self.scale
        if not math.isfinite(result):
            raise ValueError("Non-finite time")
        return result

    def from_seconds(self, value):
        return None if value is None else float(value) / self.scale

    def rate_to_per_second(self, value):
        rate = float(value)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("Rates must be finite and non-negative")
        return rate / self.scale

    def descriptor(self):
        return {**asdict(self), "direction": "forward", "runtime_unit": "second",
                "seconds_per_unit": self.scale}


def sample_duration(policy, default_seconds, rng):
    """Sample a horizon before simulation, never a target event count."""
    if not policy:
        return float(default_seconds)
    axis = TimeAxis(kind="relative", unit=policy.get("unit", "second"))
    family = policy.get("family", "fixed")
    if family == "fixed":
        value = policy["value"]
    elif family == "uniform":
        low, high = float(policy["minimum"]), float(policy["maximum"])
        if not 0 < low <= high:
            raise ValueError("Duration bounds must satisfy 0 < minimum <= maximum")
        value = rng.uniform(low, high)
    elif family == "empirical":
        values = policy["values"]
        if not values or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
            raise ValueError("Duration samples must be finite and positive")
        value = rng.choice(values)
    else:
        raise ValueError("Duration family must be fixed, uniform, or empirical")
    seconds = axis.to_seconds(value)
    if seconds <= 0:
        raise ValueError("Observation duration must be positive")
    return seconds
