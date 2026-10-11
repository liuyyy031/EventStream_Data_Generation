"""Analytic FAO-56-style single-coefficient water balance between event boundaries.

Rates are mm/day, depth is mm, and time is days. Temperature and crop coefficient
are constant inside a segment; event-driven changes create a new segment.
"""
import math


def stress_coefficient(depletion, taw, raw):
    return min(1.0, max(0.0, (taw - depletion) / (taw - raw)))


def time_to_depletion(depletion, target, potential_et, irrigation_rate, taw, raw):
    """First crossing time, or None when target is unreachable under current rates."""
    if not (all(math.isfinite(v) for v in (depletion, target, potential_et, irrigation_rate, taw, raw))
            and 0 <= target < taw and 0 <= depletion <= taw and 0 < raw < taw
            and potential_et >= 0 and irrigation_rate >= 0):
        raise ValueError("Depletion targets must be inside available root-zone water")
    if abs(target - depletion) <= 1e-10:
        return 0.0
    if depletion < raw < target or target < raw < depletion:
        first = time_to_depletion(depletion, raw, potential_et, irrigation_rate, taw, raw)
        second = time_to_depletion(raw, target, potential_et, irrigation_rate, taw, raw)
        return None if first is None or second is None else first + second
    if max(depletion, target) <= raw or potential_et == 0:
        slope = potential_et - irrigation_rate if max(depletion, target) <= raw else -irrigation_rate
        elapsed = (target - depletion) / slope if slope else -1.0
        return elapsed if elapsed >= 0 else None
    k = potential_et / (taw - raw)
    equilibrium = taw - irrigation_rate / k
    denominator = depletion - equilibrium
    ratio = (target - equilibrium) / denominator if denominator else -1.0
    return -math.log(ratio) / k if 0 < ratio < 1 else None


def advance_water(depletion, elapsed_days, potential_et, irrigation_rate, taw, raw):
    """Return final depletion, actual ET and excess deep percolation.

    dD/dt = potential_ET * Ks(D) - irrigation_rate, with D in [0, TAW].
    The solution is linear above RAW water availability and exponential in
    the water-limited regime. Saturation overflow is recorded, not discarded.
    """
    if not (all(math.isfinite(v) for v in (depletion, elapsed_days, potential_et, irrigation_rate, taw, raw))
            and 0 <= depletion <= taw and 0 < raw < taw and elapsed_days >= 0 and potential_et >= 0 and irrigation_rate >= 0):
        raise ValueError("Invalid root-zone water segment")
    initial, remaining, drainage = depletion, elapsed_days, 0.0
    for _ in range(4):
        if remaining <= 1e-12:
            break
        derivative = potential_et * stress_coefficient(depletion, taw, raw) - irrigation_rate
        if depletion <= 1e-10 and derivative <= 0:
            drainage += (irrigation_rate - potential_et) * remaining
            depletion, remaining = 0.0, 0.0
        elif depletion < raw - 1e-10 or abs(depletion - raw) <= 1e-10 and derivative <= 0:
            slope = potential_et - irrigation_rate
            boundary = raw if slope > 0 else 0.0
            until = (boundary - depletion) / slope if slope else math.inf
            step = min(remaining, max(0.0, until))
            depletion += slope * step
            remaining -= step
            if remaining and step <= 1e-12 and slope > 0:
                depletion = raw
        else:
            if potential_et == 0:
                step = min(remaining, (depletion - raw) / irrigation_rate if irrigation_rate else remaining)
                depletion -= irrigation_rate * step
            else:
                k = potential_et / (taw - raw)
                equilibrium = taw - irrigation_rate / k
                until = time_to_depletion(depletion, raw, potential_et, irrigation_rate, taw, raw) if derivative < 0 else None
                step = min(remaining, until if until is not None else remaining)
                depletion = equilibrium + (depletion - equilibrium) * math.exp(-k * step)
            remaining -= step
    if remaining > 1e-8:
        raise ArithmeticError("Water segment failed to cross its analytic regimes")
    depletion = min(taw, max(0.0, depletion))
    actual_et = depletion - initial + irrigation_rate * elapsed_days - drainage
    return depletion, max(0.0, actual_et), max(0.0, drainage)


def thermal_rate(mean_temperature_c, base_temperature_c, upper_temperature_c):
    """Capped mean-temperature thermal-time proxy, degree-days per day."""
    return max(0.0, min(mean_temperature_c, upper_temperature_c) - base_temperature_c)
