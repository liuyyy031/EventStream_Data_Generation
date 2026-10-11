"""Exact constant-forcing one-box anomaly response, in SI units."""
import math


def advance_temperature(temperature, elapsed_seconds, forcing, feedback, capacity):
    """Return T, integral(F dt), integral(lambda T dt), all per unit area.

    C dT/dt = F - lambda T. Positive feedback parameter denotes damping.
    The cumulative energy terms can be negative; this is an anomaly budget.
    """
    if not (all(math.isfinite(v) for v in (temperature, elapsed_seconds, forcing, feedback, capacity))
            and elapsed_seconds >= 0 and feedback > 0 and capacity > 0):
        raise ValueError("Invalid constant-forcing climate segment")
    tau, equilibrium = capacity / feedback, forcing / feedback
    response = -math.expm1(-elapsed_seconds / tau)
    end = temperature + (equilibrium - temperature) * response
    forced_energy = forcing * elapsed_seconds
    # Integrating the ODE avoids subtracting nearly equal time integrals.
    feedback_energy = forced_energy - capacity * (end - temperature)
    return end, forced_energy, feedback_energy


def time_to_temperature(temperature, target, forcing, feedback, capacity):
    """First finite crossing time, or None for unreachable/asymptotic targets."""
    if not (all(math.isfinite(v) for v in (temperature, target, forcing, feedback, capacity))
            and feedback > 0 and capacity > 0):
        raise ValueError("Invalid climate crossing inputs")
    if abs(target - temperature) <= 1e-12:
        return 0.0
    equilibrium = forcing / feedback
    denominator = temperature - equilibrium
    ratio = (target - equilibrium) / denominator if denominator else -1.0
    return -(capacity / feedback) * math.log(ratio) if 0 < ratio < 1 else None
