"""In-memory checkpoints and conditional stochastic continuations, not interventions."""
from dataclasses import dataclass
from typing import Any
import math


@dataclass
class SimulationSnapshot:
    state: dict[str, Any]
    model_descriptors: list[dict]
    domain: str


def resample_surviving_clock(candidate, model, now, end_time, rng):
    """Invert S(t)/S(now) for continuous clocks, retaining age and covariates.

    Policy/discrete appointments remain committed. This intentionally does not
    redraw hidden queue reservations, discrete policy choices or external plans.
    """
    origin = max(now, candidate.activation_time)
    measure = model.measure(candidate, origin)
    if measure.get("clock_kind") != "stochastic_continuous":
        candidate.provenance["continuation_clock_policy"] = "retain_committed_discrete_or_policy_clock"
        return
    survival = float(measure.get("survival_probability", 1.0))
    if survival <= 1e-290:
        raise ValueError("Conditional survival is numerically exhausted; cannot safely fork this clock")
    threshold = -math.log(max(rng.random(), 1e-15))
    previous_due = candidate.scheduled_time
    target = survival * math.exp(-threshold)
    if candidate.random_threshold is not None:
        candidate.random_threshold = float(measure["cumulative_hazard"]) + threshold
    if float(model.measure(candidate, end_time).get("survival_probability", 1.0)) > target:
        # Dynamic models may become active again on a subsequent history update.
        candidate.scheduled_time = None
    else:
        low, high = origin, end_time
        for _ in range(90):
            mid = low + (high-low) / 2
            if float(model.measure(candidate, mid)["survival_probability"]) > target:
                low = mid
            else:
                high = mid
        candidate.scheduled_time = high
    candidate.provenance["continuation_clock_policy"] = "conditional_survival_resampling"
    candidate.provenance["continuation_conditioned_at_seconds"] = origin
    candidate.temporal_provenance["continuation_resample"] = {
        "previous_scheduled_time_seconds": previous_due,
        "conditioned_at_seconds": origin,
        "conditional_exponential_threshold": threshold,
        "scheduled_time_seconds": candidate.scheduled_time,
        "activation_sample_retained_for_audit_only": True,
    }
    candidate.revision += 1
