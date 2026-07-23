"""Day-aware source/propagated time-conflict quality audit for STPP v27."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Set, Tuple

import numpy as np

from stpp_adapter_v11 import _pattern_peaks, _time_intervals
from stpp_adapter_v16 import STPPV16Config
from stpp_adapter_v24 import simulate_stpp_for_streasoner_v24


_LEGACY_CHECK = "no_self_generated_propagated_time_conflicts"


def _days(value: Any) -> Set[int]:
    try:
        parsed = {int(day) for day in (value or [])}
    except (TypeError, ValueError):
        parsed = set()
    # Missing DAYS cannot prove mutual exclusion, so treat it conservatively as
    # every day. The v19 contract normally makes this fallback unnecessary.
    return parsed or set(range(7))


def _schedule_aware_conflict_audit(
    structured_scenario: Mapping[str, Any],
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    node_types = {
        int(node["id"]): str(node.get("type", "demand_source")).lower()
        for node in structured_scenario.get("nodes", []) or []
    }
    conflicts: List[Dict[str, Any]] = []
    drift_nodes = (
        structured_scenario.get("drift_patterns", {}).get("nodes", []) or []
    )
    for node_data in drift_nodes:
        node_id = int(node_data.get("id", -1))
        if node_types.get(node_id) != "demand_source":
            continue
        self_patterns: List[Tuple[int, Mapping[str, Any]]] = []
        for pattern_index, pattern in enumerate(node_data.get("patterns", []) or []):
            amplitude = float(pattern.get("amplitude", 0) or 0)
            origin = str(pattern.get("origin", "")).lower()
            behavior = str(pattern.get("behavior", "")).lower()
            if (
                amplitude > 0
                and _pattern_peaks(pattern)
                and origin != "propagated"
                and behavior != "propagated"
            ):
                self_patterns.append((pattern_index, pattern))

        variations = node_data.get("propagated_variations", []) or []
        for pattern_index, pattern in self_patterns:
            pattern_days = _days(pattern.get("days"))
            for peak in _pattern_peaks(pattern):
                for variation_index, variation in enumerate(variations):
                    shared_days = sorted(
                        pattern_days.intersection(_days(variation.get("days")))
                    )
                    if not shared_days:
                        continue
                    intervals = _time_intervals(
                        variation.get(
                            "time",
                            variation.get(
                                "arrival_steps", variation.get("time_range")
                            ),
                        )
                    )
                    for start, end in intervals:
                        if start - tolerance <= peak <= end + tolerance:
                            conflicts.append(
                                {
                                    "node_id": node_id,
                                    "self_pattern_index": pattern_index,
                                    "self_schedule_id": pattern.get("schedule_id"),
                                    "self_peak": peak,
                                    "propagated_variation_index": variation_index,
                                    "propagated_event_id": variation.get("event_id"),
                                    "propagated_schedule_id": variation.get(
                                        "schedule_id"
                                    ),
                                    "propagated_time": variation.get("time"),
                                    "shared_days": shared_days,
                                    "parsed_interval": [start, end],
                                }
                            )
    return {
        "passed": not conflicts,
        "conflicts": conflicts,
        "semantics": "time overlap is blocking only when DAYS also intersect",
    }


def simulate_stpp_for_streasoner_v27(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV16Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Replace v11's day-agnostic conflict metric after v24 simulation."""
    ts_data, events, metadata = simulate_stpp_for_streasoner_v24(
        structured_scenario, seq_len, config
    )
    conflict_audit = _schedule_aware_conflict_audit(structured_scenario)
    quality = metadata["quality_report"]
    checks = quality["checks"]
    legacy_value = checks.pop(_LEGACY_CHECK, None)
    checks["no_same_day_self_generated_propagated_time_conflicts"] = (
        conflict_audit["passed"]
    )
    quality["passed"] = all(checks.values())
    quality["schedule_aware_self_propagated_conflict_audit"] = conflict_audit
    quality.setdefault("deprecated_checks", {})[_LEGACY_CHECK] = {
        "observed_value": legacy_value,
        "used_for_v27_pass_fail": False,
        "reason": (
            "v11 compared local times without DAYS and therefore treated "
            "mutually exclusive weekday/weekend branches as conflicts"
        ),
    }
    metadata["simulator"] = "STPPG + day-aware schedule semantics STPP v27"
    metadata["method"] = (
        "v24 schedule-quality simulation + v27 same-day source/propagated "
        "conflict audit"
    )
    return ts_data, events, metadata
