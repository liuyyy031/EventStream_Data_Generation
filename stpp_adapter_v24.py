"""Schedule-aware quality-check compatibility for STPP v24."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from stpp_adapter_v16 import STPPV16Config
from stpp_adapter_v19 import (
    audit_structured_contract_v19,
    simulate_stpp_for_streasoner_v19,
)


_LEGACY_CHECK = "exactly_one_self_pattern_per_demand_source"


def _schedule_quality_audit(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
) -> Dict[str, Any]:
    """Derive v19-compatible source-pattern checks from the full contract."""
    contract = audit_structured_contract_v19(structured_scenario, seq_len)
    temporal = contract.get("temporal_contract_audit", {}) or {}
    pattern_issues = list(temporal.get("schedule_pattern_issues", []) or [])
    route_issues = list(contract.get("route_contract_issues", []) or [])
    return {
        "passed": not pattern_issues and not route_issues,
        "exactly_one_self_pattern_per_schedule_branch": not pattern_issues,
        "one_event_family_and_day_partition_per_demand_source": not route_issues,
        "schedule_pattern_issues": pattern_issues,
        "route_contract_issues": route_issues,
        "semantics": (
            "one positive source pattern per schedule branch; one event family "
            "with mutually exclusive day branches per demand source"
        ),
    }


def simulate_stpp_for_streasoner_v24(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV16Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Replace the obsolete v11 node-global self-pattern quality metric."""
    ts_data, events, metadata = simulate_stpp_for_streasoner_v19(
        structured_scenario, seq_len, config
    )
    schedule_audit = _schedule_quality_audit(structured_scenario, seq_len)
    quality = metadata["quality_report"]
    checks = quality["checks"]
    legacy_value = checks.pop(_LEGACY_CHECK, None)
    checks.update(
        {
            "exactly_one_self_pattern_per_schedule_branch": schedule_audit[
                "exactly_one_self_pattern_per_schedule_branch"
            ],
            "one_event_family_and_day_partition_per_demand_source": (
                schedule_audit[
                    "one_event_family_and_day_partition_per_demand_source"
                ]
            ),
        }
    )
    quality["passed"] = all(checks.values())
    quality["schedule_aware_self_pattern_audit"] = schedule_audit
    quality.setdefault("deprecated_checks", {})[_LEGACY_CHECK] = {
        "observed_value": legacy_value,
        "used_for_v24_pass_fail": False,
        "reason": (
            "v11 counted mutually exclusive weekday/weekend schedule branches "
            "as multiple independent source patterns"
        ),
    }
    metadata["simulator"] = "STPPG + schedule-aware quality semantics STPP v24"
    metadata["method"] = (
        "v19 branch-aware direct-route simulation + v24 schedule-aware "
        "source-pattern quality audit"
    )
    return ts_data, events, metadata
