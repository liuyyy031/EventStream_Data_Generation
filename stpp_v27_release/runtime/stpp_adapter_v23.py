"""Focused v23 diagnostics layered over the v19 schedule-aware contract."""

from __future__ import annotations

from typing import Any, List, Mapping


def _details(issue: Mapping[str, Any]) -> str:
    values: List[str] = []
    for key in (
        "event_id",
        "schedule_id",
        "expected_stages",
        "actual_stages",
        "duplicate_stages",
        "missing_days",
        "overlapping_days",
        "count",
    ):
        if key in issue and issue.get(key) not in (None, [], {}):
            values.append(f"{key}={issue.get(key)}")
    return f" ({', '.join(values)})" if values else ""


def format_contract_feedback_v23(audit: Mapping[str, Any]) -> str:
    """Expose stage metadata that v19 collected but omitted from feedback."""
    lines = [
        "Deterministic v23 schedule-aware validation failed. Fix the root "
        "contract fields without changing already valid nodes, edges, paths, "
        "values, time span, or spatial layout."
    ]
    for issue in audit.get("blocking_issues", []) or []:
        field = issue.get("field", issue.get("source_node", "unspecified"))
        problem = issue.get("problem", "contract violation")
        lines.append(f"- {field}: {problem}{_details(issue)}")
    lines.extend(
        [
            "One (event_id, schedule_id) is one source wave and must contain "
            "each path_stage exactly once.",
            "A schedule branch has exactly one positive source peak and one "
            "arrival per receiving node; never merge morning/evening waves.",
            "Schedule branches for one source must partition days 0..6; do "
            "not repair a duplicate wave by creating overlapping day branches.",
            "Uniqueness key: (event_id, schedule_id, receiving_node).",
        ]
    )
    return "\n".join(lines)
