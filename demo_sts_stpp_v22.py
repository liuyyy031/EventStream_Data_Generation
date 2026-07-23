#!/usr/bin/env python3
"""STPP v22: audit schedule routes and arrival-family coverage before Agent 2."""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

import demo_sts_stpp_v17 as _v17
import demo_sts_stpp_v21 as _v21
from demo_sts_stpp_v18 import _pairs_from_chain_v18


_ORIGINAL_RAW_AUDIT = _v17.audit_raw_edge_fidelity_v17
_ORIGINAL_EDGE_FEEDBACK = _v17._edge_feedback  # noqa: SLF001
_ORIGINAL_WRITE_V21 = _v21._write_outputs_v21  # noqa: SLF001
_ROUTE_LINE = re.compile(
    r"EVENT\s+FULL\s+PATH|ARRIVAL\s+PATH|EDGES?\s+AFFECTED|"
    r"STAGE\s+\d+\s*:.*\bEDGE\b|EVENT\s+ID.*\bPATH\s*:",
    re.IGNORECASE,
)
_SECTION_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*([A-Z][A-Z\s]+?)"
    r"\s*:?\s*(?:\*\*|__)?\s*$"
)
_EVENT_ID = re.compile(r"EVENT\s+ID\s*:\s*([A-Za-z0-9_.-]+)", re.I)


def _normalised_heading(line: str) -> str | None:
    match = _SECTION_HEADING.match(line)
    if not match:
        return None
    return " ".join(match.group(1).upper().split())


def _section(scenario: str, heading: str) -> str:
    lines = scenario.splitlines()
    target = heading.upper()
    start = None
    for index, line in enumerate(lines):
        if _normalised_heading(line) == target:
            start = index + 1
            break
    if start is None:
        return ""
    end = len(lines)
    known = {
        "TIME AXIS",
        "NODES",
        "EDGES",
        "TEMPORAL PATTERNS",
        "PROPAGATED ARRIVALS",
        "EDGE MODULATION",
        "SPATIAL LAYOUT",
    }
    for index in range(start, len(lines)):
        current = _normalised_heading(lines[index])
        if current in known:
            end = index
            break
    return "\n".join(lines[start:end])


def _declared_route_edges(scenario: str) -> Set[Tuple[int, int]]:
    output: Set[Tuple[int, int]] = set()
    for line in scenario.splitlines():
        if _ROUTE_LINE.search(line):
            output.update(_pairs_from_chain_v18(line))
    return output


def _event_ids(section: str) -> Set[str]:
    return {match.group(1) for match in _EVENT_ID.finditer(section)}


def audit_raw_scenario_v22(
    scenario: str,
    structured_scenario: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Extend v20 edge fidelity to all scheduled routes and event families."""
    base = _ORIGINAL_RAW_AUDIT(scenario, structured_scenario)
    explicit = {
        (int(edge[0]), int(edge[1])) for edge in base.get("explicit_edges", [])
    }
    route_edges = _declared_route_edges(scenario)
    issues: List[Dict[str, Any]] = list(base.get("issues", []) or [])
    existing = {
        (str(issue.get("field")), str(issue.get("problem"))) for issue in issues
    }
    for source, target in sorted(route_edges - explicit):
        issue = {
            "field": f"scheduled route edge {source}->{target}",
            "problem": (
                "route/arrival/stage edge is absent from the top-level EDGES section"
            ),
            "error_source": "agent1",
        }
        key = (issue["field"], issue["problem"])
        if key not in existing:
            issues.append(issue)
            existing.add(key)

    scheduled_events = _event_ids(_section(scenario, "EDGE MODULATION"))
    arrival_events = _event_ids(_section(scenario, "PROPAGATED ARRIVALS"))
    for event_id in sorted(scheduled_events - arrival_events):
        issues.append(
            {
                "field": f"PROPAGATED ARRIVALS event_id={event_id}",
                "problem": (
                    "scheduled event family has no propagated-arrival declaration"
                ),
                "error_source": "agent1",
            }
        )
    for event_id in sorted(arrival_events - scheduled_events):
        issues.append(
            {
                "field": f"EDGE MODULATION event_id={event_id}",
                "problem": (
                    "propagated-arrival event family has no schedule declaration"
                ),
                "error_source": "agent1",
            }
        )

    output = dict(base)
    output.update(
        {
            "passed": not issues,
            "issues": issues,
            "declared_route_edges": [list(edge) for edge in sorted(route_edges)],
            "event_full_path_edges": [list(edge) for edge in sorted(route_edges)],
            "scheduled_event_ids": sorted(scheduled_events),
            "propagated_arrival_event_ids": sorted(arrival_events),
        }
    )
    return output


def _raw_scenario_feedback_v22(audit: Mapping[str, Any]) -> str:
    lines = [
        "Revise the same scenario without changing valid nodes, values, time "
        "span, sampling frequency, or spatial layout.",
        "Every edge used by EVENT FULL PATH, ARRIVAL PATH, schedule PATH, "
        "STAGE EDGE, or EDGES AFFECTED must appear in the top-level EDGES.",
        "Every EVENT ID in EDGE MODULATION must have a complete PROPAGATED "
        "ARRIVALS family with one record per schedule and receiving node.",
    ]
    for issue in audit.get("issues", []) or []:
        lines.append(f"- {issue.get('field')}: {issue.get('problem')}")
    lines.append("Return only the complete revised scenario.")
    return "\n".join(lines)


def _rename_v21_to_v22(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_path = old_path.with_name(
        old_path.name.replace("_stpp_v21_", "_stpp_v22_")
    )
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v22(*args, **kwargs) -> Dict[str, str | None]:
    files = _ORIGINAL_WRITE_V21(*args, **kwargs)
    renamed = {
        key: _rename_v21_to_v22(value) if key != "visualization" else value
        for key, value in files.items()
    }
    family = "source_consistent_schedule_aware_two_judge_stpp_v22"
    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = family
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v22"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = family
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v22"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Routed Schedule-Aware STPP v21",
        "STReasoner Source-Consistent Schedule-Aware STPP v22",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    original_audit = _v17.audit_raw_edge_fidelity_v17
    original_feedback = _v17._edge_feedback  # noqa: SLF001
    original_writer = _v21._write_outputs_v21  # noqa: SLF001
    added_output_argument = False
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "output_stpp_v22"])
        added_output_argument = True
    _v17.audit_raw_edge_fidelity_v17 = audit_raw_scenario_v22
    _v17._edge_feedback = _raw_scenario_feedback_v22  # noqa: SLF001
    _v21._write_outputs_v21 = _write_outputs_v22  # noqa: SLF001
    try:
        return _v21.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v21 "):
            message = "v22 " + message[len("v21 ") :]
        raise RuntimeError(message) from exc
    finally:
        _v17.audit_raw_edge_fidelity_v17 = original_audit
        _v17._edge_feedback = original_feedback  # noqa: SLF001
        _v21._write_outputs_v21 = original_writer  # noqa: SLF001
        if added_output_argument:
            del sys.argv[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
