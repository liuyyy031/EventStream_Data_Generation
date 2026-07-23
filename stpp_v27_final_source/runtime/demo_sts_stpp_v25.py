#!/usr/bin/env python3
"""STPP v25: reject cross-cycle routes and inconsistent lag units early."""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import demo_sts_stpp_v23 as _v23
import demo_sts_stpp_v24 as _v24
import demo_sts_stpp_v22 as _v22
from prompts.scenario_generation_agent_prompt_stpp_v25 import (
    STPP_V25_SCENARIO_GENERATION_PROMPT,
)


_ORIGINAL_RAW_AUDIT = _v23.audit_raw_scenario_v23
_ORIGINAL_RAW_FEEDBACK = _v23._raw_scenario_feedback_v23  # noqa: SLF001
_ORIGINAL_WRITE_V24 = _v24._write_outputs_v24  # noqa: SLF001
_SAMPLING_FREQUENCY = re.compile(
    r"SAMPLING\s+FREQUENCY\s*:\s*(?:\*\*|__)?\s*"
    r"(\d+(?:\.\d+)?)\s*"
    r"(minutes?|hours?|days?)\b",
    re.I,
)
_EDGE_LAG = re.compile(
    r"\bEDGE\s+(\d+)\s*->\s*(\d+)\s*:.*?"
    r"time_lag_steps\s*(?:=|:)\s*(\d+)",
    re.I,
)
_PHYSICAL_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*(minutes?|hours?|days?)\b",
    re.I,
)
_RANGE = re.compile(r"(?<!\d)(\d+)\s*-\s*(\d+)(?!\d)")
_REPEAT_PERIOD = re.compile(
    r"REPEAT\s+PERIOD\s*:\s*(?:\*\*|__)?\s*(\d+)", re.I
)
_TIME_FIELD = re.compile(r"\bTIME\s*:?[ \t]*(\d+)\s*-\s*(\d+)", re.I)
_AFFECTED_EDGE = re.compile(
    r"EDGES?\s+AFFECTED\s*:?[ \t]*\[?[ \t]*(\d+)\s*->\s*(\d+)",
    re.I,
)
_DESTINATION_WINDOW = re.compile(
    r"DESTINATION\s+ARRIVAL\s+WINDOW\s*:?[ \t]*(.+)$", re.I
)
_CROSS_CYCLE_LANGUAGE = re.compile(
    r"\bnext\s+day\b|\bnext\s+cycle\b|\bcross(?:es|ing)?\s+midnight\b",
    re.I,
)


def _unit_minutes(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit.startswith("minute"):
        return value
    if unit.startswith("hour"):
        return value * 60.0
    return value * 1440.0


def _sampling_minutes(scenario: str) -> float | None:
    match = _SAMPLING_FREQUENCY.search(scenario)
    if not match:
        return None
    return _unit_minutes(float(match.group(1)), match.group(2))


def _cross_cycle_issues(scenario: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    relevant = "\n".join(
        [
            _v22._section(scenario, "TEMPORAL PATTERNS"),  # noqa: SLF001
            _v22._section(scenario, "PROPAGATED ARRIVALS"),  # noqa: SLF001
            _v22._section(scenario, "EDGE MODULATION"),  # noqa: SLF001
        ]
    )
    for line_number, line in enumerate(relevant.splitlines(), 1):
        wrapping = [
            f"{start}-{end}"
            for start, end in _RANGE.findall(line)
            if int(start) > int(end)
        ]
        has_cross_language = bool(_CROSS_CYCLE_LANGUAGE.search(line))
        split_zero = bool(re.search(r"\band\s+0\s*-\s*\d+", line, re.I))
        if wrapping or has_cross_language or split_zero:
            detail = wrapping or ["cross-cycle wording"]
            issues.append(
                {
                    "field": f"cyclic window line {line_number}",
                    "problem": (
                        "event timing crosses the local-cycle boundary; move "
                        "the source window earlier so the full route ends by "
                        "repeat_period-1"
                    ),
                    "actual": detail,
                    "line": line.strip(),
                    "error_source": "agent1",
                }
            )
    return issues


def _lag_duration_issues(scenario: str) -> List[Dict[str, Any]]:
    sampling_minutes = _sampling_minutes(scenario)
    if sampling_minutes is None:
        return []
    issues: List[Dict[str, Any]] = []
    edges = _v22._section(scenario, "EDGES")  # noqa: SLF001
    for line in edges.splitlines():
        lag = _EDGE_LAG.search(line)
        if not lag:
            continue
        duration = _PHYSICAL_DURATION.search(line[lag.end() :])
        if not duration:
            issues.append(
                {
                    "field": f"EDGES {lag.group(1)}->{lag.group(2)}",
                    "problem": "edge has no parseable physical duration",
                    "error_source": "agent1",
                }
            )
            continue
        steps = int(lag.group(3))
        actual_minutes = _unit_minutes(
            float(duration.group(1)), duration.group(2)
        )
        expected_minutes = steps * sampling_minutes
        if not math.isclose(actual_minutes, expected_minutes, abs_tol=1e-9):
            issues.append(
                {
                    "field": f"EDGES {lag.group(1)}->{lag.group(2)}",
                    "problem": (
                        "physical duration does not equal time_lag_steps times "
                        "the sampling interval"
                    ),
                    "time_lag_steps": steps,
                    "sampling_minutes": sampling_minutes,
                    "expected_physical_minutes": expected_minutes,
                    "actual_physical_minutes": actual_minutes,
                    "error_source": "agent1",
                }
            )
    return issues


def _edge_lags(scenario: str) -> Dict[Tuple[int, int], int]:
    output: Dict[Tuple[int, int], int] = {}
    for line in _v22._section(scenario, "EDGES").splitlines():  # noqa: SLF001
        match = _EDGE_LAG.search(line)
        if match:
            output[(int(match.group(1)), int(match.group(2)))] = int(
                match.group(3)
            )
    return output


def _causal_window_issues(scenario: str) -> List[Dict[str, Any]]:
    period_match = _REPEAT_PERIOD.search(scenario)
    period = int(period_match.group(1)) if period_match else 24
    lags = _edge_lags(scenario)
    issues: List[Dict[str, Any]] = []
    section = _v22._section(scenario, "EDGE MODULATION")  # noqa: SLF001
    for line_number, line in enumerate(section.splitlines(), 1):
        time_match = _TIME_FIELD.search(line)
        edge_match = _AFFECTED_EDGE.search(line)
        arrival_match = _DESTINATION_WINDOW.search(line)
        if not (time_match and edge_match and arrival_match):
            continue
        edge = (int(edge_match.group(1)), int(edge_match.group(2)))
        lag = lags.get(edge)
        if lag is None:
            continue
        start, end = int(time_match.group(1)), int(time_match.group(2))
        expected = (start + lag, end + lag)
        field = f"EDGE MODULATION line {line_number} edge {edge[0]}->{edge[1]}"
        if expected[1] >= period:
            issues.append(
                {
                    "field": field,
                    "problem": (
                        "activation window plus edge lag crosses the local-cycle "
                        "boundary; move the source/stage window earlier"
                    ),
                    "time_period": [start, end],
                    "edge_lag_steps": lag,
                    "computed_destination": list(expected),
                    "repeat_period": period,
                    "error_source": "agent1",
                }
            )
            continue
        actual_ranges = [
            (int(left), int(right))
            for left, right in _RANGE.findall(arrival_match.group(1))
        ]
        if actual_ranges != [expected]:
            issues.append(
                {
                    "field": field,
                    "problem": (
                        "destination arrival window does not equal TIME shifted "
                        "by the affected edge lag"
                    ),
                    "expected_destination": list(expected),
                    "actual_destination_ranges": [
                        list(item) for item in actual_ranges
                    ],
                    "error_source": "agent1",
                }
            )
    return issues


def audit_raw_scenario_v25(
    scenario: str,
    structured_scenario: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Extend v23 raw fidelity with executable cyclic-time constraints."""
    base = _ORIGINAL_RAW_AUDIT(scenario, structured_scenario)
    issues: List[Dict[str, Any]] = list(base.get("issues", []) or [])
    existing = {
        (str(issue.get("field")), str(issue.get("problem"))) for issue in issues
    }
    timing_issues = (
        _cross_cycle_issues(scenario)
        + _lag_duration_issues(scenario)
        + _causal_window_issues(scenario)
    )
    for issue in timing_issues:
        key = (str(issue.get("field")), str(issue.get("problem")))
        if key not in existing:
            issues.append(issue)
            existing.add(key)
    output = dict(base)
    output.update(
        {
            "passed": not issues,
            "issues": issues,
            "raw_cyclic_timing_issues": timing_issues,
            "cyclic_route_semantics": "all event routes finish in one local cycle",
        }
    )
    return output


def _raw_scenario_feedback_v25(audit: Mapping[str, Any]) -> str:
    base = _ORIGINAL_RAW_FEEDBACK(audit).splitlines()
    if base and base[-1].strip() == "Return only the complete revised scenario.":
        base.pop()
    base.extend(
        [
            "Every CYCLIC_LOCAL route must finish by repeat_period-1. Do not "
            "write next-day, split, or wrapping arrival windows; move the "
            "source peak/window earlier.",
            "Make every physical edge duration equal to time_lag_steps times "
            "SAMPLING FREQUENCY.",
            "Return only the complete revised scenario.",
        ]
    )
    return "\n".join(base)


def _rename_v24_to_v25(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_path = old_path.with_name(
        old_path.name.replace("_stpp_v24_", "_stpp_v25_")
    )
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v25(*args, **kwargs) -> Dict[str, str | None]:
    files = _ORIGINAL_WRITE_V24(*args, **kwargs)
    renamed = {
        key: _rename_v24_to_v25(value) if key != "visualization" else value
        for key, value in files.items()
    }
    family = "nonwrapping_schedule_quality_compatible_stpp_v25"
    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = family
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v25"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = family
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v25"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Schedule-Quality-Compatible STPP v24",
        "STReasoner Non-Wrapping Schedule-Aware STPP v25",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    originals = {
        "audit": _v23.audit_raw_scenario_v23,
        "feedback": _v23._raw_scenario_feedback_v23,  # noqa: SLF001
        "prompt": _v23.STPP_V23_SCENARIO_GENERATION_PROMPT,
        "writer": _v24._write_outputs_v24,  # noqa: SLF001
    }
    added_output_argument = False
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "output_stpp_v25"])
        added_output_argument = True
    _v23.audit_raw_scenario_v23 = audit_raw_scenario_v25
    _v23._raw_scenario_feedback_v23 = _raw_scenario_feedback_v25  # noqa: SLF001
    _v23.STPP_V23_SCENARIO_GENERATION_PROMPT = (
        STPP_V25_SCENARIO_GENERATION_PROMPT
    )
    _v24._write_outputs_v24 = _write_outputs_v25  # noqa: SLF001
    try:
        return _v24.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v24 "):
            message = "v25 " + message[len("v24 ") :]
        raise RuntimeError(message) from exc
    finally:
        _v23.audit_raw_scenario_v23 = originals["audit"]
        _v23._raw_scenario_feedback_v23 = originals["feedback"]  # noqa: SLF001
        _v23.STPP_V23_SCENARIO_GENERATION_PROMPT = originals["prompt"]
        _v24._write_outputs_v24 = originals["writer"]  # noqa: SLF001
        if added_output_argument:
            del sys.argv[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
