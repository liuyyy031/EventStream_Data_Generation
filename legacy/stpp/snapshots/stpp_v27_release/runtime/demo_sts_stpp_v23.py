#!/usr/bin/env python3
"""STPP v23: single-wave raw audit and sampling-rate normalisation."""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import demo_sts_stpp_v19 as _v19
import demo_sts_stpp_v21 as _v21
import demo_sts_stpp_v22 as _v22
from demo_sts_sde import NetworkSDEGenerator
from prompts.scenario_generation_agent_prompt_stpp_v23 import (
    STPP_V23_SCENARIO_GENERATION_PROMPT,
)
from stpp_adapter_v23 import format_contract_feedback_v23


_ORIGINAL_RAW_AUDIT = _v22.audit_raw_scenario_v22
_ORIGINAL_WRITE_V22 = _v22._write_outputs_v22  # noqa: SLF001
_ORIGINAL_SEQUENCE_LENGTH = NetworkSDEGenerator._calculate_sequence_length
_EVENT_ID = re.compile(r"EVENT\s+ID\s*:\s*([A-Za-z0-9_.-]+)", re.I)
_SCHEDULE_ID = re.compile(r"SCHEDULE\s+ID\s*:\s*([A-Za-z0-9_.-]+)", re.I)
_PATH_STAGE = re.compile(r"PATH\s+STAGE\s*:\s*(-?\d+)", re.I)
_RECEIVING_NODE = re.compile(r"RECEIVING\s+NODE\s*:\s*(\d+)", re.I)
_ARRIVAL_TIME = re.compile(r"ARRIVAL\s+TIME\s+TO\s+NODE\s*:\s*(.+)$", re.I)
_PEAK = re.compile(r"\bPEAK\s+(?:AT\b|=)", re.I)
_RATE_FREQUENCY = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:samples?|observations?|points?)\s*"
    r"(?:per|/)\s*(minute|hour|day)s?\b",
    re.I,
)
_ONE_RATE_FREQUENCY = re.compile(
    r"\bone\s+(?:sample|observation|point)\s+(?:per|/)\s*"
    r"(minute|hour|day)s?\b",
    re.I,
)


def _append_issue(
    issues: List[Dict[str, Any]],
    existing: set[Tuple[str, str]],
    issue: Dict[str, Any],
) -> None:
    key = (str(issue.get("field")), str(issue.get("problem")))
    if key not in existing:
        issues.append(issue)
        existing.add(key)


def _modulation_wave_issues(scenario: str) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str, int]] = Counter()
    for line in _v22._section(scenario, "EDGE MODULATION").splitlines():  # noqa: SLF001
        event = _EVENT_ID.search(line)
        schedule = _SCHEDULE_ID.search(line)
        stage = _PATH_STAGE.search(line)
        if event and schedule and stage:
            counts[(event.group(1), schedule.group(1), int(stage.group(1)))] += 1

    issues: List[Dict[str, Any]] = []
    for (event_id, schedule_id, path_stage), count in sorted(counts.items()):
        if count > 1:
            issues.append(
                {
                    "field": (
                        f"EDGE MODULATION event_id={event_id}, "
                        f"schedule_id={schedule_id}, path_stage={path_stage}"
                    ),
                    "problem": (
                        "one schedule branch repeats a path stage; retain one "
                        "dominant source wave instead of morning/evening waves"
                    ),
                    "count": count,
                    "error_source": "agent1",
                }
            )
    return issues


def _temporal_peak_issues(scenario: str) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    current_schedule: str | None = None
    for line in _v22._section(scenario, "TEMPORAL PATTERNS").splitlines():  # noqa: SLF001
        schedule = _SCHEDULE_ID.search(line)
        if schedule:
            current_schedule = schedule.group(1)
            continue
        if re.search(r"\bNODE\s+\d+", line, re.I):
            current_schedule = None
        elif current_schedule and _PEAK.search(line):
            counts[current_schedule] += 1

    return [
        {
            "field": f"TEMPORAL PATTERNS schedule_id={schedule_id}",
            "problem": (
                "one schedule branch declares multiple positive source peaks; "
                "retain one dominant peak"
            ),
            "count": count,
            "error_source": "agent1",
        }
        for schedule_id, count in sorted(counts.items())
        if count > 1
    ]


def _arrival_wave_issues(scenario: str) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str, int]] = Counter()
    issues: List[Dict[str, Any]] = []
    for line in _v22._section(scenario, "PROPAGATED ARRIVALS").splitlines():  # noqa: SLF001
        event = _EVENT_ID.search(line)
        schedule = _SCHEDULE_ID.search(line)
        receiver = _RECEIVING_NODE.search(line)
        if not (event and schedule and receiver):
            continue
        key = (event.group(1), schedule.group(1), int(receiver.group(1)))
        counts[key] += 1
        arrival = _ARRIVAL_TIME.search(line)
        if arrival and re.search(r"\)\s*,\s*\d+", arrival.group(1)):
            issues.append(
                {
                    "field": (
                        f"PROPAGATED ARRIVALS event_id={key[0]}, "
                        f"schedule_id={key[1]}, receiving_node={key[2]}"
                    ),
                    "problem": (
                        "one schedule/receiver record contains multiple arrival "
                        "times; retain the arrival of the dominant source wave"
                    ),
                    "error_source": "agent1",
                }
            )
    for (event_id, schedule_id, receiver), count in sorted(counts.items()):
        if count > 1:
            issues.append(
                {
                    "field": (
                        f"PROPAGATED ARRIVALS event_id={event_id}, "
                        f"schedule_id={schedule_id}, receiving_node={receiver}"
                    ),
                    "problem": "schedule-specific receiving-node record is duplicated",
                    "count": count,
                    "error_source": "agent1",
                }
            )
    return issues


def audit_raw_scenario_v23(
    scenario: str,
    structured_scenario: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Reject ambiguous multi-wave schedules before spending an Agent 2 call."""
    base = _ORIGINAL_RAW_AUDIT(scenario, structured_scenario)
    issues: List[Dict[str, Any]] = list(base.get("issues", []) or [])
    existing = {
        (str(issue.get("field")), str(issue.get("problem"))) for issue in issues
    }
    wave_issues = (
        _modulation_wave_issues(scenario)
        + _temporal_peak_issues(scenario)
        + _arrival_wave_issues(scenario)
    )
    for issue in wave_issues:
        _append_issue(issues, existing, issue)
    output = dict(base)
    output.update(
        {
            "passed": not issues,
            "issues": issues,
            "single_wave_schedule_issues": wave_issues,
            "schedule_semantics": "one dominant source wave per schedule_id",
        }
    )
    return output


def _raw_scenario_feedback_v23(audit: Mapping[str, Any]) -> str:
    lines = [
        "Revise the same scenario without changing valid nodes, edges, paths, "
        "values, time span, or spatial layout.",
        "Every edge used by an event path, arrival path, stage, or EDGES "
        "AFFECTED field must also appear in the top-level EDGES section.",
        "Every EDGE MODULATION event family must have complete PROPAGATED "
        "ARRIVALS records for its declared receiving nodes.",
        "One (EVENT ID, SCHEDULE ID) represents exactly one dominant source "
        "wave, one positive peak, and one arrival per receiving node.",
        "For that schedule, emit every PATH STAGE exactly once. If morning and "
        "evening waves share DAYS, keep only the dominant wave; do not create "
        "another SCHEDULE ID with overlapping DAYS.",
        "Write SAMPLING FREQUENCY as an interval such as '1 hour'.",
    ]
    for issue in audit.get("issues", []) or []:
        detail = f" count={issue.get('count')}" if issue.get("count") else ""
        lines.append(
            f"- {issue.get('field')}: {issue.get('problem')}{detail}"
        )
    lines.append("Return only the complete revised scenario.")
    return "\n".join(lines)


def _calculate_sequence_length_v23(
    self: NetworkSDEGenerator,
    time_span_str: str,
    sampling_freq_str: str,
) -> int:
    """Translate rate wording into the interval wording accepted upstream."""
    text = str(sampling_freq_str or "")
    match = _RATE_FREQUENCY.search(text)
    rate = float(match.group(1)) if match else None
    unit = match.group(2).lower() if match else None
    if match is None:
        one_match = _ONE_RATE_FREQUENCY.search(text)
        if one_match:
            rate = 1.0
            unit = one_match.group(1).lower()
    if rate and rate > 0 and unit:
        unit_minutes = {"minute": 1.0, "hour": 60.0, "day": 1440.0}[unit]
        interval = unit_minutes / rate
        rounded = round(interval)
        if interval >= 1 and math.isclose(interval, rounded, abs_tol=1e-9):
            canonical = f"{int(rounded)} minutes"
            print(
                "STPP v23: normalized sampling frequency "
                f"'{sampling_freq_str}' -> '{canonical}'"
            )
            return _ORIGINAL_SEQUENCE_LENGTH(self, time_span_str, canonical)
    return _ORIGINAL_SEQUENCE_LENGTH(self, time_span_str, sampling_freq_str)


def _rename_v22_to_v23(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_path = old_path.with_name(
        old_path.name.replace("_stpp_v22_", "_stpp_v23_")
    )
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v23(*args, **kwargs) -> Dict[str, str | None]:
    files = _ORIGINAL_WRITE_V22(*args, **kwargs)
    renamed = {
        key: _rename_v22_to_v23(value) if key != "visualization" else value
        for key, value in files.items()
    }
    family = "single_wave_source_consistent_schedule_aware_stpp_v23"
    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = family
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v23"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = family
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v23"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Source-Consistent Schedule-Aware STPP v22",
        "STReasoner Single-Wave Schedule-Aware STPP v23",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    originals = {
        "audit": _v22.audit_raw_scenario_v22,
        "feedback": _v22._raw_scenario_feedback_v22,  # noqa: SLF001
        "writer": _v22._write_outputs_v22,  # noqa: SLF001
        "prompt": _v19.STPP_V19_SCENARIO_GENERATION_PROMPT,
        "v19_feedback": _v19.format_contract_feedback_v19,
        "v21_feedback": _v21.format_contract_feedback_v19,
        "sequence_length": NetworkSDEGenerator._calculate_sequence_length,
    }
    added_output_argument = False
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "output_stpp_v23"])
        added_output_argument = True
    _v22.audit_raw_scenario_v22 = audit_raw_scenario_v23
    _v22._raw_scenario_feedback_v22 = _raw_scenario_feedback_v23  # noqa: SLF001
    _v22._write_outputs_v22 = _write_outputs_v23  # noqa: SLF001
    _v19.STPP_V19_SCENARIO_GENERATION_PROMPT = (
        STPP_V23_SCENARIO_GENERATION_PROMPT
    )
    _v19.format_contract_feedback_v19 = format_contract_feedback_v23
    _v21.format_contract_feedback_v19 = format_contract_feedback_v23
    NetworkSDEGenerator._calculate_sequence_length = (
        _calculate_sequence_length_v23
    )
    try:
        return _v22.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v22 "):
            message = "v23 " + message[len("v22 ") :]
        raise RuntimeError(message) from exc
    finally:
        _v22.audit_raw_scenario_v22 = originals["audit"]
        _v22._raw_scenario_feedback_v22 = originals["feedback"]  # noqa: SLF001
        _v22._write_outputs_v22 = originals["writer"]  # noqa: SLF001
        _v19.STPP_V19_SCENARIO_GENERATION_PROMPT = originals["prompt"]
        _v19.format_contract_feedback_v19 = originals["v19_feedback"]
        _v21.format_contract_feedback_v19 = originals["v21_feedback"]
        NetworkSDEGenerator._calculate_sequence_length = originals[
            "sequence_length"
        ]
        if added_output_argument:
            del sys.argv[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
