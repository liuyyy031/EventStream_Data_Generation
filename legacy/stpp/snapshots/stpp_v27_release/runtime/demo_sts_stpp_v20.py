#!/usr/bin/env python3
"""STPP v20: Markdown-tolerant explicit-edge extraction over v19."""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict

import demo_sts_stpp_v17 as _v17
import demo_sts_stpp_v19 as _v19


_EDGE_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*EDGES"
    r"(?:\s*\([^\r\n)]*\))?\s*:?\s*(?:\*\*|__)?\s*$",
    re.IGNORECASE,
)
_FOLLOWING_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*"
    r"(?:TEMPORAL\s+PATTERNS|PROPAGATED\s+ARRIVALS|EDGE\s+MODULATION|"
    r"SPATIAL\s+LAYOUT)\s*:?\s*(?:\*\*|__)?\s*$",
    re.IGNORECASE,
)
_EXPLICIT_EDGE_LINE = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:\*\*|__)?\s*EDGE\s+"
    r"(?:NODE\s*)?\d+\s*(?:->|→)\s*(?:NODE\s*)?\d+",
    re.IGNORECASE,
)
_ORIGINAL_EXPLICIT_EDGE_SECTION = _v17._explicit_edge_section  # noqa: SLF001
_ORIGINAL_WRITE_V19 = _v19._write_outputs_v19  # noqa: SLF001


def _explicit_edge_section_v20(scenario: str) -> str:
    """Accept plain/Markdown headings, with a strict EDGE-line fallback."""
    lines = scenario.splitlines()
    start = None
    for index, line in enumerate(lines):
        if _EDGE_HEADING.match(line):
            start = index + 1
            break
    if start is not None:
        end = len(lines)
        for index in range(start, len(lines)):
            if _FOLLOWING_HEADING.match(lines[index]):
                end = index
                break
        section = "\n".join(lines[start:end])
        if _v17._ARROW_PAIR.search(section):  # noqa: SLF001
            return section

    # Fallback deliberately requires singular EDGE followed immediately by a
    # numeric source. It therefore cannot match "EDGES AFFECTED" records.
    explicit_lines = [line for line in lines if _EXPLICIT_EDGE_LINE.match(line)]
    return "\n".join(explicit_lines)


def _rename_v19_to_v20(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_path = old_path.with_name(
        old_path.name.replace("_stpp_v19_", "_stpp_v20_")
    )
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v20(*args, **kwargs) -> Dict[str, str | None]:
    files = _ORIGINAL_WRITE_V19(*args, **kwargs)
    renamed = {
        key: _rename_v19_to_v20(value) if key != "visualization" else value
        for key, value in files.items()
    }
    family = "markdown_edge_schedule_aware_two_judge_stpp_v20"

    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = family
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v20"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = family
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v20"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Schedule-Aware Two-Judge STPP v19",
        "STReasoner Markdown-Edge Schedule-Aware STPP v20",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    original_section = _v17._explicit_edge_section  # noqa: SLF001
    original_writer = _v19._write_outputs_v19  # noqa: SLF001
    added_output_argument = False
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "output_stpp_v20"])
        added_output_argument = True
    _v17._explicit_edge_section = _explicit_edge_section_v20  # noqa: SLF001
    _v19._write_outputs_v19 = _write_outputs_v20  # noqa: SLF001
    try:
        return _v19.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v19 "):
            message = "v20 " + message[len("v19 ") :]
        raise RuntimeError(message) from exc
    finally:
        _v17._explicit_edge_section = original_section  # noqa: SLF001
        _v19._write_outputs_v19 = original_writer  # noqa: SLF001
        if added_output_argument:
            del sys.argv[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
