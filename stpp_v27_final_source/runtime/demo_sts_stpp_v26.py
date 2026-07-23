#!/usr/bin/env python3
"""STPP v26: canonicalise Agent-2 variation placement by arrival path."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import demo_sts_stpp_v21 as _v21
import demo_sts_stpp_v25 as _v25
from stpp_adapter_v26 import canonicalise_agent2_json_v26


_ORIGINAL_WRITE_V25 = _v25._write_outputs_v25  # noqa: SLF001


def _rename_v25_to_v26(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_path = old_path.with_name(
        old_path.name.replace("_stpp_v25_", "_stpp_v26_")
    )
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v26(*args, **kwargs) -> Dict[str, str | None]:
    files = _ORIGINAL_WRITE_V25(*args, **kwargs)
    renamed = {
        key: _rename_v25_to_v26(value) if key != "visualization" else value
        for key, value in files.items()
    }
    family = "canonical_placement_nonwrapping_schedule_stpp_v26"
    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = family
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v26"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = family
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v26"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Non-Wrapping Schedule-Aware STPP v25",
        "STReasoner Canonical-Placement Schedule-Aware STPP v26",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    original_canonicalise = _v21._canonicalise_agent2_json_v8  # noqa: SLF001
    original_writer = _v25._write_outputs_v25  # noqa: SLF001
    added_output_argument = False
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "output_stpp_v26"])
        added_output_argument = True
    _v21._canonicalise_agent2_json_v8 = canonicalise_agent2_json_v26  # noqa: SLF001
    _v25._write_outputs_v25 = _write_outputs_v26  # noqa: SLF001
    try:
        return _v25.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v25 "):
            message = "v26 " + message[len("v25 ") :]
        raise RuntimeError(message) from exc
    finally:
        _v21._canonicalise_agent2_json_v8 = original_canonicalise  # noqa: SLF001
        _v25._write_outputs_v25 = original_writer  # noqa: SLF001
        if added_output_argument:
            del sys.argv[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
