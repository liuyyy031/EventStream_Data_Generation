#!/usr/bin/env python3
"""STPP v27: use day-aware source/propagated time-conflict quality."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import demo_sts_stpp_v24 as _v24
import demo_sts_stpp_v26 as _v26
from stpp_adapter_v27 import simulate_stpp_for_streasoner_v27


_ORIGINAL_WRITE_V26 = _v26._write_outputs_v26  # noqa: SLF001


def _rename_v26_to_v27(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_path = old_path.with_name(
        old_path.name.replace("_stpp_v26_", "_stpp_v27_")
    )
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v27(*args, **kwargs) -> Dict[str, str | None]:
    files = _ORIGINAL_WRITE_V26(*args, **kwargs)
    renamed = {
        key: _rename_v26_to_v27(value) if key != "visualization" else value
        for key, value in files.items()
    }
    family = "day_aware_conflict_canonical_placement_stpp_v27"
    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = family
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v27"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = family
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v27"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Canonical-Placement Schedule-Aware STPP v26",
        "STReasoner Day-Aware Conflict STPP v27",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    original_simulator = _v24.simulate_stpp_for_streasoner_v24
    original_writer = _v26._write_outputs_v26  # noqa: SLF001
    added_output_argument = False
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "output_stpp_v27"])
        added_output_argument = True
    _v24.simulate_stpp_for_streasoner_v24 = simulate_stpp_for_streasoner_v27
    _v26._write_outputs_v26 = _write_outputs_v27  # noqa: SLF001
    try:
        return _v26.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v26 "):
            message = "v27 " + message[len("v26 ") :]
        raise RuntimeError(message) from exc
    finally:
        _v24.simulate_stpp_for_streasoner_v24 = original_simulator
        _v26._write_outputs_v26 = original_writer  # noqa: SLF001
        if added_output_argument:
            del sys.argv[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
