#!/usr/bin/env python3
"""STPP v24: replace the obsolete node-global self-pattern quality check."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import demo_sts_stpp_v19 as _v19
import demo_sts_stpp_v23 as _v23
from stpp_adapter_v24 import simulate_stpp_for_streasoner_v24


_ORIGINAL_WRITE_V23 = _v23._write_outputs_v23  # noqa: SLF001


def _rename_v23_to_v24(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_path = old_path.with_name(
        old_path.name.replace("_stpp_v23_", "_stpp_v24_")
    )
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v24(*args, **kwargs) -> Dict[str, str | None]:
    files = _ORIGINAL_WRITE_V23(*args, **kwargs)
    renamed = {
        key: _rename_v23_to_v24(value) if key != "visualization" else value
        for key, value in files.items()
    }
    family = "schedule_quality_compatible_single_wave_stpp_v24"
    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = family
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v24"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = family
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v24"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Single-Wave Schedule-Aware STPP v23",
        "STReasoner Schedule-Quality-Compatible STPP v24",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    original_simulator = _v19.simulate_stpp_for_streasoner_v19
    original_writer = _v23._write_outputs_v23  # noqa: SLF001
    added_output_argument = False
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "output_stpp_v24"])
        added_output_argument = True
    _v19.simulate_stpp_for_streasoner_v19 = simulate_stpp_for_streasoner_v24
    _v23._write_outputs_v23 = _write_outputs_v24  # noqa: SLF001
    try:
        return _v23.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v23 "):
            message = "v24 " + message[len("v23 ") :]
        raise RuntimeError(message) from exc
    finally:
        _v19.simulate_stpp_for_streasoner_v19 = original_simulator
        _v23._write_outputs_v23 = original_writer  # noqa: SLF001
        if added_output_argument:
            del sys.argv[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
