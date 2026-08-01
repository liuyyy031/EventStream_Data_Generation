#!/usr/bin/env python3
"""STPP v18: bounded EVENT FULL PATH parsing over the v17 pipeline."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import demo_sts_stpp_v17 as _v17


_ARROW_CHAIN = re.compile(
    r"(?:(?:NODE\s*)?\d+\s*(?:->|→)\s*)+(?:NODE\s*)?\d+",
    re.IGNORECASE,
)
_ORIGINAL_PAIRS_FROM_CHAIN = _v17._pairs_from_chain  # noqa: SLF001
_ORIGINAL_WRITE_OUTPUTS = _v17._write_outputs_v17  # noqa: SLF001


def _pairs_from_chain_v18(text: str) -> List[Tuple[int, int]]:
    """Read only the first arrow chain, excluding lag/window annotations."""
    match = _ARROW_CHAIN.search(text)
    if not match:
        return []
    node_ids = [
        int(value)
        for value in re.findall(r"(?:NODE\s*)?(\d+)", match.group(0), re.I)
    ]
    return list(zip(node_ids, node_ids[1:]))


def _rename_output_version(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_name = old_path.name.replace("_stpp_v17_", "_stpp_v18_")
    new_path = old_path.with_name(new_name)
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v18(*args, **kwargs) -> Dict[str, str | None]:
    """Reuse the stable v17 writer, then version the newly written artifacts."""
    files = _ORIGINAL_WRITE_OUTPUTS(*args, **kwargs)
    renamed = {
        key: _rename_output_version(value) if key != "visualization" else value
        for key, value in files.items()
    }

    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = (
        "bounded_path_parser_two_judge_direct_route_stpp_v18"
    )
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v18"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = (
        "bounded_path_parser_two_judge_direct_route_stpp_v18"
    )
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v18"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Role-Routed Two-Judge STPP v17",
        "STReasoner Bounded-Path Two-Judge STPP v18",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    """Run v17 with the corrected bounded path parser and v18 output labels."""
    original_pairs = _v17._pairs_from_chain  # noqa: SLF001
    original_writer = _v17._write_outputs_v17  # noqa: SLF001
    _v17._pairs_from_chain = _pairs_from_chain_v18  # noqa: SLF001
    _v17._write_outputs_v17 = _write_outputs_v18  # noqa: SLF001
    try:
        return _v17.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v17 "):
            message = "v18 " + message[len("v17 ") :]
        raise RuntimeError(message) from exc
    finally:
        _v17._pairs_from_chain = original_pairs  # noqa: SLF001
        _v17._write_outputs_v17 = original_writer  # noqa: SLF001


if __name__ == "__main__":
    raise SystemExit(main())
