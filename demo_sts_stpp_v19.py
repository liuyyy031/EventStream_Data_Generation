#!/usr/bin/env python3
"""STPP v19 schedule-aware branch validation over the v18 entry point."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import demo_sts_stpp_v16 as _demo16
import demo_sts_stpp_v17 as _v17
import demo_sts_stpp_v18 as _v18
import safe_judge_generator_v17 as _safe17
from prompts.judge_scenario_parsing_prompt_stpp_v19 import (
    STPP_V19_JUDGE_SCENARIO_PARSING_PROMPT,
)
from prompts.scenario_generation_agent_prompt_stpp_v19 import (
    STPP_V19_SCENARIO_GENERATION_PROMPT,
)
from prompts.scenario_parsing_agent_prompt_stpp_v19 import (
    STPP_V19_SCENARIO_PARSING_PROMPT,
)
from stpp_adapter_v19 import (
    audit_structured_contract_v19,
    canonicalise_temporal_contract_v19,
    format_contract_feedback_v19,
    simulate_stpp_for_streasoner_v19,
)


_ORIGINAL_WRITE_V18 = _v18._write_outputs_v18  # noqa: SLF001


def _rename_v18_to_v19(path_text: str | None) -> str | None:
    if not path_text:
        return path_text
    old_path = Path(path_text)
    new_path = old_path.with_name(
        old_path.name.replace("_stpp_v18_", "_stpp_v19_")
    )
    old_path.replace(new_path)
    return str(new_path)


def _write_outputs_v19(*args, **kwargs) -> Dict[str, str | None]:
    files = _ORIGINAL_WRITE_V18(*args, **kwargs)
    renamed = {
        key: _rename_v18_to_v19(value) if key != "visualization" else value
        for key, value in files.items()
    }
    family = "schedule_aware_two_judge_direct_route_stpp_v19"

    json_path = Path(str(renamed["json"]))
    with json_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    json_data["generator_family"] = family
    json_data.setdefault("generation_info", {})["entrypoint_version"] = "v19"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    pickle_path = Path(str(renamed["pickle"]))
    with pickle_path.open("rb") as handle:
        pickle_data = pickle.load(handle)
    pickle_data["generator_family"] = family
    pickle_data.setdefault("generation_info", {})["entrypoint_version"] = "v19"
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_data, handle)

    description_path = Path(str(renamed["description"]))
    description = description_path.read_text(encoding="utf-8")
    description = description.replace(
        "STReasoner Bounded-Path Two-Judge STPP v18",
        "STReasoner Schedule-Aware Two-Judge STPP v19",
        1,
    )
    description_path.write_text(description, encoding="utf-8")
    return renamed


def main() -> int:
    """Temporarily route v17/v18 extension points through v19 semantics."""
    original_values = {
        "demo16_audit": _demo16.audit_structured_contract_v16,
        "demo16_feedback": _demo16.format_contract_feedback_v16,
        "v17_audit": _v17.audit_structured_contract_v16,
        "v17_canonicalise": _v17.canonicalise_temporal_contract_v14,
        "v17_simulate": _v17.simulate_stpp_for_streasoner_v16,
        "v18_writer": _v18._write_outputs_v18,  # noqa: SLF001
        "generation_prompt": _safe17.STPP_V17_SCENARIO_GENERATION_PROMPT,
        "parsing_prompt": _safe17.STPP_V17_SCENARIO_PARSING_PROMPT,
        "judge1_prompt": _safe17.STPP_V17_JUDGE_SCENARIO_PARSING_PROMPT,
    }
    added_output_argument = False
    if "--output_dir" not in sys.argv:
        sys.argv.extend(["--output_dir", "output_stpp_v19"])
        added_output_argument = True

    _demo16.audit_structured_contract_v16 = audit_structured_contract_v19
    _demo16.format_contract_feedback_v16 = format_contract_feedback_v19
    _v17.audit_structured_contract_v16 = audit_structured_contract_v19
    _v17.canonicalise_temporal_contract_v14 = canonicalise_temporal_contract_v19
    _v17.simulate_stpp_for_streasoner_v16 = simulate_stpp_for_streasoner_v19
    _v18._write_outputs_v18 = _write_outputs_v19  # noqa: SLF001
    _safe17.STPP_V17_SCENARIO_GENERATION_PROMPT = (
        STPP_V19_SCENARIO_GENERATION_PROMPT
    )
    _safe17.STPP_V17_SCENARIO_PARSING_PROMPT = STPP_V19_SCENARIO_PARSING_PROMPT
    _safe17.STPP_V17_JUDGE_SCENARIO_PARSING_PROMPT = (
        STPP_V19_JUDGE_SCENARIO_PARSING_PROMPT
    )
    try:
        return _v18.main()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("v18 "):
            message = "v19 " + message[len("v18 ") :]
        raise RuntimeError(message) from exc
    finally:
        _demo16.audit_structured_contract_v16 = original_values["demo16_audit"]
        _demo16.format_contract_feedback_v16 = original_values["demo16_feedback"]
        _v17.audit_structured_contract_v16 = original_values["v17_audit"]
        _v17.canonicalise_temporal_contract_v14 = original_values[
            "v17_canonicalise"
        ]
        _v17.simulate_stpp_for_streasoner_v16 = original_values["v17_simulate"]
        _v18._write_outputs_v18 = original_values["v18_writer"]  # noqa: SLF001
        _safe17.STPP_V17_SCENARIO_GENERATION_PROMPT = original_values[
            "generation_prompt"
        ]
        _safe17.STPP_V17_SCENARIO_PARSING_PROMPT = original_values[
            "parsing_prompt"
        ]
        _safe17.STPP_V17_JUDGE_SCENARIO_PARSING_PROMPT = original_values[
            "judge1_prompt"
        ]
        if added_output_argument:
            del sys.argv[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
