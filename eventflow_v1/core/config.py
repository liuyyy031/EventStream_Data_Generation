"""Configuration loading and conservative validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required = ["domain", "seed", "episode_count", "network", "simulation", "text", "validation"]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    if config["domain"] != "transportation":
        raise ValueError("EventFlow v1 currently implements only the transportation domain")
    if int(config["episode_count"]) <= 0:
        raise ValueError("episode_count must be positive")
    if int(config["network"].get("node_count", 0)) < 2:
        raise ValueError("network.node_count must be at least 2")
    if int(config["network"].get("network_count", 1)) <= 0:
        raise ValueError("network.network_count must be positive")
    if int(config["network"].get("max_topology_out_degree", 0)) < 2:
        raise ValueError("network.max_topology_out_degree must be at least 2")
    families = config["network"].get("structure_families", [])
    if not config["network"].get("external_network_file") and not families:
        raise ValueError("network.structure_families must not be empty")
    if float(config["simulation"].get("duration_seconds", 0)) <= 0:
        raise ValueError("simulation.duration_seconds must be positive")
    expected_children = float(config["simulation"].get("expected_children_per_event", 0))
    max_children = int(config["simulation"].get("max_children_per_event", 0))
    if expected_children <= 0:
        raise ValueError("simulation.expected_children_per_event must be positive")
    if max_children <= 0:
        raise ValueError("simulation.max_children_per_event must be positive")
    if expected_children > max_children:
        raise ValueError("expected_children_per_event cannot exceed max_children_per_event")
    full_recovery_probability = float(
        config["simulation"].get("full_recovery_probability", 0.35)
    )
    if not 0.0 <= full_recovery_probability <= 1.0:
        raise ValueError("simulation.full_recovery_probability must be in [0, 1]")
    rate = float(config["text"].get("paraphrase_rate", 0.0))
    if not 0.0 <= rate <= 1.0:
        raise ValueError("text.paraphrase_rate must be in [0, 1]")
    if int(config["validation"].get("llm_judge_protocol_max_attempts", 3)) <= 0:
        raise ValueError("validation.llm_judge_protocol_max_attempts must be positive")
    if int(config["validation"].get("llm_judge_reason_limit", 4)) <= 0:
        raise ValueError("validation.llm_judge_reason_limit must be positive")
    calibration = config.get("calibration", {})
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be an object")
    profile_file = calibration.get("profile_file")
    if profile_file and not Path(str(profile_file)).is_file():
        raise ValueError(f"calibration.profile_file does not exist: {profile_file}")
