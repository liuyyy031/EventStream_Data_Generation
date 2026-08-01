"""Semantics-aware lag handling and inbound-aware magnitude QA for STPP v5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from stpp_adapter_v3 import _node_baselines
from stpp_adapter_v4 import STPPV4Config, simulate_stpp_for_streasoner_v4


@dataclass(frozen=True)
class STPPV5Config(STPPV4Config):
    # A zero semantic lag needs a tiny positive simulation delay so lineage
    # remains strictly ordered. 0.01 hourly steps is 36 seconds.
    immediate_lag_epsilon_steps: float = 0.01
    propagation_jitter: float = 0.02

    def validate(self) -> None:
        super().validate()
        if not 0 < self.immediate_lag_epsilon_steps < 1:
            raise ValueError("immediate_lag_epsilon_steps must be in (0, 1)")


def _scenario_magnitude_envelope(
    structured_scenario: Mapping[str, Any],
    baselines: np.ndarray,
    background: np.ndarray,
    edge_factors: Mapping[str, np.ndarray],
    config: STPPV5Config,
) -> np.ndarray:
    """Estimate per-cell capacity including expected one-hop incoming flow."""
    envelope = baselines[:, None] * background
    for edge in structured_scenario.get("edges", []):
        source, target = int(edge["source"]), int(edge["target"])
        key = f"{source}->{target}"
        factors = edge_factors.get(key)
        if factors is None:
            factors = np.ones(background.shape[1], dtype=float)
        branching_probability = np.minimum(
            config.edge_branching_ratio * factors, 0.95
        )
        expected_incoming = (
            baselines[source]
            * background[source]
            * branching_probability
            * config.propagation_magnitude_decay
        )
        envelope[target] += expected_incoming
    return envelope


def simulate_stpp_for_streasoner_v5(
    structured_scenario: Mapping[str, Any],
    seq_len: int,
    config: STPPV5Config | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    config = config or STPPV5Config()
    config.validate()
    ts_data, events, metadata = simulate_stpp_for_streasoner_v4(
        structured_scenario, seq_len, config
    )

    baselines = _node_baselines(structured_scenario, ts_data.shape[0])
    background = np.asarray(metadata["background_profile"], dtype=float)
    edge_factors = {
        key: np.asarray(values, dtype=float)
        for key, values in metadata["edge_excitation_profiles"].items()
    }
    envelope = _scenario_magnitude_envelope(
        structured_scenario, baselines, background, edge_factors, config
    )
    ratios = np.divide(
        ts_data,
        envelope,
        out=np.zeros_like(ts_data),
        where=envelope > 0,
    )
    envelope_ratio = float(ratios.max())

    quality = metadata["quality_report"]
    source_only_ratio = float(quality["max_matrix_target_ratio"])
    quality["checks"]["matrix_peak_ratio_within_limit"] = (
        envelope_ratio <= config.max_matrix_target_ratio
    )
    quality["passed"] = all(quality["checks"].values())
    quality["source_only_max_matrix_target_ratio"] = source_only_ratio
    quality["max_matrix_envelope_ratio"] = envelope_ratio
    # Keep this compatibility field aligned with the profile used by the gate.
    quality["max_matrix_target_ratio"] = envelope_ratio

    metadata.update(
        {
            "simulator": "STPPG + semantics-aware inbound-envelope graph STPP v5",
            "method": (
                "node-quota STPPG roots + calibrated marks + semantics-aware "
                "lag branching + inbound magnitude envelope"
            ),
            "integration_method": (
                "quota control + quality-gated marked branching + inbound-aware QA"
            ),
            "scenario_magnitude_envelope": envelope.tolist(),
            "source_only_max_matrix_target_ratio": source_only_ratio,
            "max_matrix_envelope_ratio": envelope_ratio,
        }
    )
    return ts_data, events, metadata
