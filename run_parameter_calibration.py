"""Fit one declared mechanism from normalized reference-data windows."""

from __future__ import annotations

import argparse
import json

from generation_core.calibration import CalibrationEngine, build_root_event_observations
from generation_core.mechanisms import load_mechanism_registry
from generation_core.reference_data import NormalizedJsonlReferenceAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-spec", required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    registry = load_mechanism_registry(args.mechanism_spec)
    mechanism = registry.get(args.mechanism_id)
    dataset = NormalizedJsonlReferenceAdapter().load(args.reference_dir)
    if dataset.domain != mechanism.domain:
        raise ValueError("Reference dataset and mechanism use different domains")
    observations = build_root_event_observations(dataset, mechanism)
    train = [item for item in observations if item.split == "train"]
    holdout = [item for item in observations if item.split == "holdout"]
    bundle = CalibrationEngine().fit(
        mechanism,
        train,
        holdout_observations=holdout,
        reference_data_fingerprint=dataset.fingerprint(),
    )
    bundle.write(args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "domain": bundle.domain,
                "mechanism_id": bundle.mechanism_id,
                "parameter_status": bundle.parameter_status,
                "parameter_estimates": bundle.parameter_estimates,
                "holdout_metrics": bundle.holdout_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
