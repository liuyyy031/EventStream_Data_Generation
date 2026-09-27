"""Fit mechanism or sequence priors and evaluate calibrated generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_generation.generation_core.calibration import (
    CalibrationEngine,
    build_root_event_observations,
)
from data_generation.generation_core.distribution_calibration import (
    compare_reference_to_generation,
    fit_sequence_calibration_bundle,
)
from data_generation.generation_core.mechanisms import load_mechanism_registry
from data_generation.generation_core.reference_data import NormalizedJsonlReferenceAdapter
from data_generation.generation_core.tabular_reference_adapter import (
    GenericTabularReferenceAdapter,
    write_normalized_reference_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("mechanism-rate", "sequence-prior", "compare"),
        default="mechanism-rate",
    )
    parser.add_argument("--mechanism-spec")
    parser.add_argument("--mechanism-id")
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument(
        "--reference-dir",
        help="Directory already using the normalized reference-data contract",
    )
    reference_group.add_argument(
        "--input-file",
        help="Raw CSV, JSONL, or Parquet event table",
    )
    parser.add_argument(
        "--mapping-config",
        help="Column/type mapping JSON required with --input-file",
    )
    parser.add_argument(
        "--normalized-output-dir",
        help="Optional directory for the normalized audit snapshot and report",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-transition-samples", type=int, default=5)
    parser.add_argument("--generic-output-dir")
    parser.add_argument("--calibrated-output-dir")
    parser.add_argument(
        "--reference-split",
        choices=("train", "validation", "holdout"),
        default="holdout",
    )
    args = parser.parse_args()

    if args.input_file and not args.mapping_config:
        parser.error("--mapping-config is required with --input-file")
    if args.mapping_config and not args.input_file:
        parser.error("--mapping-config is only valid with --input-file")
    if args.reference_dir:
        dataset = NormalizedJsonlReferenceAdapter().load(args.reference_dir)
        normalization_report = None
        reference_input_summary = {
            "mode": "normalized_directory",
            "source": args.reference_dir,
            "dataset_fingerprint": dataset.fingerprint(),
        }
    else:
        adapter = GenericTabularReferenceAdapter.from_json(args.mapping_config)
        dataset = adapter.load(args.input_file)
        normalization_report = adapter.last_report
        reference_input_summary = {
            "mode": "raw_tabular_file",
            "source": args.input_file,
            "mapping_config": args.mapping_config,
            "normalization": normalization_report,
        }
    if args.normalized_output_dir:
        write_normalized_reference_dataset(
            dataset,
            args.normalized_output_dir,
            normalization_report=normalization_report,
        )
    if args.mode == "sequence-prior":
        if not args.mechanism_spec:
            parser.error("--mechanism-spec is required for --mode sequence-prior")
        registry = load_mechanism_registry(args.mechanism_spec)
        bundle = fit_sequence_calibration_bundle(
            dataset,
            registry,
            min_transition_samples=args.min_transition_samples,
        )
        _write_json(args.output, bundle)
        print(
            json.dumps(
                {
                    "output": args.output,
                    "domain": bundle["domain"],
                    "parameter_status": bundle["parameter_status"],
                    "temporal_model_override_count": len(
                        bundle["temporal_model_overrides"]
                    ),
                    "holdout_diagnostics": bundle["holdout_diagnostics"],
                    "bundle_fingerprint": bundle["bundle_fingerprint"],
                    "reference_input": reference_input_summary,
                    "normalized_output_dir": args.normalized_output_dir,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.mode == "compare":
        if not args.generic_output_dir or not args.calibrated_output_dir:
            parser.error(
                "--generic-output-dir and --calibrated-output-dir are required "
                "for --mode compare"
            )
        report = compare_reference_to_generation(
            dataset,
            generic_output_dir=args.generic_output_dir,
            calibrated_output_dir=args.calibrated_output_dir,
            split=args.reference_split,
        )
        _write_json(args.output, report)
        print(
            json.dumps(
                {
                    **report,
                    "reference_input": reference_input_summary,
                    "normalized_output_dir": args.normalized_output_dir,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not args.mechanism_spec or not args.mechanism_id:
        parser.error(
            "--mechanism-spec and --mechanism-id are required for "
            "--mode mechanism-rate"
        )
    registry = load_mechanism_registry(args.mechanism_spec)
    mechanism = registry.get(args.mechanism_id)
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
                "reference_input": reference_input_summary,
                "normalized_output_dir": args.normalized_output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_json(path: str | Path, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
