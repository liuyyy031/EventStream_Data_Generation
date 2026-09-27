"""Tests for sequence calibration migrated into the active generation core."""

from __future__ import annotations

import unittest

from data_generation.domain_packages.healthcare import (
    build_healthcare_temporal_models,
)
from data_generation.generation_core.distribution_calibration import (
    apply_sequence_calibration_bundle,
    fit_sequence_calibration_bundle,
    summarize_reference_sequences,
)
from data_generation.generation_core.mechanisms import load_mechanism_registry
from data_generation.generation_core.models import EventParticipant
from data_generation.generation_core.reference_data import (
    ReferenceDataset,
    ReferenceEvent,
    ReferenceWindow,
)
from pathlib import Path


HEALTHCARE_DIR = (
    Path(__file__).resolve().parents[1] / "domain_packages" / "healthcare"
)
ALERT = "healthcare.workflow.monitoring_alert.created"
REVIEW = "healthcare.workflow.clinical_review.completed"
CARE_PLAN = "healthcare.intervention.care_plan.adjusted"


class DistributionCalibrationTests(unittest.TestCase):
    def test_zero_event_reference_windows_are_not_dropped(self) -> None:
        dataset = _reference_dataset()
        dataset.windows.append(
            ReferenceWindow(
                window_id="empty_train_window",
                context_id="empty_context",
                domain="healthcare",
                start_time="2026-02-01T00:00:00+00:00",
                duration_seconds=3600.0,
                split="train",
                attributes={"scenario_family": "temperature_escalation"},
            )
        )

        summary = summarize_reference_sequences(dataset, split="train")

        self.assertEqual(summary["stream_count"], 7)
        self.assertEqual(summary["sequence_length_distribution"]["0"], 1)

    def test_sequence_bundle_uses_train_and_fits_explicit_parent_delays(self) -> None:
        dataset = _reference_dataset()
        registry = load_mechanism_registry(
            HEALTHCARE_DIR / "mechanism_specs.json"
        )

        bundle = fit_sequence_calibration_bundle(
            dataset,
            registry,
            min_transition_samples=5,
        )

        self.assertEqual(bundle["schema_version"], "sequence-calibration-bundle-v1")
        train = bundle["sequence_statistics"]["train"]
        self.assertEqual(train["stream_count"], 6)
        self.assertEqual(
            train["first_order_transitions"][ALERT][REVIEW]["probability"],
            1.0,
        )
        self.assertEqual(
            train["second_order_transitions"][f"{ALERT}|{REVIEW}"][CARE_PLAN][
                "probability"
            ],
            1.0,
        )
        self.assertIn(
            "healthcare.review_delay.prior_v1",
            bundle["temporal_model_overrides"],
        )
        self.assertEqual(
            bundle["domain_config_overrides"]["scenario_weights"],
            {"temperature_escalation": 1.0},
        )
        self.assertEqual(bundle["holdout_diagnostics"]["status"], "diagnostic_only_no_universal_acceptance_threshold")

    def test_pending_bundle_requires_explicit_evaluation_opt_in(self) -> None:
        dataset = _reference_dataset()
        registry = load_mechanism_registry(
            HEALTHCARE_DIR / "mechanism_specs.json"
        )
        bundle = fit_sequence_calibration_bundle(
            dataset,
            registry,
            min_transition_samples=5,
        )
        temporal_models = build_healthcare_temporal_models()
        domain_config = {}
        with self.assertRaises(ValueError):
            apply_sequence_calibration_bundle(
                bundle,
                domain="healthcare",
                domain_config=domain_config,
                temporal_models=temporal_models,
            )

        application = apply_sequence_calibration_bundle(
            bundle,
            domain="healthcare",
            domain_config=domain_config,
            temporal_models=temporal_models,
            allow_pending=True,
        )

        self.assertTrue(application["pending_bundle_explicitly_allowed"])
        self.assertNotIn(
            "healthcare.review_delay.prior_v1",
            application["temporal_model_ids_applied"],
        )
        self.assertIn("healthcare.review_delay.prior_v1", application["temporal_models_skipped"])
        descriptor = next(
            item
            for item in temporal_models.descriptors()
            if item["model_id"] == "healthcare.review_delay.prior_v1"
        )
        self.assertFalse(descriptor.get("calibration_applied", False))

    def test_modified_bundle_is_rejected_by_fingerprint(self) -> None:
        dataset = _reference_dataset()
        registry = load_mechanism_registry(
            HEALTHCARE_DIR / "mechanism_specs.json"
        )
        bundle = fit_sequence_calibration_bundle(dataset, registry)
        bundle["domain_config_overrides"]["scenario_weights"] = {
            "temperature_escalation": 0.5,
            "tampered": 0.5,
        }

        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            apply_sequence_calibration_bundle(
                bundle,
                domain="healthcare",
                domain_config={},
                temporal_models=build_healthcare_temporal_models(),
                allow_pending=True,
            )

    def test_component_ablation_applies_only_selected_parts(self) -> None:
        dataset = _reference_dataset()
        registry = load_mechanism_registry(
            HEALTHCARE_DIR / "mechanism_specs.json"
        )
        bundle = fit_sequence_calibration_bundle(dataset, registry)
        temporal_models = build_healthcare_temporal_models()
        domain_config = {"scenario_weights": {"routine_observation": 1.0}}

        application = apply_sequence_calibration_bundle(
            bundle,
            domain="healthcare",
            domain_config=domain_config,
            temporal_models=temporal_models,
            allow_pending=True,
            components={"mechanism_reweighting"},
        )

        self.assertEqual(
            domain_config["scenario_weights"], {"routine_observation": 1.0}
        )
        self.assertEqual(application["temporal_model_ids_applied"], [])
        self.assertTrue(application["mechanism_reweighting_enabled"])
        self.assertEqual(
            application["selected_components"], ["mechanism_reweighting"]
        )


def _reference_dataset() -> ReferenceDataset:
    windows = []
    events = []
    participant = EventParticipant("patient_1", "subject")
    for index in range(8):
        split = "train" if index < 6 else "holdout"
        window_id = f"window_{index}"
        context_id = f"context_{index}"
        windows.append(
            ReferenceWindow(
                window_id=window_id,
                context_id=context_id,
                domain="healthcare",
                start_time=f"2026-01-{index + 1:02d}T00:00:00+00:00",
                duration_seconds=3600.0,
                split=split,
                attributes={"scenario_family": "temperature_escalation"},
            )
        )
        alert_id = f"alert_{index}"
        review_id = f"review_{index}"
        care_plan_id = f"care_plan_{index}"
        events.extend(
            [
                ReferenceEvent(
                    source_record_id=alert_id,
                    window_id=window_id,
                    context_id=context_id,
                    domain="healthcare",
                    event_type_id=ALERT,
                    occurrence_offset_seconds=100.0,
                    participants=[participant],
                ),
                ReferenceEvent(
                    source_record_id=review_id,
                    window_id=window_id,
                    context_id=context_id,
                    domain="healthcare",
                    event_type_id=REVIEW,
                    occurrence_offset_seconds=300.0 + index * 10.0,
                    participants=[participant],
                    explicit_parent_record_ids=[alert_id],
                ),
                ReferenceEvent(
                    source_record_id=care_plan_id,
                    window_id=window_id,
                    context_id=context_id,
                    domain="healthcare",
                    event_type_id=CARE_PLAN,
                    occurrence_offset_seconds=500.0 + index * 20.0,
                    participants=[participant],
                    explicit_parent_record_ids=[review_id],
                ),
            ]
        )
    return ReferenceDataset(
        domain="healthcare",
        windows=windows,
        events=events,
        source_metadata={
            "domain": "healthcare",
            "source_dataset_id": "unit-test",
            "source_version": "1",
            "split_policy": {
                "unit": "context",
                "assignment_method": "fixed",
                "allow_overlapping_windows": False,
            },
        },
    )


if __name__ == "__main__":
    unittest.main()
