"""Tests for the controlled healthcare correction evaluation."""

from __future__ import annotations

import unittest

from data_generation.domain_packages.healthcare import (
    HealthcarePackage,
    build_healthcare_temporal_models,
)
from data_generation.evaluation.run_healthcare_correction_challenge import (
    INJECTED_CLAIM,
    grounded_record_ids,
    inject_unsupported_clinical_claim,
    structured_truth_sha256,
)
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.validation import validate_text_alignment


class HealthcareCorrectionEvaluationTests(unittest.TestCase):
    def test_controlled_challenge_changes_only_grounded_text(self) -> None:
        package = HealthcarePackage(
            {
                "nodes_per_context": 24,
                "context_count": 1,
                "duration_seconds": 3600.0,
            }
        )
        result = SimulationEngine(
            package,
            build_healthcare_temporal_models(),
            max_events=64,
        ).run(0, 20260919)
        original = package.render_episode_text(result)
        structured_before = structured_truth_sha256(result)

        challenged, injection = inject_unsupported_clinical_claim(original)

        self.assertNotEqual(original["text"], challenged["text"])
        self.assertIn(INJECTED_CLAIM.strip(), challenged["text"])
        self.assertTrue(
            set(injection["grounded_record_ids"])
            <= grounded_record_ids(challenged)
        )
        self.assertTrue(
            validate_text_alignment(result, challenged, package.catalog())["passed"]
        )
        self.assertEqual(structured_before, structured_truth_sha256(result))


if __name__ == "__main__":
    unittest.main()

