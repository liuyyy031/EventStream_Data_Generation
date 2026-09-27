"""Tests for safe transition-informed legal-mechanism reweighting."""

from __future__ import annotations

import math
import random
import unittest

from data_generation.domain_packages.healthcare import (
    HealthcarePackage,
    build_healthcare_temporal_models,
)
from data_generation.generation_core.mechanism_reweighting import (
    MECHANISM_HAZARD_MULTIPLIER,
    LegalMechanismReweighter,
)
from data_generation.generation_core.models import (
    Candidate,
    EventRecord,
    TemporalExtent,
)
from data_generation.generation_core.temporal import (
    ConditionalGammaModel,
    ConditionalLogNormalModel,
    ConditionalWeibullModel,
)
from data_generation.generation_core.scheduler import SimulationEngine


class LegalMechanismReweightingTests(unittest.TestCase):
    def test_only_declared_parent_target_mechanism_is_reweighted(self) -> None:
        parent = _event("parent", "domain.event.a", 10.0)
        candidate = _candidate("domain.a_to_b", "domain.event.b", parent.event_id)
        reweighter = LegalMechanismReweighter(
            first_order_transitions={
                "domain.event.a": {
                    "domain.event.b": {"count": 80, "probability": 0.8},
                    "domain.event.c": {"count": 20, "probability": 0.2},
                }
            },
            second_order_transitions={},
            mechanism_descriptors=[
                {
                    "mechanism_id": "domain.a_to_b",
                    "source_event_type_ids": ["domain.event.a"],
                    "target_event_type_id": "domain.event.b",
                }
            ],
            min_history_count=5,
        )

        audit = reweighter.apply(
            candidate,
            event_by_id={parent.event_id: parent},
            event_history=[parent],
            temporal_family="conditional_lognormal",
        )

        self.assertEqual(audit["status"], "applied")
        self.assertAlmostEqual(
            candidate.temporal_inputs[MECHANISM_HAZARD_MULTIPLIER], 1.6
        )
        self.assertFalse(audit["creates_candidate_or_parent_edge"])

    def test_second_order_is_used_only_when_parent_is_latest_event(self) -> None:
        previous = _event("previous", "domain.event.x", 5.0)
        parent = _event("parent", "domain.event.a", 10.0)
        candidate = _candidate("domain.a_to_b", "domain.event.b", parent.event_id)
        reweighter = LegalMechanismReweighter(
            first_order_transitions={
                "domain.event.a": {
                    "domain.event.b": {"count": 50},
                    "domain.event.c": {"count": 50},
                }
            },
            second_order_transitions={
                "domain.event.x|domain.event.a": {
                    "domain.event.b": {"count": 90},
                    "domain.event.c": {"count": 10},
                }
            },
            mechanism_descriptors=[
                {
                    "mechanism_id": "domain.a_to_b",
                    "source_event_type_ids": ["domain.event.a"],
                    "target_event_type_id": "domain.event.b",
                }
            ],
        )

        audit = reweighter.apply(
            candidate,
            event_by_id={previous.event_id: previous, parent.event_id: parent},
            event_history=[previous, parent],
            temporal_family="conditional_lognormal",
        )

        self.assertEqual(audit["history_order"], "second_order")
        self.assertAlmostEqual(audit["applied_hazard_multiplier"], 1.8)

    def test_unobserved_legal_transition_keeps_original_prior(self) -> None:
        parent = _event("parent", "domain.event.a", 10.0)
        candidate = _candidate("domain.a_to_d", "domain.event.d", parent.event_id)
        reweighter = LegalMechanismReweighter(
            first_order_transitions={
                "domain.event.a": {"domain.event.b": {"count": 10}}
            },
            second_order_transitions={},
            mechanism_descriptors=[
                {
                    "mechanism_id": "domain.a_to_d",
                    "source_event_type_ids": ["domain.event.a"],
                    "target_event_type_id": "domain.event.d",
                }
            ],
        )

        audit = reweighter.apply(
            candidate,
            event_by_id={parent.event_id: parent},
            event_history=[parent],
            temporal_family="conditional_lognormal",
        )

        self.assertEqual(audit["reason"], "legal_transition_unobserved")
        self.assertNotIn(MECHANISM_HAZARD_MULTIPLIER, candidate.temporal_inputs)

    def test_proportional_hazards_multiplier_changes_supported_clocks(self) -> None:
        models = [
            ConditionalLogNormalModel(
                "lognormal", math.log(100.0), 0.4, {}
            ),
            ConditionalGammaModel("gamma", 2.0, math.log(1.0 / 50.0), {}),
            ConditionalWeibullModel("weibull", 1.5, math.log(100.0), {}),
        ]
        for model in models:
            baseline = _clock_candidate(model.model_id, 1.0)
            faster = _clock_candidate(model.model_id, 2.0)
            model.initialize(baseline, random.Random(42))
            model.initialize(faster, random.Random(42))
            self.assertLess(
                faster.scheduled_time,
                baseline.scheduled_time,
                msg=model.model_id,
            )

    def test_scheduler_records_reweighting_without_breaking_contract(self) -> None:
        package = HealthcarePackage(
            {
                "nodes_per_context": 50,
                "scenario_weights": {"temperature_escalation": 1.0},
            }
        )
        alert = "healthcare.workflow.monitoring_alert.created"
        review = "healthcare.workflow.clinical_review.completed"
        reweighter = LegalMechanismReweighter(
            first_order_transitions={
                alert: {
                    review: {"count": 80},
                    "healthcare.observation.follow_up.recorded": {"count": 20},
                }
            },
            second_order_transitions={},
            mechanism_descriptors=package.catalog()["mechanisms"],
        )
        result = SimulationEngine(
            package,
            build_healthcare_temporal_models(),
            mechanism_reweighter=reweighter,
        ).run(episode_index=0, seed=20260918)

        review_candidates = [
            item
            for item in result.candidates
            if item.mechanism_id == "healthcare.alert_to_clinical_review"
        ]
        self.assertTrue(result.validation["passed"])
        self.assertTrue(review_candidates)
        self.assertEqual(
            review_candidates[0].provenance["mechanism_reweighting"]["status"],
            "applied",
        )
        self.assertAlmostEqual(
            review_candidates[0].temporal_inputs[MECHANISM_HAZARD_MULTIPLIER],
            1.6,
        )


def _event(event_id: str, event_type_id: str, at_time: float) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        episode_id="episode",
        domain="domain",
        event_type_id=event_type_id,
        temporal=TemporalExtent(at_time),
        participants=[],
    )


def _candidate(mechanism_id: str, target: str, parent_id: str) -> Candidate:
    return Candidate(
        candidate_id="candidate",
        episode_id="episode",
        mechanism_id=mechanism_id,
        target_event_type_id=target,
        parent_event_ids=[parent_id],
        participants=[],
        activation_time=10.0,
        temporal_model_ref="model",
        temporal_inputs={},
    )


def _clock_candidate(model_id: str, multiplier: float) -> Candidate:
    return Candidate(
        candidate_id=f"candidate_{multiplier}",
        episode_id="episode",
        mechanism_id="mechanism",
        target_event_type_id="target",
        parent_event_ids=["parent"],
        participants=[],
        activation_time=0.0,
        temporal_model_ref=model_id,
        temporal_inputs={MECHANISM_HAZARD_MULTIPLIER: multiplier},
    )


if __name__ == "__main__":
    unittest.main()
