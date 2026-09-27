from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path


DATA_GENERATION = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = DATA_GENERATION.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_generation.generation_core.advanced_temporal import (  # noqa: E402
    AvailabilityWindowDelayModel,
    BatchScheduledTimeModel,
    ConditionalLogNormalMixtureModel,
    IntervalTimeModel,
    LogNormalMixtureComponent,
    PeriodicPiecewiseHazardModel,
    RefractoryHazardModel,
    ResourceQueueDelayModel,
    SelfExcitingHazardModel,
)
from data_generation.generation_core.models import Candidate  # noqa: E402
from data_generation.generation_core.temporal import TemporalModelRegistry  # noqa: E402


def _candidate(model_id: str, inputs: dict, *, activation: float = 0.0) -> Candidate:
    return Candidate(
        candidate_id="candidate",
        episode_id="episode",
        mechanism_id="mechanism",
        target_event_type_id="test.event",
        parent_event_ids=[],
        participants=[],
        activation_time=activation,
        temporal_model_ref=model_id,
        temporal_inputs=inputs,
    )


class AdvancedTemporalModelTests(unittest.TestCase):
    def test_periodic_hazard_samples_and_measures_a_nonhomogeneous_clock(self) -> None:
        model = PeriodicPiecewiseHazardModel(
            "test.periodic", 24.0, (0.01, 0.10, 0.02, 0.20)
        )
        candidate = _candidate(model.model_id, {})
        model.initialize(candidate, random.Random(7))
        self.assertIsNotNone(candidate.scheduled_time)
        measure = model.measure(candidate, float(candidate.scheduled_time))
        self.assertEqual(measure["clock_kind"], "stochastic_continuous")
        self.assertGreater(measure["hazard_per_second"], 0.0)
        self.assertGreater(measure["cumulative_hazard"], 0.0)

    def test_self_exciting_history_increases_current_hazard(self) -> None:
        model = SelfExcitingHazardModel(
            "test.hawkes",
            baseline_rate_per_second=0.01,
            excitation_per_second=0.3,
            decay_per_second=0.1,
        )
        with_history = _candidate(
            model.model_id,
            {"history_event_offsets": [0.0, 8.0]},
            activation=10.0,
        )
        without_history = _candidate(
            model.model_id, {"history_event_offsets": []}, activation=10.0
        )
        model.initialize(with_history, random.Random(3))
        model.initialize(without_history, random.Random(3))
        history_measure = model.measure(with_history, 10.0)
        baseline_measure = model.measure(without_history, 10.0)
        self.assertGreater(
            history_measure["hazard_per_second"],
            baseline_measure["hazard_per_second"],
        )
        self.assertLess(
            float(with_history.scheduled_time), float(without_history.scheduled_time)
        )

    def test_refractory_clock_cannot_fire_during_cooldown(self) -> None:
        model = RefractoryHazardModel(
            "test.refractory", cooldown_seconds=30.0, baseline_rate_per_second=0.5
        )
        candidate = _candidate(
            model.model_id,
            {"last_event_offset_seconds": 8.0},
            activation=10.0,
        )
        model.initialize(candidate, random.Random(5))
        self.assertGreaterEqual(float(candidate.scheduled_time), 38.0)
        self.assertEqual(model.measure(candidate, 37.0)["hazard_per_second"], 0.0)

    def test_lognormal_mixture_represents_multimodal_positive_delays(self) -> None:
        model = ConditionalLogNormalMixtureModel(
            "test.mixture",
            (
                LogNormalMixtureComponent(0.7, math.log(10.0), 0.15),
                LogNormalMixtureComponent(0.3, math.log(100.0), 0.20),
            ),
            minimum_seconds=1.0,
            maximum_seconds=300.0,
        )
        candidate = _candidate(model.model_id, {})
        model.initialize(candidate, random.Random(13))
        delay = float(candidate.scheduled_time) - candidate.activation_time
        self.assertGreaterEqual(delay, 1.0)
        self.assertLessEqual(delay, 300.0)
        measure = model.measure(candidate, float(candidate.scheduled_time))
        self.assertEqual(measure["component_count"], 2)
        self.assertGreater(measure["hazard_per_second"], 0.0)

    def test_supported_advanced_hazards_honor_mechanism_reweighting(self) -> None:
        models_and_inputs = [
            (
                PeriodicPiecewiseHazardModel("periodic", 10.0, (0.1, 0.2)),
                {},
            ),
            (
                SelfExcitingHazardModel("hawkes", 0.1, 0.2, 0.3),
                {"history_event_offsets": [0.0]},
            ),
            (RefractoryHazardModel("refractory", 1.0, 0.1), {}),
            (
                ConditionalLogNormalMixtureModel(
                    "mixture",
                    (LogNormalMixtureComponent(1.0, math.log(10.0), 0.4),),
                ),
                {},
            ),
        ]
        for model, inputs in models_and_inputs:
            with self.subTest(model=model.model_id):
                neutral = _candidate(model.model_id, dict(inputs))
                weighted = _candidate(
                    model.model_id,
                    {**inputs, "_mechanism_hazard_multiplier": 2.0},
                )
                model.initialize(neutral, random.Random(31))
                model.initialize(weighted, random.Random(31))
                self.assertLessEqual(
                    float(weighted.scheduled_time), float(neutral.scheduled_time)
                )

    def test_resource_queue_waits_for_capacity_then_service(self) -> None:
        model = ResourceQueueDelayModel(
            "test.queue", default_mean_service_seconds=20.0, service_shape=2.0
        )
        candidate = _candidate(
            model.model_id,
            {
                "resource_available_offset_seconds": 50.0,
                "queue_delay_seconds": 10.0,
            },
            activation=5.0,
        )
        model.initialize(candidate, random.Random(11))
        self.assertGreater(float(candidate.scheduled_time), 60.0)
        self.assertEqual(model.measure(candidate, 59.0)["survival_probability"], 1.0)
        self.assertTrue(
            candidate.temporal_provenance["details"][
                "resource_state_is_domain_supplied"
            ]
        )

    def test_availability_windows_pause_nonworking_time(self) -> None:
        model = AvailabilityWindowDelayModel("test.calendar")
        candidate = _candidate(
            model.model_id,
            {
                "required_work_seconds": 10.0,
                "available_intervals": [[0.0, 10.0], [20.0, 30.0]],
            },
            activation=5.0,
        )
        model.initialize(candidate, random.Random(1))
        self.assertEqual(candidate.scheduled_time, 25.0)
        self.assertEqual(
            model.measure(candidate, 25.0)["clock_kind"], "calendar_policy_atom"
        )

    def test_interval_time_preserves_reported_uncertainty(self) -> None:
        model = IntervalTimeModel("test.interval", selection_policy="uniform")
        candidate = _candidate(
            model.model_id,
            {"earliest_offset_seconds": 20.0, "latest_offset_seconds": 40.0},
            activation=10.0,
        )
        model.initialize(candidate, random.Random(17))
        self.assertGreaterEqual(float(candidate.scheduled_time), 20.0)
        self.assertLessEqual(float(candidate.scheduled_time), 40.0)
        measure = model.measure(candidate, 30.0)
        self.assertAlmostEqual(measure["survival_probability"], 0.5)
        self.assertEqual(measure["reported_interval_offsets"], [20.0, 40.0])

    def test_batch_candidates_share_exact_timestamp_and_batch_identity(self) -> None:
        model = BatchScheduledTimeModel("test.batch")
        candidates = [
            _candidate(
                model.model_id,
                {"batch_id": "batch-1", "batch_offset_seconds": 45.0},
            )
            for _ in range(3)
        ]
        for index, candidate in enumerate(candidates):
            model.initialize(candidate, random.Random(index))
        self.assertEqual({item.scheduled_time for item in candidates}, {45.0})
        measure = model.measure(candidates[0], 45.0)
        self.assertEqual(measure["clock_kind"], "batch_atom")
        self.assertEqual(measure["batch_id"], "batch-1")

    def test_all_models_register_and_publish_auditable_descriptors(self) -> None:
        models = [
            PeriodicPiecewiseHazardModel("periodic", 10.0, (0.1, 0.2)),
            SelfExcitingHazardModel("hawkes", 0.1, 0.2, 0.3),
            RefractoryHazardModel("refractory", 1.0, 0.1),
            ConditionalLogNormalMixtureModel(
                "mixture", (LogNormalMixtureComponent(1.0, 0.0, 1.0),)
            ),
            ResourceQueueDelayModel("queue", 1.0),
            AvailabilityWindowDelayModel("calendar"),
            IntervalTimeModel("interval"),
            BatchScheduledTimeModel("batch"),
        ]
        registry = TemporalModelRegistry(models)
        descriptors = registry.descriptors()
        self.assertEqual(len(descriptors), len(models))
        self.assertTrue(all(item["mathematical_form"] for item in descriptors))


if __name__ == "__main__":
    unittest.main()
