from __future__ import annotations

import random
import sys
import unittest
import uuid
from pathlib import Path


DATA_GENERATION = Path(__file__).resolve().parents[1]
if str(DATA_GENERATION) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION))

from domain_packages.transportation import (  # noqa: E402
    TransportationPackage,
    build_transportation_temporal_models,
)
from domain_packages.transportation.package import (  # noqa: E402
    COLLISION,
    HEAVY_RAIN_START,
    ROAD_CLOSURE_START,
)
from generation_core.domain import (  # noqa: E402
    CandidateSpec,
    CandidateUpdate,
    EpisodeContext,
    ObservationPlan,
)
from generation_core.models import (  # noqa: E402
    Candidate,
    CandidateStatus,
    Entity,
    EventParticipant,
    EventRecord,
    TemporalExtent,
)
from generation_core.pipeline import GenerationPipeline  # noqa: E402
from generation_core.scheduler import SimulationEngine, _resolve_activation_time  # noqa: E402
from generation_core.temporal import (  # noqa: E402
    DeterministicDelayModel,
    PiecewiseExponentialHazardModel,
    TemporalModelRegistry,
)


class TemporalModelTest(unittest.TestCase):
    def test_time_varying_hazard_preserves_threshold_and_accumulates_hazard(self) -> None:
        model = PiecewiseExponentialHazardModel(
            "test.hazard", baseline_rate_per_second=0.1, coefficients={"load": 1.0}
        )
        candidate = Candidate(
            candidate_id="candidate",
            episode_id="episode",
            mechanism_id="mechanism",
            target_event_type_id="test.target",
            parent_event_ids=[],
            participants=[],
            activation_time=0.0,
            temporal_model_ref=model.model_id,
            temporal_inputs={"load": 0.0},
        )
        model.initialize(candidate, random.Random(7))
        threshold = candidate.random_threshold
        first_rate = candidate.current_hazard_rate
        model.update(candidate, 2.5, {"load": 0.5})
        self.assertEqual(candidate.random_threshold, threshold)
        self.assertAlmostEqual(candidate.accumulated_hazard, first_rate * 2.5)
        self.assertGreater(candidate.current_hazard_rate, first_rate)

    def test_hazard_onset_delay_enforces_physical_minimum(self) -> None:
        model = PiecewiseExponentialHazardModel(
            "test.delayed_hazard", baseline_rate_per_second=0.1, coefficients={}
        )
        candidate = Candidate(
            candidate_id="candidate",
            episode_id="episode",
            mechanism_id="mechanism",
            target_event_type_id="test.target",
            parent_event_ids=[],
            participants=[],
            activation_time=10.0,
            temporal_model_ref=model.model_id,
            temporal_inputs={"hazard_start_delay_seconds": 30.0},
        )
        model.initialize(candidate, random.Random(2))
        self.assertEqual(candidate.hazard_active_from, 40.0)
        self.assertGreaterEqual(float(candidate.scheduled_time), 40.0)
        model.update(candidate, 20.0, candidate.temporal_inputs)
        self.assertEqual(candidate.accumulated_hazard, 0.0)

    def test_all_of_activation_uses_latest_parent(self) -> None:
        left = EventRecord(
            "left",
            "episode",
            "test",
            "test.left",
            TemporalExtent(10.0),
            [EventParticipant("entity", "subject")],
        )
        right = EventRecord(
            "right",
            "episode",
            "test",
            "test.right",
            TemporalExtent(14.0),
            [EventParticipant("entity", "subject")],
        )
        spec = CandidateSpec(
            mechanism_id="joint",
            target_event_type_id="test.target",
            parent_event_ids=["left", "right"],
            participants=[EventParticipant("entity", "subject")],
            temporal_model_ref="core.immediate",
            temporal_inputs={},
            combination="all_of",
        )
        self.assertEqual(_resolve_activation_time(spec, {"left": left, "right": right}), 14.0)

    def test_any_of_records_only_the_realized_trigger(self) -> None:
        parent = EventRecord(
            "parent",
            "episode",
            "test",
            "test.parent",
            TemporalExtent(4.0),
            [EventParticipant("entity", "subject")],
        )
        spec = CandidateSpec(
            mechanism_id="alternative",
            target_event_type_id="test.target",
            parent_event_ids=["parent", "unrealized_alternative"],
            participants=[EventParticipant("entity", "subject")],
            temporal_model_ref="core.immediate",
            temporal_inputs={},
            combination="any_of",
        )
        with self.assertRaises(ValueError):
            _resolve_activation_time(spec, {"parent": parent})


class SchedulerTest(unittest.TestCase):
    def test_same_time_priority_revalidates_and_cancels_later_candidate(self) -> None:
        registry = TemporalModelRegistry([DeterministicDelayModel("test.delay", 5.0)])
        result = SimulationEngine(_PriorityDomain(), registry).run(0, 11)
        self.assertTrue(result.validation["passed"])
        self.assertEqual([event.event_type_id for event in result.events], ["test.stop"])
        status = {item.attributes["kind"]: item.status for item in result.candidates}
        self.assertEqual(status["stop"], CandidateStatus.FIRED)
        self.assertEqual(status["dependent"], CandidateStatus.CANCELLED)

    def test_transportation_package_is_reproducible_and_valid(self) -> None:
        config = {
            "nodes_per_context": 24,
            "network_count": 1,
            "structure_families": ["corridor"],
            "duration_seconds": 3600.0,
            "root_rate_per_hour": 6.0,
            "max_root_events": 2,
        }
        package = TransportationPackage(config)
        registry = build_transportation_temporal_models()
        first = SimulationEngine(package, registry, max_events=100).run(0, 31)
        second = SimulationEngine(package, registry, max_events=100).run(0, 31)
        self.assertTrue(first.validation["passed"])
        self.assertEqual(
            [event.to_dict() for event in first.events],
            [event.to_dict() for event in second.events],
        )
        self.assertTrue(first.event_relations)
        self.assertTrue(
            all(relation.temporal_link.temporal_model_ref for relation in first.event_relations)
        )
        context_relation_by_id = {
            relation.relation_id: relation for relation in first.context_relations
        }
        for relation in first.event_relations:
            context_relation_id = relation.attributes.get("context_relation_id")
            if context_relation_id:
                minimum = context_relation_by_id[
                    context_relation_id
                ].attributes["free_flow_seconds"]
                self.assertGreaterEqual(
                    relation.temporal_link.lag_seconds + 1e-6,
                    minimum,
                )
        self.assertTrue(
            all(
                event.temporal.observed_at_offset_seconds
                is None
                or event.temporal.observed_at_offset_seconds
                >= event.temporal.occurrence_start_offset_seconds
                for event in first.events
            )
        )

    def test_transportation_mechanism_families_and_compound_parentage(self) -> None:
        expected_root_type = {
            "accident_propagation": COLLISION,
            "weather_disruption": HEAVY_RAIN_START,
            "planned_closure": ROAD_CLOSURE_START,
            "compound_weather_accident": HEAVY_RAIN_START,
        }
        for family, root_type in expected_root_type.items():
            with self.subTest(family=family):
                package = TransportationPackage(
                    {
                        "nodes_per_context": 12,
                        "network_count": 1,
                        "duration_seconds": 14_400.0,
                        "root_rate_per_hour": 20.0,
                        "max_root_events": 1,
                        "scenario_weights": {family: 1.0},
                    }
                )
                result = SimulationEngine(
                    package, build_transportation_temporal_models(), max_events=50
                ).run(0, 31)
                self.assertTrue(result.validation["passed"])
                self.assertIn(root_type, {event.event_type_id for event in result.events})
                self.assertEqual(result.episode_attributes["scenario_family"], family)
                if family == "compound_weather_accident":
                    multi_parent = [
                        item for item in result.candidates
                        if item.status is CandidateStatus.FIRED
                        and len(item.parent_event_ids) > 1
                    ]
                    self.assertTrue(multi_parent)
                    target_event_id = multi_parent[0].fired_event_id
                    grouped = [
                        relation for relation in result.event_relations
                        if relation.target_event_id == target_event_id
                    ]
                    self.assertEqual(len(grouped), 2)
                    self.assertEqual(
                        len({relation.mechanism_group_id for relation in grouped}), 1
                    )

    def test_short_window_right_censors_candidate_instead_of_dropping_it(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 8,
                "network_count": 1,
                "duration_seconds": 0.001,
                "root_rate_per_hour": 0.0001,
            }
        )
        result = SimulationEngine(
            package, build_transportation_temporal_models(), max_events=10
        ).run(0, 5)
        self.assertTrue(result.validation["passed"])
        self.assertEqual(len(result.events), 0)
        self.assertEqual(result.candidates[0].status, CandidateStatus.RIGHT_CENSORED)

    def test_event_limit_uses_administrative_censoring_time(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 8,
                "network_count": 1,
                "duration_seconds": 3600.0,
                "root_rate_per_hour": 100.0,
            }
        )
        result = SimulationEngine(
            package, build_transportation_temporal_models(), max_events=1
        ).run(0, 13)
        self.assertTrue(result.validation["passed"])
        event_time = result.events[0].temporal.occurrence_start_offset_seconds
        censored = [
            item for item in result.candidates
            if item.status is CandidateStatus.RIGHT_CENSORED
        ]
        self.assertTrue(censored)
        self.assertTrue(all(item.censored_at == event_time for item in censored))
        self.assertTrue(
            all(item.terminal_reason == "event_limit_reached" for item in censored)
        )

    def test_pipeline_writes_normalized_streams(self) -> None:
        config = {
            "seed": 7,
            "episode_count": 2,
            "max_events_per_episode": 32,
        }
        package = TransportationPackage(
            {
                "nodes_per_context": 12,
                "network_count": 1,
                "duration_seconds": 1200.0,
                "root_rate_per_hour": 10.0,
            }
        )
        test_output_root = DATA_GENERATION / "test_runtime_output"
        test_output_root.mkdir(exist_ok=True)
        output = test_output_root / f"generated_{uuid.uuid4().hex}"
        report = GenerationPipeline(
            package,
            build_transportation_temporal_models(),
            config,
            output,
        ).run()
        self.assertTrue(report["quality_report"]["passed"])
        for filename in (
                "manifest.json",
                "domain_catalog.json",
                "temporal_model_registry.json",
            "contexts.jsonl",
            "entities.jsonl",
            "context_relations.jsonl",
            "episodes.jsonl",
            "events.jsonl",
            "event_relations.jsonl",
                "candidates.jsonl",
                "episode_texts.jsonl",
                "validation.jsonl",
            "quality_report.json",
        ):
                self.assertTrue((output / filename).is_file(), filename)

    def test_pipeline_quality_gate_rejects_administrative_truncation(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 8,
                "network_count": 1,
                "duration_seconds": 3600.0,
                "root_rate_per_hour": 100.0,
                "scenario_weights": {"accident_propagation": 1.0},
            }
        )
        test_output_root = DATA_GENERATION / "test_runtime_output"
        test_output_root.mkdir(exist_ok=True)
        output = test_output_root / f"truncation_{uuid.uuid4().hex}"
        report = GenerationPipeline(
            package,
            build_transportation_temporal_models(),
            {"seed": 13, "episode_count": 1, "max_events_per_episode": 1},
            output,
        ).run()["quality_report"]
        self.assertFalse(report["passed"])
        self.assertEqual(report["administratively_truncated_episode_count"], 1)

    def test_neutral_core_contains_no_transportation_semantics(self) -> None:
        core_root = DATA_GENERATION / "generation_core"
        payload = "\n".join(
            path.read_text(encoding="utf-8")
            for path in core_root.glob("*.py")
        ).lower()
        for token in ("transportation", "congestion", "road_segment", "bpr"):
            self.assertNotIn(token, payload)


class _PriorityDomain:
    domain_id = "test"

    def catalog(self):
        return {
            "domain": "test",
            "entity_types": [
                {"entity_type_id": "test.entity", "required_attributes": []}
            ],
            "event_types": [
                {"event_type_id": "test.stop", "required_attributes": []},
                {"event_type_id": "test.dependent", "required_attributes": []},
            ],
            "context_relation_types": [],
            "event_relation_types": [],
            "mechanisms": [
                {"mechanism_id": "stop_rule"},
                {"mechanism_id": "dependent_rule"},
            ],
        }

    def create_episode_context(self, episode_index: int, seed: int) -> EpisodeContext:
        del episode_index, seed
        return EpisodeContext(
            episode_id="priority_episode",
            context_id="priority_context",
            domain="test",
            start_time="2026-01-01T00:00:00+00:00",
            duration_seconds=20.0,
            entities=[Entity("entity", "test", "test.entity")],
            context_relations=[],
            state={"active": True},
        )

    def seed_candidates(self, context: EpisodeContext, rng: random.Random):
        del context, rng
        participant = [EventParticipant("entity", "subject")]
        return [
            CandidateSpec(
                "stop_rule",
                "test.stop",
                [],
                participant,
                "test.delay",
                {},
                phase_priority=10,
                attributes={"kind": "stop"},
            ),
            CandidateSpec(
                "dependent_rule",
                "test.dependent",
                [],
                participant,
                "test.delay",
                {},
                phase_priority=20,
                attributes={"kind": "dependent"},
            ),
        ]

    def revalidate_candidate(self, candidate, context, event_by_id, now):
        del event_by_id, now
        if candidate.attributes["kind"] == "dependent" and not context.state["active"]:
            return CandidateUpdate(False, "state_was_stopped")
        return CandidateUpdate(True)

    def materialize_event(self, event_id, candidate, context, rng):
        del rng
        return EventRecord(
            event_id,
            context.episode_id,
            "test",
            candidate.target_event_type_id,
            TemporalExtent(float(candidate.scheduled_time)),
            list(candidate.participants),
        )

    def apply_event(self, event, candidate, context):
        del candidate
        if event.event_type_id == "test.stop":
            context.state["active"] = False

    def spawn_candidates(self, event, candidate, context, event_by_id, rng):
        del event, candidate, context, event_by_id, rng
        return []

    def relation_specs(self, event, candidate, context, event_by_id):
        del event, candidate, context, event_by_id
        return []

    def observation_plan(self, event, candidate, context):
        del event, candidate, context
        return ObservationPlan()


if __name__ == "__main__":
    unittest.main()
