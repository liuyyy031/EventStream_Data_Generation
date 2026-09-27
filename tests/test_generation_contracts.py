"""Regression tests for branching, scientific time, topology and safe priors."""
import copy
import math
import random
import unittest
from data_generation.domain_packages.healthcare import HealthcarePackage, build_healthcare_temporal_models
from data_generation.generation_core.scheduler import SimulationEngine, _build_risk_set
from data_generation.generation_core.models import Candidate, Entity, EventParticipant
from data_generation.generation_core.temporal import ExponentialArrivalModel, TemporalModelRegistry
from data_generation.generation_core.advanced_temporal import SelfExcitingHazardModel
from data_generation.generation_core.continuation import SimulationSnapshot, resample_surviving_clock
from data_generation.generation_core.time_axis import TimeAxis, sample_duration
from data_generation.generation_core.topology import RelationLayerSpec, SparseHeterogeneousTopologyGenerator
from data_generation.generation_core.distribution_calibration import summarize_streams, summarize_reference_sequences
from data_generation.generation_core.reference_data import ReferenceDataset, ReferenceWindow, ReferenceEvent
from data_generation.generation_core.history import HistoryIndex
from data_generation.generation_core.models import EventRecord, TemporalExtent


def clock(name="c", model="m", inputs=None):
    return Candidate(name, "episode", "mechanism", "test.event", [], [], 0., model, inputs or {}, combination="none")


class GenerationContracts(unittest.TestCase):
    def engine(self, **kwargs):
        return SimulationEngine(HealthcarePackage({"nodes_per_context": 32}), build_healthcare_temporal_models(), **kwargs)

    def test_checkpoint_exact_resume(self):
        engine = self.engine()
        whole = engine.run(0, 79)
        snapshot = engine.run(0, 79, stop_after_events=2)
        self.assertIsInstance(snapshot, SimulationSnapshot)
        resumed = engine.run(0, 79, snapshot=snapshot)
        self.assertEqual([e.to_dict() for e in whole.events], [e.to_dict() for e in resumed.events])
        self.assertEqual([c.to_dict() for c in whole.candidates], [c.to_dict() for c in resumed.candidates])
        self.assertEqual(whole.final_state, resumed.final_state)

    def test_continuations_preserve_prefix_and_snapshot(self):
        engine = self.engine()
        snapshot = engine.run(0, 79, stop_after_events=2)
        before = copy.deepcopy(snapshot.state)
        a = engine.run(0, 79, snapshot=snapshot, continuation_seed=100)
        b = engine.run(0, 79, snapshot=snapshot, continuation_seed=200)
        self.assertTrue(a.validation["passed"])
        self.assertTrue(b.validation["passed"])
        self.assertEqual([e.to_dict() for e in a.events[:2]], [e.to_dict() for e in b.events[:2]])
        self.assertNotEqual([e.to_dict() for e in a.events[2:]], [e.to_dict() for e in b.events[2:]])
        self.assertEqual(a.episode_attributes["prefix_group_id"], b.episode_attributes["prefix_group_id"])
        self.assertEqual(before["context"].state, snapshot.state["context"].state)

    def test_conditional_exponential_remaining_wait(self):
        model = ExponentialArrivalModel("m", .1)
        candidate = clock()
        model.initialize(candidate, random.Random(9))
        rng = random.Random(17)
        expected = 30 - math.log(random.Random(17).random()) / .1
        resample_surviving_clock(candidate, model, 30, 1000, rng)
        self.assertAlmostEqual(candidate.scheduled_time, expected, places=7)
        self.assertEqual(candidate.activation_time, 0)

    def test_snapshot_rejects_changed_models(self):
        engine = self.engine()
        snapshot = engine.run(0, 79, stop_after_events=1)
        engine.temporal_models.register(ExponentialArrivalModel("extra", .1))
        with self.assertRaises(ValueError):
            engine.run(0, 79, snapshot=snapshot)

    def test_time_units_and_rates_preserve_integrated_hazard(self):
        axis = TimeAxis(kind="deep_time", unit="Myr")
        self.assertAlmostEqual(axis.rate_to_per_second(.2) * axis.to_seconds(5), 1)
        self.assertEqual(axis.from_seconds(axis.to_seconds(2)), 2)
        with self.assertRaises(ValueError):
            TimeAxis(kind="calendar", unit="year")

    def test_horizon_sampling_is_pre_generation_and_variable(self):
        policy = {"family": "uniform", "unit": "hour", "minimum": 1, "maximum": 4}
        durations = [sample_duration(policy, 1, random.Random(seed)) for seed in range(20)]
        self.assertTrue(all(3600 <= d <= 14400 for d in durations))
        self.assertGreater(len(set(durations)), 1)
        engine = self.engine(observation_duration=policy, time_axis={"unit": "hour"})
        result = engine.run(0, 1)
        self.assertLessEqual(result.duration_seconds, 14400)
        self.assertEqual(result.time_axis["unit"], "hour")

    def test_support_horizon_cannot_be_silently_extended(self):
        with self.assertRaises(ValueError):
            self.engine(observation_duration={"family": "fixed", "unit": "year", "value": 1}).run(0, 1)

    def test_forbidden_mixing_applies_to_every_edge(self):
        entities = [Entity(str(i), "test", "test.node", attributes={"group": "a" if i < 12 else "b"}) for i in range(24)]
        layer = RelationLayerSpec("test.link", ("test.node",), ("test.node",), community_mode="semantic", community_attribute="group", mixing_weights={"a": {"a": 1}, "b": {"b": 1}}, expected_out_degree=1)
        graph = SparseHeterogeneousTopologyGenerator().generate(context_id="c", domain="test", entities=entities, layers=[layer], seed=7)
        self.assertTrue(graph.relations)
        self.assertTrue(all(e.provenance["source_group"] == e.provenance["target_group"] for e in graph.relations))
        from dataclasses import replace
        with self.assertRaises(ValueError):
            SparseHeterogeneousTopologyGenerator().generate(context_id="c", domain="test", entities=entities, layers=[replace(layer, ensure_weak_connectivity=True)], seed=7)

    def test_no_community_does_not_require_attributes(self):
        entities = [Entity(str(i), "test", "test.node") for i in range(15)]
        layer = RelationLayerSpec("test.link", ("test.node",), ("test.node",), community_mode="none")
        graph = SparseHeterogeneousTopologyGenerator().generate(context_id="c", domain="test", entities=entities, layers=[layer], seed=7)
        self.assertTrue(all(e.provenance["source_group"] == 0 for e in graph.relations))

    def test_simultaneous_records_do_not_invent_transition_order(self):
        result = summarize_streams([[(0, "a"), (1, "b"), (1, "c"), (2, "d"), (3, "e")]])
        self.assertEqual(result["transition_counts"], {"d->e": 1})
        self.assertEqual(result["second_order_transitions"], {})

    def test_entity_scoped_statistics_do_not_cross_people(self):
        window = ReferenceWindow("w", "c", "test", "2020-01-01T00:00:00+00:00", 10, "train")
        events = [ReferenceEvent("a", "w", "c", "test", "a", 0, [EventParticipant("p1", "subject")]), ReferenceEvent("b", "w", "c", "test", "b", 1, [EventParticipant("p2", "subject")])]
        result = summarize_reference_sequences(ReferenceDataset("test", [window], events, source_metadata={"sequence_scope": "entity"}), split="train")
        self.assertEqual(result["transition_counts"], {})

    def test_hawkes_update_preserves_accumulated_hazard(self):
        model = SelfExcitingHazardModel("m", .001, .1, .1)
        candidate = clock(inputs={"history_event_offsets": []})
        model.initialize(candidate, random.Random(2))
        old_threshold = candidate.random_threshold
        old_due = candidate.scheduled_time
        model.update(candidate, 5, {"history_event_offsets": [5]})
        self.assertAlmostEqual(candidate.accumulated_hazard, .005)
        self.assertEqual(candidate.random_threshold, old_threshold)
        self.assertLess(candidate.scheduled_time, old_due)
        self.assertAlmostEqual(model.measure(candidate, 5)["hazard_per_second"], .101)

    def test_risk_set_survival_is_conditioned_on_prefix(self):
        model = ExponentialArrivalModel("m", .1)
        candidate = clock()
        model.initialize(candidate, random.Random(2))
        risk = _build_risk_set(risk_set_id="r", event_id="e", selected=candidate, candidates={"c": candidate}, at_time=20, temporal_models=TemporalModelRegistry([model]), survival_baselines={"c": math.exp(-1)}, conditioning_time=10)
        self.assertAlmostEqual(risk.factorization["time"]["joint_survival_probability"], math.exp(-1))

    def test_observable_history_has_no_future_or_cross_entity_leakage(self):
        event = EventRecord("e", "episode", "test", "test.event", TemporalExtent(1, observed_at_offset_seconds=8), [EventParticipant("p", "subject")])
        candidate = clock(inputs={"history_spec": {"scope": "entity", "available_at": "observed_at"}})
        candidate.participants = [EventParticipant("p", "subject")]
        history = HistoryIndex([event])
        self.assertEqual(history.inputs(candidate, 7)["history_count"], 0)
        self.assertEqual(history.inputs(candidate, 8)["history_count"], 1)
        candidate.participants = [EventParticipant("other", "subject")]
        self.assertEqual(history.inputs(candidate, 9)["history_count"], 0)

    def test_explicit_zero_event_windows_survive_normalization(self):
        from unittest.mock import patch
        from data_generation.generation_core.tabular_reference_adapter import GenericTabularReferenceAdapter
        mapping = {"schema_version": "generic-tabular-reference-mapping-v1", "domain": "test", "source_dataset_id": "test", "source_version": "1", "format": "csv",
            "columns": {"stream_id": "stream", "timestamp": "time", "event_type": "type"},
            "window": {"records": [{"stream_id": "empty", "context_id": "subject", "split": "train", "start_time": "2020-01-01T00:00:00+00:00", "duration_seconds": 100, "observation_complete": True}]}}
        module = "data_generation.generation_core.tabular_reference_adapter"
        with patch(module+"._read_records", return_value=[]), patch(module+"._file_sha256", return_value="test"):
            dataset = GenericTabularReferenceAdapter(mapping).load("unused.csv")
        self.assertEqual(len(dataset.windows), 1)
        self.assertEqual(dataset.events, [])
        self.assertTrue(dataset.windows[0].provenance["observation_complete"])
        self.assertEqual(summarize_reference_sequences(dataset, split="train")["zero_event_window_count"], 1)

    def test_root_rate_refuses_event_only_span(self):
        from data_generation.generation_core.calibration import build_root_event_observations
        from data_generation.generation_core.mechanisms import load_mechanism_registry
        from pathlib import Path
        registry = load_mechanism_registry(Path(__file__).resolve().parents[1]/"domain_packages/transportation/mechanism_specs.json")
        mechanism = registry.get("transportation.exogenous_incident_arrival")
        window = ReferenceWindow("w", "c", "transportation", "2020-01-01T00:00:00+00:00", 10, "train")
        with self.assertRaisesRegex(ValueError, "complete observation"):
            build_root_event_observations(ReferenceDataset("transportation", [window], []), mechanism)

    def test_required_edges_cannot_bypass_degree_limits(self):
        entities = [Entity(str(i), "test", "test.node") for i in range(4)]
        layer = RelationLayerSpec("test.link", ("test.node",), ("test.node",), community_mode="none", max_out_degree=1, required_pairs=(("0", "1"), ("0", "2")))
        with self.assertRaises(ValueError):
            SparseHeterogeneousTopologyGenerator().generate(context_id="c", domain="test", entities=entities, layers=[layer], seed=7)

    def test_delayed_observation_wakes_previously_zero_hazard(self):
        from data_generation.tests.test_generation_core import _PriorityDomain
        from data_generation.generation_core.domain import ObservationPlan
        from data_generation.generation_core.temporal import DeterministicDelayModel

        class ObservedDomain(_PriorityDomain):
            def seed_candidates(self, context, rng):
                specs = super().seed_candidates(context, rng)
                specs[0].temporal_model_ref = "start"
                specs[1].temporal_model_ref = "excite"
                specs[1].temporal_inputs = {"history_spec": {"scope": "entity", "available_at": "observed_at", "event_types": ["test.stop"]}}
                return specs

            def apply_event(self, event, candidate, context):
                pass

            def observation_plan(self, event, candidate, context):
                return ObservationPlan(observation_model_ref="observe") if event.event_type_id == "test.stop" else ObservationPlan()

        models = TemporalModelRegistry([DeterministicDelayModel("start", 1), DeterministicDelayModel("observe", 5), SelfExcitingHazardModel("excite", 0, 5, .1)])
        result = SimulationEngine(ObservedDomain(), models).run(0, 2)
        self.assertTrue(result.validation["passed"])
        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.events[0].temporal.observed_at_offset_seconds, 6)
        self.assertGreaterEqual(result.events[1].temporal.occurrence_start_offset_seconds, 6)

    def test_explicit_mutual_exclusion_only_cancels_declared_group(self):
        from data_generation.tests.test_generation_core import _PriorityDomain
        from data_generation.generation_core.temporal import DeterministicDelayModel

        class ExclusiveDomain(_PriorityDomain):
            def seed_candidates(self, context, rng):
                specs = super().seed_candidates(context, rng)
                for spec in specs:
                    spec.attributes["exclusive_group_id"] = "one_choice"
                return specs

            def apply_event(self, event, candidate, context):
                pass

        result = SimulationEngine(ExclusiveDomain(), TemporalModelRegistry([DeterministicDelayModel("test.delay", 1)])).run(0, 2)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.candidates[1].terminal_reason, "explicit_mutual_exclusion")
        self.assertEqual(result.termination_reason, "candidate_set_exhausted")


if __name__ == "__main__":
    unittest.main()
