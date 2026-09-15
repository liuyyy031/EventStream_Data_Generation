from __future__ import annotations

import copy
import random
import json
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
    CONGESTION,
    HEAVY_RAIN_START,
    ROAD_CLOSURE_START,
)
from generation_core.domain import (  # noqa: E402
    CandidateSpec,
    CandidateUpdate,
    EpisodeContext,
    ObservationPlan,
)
from generation_core.domain_spec import load_domain_spec  # noqa: E402
from generation_core.calibration import (  # noqa: E402
    CalibrationEngine,
    build_root_event_observations,
)
from generation_core.mechanisms import load_mechanism_registry  # noqa: E402
from generation_core.models import (  # noqa: E402
    Candidate,
    CandidateStatus,
    ContextRelation,
    Entity,
    EventParticipant,
    EventRecord,
    TemporalExtent,
)
from generation_core.reference_data import (  # noqa: E402
    NormalizedJsonlReferenceAdapter,
    ReferenceDataset,
    ReferenceEvent,
    ReferenceWindow,
    validate_reference_dataset,
)
from generation_core.pipeline import GenerationPipeline  # noqa: E402
from generation_core.scheduler import SimulationEngine, _resolve_activation_time  # noqa: E402
from generation_core.semantic_judge import (  # noqa: E402
    LLMSemanticJudge,
    SemanticJudgeResult,
    _build_prompt,
)
from generation_core.temporal import (  # noqa: E402
    ConditionalGammaModel,
    ConditionalWeibullModel,
    DeterministicDelayModel,
    PiecewiseExponentialHazardModel,
    TemporalModelRegistry,
)
from generation_core.validation import validate_text_alignment  # noqa: E402
from generation_core.topology import (  # noqa: E402
    ContextRelationIndex,
    RelationLayerSpec,
    SparseHeterogeneousTopologyGenerator,
)


class TopologyTest(unittest.TestCase):
    def test_all_domain_topology_specs_compile_without_event_priors(self) -> None:
        package_root = DATA_GENERATION / "domain_packages"
        for domain in ("transportation", "healthcare", "distributed_systems"):
            with self.subTest(domain=domain):
                spec = load_domain_spec(package_root / domain / "domain_spec.json")
                builders = {}
                if domain == "transportation":
                    builders["transportation.downstream_relation.v1"] = (
                        lambda _source, _target, _rng: {}
                    )
                layers = spec.compile_relation_layers(attribute_builders=builders)
                self.assertTrue(layers)
                self.assertTrue(
                    all(
                        set(layer.source_type_ids + layer.target_type_ids)
                        <= set(spec.entity_type_ids)
                        for layer in layers
                    )
                )
                if spec.raw["implementation_status"] == "structure_only":
                    self.assertNotIn("event_probabilities", spec.raw)

    def test_domain_spec_rejects_undeclared_layer_override(self) -> None:
        spec = load_domain_spec(
            DATA_GENERATION
            / "domain_packages"
            / "healthcare"
            / "domain_spec.json"
        )
        with self.assertRaises(ValueError):
            spec.compile_relation_layers(
                parameter_overrides={"healthcare.unknown_relation": {"max_out_degree": 3}}
            )

    def test_relation_index_filters_dynamic_edges_by_time_and_direction(self) -> None:
        relation = ContextRelation(
            relation_id="r1",
            domain="test",
            relation_type_id="test.connects",
            source_entity_id="source",
            target_entity_id="target",
            valid_from_offset_seconds=10.0,
            valid_to_offset_seconds=20.0,
        )
        index = ContextRelationIndex([relation])
        self.assertEqual(index.neighbors("source", "test.connects", at_time=9.9), [])
        self.assertEqual(
            index.neighbors("source", "test.connects", at_time=10.0)[0].neighbor_entity_id,
            "target",
        )
        self.assertEqual(
            index.neighbors(
                "target", "test.connects", at_time=20.0, direction="incoming"
            )[0].neighbor_entity_id,
            "source",
        )
        self.assertEqual(index.neighbors("source", "test.connects", at_time=20.1), [])


class CalibrationContractTest(unittest.TestCase):
    def _transport_registry(self):
        return load_mechanism_registry(
            DATA_GENERATION
            / "domain_packages"
            / "transportation"
            / "mechanism_specs.json"
        )

    def _reference_dataset(self) -> ReferenceDataset:
        windows = [
            ReferenceWindow("train_a", "context_a", "transportation", "2026-01-01T00:00:00+00:00", 100.0, "train"),
            ReferenceWindow("train_b", "context_b", "transportation", "2026-01-01T00:00:00+00:00", 100.0, "train"),
            ReferenceWindow("holdout_a", "context_c", "transportation", "2026-01-01T00:00:00+00:00", 100.0, "holdout"),
        ]
        participant = [EventParticipant("road_1", "affected_entity")]
        events = [
            ReferenceEvent("e1", "train_a", "context_a", "transportation", COLLISION, 10.0, participant),
            ReferenceEvent("e2", "train_a", "context_a", "transportation", COLLISION, 40.0, participant),
            ReferenceEvent("e3", "train_b", "context_b", "transportation", COLLISION, 30.0, participant),
            ReferenceEvent("e4", "holdout_a", "context_c", "transportation", COLLISION, 20.0, participant),
            ReferenceEvent("e5", "holdout_a", "context_c", "transportation", COLLISION, 70.0, participant),
        ]
        return ReferenceDataset(
            "transportation",
            windows,
            events,
            source_metadata={
                "domain": "transportation",
                "source_dataset_id": "unit_test_reference",
                "source_version": "1",
                "split_policy": {
                    "unit": "context",
                    "assignment_method": "fixed_unit_test_assignment",
                    "allow_overlapping_windows": False,
                },
            },
        )

    def test_all_domain_mechanism_files_compile_without_cross_domain_core_logic(self) -> None:
        root = DATA_GENERATION / "domain_packages"
        transportation = self._transport_registry()
        healthcare = load_mechanism_registry(root / "healthcare" / "mechanism_specs.json")
        distributed = load_mechanism_registry(root / "distributed_systems" / "mechanism_specs.json")
        self.assertEqual(len(transportation), 11)
        self.assertEqual(len(healthcare), 0)
        self.assertEqual(len(distributed), 0)
        joint = transportation.get(
            "transportation.weather_collision_joint_to_congestion"
        )
        self.assertEqual(joint.parent_combination, "all_of")
        self.assertEqual(
            joint.selection_semantics,
            "cause_specific_competing_hazard",
        )
        self.assertEqual(
            joint.parent_attribution_mode,
            "explicit_realized_parents",
        )
        root = transportation.get("transportation.exogenous_incident_arrival")
        self.assertEqual(root.parent_attribution_mode, "independent_background")

    def test_reference_contract_and_constant_rate_mle_use_holdout_separately(self) -> None:
        dataset = self._reference_dataset()
        report = validate_reference_dataset(dataset)
        self.assertTrue(report["passed"])
        mechanism = self._transport_registry().get(
            "transportation.exogenous_incident_arrival"
        )
        observations = build_root_event_observations(dataset, mechanism)
        train = [item for item in observations if item.split == "train"]
        holdout = [item for item in observations if item.split == "holdout"]
        bundle = CalibrationEngine().fit(
            mechanism,
            train,
            holdout_observations=holdout,
            reference_data_fingerprint=dataset.fingerprint(),
        )
        self.assertAlmostEqual(bundle.parameter_estimates["rate_per_second"], 3.0 / 200.0)
        self.assertEqual(bundle.training_scope["window_count"], 2)
        self.assertEqual(bundle.holdout_metrics["observed_event_count"], 2)
        self.assertAlmostEqual(bundle.holdout_metrics["expected_event_count"], 1.5)
        self.assertEqual(
            bundle.parameter_status,
            "empirically_estimated_pending_domain_acceptance",
        )

    def test_normalized_jsonl_adapter_preserves_source_fingerprint(self) -> None:
        dataset = self._reference_dataset()
        root = DATA_GENERATION / "test_runtime_output" / f"reference_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        (root / "reference_manifest.json").write_text(
            json.dumps(dataset.source_metadata),
            encoding="utf-8",
        )
        (root / "reference_windows.jsonl").write_text(
            "\n".join(json.dumps(item.to_dict()) for item in dataset.windows) + "\n",
            encoding="utf-8",
        )
        (root / "reference_events.jsonl").write_text(
            "\n".join(json.dumps(item.to_dict()) for item in dataset.events) + "\n",
            encoding="utf-8",
        )
        loaded = NormalizedJsonlReferenceAdapter().load(root)
        self.assertEqual(len(loaded.windows), 3)
        self.assertEqual(len(loaded.events), 5)
        self.assertTrue(validate_reference_dataset(loaded)["passed"])
        loaded_again = NormalizedJsonlReferenceAdapter().load(root)
        self.assertEqual(loaded.fingerprint(), loaded_again.fingerprint())

    def test_reference_contract_rejects_future_explicit_parent(self) -> None:
        dataset = self._reference_dataset()
        participant = [EventParticipant("road_1", "affected_entity")]
        dataset.events.append(
            ReferenceEvent(
                "bad_child",
                "train_a",
                "context_a",
                "transportation",
                COLLISION,
                5.0,
                participant,
                explicit_parent_record_ids=["e2"],
            )
        )
        report = validate_reference_dataset(dataset)
        self.assertFalse(report["passed"])
        self.assertTrue(any("later explicit parent" in item for item in report["errors"]))

    def test_context_split_policy_rejects_context_leakage(self) -> None:
        dataset = self._reference_dataset()
        dataset.windows.append(
            ReferenceWindow(
                "leaked_holdout",
                "context_a",
                "transportation",
                "2026-01-02T00:00:00+00:00",
                100.0,
                "holdout",
            )
        )
        report = validate_reference_dataset(dataset)
        self.assertFalse(report["passed"])
        self.assertTrue(any("leaks across" in item for item in report["errors"]))


class TopologyGenerationTest(unittest.TestCase):
    def test_sparse_irregular_topology_is_reproducible_and_not_a_chain(self) -> None:
        entities = [
            Entity(
                entity_id=f"entity_{index:05d}",
                domain="test",
                entity_type_id="test.node",
                context_id="context",
            )
            for index in range(1000)
        ]
        layer = RelationLayerSpec(
            relation_type_id="test.depends_on",
            source_type_ids=("test.node",),
            target_type_ids=("test.node",),
            expected_out_degree=2.4,
            max_out_degree=7,
            community_count=8,
            within_community_bias=3.0,
            degree_sigma=0.8,
            hub_fraction=0.05,
            hub_multiplier=4.0,
            ensure_weak_connectivity=True,
        )
        generator = SparseHeterogeneousTopologyGenerator()
        first = generator.generate(
            context_id="context",
            domain="test",
            entities=entities,
            layers=[layer],
            seed=19,
        )
        second = generator.generate(
            context_id="context",
            domain="test",
            entities=entities,
            layers=[layer],
            seed=19,
        )
        self.assertEqual(
            [item.to_dict() for item in first.relations],
            [item.to_dict() for item in second.relations],
        )
        stats = first.statistics["layers"][0]
        self.assertEqual(stats["weak_component_count"], 1)
        self.assertGreater(stats["max_out_degree"], 2)
        self.assertGreater(stats["unique_out_degree_count"], 3)
        self.assertLess(len(first.relations), len(entities) * 3)
        self.assertFalse(first.statistics["dense_matrix_materialized"])

    def test_typed_layer_never_creates_semantically_invalid_edges(self) -> None:
        entities = [
            *[
                Entity(f"device_{index}", "healthcare", "healthcare.device")
                for index in range(20)
            ],
            *[
                Entity(f"patient_{index}", "healthcare", "healthcare.patient")
                for index in range(40)
            ],
        ]
        layer = RelationLayerSpec(
            relation_type_id="healthcare.device_monitors_patient",
            source_type_ids=("healthcare.device",),
            target_type_ids=("healthcare.patient",),
            family="typed_bipartite",
            expected_out_degree=2.0,
            max_out_degree=4,
            ensure_weak_connectivity=False,
        )
        result = SparseHeterogeneousTopologyGenerator().generate(
            context_id="clinical_context",
            domain="healthcare",
            entities=entities,
            layers=[layer],
            seed=23,
        )
        type_by_id = {item.entity_id: item.entity_type_id for item in entities}
        self.assertTrue(result.relations)
        self.assertTrue(
            all(
                type_by_id[item.source_entity_id] == "healthcare.device"
                and type_by_id[item.target_entity_id] == "healthcare.patient"
                for item in result.relations
            )
        )

    def test_directed_acyclic_layer_and_temporal_validity(self) -> None:
        entities = [
            Entity(f"step_{index:03d}", "test", "test.workflow_step")
            for index in range(100)
        ]
        layer = RelationLayerSpec(
            relation_type_id="test.precedes",
            source_type_ids=("test.workflow_step",),
            target_type_ids=("test.workflow_step",),
            family="directed_acyclic",
            directed=True,
            allow_cycles=False,
            expected_out_degree=1.6,
            max_out_degree=4,
            ensure_weak_connectivity=True,
            temporal_edge_fraction=1.0,
        )
        result = SparseHeterogeneousTopologyGenerator().generate(
            context_id="workflow",
            domain="test",
            entities=entities,
            layers=[layer],
            seed=29,
            observation_window_seconds=1000.0,
        )
        rank = {
            entity_id: index
            for index, entity_id in enumerate(
                sorted(entity.entity_id for entity in entities)
            )
        }
        self.assertEqual(
            result.statistics["layers"][0]["weak_component_count"], 1
        )
        self.assertTrue(
            all(
                rank[item.source_entity_id] < rank[item.target_entity_id]
                and item.valid_from_offset_seconds is not None
                and item.valid_to_offset_seconds is not None
                and item.valid_from_offset_seconds <= item.valid_to_offset_seconds
                for item in result.relations
            )
        )


class TemporalModelTest(unittest.TestCase):
    def test_gamma_and_weibull_expose_valid_continuous_clock_measures(self) -> None:
        models = [
            ConditionalGammaModel(
                "test.gamma",
                shape=2.0,
                log_rate_intercept=-2.0,
                coefficients={"load": 0.2},
                minimum_seconds=0.1,
                maximum_seconds=100.0,
            ),
            ConditionalWeibullModel(
                "test.weibull",
                shape=1.5,
                log_scale_intercept=2.0,
                coefficients={"load": -0.1},
                minimum_seconds=0.1,
                maximum_seconds=100.0,
            ),
        ]
        for model in models:
            with self.subTest(model=model.model_id):
                candidate = Candidate(
                    candidate_id="candidate",
                    episode_id="episode",
                    mechanism_id="mechanism",
                    target_event_type_id="test.target",
                    parent_event_ids=[],
                    participants=[],
                    activation_time=0.0,
                    temporal_model_ref=model.model_id,
                    temporal_inputs={"load": 0.4},
                )
                model.initialize(candidate, random.Random(17))
                measure = model.measure(candidate, float(candidate.scheduled_time))
                self.assertEqual(measure["clock_kind"], "stochastic_continuous")
                self.assertGreater(measure["hazard_per_second"], 0.0)
                self.assertGreaterEqual(measure["survival_probability"], 0.0)
                self.assertLessEqual(measure["survival_probability"], 1.0)

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


class SemanticJudgeTest(unittest.TestCase):
    def test_prompt_defines_root_and_cancelled_candidate_semantics(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 8,
                "context_count": 1,
                "duration_seconds": 1200.0,
                "root_rate_per_hour": 10.0,
                "scenario_weights": {"accident_propagation": 1.0},
            }
        )
        result = SimulationEngine(
            package, build_transportation_temporal_models(), max_events=32
        ).run(0, 41)
        prompt = _build_prompt(
            result,
            package.render_episode_text(result),
            package.catalog(),
            reason_limit=5,
        )
        payload = json.loads(prompt.split("\nDATA:\n", 1)[1])
        root_contract = payload["review_contract"]["root_event_semantics"]
        self.assertFalse(root_contract["entity_temporal_primacy"])
        self.assertTrue(root_contract["prior_events_on_the_same_entity_are_allowed"])
        cancellation_contract = payload["review_contract"][
            "cancelled_candidate_semantics"
        ]
        self.assertTrue(cancellation_contract["does_not_invalidate_parent_event"])
        root_mechanism = next(
            item
            for item in payload["mechanism_catalog"]
            if item["mechanism_id"]
            == "transportation.exogenous_incident_arrival"
        )
        self.assertIn("not the first event", root_mechanism["notes"])

    def test_llm_protocol_retries_invalid_json_without_regenerating(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 8,
                "context_count": 1,
                "duration_seconds": 1200.0,
                "root_rate_per_hour": 10.0,
            }
        )
        result = SimulationEngine(
            package, build_transportation_temporal_models(), max_events=32
        ).run(0, 41)
        text_alignment = package.render_episode_text(result)
        client = _ResponseClient(
            [
                "not-json",
                json.dumps(
                    {
                        "passed": True,
                        "reasons": ["The supplied records are coherent."],
                        "flagged_record_ids": [],
                    }
                ),
            ]
        )
        judged = LLMSemanticJudge(
            client, protocol_max_attempts=2
        ).evaluate(result, text_alignment, package.catalog())
        self.assertTrue(judged.passed)
        self.assertEqual(judged.protocol_attempt_count, 2)
        self.assertEqual(len(judged.protocol_errors), 1)


class SchedulerTest(unittest.TestCase):
    def test_later_independent_root_on_congested_entity_is_a_valid_chain(self) -> None:
        base_seed = 20260915
        episode_index = 80
        episode_seed = base_seed + episode_index * 104729
        package = TransportationPackage(
            {
                "base_seed": base_seed,
                "context_count": 4,
                "nodes_per_context": 1000,
                "duration_seconds": 7200.0,
                "root_rate_per_hour": 1.5,
                "max_root_events": 3,
                "max_propagation_depth": 4,
                "scenario_weights": {
                    "accident_propagation": 0.35,
                    "weather_disruption": 0.25,
                    "planned_closure": 0.2,
                    "compound_weather_accident": 0.2,
                },
            }
        )
        result = SimulationEngine(
            package, build_transportation_temporal_models(), max_events=512
        ).run(episode_index, episode_seed)
        self.assertTrue(result.validation["passed"])
        events_by_id = {event.event_id: event for event in result.events}
        qualifying_roots = []
        for event in result.events:
            if event.event_type_id != COLLISION or event.event_role != "root":
                continue
            road_id = event.participants[0].entity_id
            event_time = event.temporal.occurrence_start_offset_seconds
            earlier_congestion = any(
                other.event_type_id == CONGESTION
                and other.participants[0].entity_id == road_id
                and other.temporal.occurrence_start_offset_seconds < event_time
                for other in result.events
            )
            cancelled_child = any(
                event.event_id in candidate.parent_event_ids
                and candidate.target_event_type_id == CONGESTION
                and candidate.status is CandidateStatus.CANCELLED
                and candidate.terminal_reason == "target_road_is_not_normal"
                for candidate in result.candidates
            )
            if earlier_congestion and cancelled_child:
                qualifying_roots.append(event.event_id)
        self.assertEqual(
            qualifying_roots,
            ["transport_episode_0000080_event_000020"],
        )
        self.assertIn(qualifying_roots[0], events_by_id)

    def test_independent_root_remains_valid_on_an_already_affected_entity(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 8,
                "context_count": 1,
                "duration_seconds": 1200.0,
                "scenario_weights": {"accident_propagation": 1.0},
            }
        )
        context = package.create_episode_context(0, 71)
        spec = package.seed_candidates(context, random.Random(71))[0]
        road_id = spec.participants[0].entity_id
        context.state["road_status"][road_id] = "congested"
        candidate = Candidate(
            candidate_id="root_candidate",
            episode_id=context.episode_id,
            mechanism_id=spec.mechanism_id,
            target_event_type_id=spec.target_event_type_id,
            parent_event_ids=list(spec.parent_event_ids),
            participants=list(spec.participants),
            activation_time=float(spec.activation_time or 0.0),
            temporal_model_ref=spec.temporal_model_ref,
            temporal_inputs=dict(spec.temporal_inputs),
            combination=spec.combination,
            attributes=dict(spec.attributes),
            provenance=dict(spec.provenance),
        )
        decision = package.revalidate_candidate(candidate, context, {}, 100.0)
        self.assertTrue(decision.valid)

    def test_relation_text_semantics_are_deterministically_aligned(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 12,
                "context_count": 1,
                "duration_seconds": 7200.0,
                "root_rate_per_hour": 10.0,
                "scenario_weights": {"weather_disruption": 1.0},
            }
        )
        result = SimulationEngine(
            package, build_transportation_temporal_models(), max_events=32
        ).run(0, 1)
        statistical_relation = next(
            relation
            for relation in result.event_relations
            if relation.relation_class == "statistical_influence"
        )
        text_alignment = package.render_episode_text(result)
        self.assertTrue(
            validate_text_alignment(
                result, text_alignment, package.catalog()
            )["passed"]
        )
        relation_claim = next(
            claim
            for claim in text_alignment["claims"]
            if statistical_relation.relation_id in claim["relation_ids"]
        )
        assertion = relation_claim["relation_assertions"][0]
        self.assertEqual(
            assertion["asserted_relation_class"], "statistical_influence"
        )
        self.assertIn("statistically", assertion["surface_predicate"])
        self.assertNotIn(
            "induced",
            text_alignment["sentences"][relation_claim["sentence_index"]],
        )

        tampered = copy.deepcopy(text_alignment)
        tampered_claim = next(
            claim
            for claim in tampered["claims"]
            if statistical_relation.relation_id in claim["relation_ids"]
        )
        tampered_claim["relation_assertions"][0][
            "asserted_relation_class"
        ] = "causal"
        checked = validate_text_alignment(result, tampered, package.catalog())
        self.assertFalse(checked["passed"])
        self.assertTrue(
            any("expected statistical_influence" in error for error in checked["errors"])
        )

    def test_same_time_priority_revalidates_and_cancels_later_candidate(self) -> None:
        registry = TemporalModelRegistry([DeterministicDelayModel("test.delay", 5.0)])
        result = SimulationEngine(_PriorityDomain(), registry).run(0, 11)
        self.assertTrue(result.validation["passed"])
        self.assertEqual([event.event_type_id for event in result.events], ["test.stop"])
        status = {item.attributes["kind"]: item.status for item in result.candidates}
        self.assertEqual(status["stop"], CandidateStatus.FIRED)
        self.assertEqual(status["dependent"], CandidateStatus.CANCELLED)
        self.assertEqual(len(result.risk_sets), 1)
        self.assertEqual(
            result.risk_sets[0].selection_mode,
            "deterministic_atom_priority",
        )
        self.assertFalse(
            result.risk_sets[0].factorization[
                "continuous_hazard_factorization_applicable"
            ]
        )

    def test_transportation_package_is_reproducible_and_valid(self) -> None:
        config = {
            "nodes_per_context": 24,
            "context_count": 1,
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
        self.assertEqual(len(first.risk_sets), len(first.events))
        continuous = [
            item
            for item in first.risk_sets
            if item.selection_mode == "cause_specific_competing_hazards"
        ]
        self.assertTrue(continuous)
        for risk_set in continuous:
            probabilities = [
                item.candidate_probability_given_time
                for item in risk_set.alternatives
                if item.candidate_probability_given_time is not None
            ]
            self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertIn(
                risk_set.selected_candidate_id,
                {item.candidate_id for item in risk_set.alternatives},
            )
            factorization = risk_set.factorization
            decomposed = (
                factorization["type_given_time"]["probability"]
                * factorization["entity_given_time_and_type"]["probability"]
                * factorization["mechanism_given_time_type_and_entity"]["probability"]
            )
            self.assertAlmostEqual(
                decomposed,
                factorization["selected_candidate_probability_given_time"],
            )
        road_entities = [
            item for item in first.entities
            if item.entity_type_id == "transportation.road.segment"
        ]
        self.assertTrue(
            all("x" not in item.attributes and "y" not in item.attributes for item in road_entities)
        )
        topology = first.context_attributes["topology_profile"]
        self.assertEqual(
            topology["generator_family"], "constrained_degree_weighted_block"
        )
        self.assertFalse(topology["spatial_coordinates_required"])
        self.assertEqual(
            topology["observed_statistics"]["layers"][0]["weak_component_count"],
            1,
        )
        self.assertTrue(
            all(relation.temporal_link.temporal_model_ref for relation in first.event_relations)
        )
        context_relation_by_id = {
            relation.relation_id: relation for relation in first.context_relations
        }
        propagation_relations = [
            relation
            for relation in first.event_relations
            if relation.relation_type_id == "transportation.propagates_downstream"
        ]
        self.assertTrue(propagation_relations)
        self.assertNotIn("outgoing", first.final_state)
        for relation in propagation_relations:
            evidence = relation.context_evidence
            self.assertEqual(len(evidence.context_relation_ids), 1)
            self.assertEqual(len(evidence.entity_ids), 2)
            self.assertTrue(evidence.state_predicates)
            self.assertTrue(all(item.passed for item in evidence.state_predicates))
            context_relation_id = evidence.context_relation_ids[0]
            minimum = context_relation_by_id[
                context_relation_id
            ].attributes["free_flow_seconds"]
            self.assertGreaterEqual(
                relation.temporal_link.lag_seconds + 1e-6,
                minimum,
            )
            self.assertNotIn("context_relation_id", relation.attributes)
        self.assertTrue(
            all(
                event.temporal.observed_at_offset_seconds
                is None
                or event.temporal.observed_at_offset_seconds
                >= event.temporal.occurrence_start_offset_seconds
                for event in first.events
            )
        )

    def test_transportation_contexts_sample_distinct_irregular_profiles(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 200,
                "context_count": 4,
                "base_seed": 20260822,
            }
        )
        profiles = []
        for episode_index in range(4):
            context = package.create_episode_context(
                episode_index, 20260822 + episode_index
            )
            profile = context.context_attributes["topology_profile"]
            profiles.append(profile)
            layer_stats = profile["observed_statistics"]["layers"][0]
            self.assertEqual(layer_stats["weak_component_count"], 1)
            self.assertGreater(layer_stats["unique_out_degree_count"], 2)
            self.assertLess(
                layer_stats["edge_count"],
                200 * profile["max_out_degree"],
            )
        signatures = {
            (
                item["expected_out_degree"],
                item["community_count"],
                item["degree_sigma"],
            )
            for item in profiles
        }
        self.assertGreater(len(signatures), 1)

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
                        "context_count": 1,
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
                "context_count": 1,
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
                "context_count": 1,
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
                "context_count": 1,
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
        distribution = report["quality_report"][
            "episode_event_count_distribution"
        ]
        self.assertEqual(distribution["summary"]["count"], 2)
        self.assertEqual(
            sum(bucket["count"] for bucket in distribution["buckets"]), 2
        )
        self.assertTrue(distribution["bucket_boundaries_are_reporting_only"])
        manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["configuration"]["output_directory"],
            str(output.resolve()),
        )
        self.assertEqual(
            manifest["output"]["resolved_directory"], str(output.resolve())
        )
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
                "risk_sets.jsonl",
                "judge_results.jsonl",
                "episode_texts.jsonl",
                "validation.jsonl",
            "quality_report.json",
        ):
                self.assertTrue((output / filename).is_file(), filename)

    def test_pipeline_llm_rejection_regenerates_with_injected_judge(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 12,
                "context_count": 1,
                "duration_seconds": 1200.0,
                "root_rate_per_hour": 10.0,
            }
        )
        test_output_root = DATA_GENERATION / "test_runtime_output"
        test_output_root.mkdir(exist_ok=True)
        output = test_output_root / f"judge_{uuid.uuid4().hex}"
        judge = _RejectOnceJudge()
        report = GenerationPipeline(
            package,
            build_transportation_temporal_models(),
            {
                "seed": 19,
                "episode_count": 2,
                "max_events_per_episode": 32,
                "semantic_judge": {
                    "mode": "llm",
                    "model": "fake-model",
                    "base_url": "https://unit.test",
                    "workers": 2,
                    "max_generation_attempts": 2,
                },
            },
            output,
            semantic_judge=judge,
        ).run()["quality_report"]
        self.assertTrue(report["passed"])
        self.assertEqual(report["accepted_episode_count"], 2)
        self.assertEqual(report["semantic_judged_episode_count"], 2)
        self.assertEqual(report["semantic_judge_call_count"], 4)
        self.assertEqual(report["semantic_rejection_count"], 2)
        self.assertEqual(report["generation_retry_count"], 2)
        self.assertEqual(report["judge_protocol_failure_count"], 0)
        self.assertEqual(report["semantic_judge_base_url"], "https://unit.test")
        manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["semantic_judge"]["base_url"], "https://unit.test"
        )
        self.assertEqual(
            manifest["configuration"]["semantic_judge"]["base_url"],
            "https://unit.test",
        )
        self.assertEqual(
            len((output / "judge_results.jsonl").read_text(encoding="utf-8").splitlines()),
            4,
        )

    def test_pipeline_quality_gate_rejects_administrative_truncation(self) -> None:
        package = TransportationPackage(
            {
                "nodes_per_context": 8,
                "context_count": 1,
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


class _ResponseClient:
    model = "fake-model"
    base_url = "https://unit.test"

    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, prompt, max_tokens=4096, temperature=1.0):
        del prompt, max_tokens, temperature
        return self.responses.pop(0)


class _RejectOnceJudge:
    def __init__(self):
        self.calls = {}

    def evaluate(self, result, text_alignment, domain_catalog):
        del text_alignment, domain_catalog
        count = self.calls.get(result.episode_id, 0)
        self.calls[result.episode_id] = count + 1
        if count == 0:
            record_id = (
                result.events[0].event_id
                if result.events
                else result.candidates[0].candidate_id
            )
            return SemanticJudgeResult(
                outcome="semantic_rejected",
                passed=False,
                reasons=[f"Synthetic retry requested for {record_id}."],
                flagged_record_ids=[record_id],
                model_id="fake-model",
                provider_base_url="https://unit.test",
            )
        return SemanticJudgeResult(
            outcome="passed",
            passed=True,
            reasons=["The supplied records are coherent."],
            flagged_record_ids=[],
            model_id="fake-model",
            provider_base_url="https://unit.test",
        )


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
                combination="none",
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
                combination="none",
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
