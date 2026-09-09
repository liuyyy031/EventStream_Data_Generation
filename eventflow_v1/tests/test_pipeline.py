from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
import uuid
from dataclasses import replace
from pathlib import Path


DATA_GENERATION = Path(__file__).resolve().parents[2]
if str(DATA_GENERATION) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION))

from eventflow_v1.core.config import load_config  # noqa: E402
from eventflow_v1.core.judge import (  # noqa: E402
    HeuristicSemanticJudge,
    JudgeResult,
    LLMSemanticJudge,
)
from eventflow_v1.core.pipeline import EventFlowPipeline  # noqa: E402
from eventflow_v1.domains.transportation.generator import TransportationEpisodeGenerator  # noqa: E402
from eventflow_v1.domains.transportation.event_types import (  # noqa: E402
    CONGESTION_EASING,
    HEAVY_RAIN_ONSET,
    NORMAL_FLOW_RESTORED,
    ROAD_DEBRIS,
    TRAFFIC_SIGNAL_FAILURE,
    VEHICLE_BREAKDOWN,
    VEHICLE_COLLISION,
    load_event_type_registry,
)
from eventflow_v1.domains.transportation.network import generate_network  # noqa: E402
from eventflow_v1.domains.transportation.text import TransportationTextGenerator  # noqa: E402
from eventflow_v1.domains.transportation.validation import TransportationValidator  # noqa: E402
from eventflow_v1.experiments.judge_challenge import run_judge_challenge  # noqa: E402
from eventflow_v1.experiments.stability import run_stability_experiment  # noqa: E402


class EventFlowPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        config_path = DATA_GENERATION / "eventflow_v1" / "config" / "transportation_default.json"
        self.config = load_config(config_path)
        self.test_temp_root = DATA_GENERATION / "eventflow_v1" / ".test_output"
        self.test_temp_root.mkdir(exist_ok=True)

    def output_path(self, label: str) -> Path:
        return self.test_temp_root / f"{label}_{uuid.uuid4().hex}"

    def test_sparse_network_does_not_build_dense_matrix(self) -> None:
        network = generate_network("test_network", "random_connected", 1000, 0.1, 6, 7)
        self.assertEqual(len(network.road_ids), 1000)
        self.assertLess(len(network.edges), 2000)
        self.assertFalse(hasattr(network, "adjacency_matrix"))

    def test_v14_output_uses_registry_ids_and_qualified_participants(self) -> None:
        config = copy.deepcopy(self.config)
        config["episode_count"] = 1
        config["network"]["network_count"] = 1
        config["network"]["node_count"] = 20
        config["network"]["structure_families"] = ["corridor"]
        config["simulation"]["scenario_families"] = ["vehicle_breakdown"]
        output = self.output_path("v14_registry_contract")
        result = EventFlowPipeline(config, output).run()
        self.assertTrue(result["quality_report"]["passed"])
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        registry = json.loads(
            (output / "event_type_registry.json").read_text(encoding="utf-8")
        )
        events = _read_jsonl(output / "events.jsonl")
        self.assertEqual(manifest["schema_version"], "eventflow-v1")
        self.assertEqual(
            manifest["event_type_registry"]["registry_version"],
            registry["registry_version"],
        )
        self.assertEqual(
            manifest["event_type_registry"]["sha256"],
            hashlib.sha256((output / "event_type_registry.json").read_bytes()).hexdigest(),
        )
        self.assertTrue(all("event_type_id" in event for event in events))
        self.assertTrue(all("event_type" not in event for event in events))
        self.assertTrue(all("participant_ids" not in event for event in events))
        self.assertTrue(
            all(
                participant.get("entity_id") and participant.get("role")
                for event in events
                for participant in event["participants"]
            )
        )

    def test_registry_rejects_unknown_type_and_unregistered_attribute(self) -> None:
        network = generate_network("registry_reject", "corridor", 20, 0.0, 6, 5)
        simulation = copy.deepcopy(self.config["simulation"])
        simulation["scenario_families"] = ["vehicle_breakdown"]
        bundle = TransportationEpisodeGenerator(network, simulation).generate(0, 7)
        first = bundle.events[0]
        bundle.events[0] = replace(
            first,
            event_type_id="transportation.road.incident.unregistered_type",
        )
        validator = TransportationValidator(network, simulation, self.config["network"])
        result = validator.validate(bundle)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["valid_event_type_contract"])

        bundle = TransportationEpisodeGenerator(network, simulation).generate(0, 7)
        first = bundle.events[0]
        bundle.events[0] = replace(
            first,
            attributes={**first.attributes, "unregistered_attribute": 1},
        )
        result = validator.validate(bundle)
        self.assertFalse(result.passed)
        self.assertTrue(any("unregistered attributes" in error for error in result.errors))

    def test_transport_validator_rejects_more_blocked_lanes_than_road_has(self) -> None:
        network = generate_network("lane_constraint", "corridor", 20, 0.0, 6, 11)
        simulation = copy.deepcopy(self.config["simulation"])
        simulation["scenario_families"] = ["accident_propagation"]
        bundle = TransportationEpisodeGenerator(network, simulation).generate(0, 17)
        collision_index = next(
            index
            for index, event in enumerate(bundle.events)
            if event.event_type_id == VEHICLE_COLLISION
        )
        collision = bundle.events[collision_index]
        road_id = collision.participant_ids_for_role("affected_road")[0]
        road_lanes = int(network.entity_by_id[road_id].attributes["lanes"])
        bundle.events[collision_index] = replace(
            collision,
            attributes={**collision.attributes, "lanes_blocked": road_lanes + 1},
        )
        result = TransportationValidator(
            network, simulation, self.config["network"]
        ).validate(bundle)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["valid_transport_constraints"])

    def test_each_new_scenario_emits_its_registered_root_family(self) -> None:
        network = generate_network("scenario_types", "grid", 64, 0.1, 6, 13)
        expected = {
            "accident_propagation": {VEHICLE_COLLISION},
            "vehicle_breakdown": {VEHICLE_BREAKDOWN},
            "road_obstruction": {ROAD_DEBRIS},
            "signal_failure": {TRAFFIC_SIGNAL_FAILURE},
            "mixed_scenario": {VEHICLE_COLLISION, HEAVY_RAIN_ONSET},
        }
        registry = load_event_type_registry()
        for index, (scenario, expected_types) in enumerate(expected.items()):
            simulation = copy.deepcopy(self.config["simulation"])
            simulation["scenario_families"] = [scenario]
            bundle = TransportationEpisodeGenerator(
                network, simulation, event_type_registry=registry
            ).generate(index, 101 + index)
            primary_ids = set(bundle.program["primary_root_event_ids"])
            primary_types = {
                event.event_type_id
                for event in bundle.events
                if event.event_id in primary_ids
            }
            self.assertEqual(primary_types, expected_types)

    def test_partial_and_full_recovery_have_distinct_event_types(self) -> None:
        network = generate_network("recovery_types", "corridor", 32, 0.0, 6, 23)
        simulation = copy.deepcopy(self.config["simulation"])
        simulation["scenario_families"] = ["vehicle_breakdown"]
        simulation["duration_seconds"] = 14400.0
        simulation["recovery_probability"] = 1.0

        simulation["full_recovery_probability"] = 0.0
        partial = TransportationEpisodeGenerator(network, simulation).generate(0, 29)
        partial_recoveries = [event for event in partial.events if event.event_role == "recovery"]
        self.assertTrue(partial_recoveries)
        self.assertTrue(
            all(event.event_type_id == CONGESTION_EASING for event in partial_recoveries)
        )
        self.assertTrue(
            all(float(event.attributes["recovery_fraction"]) < 1.0 for event in partial_recoveries)
        )

        simulation["full_recovery_probability"] = 1.0
        full = TransportationEpisodeGenerator(network, simulation).generate(0, 29)
        full_recoveries = [event for event in full.events if event.event_role == "recovery"]
        self.assertTrue(full_recoveries)
        self.assertTrue(
            all(event.event_type_id == NORMAL_FLOW_RESTORED for event in full_recoveries)
        )
        self.assertTrue(
            all(float(event.attributes["recovery_fraction"]) == 1.0 for event in full_recoveries)
        )

    def test_hub_topology_and_event_branching_are_bounded(self) -> None:
        network = generate_network("bounded_hub", "hub", 1000, 0.15, 6, 19)
        self.assertLessEqual(max(len(edges) for edges in network.outgoing.values()), 6)
        simulation = copy.deepcopy(self.config["simulation"])
        simulation["scenario_families"] = ["mixed_scenario"]
        bundle = TransportationEpisodeGenerator(network, simulation).generate(0, 29)
        outgoing = {}
        for relation in bundle.relations:
            if relation.relation_type == "propagates_downstream":
                outgoing[relation.source_event_id] = outgoing.get(relation.source_event_id, 0) + 1
        self.assertLessEqual(max(outgoing.values(), default=0), simulation["max_children_per_event"])

    def test_truncated_episode_is_rejected_before_judge_acceptance(self) -> None:
        network = generate_network("truncation_test", "hub", 60, 0.0, 6, 31)
        simulation = copy.deepcopy(self.config["simulation"])
        simulation["scenario_families"] = ["mixed_scenario"]
        simulation["max_events_per_episode"] = 3
        simulation["expected_children_per_event"] = 3.0
        simulation["max_children_per_event"] = 3
        bundle = TransportationEpisodeGenerator(network, simulation).generate(0, 37)
        bundle.texts = TransportationTextGenerator(network, self.config["text"]).generate(bundle, 41)
        validator = TransportationValidator(network, simulation, self.config["network"])
        result = validator.validate(bundle)
        self.assertTrue(bundle.program["termination"]["truncated"])
        self.assertFalse(result.passed)
        self.assertFalse(HeuristicSemanticJudge().evaluate(bundle).passed)

    def test_llm_judge_retries_protocol_errors_and_requires_reasons(self) -> None:
        bundle = self._small_bundle()
        client = _SequenceLLMClient(
            [
                '{"passed":true,"reasons":[]}',
                '{"passed":true,"reasons":["unterminated]',
                '{"passed":true,"reasons":["The event flow is coherent."]}',
            ]
        )
        judge = LLMSemanticJudge(client=client, protocol_max_attempts=3)
        result = judge.evaluate(bundle)
        self.assertTrue(result.passed)
        self.assertEqual(result.outcome, "passed")
        self.assertEqual(result.protocol_attempt_count, 3)
        self.assertEqual(client.call_count, 3)
        self.assertEqual(len(result.raw["protocol_errors_before_success"]), 2)

    def test_pipeline_judge_protocol_retry_keeps_episode_seed(self) -> None:
        config = copy.deepcopy(self.config)
        config["episode_count"] = 1
        config["network"]["network_count"] = 1
        config["network"]["node_count"] = 20
        config["network"]["structure_families"] = ["corridor"]
        config["simulation"]["scenario_families"] = ["road_closure"]
        client = _SequenceLLMClient(
            [
                "not json",
                '{"passed":true,"reasons":["The rule-grounded event flow is coherent."]}',
            ]
        )
        judge = LLMSemanticJudge(client=client, protocol_max_attempts=3)
        output = self.output_path("judge_protocol_retry")
        result = EventFlowPipeline(config, output, judge=judge).run()
        episode = _read_jsonl(output / "episodes.jsonl")[0]
        self.assertEqual(episode["program"]["seed"], config["seed"])
        self.assertEqual(episode["validation"]["accepted_attempt"], 1)
        self.assertEqual(result["quality_report"]["judge_protocol_retry_count"], 1)
        self.assertEqual(result["quality_report"]["rejected_attempt_count"], 0)

    def test_pipeline_stops_after_exhausted_judge_protocol_retries(self) -> None:
        config = copy.deepcopy(self.config)
        config["episode_count"] = 1
        config["network"]["network_count"] = 1
        config["network"]["node_count"] = 20
        config["network"]["structure_families"] = ["corridor"]
        client = _SequenceLLMClient(["not json"])
        judge = LLMSemanticJudge(client=client, protocol_max_attempts=3)
        output = self.output_path("judge_protocol_failure")
        with self.assertRaisesRegex(RuntimeError, "episode was not regenerated"):
            EventFlowPipeline(config, output, judge=judge).run()
        self.assertEqual(client.call_count, 3)
        failure = json.loads(
            (output / "judge_protocol_failure_0000000.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failure["episode_seed"], config["seed"])
        self.assertEqual(failure["routing"], "pipeline_stopped_without_regenerating_episode")

    def test_semantic_rejection_regenerates_episode(self) -> None:
        config = copy.deepcopy(self.config)
        config["episode_count"] = 1
        config["network"]["network_count"] = 1
        config["network"]["node_count"] = 20
        config["network"]["structure_families"] = ["corridor"]
        client = _SequenceLLMClient(
            [
                '{"passed":false,"reasons":["The event flow is semantically implausible."]}',
                '{"passed":true,"reasons":["The regenerated event flow is coherent."]}',
            ]
        )
        judge = LLMSemanticJudge(client=client, protocol_max_attempts=3)
        output = self.output_path("semantic_retry")
        result = EventFlowPipeline(config, output, judge=judge).run()
        episode = _read_jsonl(output / "episodes.jsonl")[0]
        self.assertEqual(episode["validation"]["accepted_attempt"], 2)
        self.assertEqual(episode["program"]["seed"], config["seed"] + 104729)
        self.assertEqual(result["quality_report"]["semantic_rejection_count"], 1)
        self.assertEqual(result["quality_report"]["judge_protocol_retry_count"], 0)

    def _small_bundle(self):
        network = generate_network("judge_test", "corridor", 12, 0.0, 6, 43)
        simulation = copy.deepcopy(self.config["simulation"])
        simulation["scenario_families"] = ["road_closure"]
        bundle = TransportationEpisodeGenerator(network, simulation).generate(0, 47)
        bundle.texts = TransportationTextGenerator(network, self.config["text"]).generate(
            bundle, 53
        )
        return bundle

    def test_pipeline_writes_grounded_multi_parent_event_flow(self) -> None:
        config = copy.deepcopy(self.config)
        config["episode_count"] = 3
        config["network"]["node_count"] = 24
        config["network"]["network_count"] = 1
        config["network"]["structure_families"] = ["grid"]
        config["simulation"]["scenario_families"] = ["mixed_scenario"]
        config["simulation"]["duration_seconds"] = 3600.0
        config["text"]["paraphrase_rate"] = 1.0
        config["validation"]["judge_mode"] = "heuristic"
        output = self.output_path("multi_parent")
        result = EventFlowPipeline(config, output).run()
        self.assertEqual(result["accepted_episode_count"], 3)
        self.assertTrue(result["quality_report"]["passed"])

        events = _read_jsonl(output / "events.jsonl")
        relations = _read_jsonl(output / "relations.jsonl")
        mechanisms = _read_jsonl(output / "mechanisms.jsonl")
        texts = _read_jsonl(output / "texts.jsonl")
        self.assertTrue(events)
        self.assertTrue(relations)
        self.assertEqual(len(mechanisms), 3)
        self.assertTrue(all("parent_event_ids" not in event for event in events))
        self.assertTrue(
            all(relation["provenance"]["recorded_when_rule_executed"] for relation in relations)
        )
        self.assertTrue(all(mechanism["combination"] == "all_of" for mechanism in mechanisms))

        canonical_event_ids = {
            event_id
            for record in texts
            if record["text_role"] == "event" and record["variant"] == "canonical"
            for event_id in record["aligned_event_ids"]
        }
        self.assertEqual(canonical_event_ids, {event["event_id"] for event in events})
        for record in texts:
            if record["text_role"] == "event" and record["variant"] == "canonical":
                self.assertTrue(
                    all(
                        f"attributes.{key}" in record["verbalized_fact_keys"]
                        for key in record["grounded_facts"]["attributes"]
                    )
                )

    def test_external_network_example_is_accepted(self) -> None:
        config = copy.deepcopy(self.config)
        config["episode_count"] = 1
        config["network"]["external_network_file"] = str(
            DATA_GENERATION / "eventflow_v1" / "config" / "external_network_example.json"
        )
        config["simulation"]["scenario_families"] = ["accident_propagation"]
        result = EventFlowPipeline(config, self.output_path("external_run")).run()
        self.assertEqual(result["accepted_episode_count"], 1)

    def test_multiple_networks_have_distinct_structures_and_global_ids(self) -> None:
        config = copy.deepcopy(self.config)
        config["episode_count"] = 4
        config["network"]["node_count"] = 18
        config["network"]["network_count"] = 4
        config["simulation"]["scenario_families"] = ["road_closure"]
        output = self.output_path("multi_network")
        result = EventFlowPipeline(config, output).run()
        self.assertEqual(result["quality_report"]["network_count"], 4)
        networks = _read_jsonl(output / "networks.jsonl")
        entities = _read_jsonl(output / "entities.jsonl")
        edges = _read_jsonl(output / "network_edges.jsonl")
        self.assertEqual(
            {network["structure_family"] for network in networks},
            {"corridor", "grid", "hub", "random_connected"},
        )
        self.assertEqual(len(entities), len({entity["entity_id"] for entity in entities}))
        self.assertEqual(len(edges), len({edge["edge_id"] for edge in edges}))

    def test_calibrated_lags_are_recorded_and_reproducible(self) -> None:
        network = generate_network("calibration_test", "grid", 64, 0.1, 6, 71)
        simulation = copy.deepcopy(self.config["simulation"])
        simulation["scenario_families"] = ["road_closure"]
        simulation["recovery_probability"] = 1.0
        simulation["duration_seconds"] = 14400.0
        bundle = TransportationEpisodeGenerator(network, simulation).generate(0, 73)
        bundle.texts = TransportationTextGenerator(network, self.config["text"]).generate(
            bundle, 79
        )
        calibrated = [
            relation
            for relation in bundle.relations
            if relation.relation_type in {"propagates_downstream", "transitions_to_recovery"}
        ]
        self.assertTrue(calibrated)
        self.assertTrue(
            all("lag_calibration" in relation.attributes for relation in calibrated)
        )
        result = TransportationValidator(
            network, simulation, self.config["network"]
        ).validate(bundle)
        self.assertTrue(result.passed, result.errors)
        self.assertTrue(result.checks["calibrated_lag_provenance"])

    def test_small_multi_seed_stability_report_is_reproducible(self) -> None:
        config = copy.deepcopy(self.config)
        config["episode_count"] = 3
        config["network"]["network_count"] = 1
        config["network"]["node_count"] = 32
        config["network"]["structure_families"] = ["corridor"]
        config["simulation"]["scenario_families"] = ["road_closure"]
        report = run_stability_experiment(
            config,
            [101, 202],
            self.output_path("stability"),
        )
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["run_count"], 2)
        self.assertEqual(report["total_episode_count"], 6)

    def test_layered_judge_challenge_assigns_counterexamples(self) -> None:
        report = run_judge_challenge(
            copy.deepcopy(self.config),
            "llm",
            self.output_path("judge_challenge"),
            judge_override=_ChallengeAwareJudge(),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["unassessed_case_count"], 0)
        self.assertEqual(report["detected_as_expected_count"], 7)


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class _SequenceLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def complete(self, *_args, **_kwargs):
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[index]


class _ChallengeAwareJudge:
    def evaluate(self, bundle):
        incoming = {relation.target_event_id for relation in bundle.relations}
        orphan_recovery = any(
            event.event_role == "recovery" and event.event_id not in incoming
            for event in bundle.events
        )
        unsupported_claim = any(
            "The data establishes that" in text.content for text in bundle.texts
        )
        rejected = orphan_recovery or unsupported_claim
        return JudgeResult(
            outcome="semantic_rejected" if rejected else "passed",
            passed=not rejected,
            reasons=[
                "Counterexample contains an unsupported semantic claim."
                if rejected
                else "The valid control is coherent."
            ],
            mode="llm",
            protocol_attempt_count=1,
            raw={},
        )


if __name__ == "__main__":
    unittest.main()
