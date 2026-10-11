"""Domain semantics, clocks, exports and cross-domain entry point regression."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import uuid
import unittest

from data_generation.domain_packages.distributed_systems import (
    DistributedSystemsPackage, build_distributed_systems_temporal_models,
)
from data_generation.domain_packages.distributed_systems.package import (
    NS, SENT, RECEIVED, PROCESSED, RESPONSE, TIMEOUT, EXHAUSTED, SETTLED, SCENARIOS,
)
from data_generation.domain_packages.registry import DOMAIN_BUILDERS
from data_generation.generation_core.models import EventParticipant
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.validation import validate_text_alignment

ROOT = Path(__file__).resolve().parents[2]


class DistributedSystemsTests(unittest.TestCase):
    def engine(self, scenario="retry_pressure", **config):
        package = DistributedSystemsPackage({"nodes_per_context": 40,
            "scenario_weights": {scenario: 1.0}, **config})
        return package, SimulationEngine(package, build_distributed_systems_temporal_models(), max_events=128)

    def test_all_domains_registered(self):
        self.assertEqual(set(DOMAIN_BUILDERS), {"transportation", "healthcare", "distributed_systems", "energy", "environment_pollution", "ecology", "public_health", "hydrology", "agriculture", "climate"})

    def test_configuration_rejects_invalid_values(self):
        for config in ({"nodes_per_context": 10}, {"fanout": 1}, {"timeout_seconds": float("nan")},
                       {"max_retries": -1}, {"scenario_weights": {"unknown": 1}},
                       {"scenario_weights": {"normal_rpc": float("inf")}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                DistributedSystemsPackage(config)

    def test_scenarios_many_seeds_validate_and_align(self):
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario)
            for seed in range(12):
                result = engine.run(seed, seed)
                self.assertTrue(result.validation["passed"], result.validation)
                self.assertTrue(result.validation["checks"]["domain_contracts"])
                alignment = validate_text_alignment(result, package.render_episode_text(result), package.catalog())
                self.assertTrue(alignment["passed"], alignment)

    def test_thousand_typed_nodes_and_consistent_bindings(self):
        package, engine = self.engine(nodes_per_context=1000)
        result = engine.run(0, 13)
        self.assertEqual(len(result.entities), 1000)
        self.assertEqual({e.entity_type_id for e in result.entities},
                         {NS + t for t in ("service", "host", "container", "endpoint")})
        self.assertLess(len(result.context_relations), 10000)
        self.assertEqual(package.validate_result(result), [])
        self.assertEqual(len({e.entity_id for e in result.entities}), 1000)
        degree = {}
        for relation in result.context_relations:
            if relation.relation_type_id == NS + "service_calls_service":
                degree[relation.source_entity_id] = degree.get(relation.source_entity_id, 0) + 1
        self.assertGreater(len(set(degree.values())), 1)

    def test_timeout_does_not_cancel_server_processing(self):
        package, engine = self.engine(timeout_seconds=.02, duration_seconds=40, max_retries=1)
        result = engine.run(0, 17)
        timeouts = {e.attributes["attempt_id"]: e for e in result.events if e.event_type_id == TIMEOUT}
        late = [e for e in result.events if e.event_type_id == RESPONSE and e.attributes["late_response"]]
        self.assertTrue(late)
        for event in late:
            self.assertIn(event.attributes["attempt_id"], timeouts)
            self.assertFalse(event.attributes["response_accepted"])
        self.assertTrue(any(e.event_type_id == PROCESSED and e.attributes["attempt_id"] in timeouts
                            and e.temporal.anchor("occurrence_start") > timeouts[e.attributes["attempt_id"]].temporal.anchor("occurrence_start")
                            for e in result.events))
        join = next(e for e in result.events if e.event_type_id == SETTLED)
        parents = next(c.parent_event_ids for c in result.candidates if c.fired_event_id == join.event_id)
        self.assertFalse(set(parents) & {e.event_id for e in late})
        self.assertEqual(package.validate_result(result), [])

    def test_attempt_identity_deadlines_and_backoff(self):
        package, engine = self.engine(timeout_seconds=.025, max_retries=2)
        result = engine.run(0, 17)
        sends = [e for e in result.events if e.event_type_id == SENT]
        self.assertTrue(any(e.attributes["attempt"] > 0 for e in sends))
        self.assertEqual(len(sends), len({e.attributes["attempt_id"] for e in sends}))
        events = {e.event_id: e for e in result.events}
        for candidate in result.candidates:
            if candidate.mechanism_id != NS + "retry_backoff" or not candidate.fired_event_id:
                continue
            event, parent = events[candidate.fired_event_id], events[candidate.parent_event_ids[0]]
            self.assertEqual(event.attributes["idempotency_key"], parent.attributes["idempotency_key"])
            self.assertEqual(event.attributes["attempt"], parent.attributes["attempt"] + 1)
            base = min(2, .15 * 2 ** parent.attributes["attempt"])
            lag = event.temporal.anchor("occurrence_start") - parent.temporal.anchor("observed_at")
            self.assertGreaterEqual(lag, base * .8 - 1e-9)
            self.assertLessEqual(lag, base * 1.2 + 1e-9)

    def test_local_availability_is_not_collector_recording_time(self):
        _, engine = self.engine("normal_rpc")
        result = engine.run(0, 5)
        for event in result.events:
            self.assertEqual(event.temporal.anchor("observed_at"), event.temporal.anchor("occurrence_start"))
            self.assertGreater(event.temporal.anchor("recorded_at"), event.temporal.anchor("observed_at"))
        for relation in result.event_relations:
            self.assertEqual(relation.temporal_link.source_anchor, "observed_at")
            if relation.relation_type_id in (NS + "request_delivery", NS + "response_delivery"):
                self.assertGreater(relation.temporal_link.lag_seconds, 0)

    def test_exact_response_timeout_tie_is_response_first(self):
        package, engine = self.engine("normal_rpc", timeout_seconds=.5)
        # Deterministic test models retain declared IDs; production clocks remain stochastic.
        from data_generation.generation_core.temporal import DeterministicDelayModel
        engine.temporal_models.replace(DeterministicDelayModel(NS + "network_delay", .125))
        engine.temporal_models.replace(DeterministicDelayModel(NS + "processing_hazard", .25))
        result = engine.run(0, 1)
        self.assertFalse(any(e.event_type_id == TIMEOUT for e in result.events))
        self.assertTrue(all(e.attributes["response_accepted"] for e in result.events if e.event_type_id == RESPONSE))
        self.assertEqual(package.validate_result(result), [])

    def test_dynamic_processing_exposes_capacity_and_retains_threshold(self):
        _, engine = self.engine(timeout_seconds=.05)
        result = engine.run(0, 9)
        clocks = [c for c in result.candidates if c.mechanism_id == NS + "server_processing"]
        self.assertTrue(clocks)
        for clock in clocks:
            self.assertGreater(clock.temporal_inputs["log_capacity"], 0)
            self.assertEqual(clock.random_threshold, clock.temporal_provenance["details"]["random_threshold"])

    def test_domain_validator_rejects_wrong_node_and_late_response_claims(self):
        package, engine = self.engine(timeout_seconds=.01, duration_seconds=40)
        result = engine.run(0, 7)
        broken = copy.deepcopy(result)
        response = next(e for e in broken.events if e.event_type_id == RESPONSE and e.attributes["late_response"])
        response.attributes["response_accepted"] = True
        self.assertTrue(any("Late response" in e for e in package.validate_result(broken)))
        broken = copy.deepcopy(result)
        request = next(e for e in broken.events if e.event_type_id == SENT)
        host_id = next(e.entity_id for e in broken.entities if e.entity_type_id == NS + "host")
        request.participants = [EventParticipant(host_id, p.role) if p.role == "callee" else p for p in request.participants]
        self.assertTrue(any("node meaning" in e for e in package.validate_result(broken)))

    def test_recovery_changes_running_clock_without_resetting_threshold(self):
        _, engine = self.engine(timeout_seconds=.05)
        changed = []
        for seed in range(5):
            for clock in engine.run(0, seed).candidates:
                if clock.mechanism_id != NS + "server_processing":
                    continue
                initial = clock.temporal_provenance["details"]
                latest = clock.temporal_provenance.get("last_update", {})
                if initial["conditioning_inputs"]["degraded"] == 1 and latest.get("conditioning_inputs", {}).get("degraded") == 0:
                    changed.append(clock)
                    self.assertEqual(clock.random_threshold, initial["random_threshold"])
                    self.assertGreater(latest["new_rate_per_second"], initial["initial_rate_per_second"])
        self.assertTrue(changed, "No running processing clock experienced recovery")

    def test_workflow_join_has_distinct_operations_not_topology_neighbors(self):
        _, engine = self.engine()
        result = engine.run(0, 19)
        events = {e.event_id: e for e in result.events}
        join = next(c for c in result.candidates if c.target_event_type_id == SETTLED)
        self.assertEqual(join.combination, "all_of")
        self.assertEqual(len(join.parent_event_ids), 3)
        self.assertEqual(len({events[p].attributes["operation_id"] for p in join.parent_event_ids}), 3)
        self.assertTrue(all(events[p].event_type_id in (RESPONSE, EXHAUSTED) for p in join.parent_event_ids))
        self.assertFalse(join.context_evidence.context_relation_ids)

    def test_short_horizon_censors_without_fabricating_closure(self):
        _, engine = self.engine(duration_seconds=.001)
        result = engine.run(0, 2)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertTrue(any(c.status.value == "right_censored" for c in result.candidates))
        self.assertFalse(any(e.event_type_id == SETTLED for e in result.events))

    def test_continuations_preserve_prefix_and_vary_suffix(self):
        _, engine = self.engine("normal_rpc")
        snapshot = engine.run(0, 10, stop_after_events=2)
        results = [engine.run(0, 10, snapshot=snapshot, continuation_seed=s) for s in (101, 102, 103)]
        self.assertTrue(all(r.validation["passed"] for r in results))
        self.assertEqual([e.to_dict() for e in results[0].events[:2]], [e.to_dict() for e in results[1].events[:2]])
        self.assertNotEqual([e.to_dict() for e in results[0].events[2:]], [e.to_dict() for e in results[1].events[2:]])

    def test_cli_exports_qa_and_training_without_llm(self):
        output = ROOT / "data_generation" / "artifacts" / "test_runtime_output" / f"distributed_{uuid.uuid4().hex}"
        process = subprocess.run([sys.executable, "data_generation/run_data_generation.py",
            "--domain", "distributed_systems", "--episode-count", "2", "--nodes-per-context", "40",
            "--judge-mode", "none", "--output-dir", str(output)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"], report)
        self.assertTrue((output / "qa_pairs.jsonl").stat().st_size > 0)
        self.assertTrue((output / "training" / "qa_instruction.jsonl").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
