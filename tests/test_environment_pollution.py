"""River topology, pollutant mass/time and operational response checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from data_generation.domain_packages.environment_pollution import (
    EnvironmentPollutionPackage, build_environment_pollution_temporal_models,
)
from data_generation.domain_packages.environment_pollution.package import (
    NS, SCENARIOS, SPILL, BYPASS, RAIN, LOADING, END, ARRIVAL, CLEAR,
    ALERT, NORMAL, INTAKE_CLOSED, INTAKE_REOPENED, SETTLED,
)
from data_generation.generation_core.models import EventParticipant
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.validation import validate_text_alignment

ROOT = Path(__file__).resolve().parents[2]


class EnvironmentPollutionTests(unittest.TestCase):
    def engine(self, scenario="industrial_spill", **config):
        package = EnvironmentPollutionPackage({"nodes_per_context": 40,
            "scenario_weights": {scenario: 1.0}, **config})
        return package, SimulationEngine(package, build_environment_pollution_temporal_models(), max_events=512)

    def test_configuration_and_registry(self):
        for config in ({"nodes_per_context": 10}, {"context_count": 0},
                       {"duration_seconds": float("nan")},
                       {"scenario_weights": {"unknown": 1}},
                       {"scenario_weights": {"storm_runoff": -1}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                EnvironmentPollutionPackage(config)
        package, _ = self.engine()
        self.assertEqual(len(package.mechanism_registry), 16)

    def test_scenarios_validate_and_text_align(self):
        roots = {"industrial_spill": SPILL, "wastewater_bypass": BYPASS,
                 "storm_runoff": RAIN}
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario)
            for seed in range(12):
                result = engine.run(seed, seed + 100)
                self.assertTrue(result.validation["passed"], result.validation)
                self.assertTrue(result.validation["checks"]["domain_contracts"])
                self.assertEqual(package.validate_result(result), [])
                self.assertTrue(validate_text_alignment(result, package.render_episode_text(result), package.catalog())["passed"])
                kinds = {event.event_type_id for event in result.events}
                self.assertIn(roots[scenario], kinds)
                self.assertIn(LOADING, kinds)
                self.assertIn(END, kinds)
                self.assertIn(ARRIVAL, kinds)

    def test_thousand_nodes_sparse_downstream_mass_and_monitor_response(self):
        package, engine = self.engine(nodes_per_context=1000)
        result = engine.run(0, 30)
        self.assertEqual(len(result.entities), 1000)
        self.assertLess(len(result.context_relations), 1500)
        self.assertEqual({entity.entity_type_id for entity in result.entities},
                         {NS + kind for kind in ("reach", "outfall", "catchment", "monitor", "intake")})
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertEqual(package.validate_result(result), [])
        entity_by_id = {entity.entity_id: entity for entity in result.entities}
        for event in result.events:
            if event.event_type_id != ARRIVAL:
                continue
            attrs = event.attributes
            q = entity_by_id[attrs["reach_id"]].attributes["reference_flow_m3_s"] + attrs["source_added_flow_m3_s"]
            self.assertAlmostEqual(attrs["concentration_increment_mg_l"],
                                   1000 * attrs["mass_flux_kg_s"] / q, places=5)
        self.assertTrue(any(event.event_type_id == ALERT for event in result.events))
        self.assertTrue(any(event.event_type_id == INTAKE_CLOSED for event in result.events))

    def test_monitor_and_intake_sequence(self):
        _, engine = self.engine("industrial_spill")
        for seed in range(20):
            result = engine.run(seed, seed + 400)
            self.assertTrue(result.validation["passed"], result.validation)
            seen = {}
            for event in result.events:
                reach = event.attributes["reach_id"]
                seen.setdefault(reach, {})[event.event_type_id] = event.temporal.anchor("occurrence_start")
            for reach, times in seen.items():
                if NORMAL in times:
                    self.assertLessEqual(times[CLEAR], times[NORMAL])
                    self.assertLessEqual(times[ALERT], times[NORMAL])
                if INTAKE_REOPENED in times:
                    self.assertLessEqual(times[INTAKE_CLOSED], times[INTAKE_REOPENED])
                    self.assertLessEqual(times[NORMAL], times[INTAKE_REOPENED])

    def test_tampering_mass_or_route_fails_domain_validation(self):
        package, engine = self.engine(nodes_per_context=1000)
        result = engine.run(0, 30)
        broken = copy.deepcopy(result)
        arrival = next(event for event in broken.events if event.event_type_id == ARRIVAL)
        arrival.attributes["concentration_increment_mg_l"] += 2
        self.assertTrue(any("concentration/flow mismatch" in error
                            for error in package.validate_result(broken)))
        broken = copy.deepcopy(result)
        downstream = next(event for event in broken.events
                          if event.event_type_id == ARRIVAL and event.attributes["cumulative_travel_seconds"] > 0)
        downstream.attributes["mass_flux_kg_s"] *= 1.5
        self.assertTrue(any("downstream mass transport" in error
                            for error in package.validate_result(broken)))
        broken = copy.deepcopy(result)
        downstream = next(event for event in broken.events
                          if event.event_type_id == ARRIVAL and event.attributes["cumulative_travel_seconds"] > 0)
        wrong_reach = next(entity.entity_id for entity in broken.entities
                           if entity.entity_type_id == NS + "reach" and entity.entity_id != downstream.attributes["reach_id"])
        downstream.participants = [EventParticipant(wrong_reach, role.role)
                                   if role.role == "upstream_reach" else role
                                   for role in downstream.participants]
        self.assertTrue(any("downstream mass transport" in error
                            for error in package.validate_result(broken)))

    def test_short_horizon_and_continuation(self):
        _, engine = self.engine(duration_seconds=.001)
        result = engine.run(0, 77)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertFalse(any(event.event_type_id == SETTLED for event in result.events))
        _, engine = self.engine()
        snapshot = engine.run(0, 111, stop_after_events=2)
        left = engine.run(0, 111, snapshot=snapshot, continuation_seed=10)
        right = engine.run(0, 111, snapshot=snapshot, continuation_seed=11)
        self.assertTrue(left.validation["passed"], left.validation)
        self.assertTrue(right.validation["passed"], right.validation)
        self.assertEqual([event.to_dict() for event in left.events[:2]],
                         [event.to_dict() for event in right.events[:2]])

    def test_cli_qa_and_training_export(self):
        output = ROOT / "data_generation" / "artifacts" / "test_runtime_output" / f"pollution_{uuid.uuid4().hex}"
        process = subprocess.run([sys.executable, "data_generation/run_data_generation.py",
            "--domain", "environment_pollution", "--episode-count", "3",
            "--nodes-per-context", "40", "--judge-mode", "none",
            "--output-dir", str(output)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"], report)
        self.assertGreater((output / "qa_pairs.jsonl").stat().st_size, 0)
        self.assertGreater((output / "training" / "qa_instruction.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
