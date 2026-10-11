"""Energy topology, event logic, clocks and export regression tests."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from data_generation.domain_packages.energy import EnergyPackage, build_energy_temporal_models
from data_generation.domain_packages.energy.package import (
    NS, SCENARIOS, OPEN, INTERRUPTED, REPAIRED, CLOSED, RESTORED,
    SURGE, HIGH, NORMAL, SHORTFALL, DISPATCHED, RECOVERED, RESERVE_RESTORED, SETTLED,
)
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.validation import validate_text_alignment

ROOT = Path(__file__).resolve().parents[2]


class EnergyTests(unittest.TestCase):
    def engine(self, scenario="line_fault", **config):
        package = EnergyPackage({"nodes_per_context": 40,
                                 "scenario_weights": {scenario: 1.0}, **config})
        return package, SimulationEngine(package, build_energy_temporal_models(), max_events=128)

    def test_invalid_configuration(self):
        for config in ({"nodes_per_context": 10}, {"context_count": 0},
                       {"duration_seconds": float("nan")},
                       {"scenario_weights": {"unknown": 1}},
                       {"scenario_weights": {"line_fault": -1}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                EnergyPackage(config)

    def test_three_scenarios_are_valid_and_grounded(self):
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario)
            for seed in range(15):
                result = engine.run(seed, seed + 100)
                self.assertTrue(result.validation["passed"], result.validation)
                self.assertTrue(result.validation["checks"]["domain_contracts"])
                self.assertEqual(package.validate_result(result), [])
                alignment = validate_text_alignment(result, package.render_episode_text(result), package.catalog())
                self.assertTrue(alignment["passed"], alignment)
                self.assertEqual(sum(e.event_type_id == SETTLED for e in result.events), 1)

    def test_thousand_nodes_sparse_and_outage_downstream(self):
        _, engine = self.engine(nodes_per_context=1000)
        result = engine.run(0, 42)
        self.assertEqual(len(result.entities), 1000)
        self.assertLess(len(result.context_relations), 2000)
        self.assertEqual({e.entity_type_id for e in result.entities},
                         {NS + t for t in ("bus", "line", "breaker", "load", "generator")})
        self.assertTrue(result.validation["passed"], result.validation)
        opens = [e for e in result.events if e.event_type_id == OPEN]
        interrupted = {e.attributes["load_id"] for e in result.events if e.event_type_id == INTERRUPTED}
        restored = {e.attributes["load_id"] for e in result.events if e.event_type_id == RESTORED}
        self.assertEqual(len(opens), 1)
        self.assertEqual(interrupted, restored)
        self.assertEqual(len(interrupted), opens[0].attributes["downstream_load_count"])
        self.assertTrue(any(e.event_type_id == REPAIRED for e in result.events))
        self.assertTrue(any(e.event_type_id == CLOSED for e in result.events))

    def test_surge_may_trip_or_normalize_before_trip(self):
        _, engine = self.engine("demand_surge")
        results = [engine.run(i, i + 300) for i in range(40)]
        self.assertTrue(any(any(e.event_type_id == OPEN for e in r.events) for r in results))
        self.assertTrue(any(not any(e.event_type_id == OPEN for e in r.events) for r in results))
        for result in results:
            self.assertTrue(result.validation["passed"], result.validation)
            kinds = [e.event_type_id for e in result.events]
            self.assertIn(SURGE, kinds)
            self.assertIn(NORMAL, kinds)
            if OPEN in kinds:
                self.assertLess(kinds.index(HIGH), kinds.index(OPEN))
                self.assertLess(kinds.index(NORMAL), kinds.index(CLOSED))
            else:
                self.assertNotIn(INTERRUPTED, kinds)

    def test_shortfall_single_restoration(self):
        _, engine = self.engine("generation_shortfall")
        dispatch_branches = set()
        for seed in range(30):
            result = engine.run(seed, seed + 500)
            self.assertTrue(result.validation["passed"], result.validation)
            kinds = [e.event_type_id for e in result.events]
            self.assertIn(SHORTFALL, kinds)
            self.assertIn(RECOVERED, kinds)
            self.assertEqual(kinds.count(RESERVE_RESTORED), 1)
            self.assertLess(kinds.index(RESERVE_RESTORED), kinds.index(SETTLED))
            dispatch_branches.add(DISPATCHED in kinds)
            if DISPATCHED in kinds:
                self.assertLess(kinds.index(DISPATCHED), kinds.index(RESERVE_RESTORED))
        self.assertEqual(dispatch_branches, {False, True})

    def test_short_horizon_and_continuation(self):
        _, engine = self.engine(duration_seconds=.001)
        result = engine.run(0, 99)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertFalse(any(e.event_type_id == SETTLED for e in result.events))
        _, engine = self.engine()
        snapshot = engine.run(0, 111, stop_after_events=2)
        left = engine.run(0, 111, snapshot=snapshot, continuation_seed=10)
        right = engine.run(0, 111, snapshot=snapshot, continuation_seed=11)
        self.assertTrue(left.validation["passed"], left.validation)
        self.assertTrue(right.validation["passed"], right.validation)
        self.assertEqual([e.to_dict() for e in left.events[:2]],
                         [e.to_dict() for e in right.events[:2]])

    def test_domain_validator_rejects_wrong_load_and_unsafe_reclose(self):
        package, engine = self.engine()
        result = engine.run(0, 45)
        broken = copy.deepcopy(result)
        event = next(e for e in broken.events if e.event_type_id == INTERRUPTED)
        event.attributes["load_id"] = next(e.entity_id for e in broken.entities
                                           if e.entity_type_id == NS + "generator")
        self.assertTrue(any("load identity mismatch" in issue
                            for issue in package.validate_result(broken)))
        broken = copy.deepcopy(result)
        next(e for e in broken.events if e.event_type_id == REPAIRED).event_type_id = NS + "line.fault.occurred"
        self.assertTrue(any("Unsafe breaker reclose" in issue
                            for issue in package.validate_result(broken)))

    def test_cli_exports_qa_and_training(self):
        output = ROOT / "data_generation" / "artifacts" / "test_runtime_output" / f"energy_{uuid.uuid4().hex}"
        process = subprocess.run([sys.executable, "data_generation/run_data_generation.py",
            "--domain", "energy", "--episode-count", "3", "--nodes-per-context", "40",
            "--judge-mode", "none", "--output-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"], report)
        self.assertGreater((output / "qa_pairs.jsonl").stat().st_size, 0)
        self.assertGreater((output / "training" / "qa_instruction.jsonl").stat().st_size, 0)
