"""Ecology patch, trophic, temporal and export contracts."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from data_generation.domain_packages.ecology import EcologyPackage, build_ecology_temporal_models
from data_generation.domain_packages.ecology.package import (
    NS, SCENARIOS, DROUGHT, PULSE, ARRIVED, ESTABLISHED, FAILED,
    LOW, HIGH, HERB_LOW, PRED_LOW,
)
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.validation import validate_text_alignment

ROOT = Path(__file__).resolve().parents[2]


class EcologyTests(unittest.TestCase):
    def engine(self, scenario="regional_drought", **config):
        package = EcologyPackage({"nodes_per_context": 40,
            "scenario_weights": {scenario: 1.0}, **config})
        return package, SimulationEngine(package, build_ecology_temporal_models(), max_events=512)

    def test_configuration_and_registry(self):
        for config in ({"nodes_per_context": 10}, {"context_count": 0},
                       {"duration_seconds": float("nan")},
                       {"scenario_weights": {"unknown": 1}},
                       {"scenario_weights": {"resource_pulse": -1}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                EcologyPackage(config)
        package, _ = self.engine()
        self.assertEqual(len(package.mechanism_registry), 19)

    def test_three_scenarios_validate_and_text_align(self):
        roots = {"regional_drought": DROUGHT, "resource_pulse": PULSE,
                 "invasive_colonization": ARRIVED}
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario)
            for seed in range(15):
                result = engine.run(seed, seed + 100)
                self.assertTrue(result.validation["passed"], result.validation)
                self.assertEqual(package.validate_result(result), [])
                self.assertTrue(validate_text_alignment(
                    result, package.render_episode_text(result), package.catalog())["passed"])
                self.assertIn(roots[scenario], {event.event_type_id for event in result.events})
                self.assertFalse(result.episode_attributes["empirical_realism_claimed"])

    def test_thousand_nodes_sparse_and_conditional_trophic_links(self):
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario, nodes_per_context=1000)
            result = engine.run(0, 42)
            self.assertEqual(len(result.entities), 1000)
            self.assertLess(len(result.context_relations), 2000)
            self.assertTrue(result.validation["passed"], result.validation)
            self.assertEqual(package.validate_result(result), [])
            kinds = {event.event_type_id for event in result.events}
            if scenario == "regional_drought":
                self.assertIn(LOW, kinds)
                self.assertIn(HERB_LOW, kinds)
                self.assertIn(PRED_LOW, kinds)
                drought_losses = sum(event.event_type_id == LOW for event in result.events)
                herb_losses = sum(event.event_type_id == HERB_LOW for event in result.events)
                self.assertLess(herb_losses, drought_losses)
            if scenario == "resource_pulse":
                self.assertIn(HIGH, kinds)
                self.assertLess(sum(event.event_type_id == NS + "population.herbivore.increased"
                                    for event in result.events),
                                sum(event.event_type_id == HIGH for event in result.events))
            if scenario == "invasive_colonization":
                by_patch = {}
                for event in result.events:
                    if event.event_type_id in (ESTABLISHED, FAILED):
                        by_patch.setdefault(event.attributes["patch_id"], set()).add(event.event_type_id)
                self.assertTrue(all(len(outcomes) == 1 for outcomes in by_patch.values()))

    def test_competing_outcomes_and_cancellation(self):
        _, engine = self.engine("invasive_colonization")
        saw_failure = saw_establishment = saw_cancel = False
        saw_competition_drop = False
        for seed in range(40):
            result = engine.run(seed, seed + 1000)
            kinds = {event.event_type_id for event in result.events}
            saw_failure |= FAILED in kinds
            saw_establishment |= ESTABLISHED in kinds
            saw_cancel |= any(candidate.status.value == "cancelled" for candidate in result.candidates)
            for event in result.events:
                if event.event_type_id == LOW and event.attributes.get("driver") == "invader_competition":
                    self.assertGreater(event.attributes["invader_abundance_after"],
                                       event.attributes["invader_abundance_before"])
                    self.assertGreater(event.attributes["availability_before_kg_per_effective_consumer"],
                                       event.attributes["availability_after_kg_per_effective_consumer"])
                    saw_competition_drop = True
        self.assertTrue(saw_failure)
        self.assertTrue(saw_establishment)
        self.assertTrue(saw_cancel)
        self.assertTrue(saw_competition_drop)

    def test_tampered_population_and_resource_rejected(self):
        package, engine = self.engine("regional_drought")
        result = engine.run(0, 100)
        broken = copy.deepcopy(result)
        herb = next(event for event in broken.events if event.event_type_id == HERB_LOW)
        herb.attributes["abundance_after"] = herb.attributes["abundance_before"] + 100
        self.assertTrue(package.validate_result(broken))
        broken = copy.deepcopy(result)
        low = next(event for event in broken.events if event.event_type_id == LOW)
        low.attributes["resource_factor_after"] = 2.0
        self.assertTrue(package.validate_result(broken))

    def test_short_horizon_and_continuation(self):
        _, engine = self.engine(duration_seconds=.001)
        result = engine.run(0, 77)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertEqual(result.events, [])
        _, engine = self.engine("invasive_colonization")
        snapshot = engine.run(0, 111, stop_after_events=2)
        left = engine.run(0, 111, snapshot=snapshot, continuation_seed=10)
        right = engine.run(0, 111, snapshot=snapshot, continuation_seed=11)
        self.assertTrue(left.validation["passed"], left.validation)
        self.assertTrue(right.validation["passed"], right.validation)
        self.assertEqual([event.to_dict() for event in left.events[:2]],
                         [event.to_dict() for event in right.events[:2]])

    def test_cli_qa_and_training_export(self):
        output = ROOT / "data_generation" / "artifacts" / "test_runtime_output" / f"ecology_{uuid.uuid4().hex}"
        process = subprocess.run([sys.executable, "data_generation/run_data_generation.py",
            "--domain", "ecology", "--episode-count", "3", "--nodes-per-context", "40",
            "--judge-mode", "none", "--output-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"], report)
        self.assertGreater((output / "qa_pairs.jsonl").stat().st_size, 0)
        self.assertGreater((output / "training" / "qa_instruction.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
