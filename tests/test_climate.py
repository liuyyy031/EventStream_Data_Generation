"""Energy budgets, finite threshold clocks, forcing sums and sampled evidence."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from data_generation.domain_packages.climate import ClimatePackage, build_climate_temporal_models
from data_generation.domain_packages.climate.package import (
    NS, YEAR, SCENARIOS, START, END, SUM, WARM, WARM_END, WARM_PERSIST,
    COLD, COLD_END, COLD_PERSIST, SAMPLE, REPORT, BULLETIN, WITHDRAWN, SAMPLE_FIELDS,
)
from data_generation.domain_packages.climate.physics import advance_temperature, time_to_temperature
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.temporal import ConditionalLogNormalModel, ExponentialArrivalModel
from data_generation.generation_core.validation import validate_text_alignment

ROOT = Path(__file__).resolve().parents[2]


class ClimateTests(unittest.TestCase):
    def engine(self, scenario="warming_pulse", **config):
        package = ClimatePackage({"nodes_per_context": 40, "scenario_weights": {scenario: 1}, **config})
        return package, SimulationEngine(package, build_climate_temporal_models(), max_events=2048)

    @staticmethod
    def unit_fixture(package):
        context = package.create_episode_context(0, 42)
        for entity in context.entities:
            if entity.entity_type_id == NS + "climate_unit":
                entity.attributes.update(damping_w_m2_k=1., heat_capacity_j_m2_k=YEAR)

    def test_configuration_and_registry(self):
        for config in ({"nodes_per_context": 10}, {"context_count": 0}, {"nodes_per_context": float("inf")},
            {"duration_seconds": float("nan")}, {"positive_forcing_range_w_m2": [-1, 2]},
            {"negative_forcing_range_w_m2": [-1, 0]}, {"forcing_duration_range_years": [0, 1]},
            {"sample_period_years": 0}, {"persistence_years": -1}, {"scenario_weights": {"unknown": 1}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ClimatePackage(config)
        package, _ = self.engine()
        self.assertEqual(len(package.mechanism_registry), 14)

    def test_analytic_step_relaxation_and_signed_energy(self):
        t, forced, feedback = advance_temperature(0, 2 * YEAR, 2, 1, YEAR)
        self.assertAlmostEqual(t, 2 * (1 - math.exp(-2)))
        self.assertAlmostEqual(YEAR * t, forced - feedback)
        cooling, f, b = advance_temperature(0, 2 * YEAR, -2, 1, YEAR)
        self.assertAlmostEqual(cooling, -t)
        self.assertLess(f, 0)
        self.assertLess(b, 0)
        relaxed, forced, _ = advance_temperature(t, YEAR, 0, 1, YEAR)
        self.assertAlmostEqual(relaxed, t / math.e)
        self.assertEqual(forced, 0)
        self.assertEqual(advance_temperature(t, 0, 9, 1, YEAR)[0], t)

    def test_crossing_unreachable_targets_and_segment_split_invariance(self):
        crossing = time_to_temperature(0, 1, 2, 1, YEAR)
        self.assertAlmostEqual(crossing, YEAR * math.log(2))
        self.assertAlmostEqual(advance_temperature(0, crossing, 2, 1, YEAR)[0], 1)
        self.assertIsNone(time_to_temperature(0, 2, 2, 1, YEAR))
        self.assertIsNone(time_to_temperature(0, -1, 2, 1, YEAR))
        for initial, forcing in ((0, 2), (2, -1), (-2, 0), (0, 0)):
            whole = advance_temperature(initial, 10 * YEAR, forcing, 1.2, 5 * YEAR)
            first = advance_temperature(initial, 3 * YEAR, forcing, 1.2, 5 * YEAR)
            second = advance_temperature(first[0], 7 * YEAR, forcing, 1.2, 5 * YEAR)
            self.assertAlmostEqual(whole[0], second[0])
            self.assertAlmostEqual(whole[1], first[1] + second[1], delta=1e-5)
            self.assertAlmostEqual(whole[2], first[2] + second[2], delta=1e-5)
        for args in ((0, -1, 2, 1, YEAR), (0, YEAR, 2, 0, YEAR), (0, YEAR, float("inf"), 1, YEAR)):
            with self.assertRaises(ValueError):
                advance_temperature(*args)

    def test_four_scenarios_many_seeds_validate_and_align(self):
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario)
            for seed in range(20):
                result = engine.run(seed, seed + 100)
                self.assertTrue(result.validation["passed"], result.validation)
                self.assertEqual(package.validate_result(result), [])
                self.assertTrue(validate_text_alignment(result, package.render_episode_text(result), package.catalog())["passed"])
                self.assertFalse(result.episode_attributes["empirical_realism_claimed"])

    def test_thousand_nodes_sparse_functional_bindings(self):
        for scenario in SCENARIOS:
            _, engine = self.engine(scenario, nodes_per_context=1000)
            result = engine.run(0, 42)
            self.assertTrue(result.validation["passed"], result.validation)
            self.assertEqual(len(result.entities), 1000)
            self.assertEqual(len(result.context_relations), 980)
            self.assertEqual({e.entity_type_id for e in result.entities}, {NS + k for k in
                ("climate_unit", "forcing_driver", "anomaly_observer", "climate_service")})
            self.assertEqual({e.attributes["unit_id"] for e in result.events}, {result.episode_attributes["selected_unit"]})

    def test_weak_forcing_does_not_force_exceedance(self):
        _, engine = self.engine(positive_forcing_range_w_m2=[.01, .01])
        result = engine.run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertFalse(any(e.event_type_id in (WARM, COLD, BULLETIN, WARM_PERSIST, COLD_PERSIST) for e in result.events))
        self.assertTrue(any(e.event_type_id == START for e in result.events))
        self.assertTrue(any(e.event_type_id == SAMPLE for e in result.events))

    def test_superposition_uses_actual_overlapping_forcings_not_edges(self):
        package, engine = self.engine("combined_warming", positive_forcing_range_w_m2=[.4, .4], forcing_duration_range_years=[12, 12])
        self.unit_fixture(package)
        result = engine.run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        events = {e.event_id: e for e in result.events}
        combined = [e for e in result.events if e.event_type_id == SUM]
        self.assertTrue(combined)
        self.assertTrue(any(e.event_type_id == WARM for e in result.events))
        for event in combined:
            candidate = next(c for c in result.candidates if c.fired_event_id == event.event_id)
            self.assertEqual(len(candidate.parent_event_ids), 2)
            self.assertTrue(all(events[p].event_type_id == START for p in candidate.parent_event_ids))
            self.assertEqual(set(candidate.parent_event_ids), set(event.attributes["active_forcing_components_w_m2"]))
            self.assertAlmostEqual(event.attributes["total_forcing_before_w_m2"], .8)
            self.assertEqual({p.role for p in event.participants}, {"unit", "driver_a", "driver_b"})
        # Either .4 forcing alone has equilibrium below the .7 threshold.
        _, single_engine = self.engine(positive_forcing_range_w_m2=[.4, .4], forcing_duration_range_years=[12, 12])
        single_result = single_engine.run(0, 42)
        self.assertFalse(any(e.event_type_id == WARM for e in single_result.events))

    def test_opposite_forcings_cancel_without_instant_reset(self):
        package, _ = self.engine("opposing_forcings", positive_forcing_range_w_m2=[1, 1], negative_forcing_range_w_m2=[-1, -1], forcing_duration_range_years=[12, 12])
        self.unit_fixture(package)
        models = build_climate_temporal_models()
        models.replace(ExponentialArrivalModel(NS + "forcing_arrival.prior_v1", 1e6), parameter_status="synthetic_test_fixture")
        result = SimulationEngine(package, models, max_events=2048).run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        combined = next(e for e in result.events if e.event_type_id == SUM)
        self.assertEqual(combined.attributes["total_forcing_before_w_m2"], 0)
        self.assertFalse(any(e.event_type_id in (WARM, COLD, BULLETIN) for e in result.events))
        # A single pulse leaves a decaying nonzero anomaly after its forcing ends.
        _, engine = self.engine()
        result = engine.run(0, 42)
        end = next(e for e in result.events if e.event_type_id == END)
        self.assertGreater(end.attributes["temperature_anomaly_before_k"], 0)
        later = next(e for e in result.events if e.event_type_id == SAMPLE and e.temporal.anchor("occurrence_start") > end.temporal.anchor("occurrence_start"))
        self.assertGreater(later.attributes["temperature_anomaly_before_k"], 0)
        self.assertLess(later.attributes["temperature_anomaly_before_k"], end.attributes["temperature_anomaly_before_k"])

    def test_insufficient_duration_cancels_persistence_confirmation(self):
        package, engine = self.engine(positive_forcing_range_w_m2=[10, 10], forcing_duration_range_years=[.1, .1], duration_seconds=4 * YEAR)
        self.unit_fixture(package)
        result = engine.run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertTrue(any(e.event_type_id == WARM for e in result.events))
        self.assertTrue(any(e.event_type_id == WARM_END for e in result.events))
        self.assertFalse(any(e.event_type_id == WARM_PERSIST for e in result.events))
        self.assertTrue(any(c.mechanism_id == NS + "warm_persistence" and c.status.value == "cancelled" for c in result.candidates))

    def test_persistence_and_observation_schedules_are_distinct(self):
        package, engine = self.engine()
        result = engine.run(0, 42)
        events = {e.event_id: e for e in result.events}
        samples = [e for e in result.events if e.event_type_id == SAMPLE]
        self.assertEqual([e.temporal.anchor("occurrence_start") for e in samples], [i * YEAR for i in range(1, 31)])
        confirmations = [e for e in result.events if e.event_type_id == WARM_PERSIST]
        self.assertTrue(confirmations)
        for event in confirmations:
            candidate = next(c for c in result.candidates if c.fired_event_id == event.event_id)
            parent = events[candidate.parent_event_ids[0]]
            self.assertAlmostEqual(event.temporal.anchor("occurrence_start") - parent.temporal.anchor("occurrence_start"), YEAR)
        self.assertTrue(all(e.attributes["sampling_semantics"].startswith("scheduled_point_snapshot") for e in samples))
        self.assertEqual(package.validate_result(result), [])

    def test_late_out_of_order_reports_preserve_snapshots_and_latest_policy(self):
        package, _ = self.engine("opposing_forcings")
        models = build_climate_temporal_models()
        models.replace(ConditionalLogNormalModel(NS + "reporting_delay.prior_v1", math.log(2 * YEAR), 2., {}, minimum_seconds=1),
            parameter_status="synthetic_test_fixture")
        engine = SimulationEngine(package, models, max_events=2048)
        saw_stale, saw_difference = False, False
        for seed in range(8):
            result = engine.run(seed, seed + 100)
            self.assertTrue(result.validation["passed"], result.validation)
            events, latest = {e.event_id: e for e in result.events}, 0
            for event in result.events:
                if event.event_type_id != REPORT:
                    continue
                stale = event.attributes["sample_sequence"] <= latest
                self.assertEqual(event.attributes["stale_report"], stale)
                saw_stale |= stale
                if not stale:
                    latest = event.attributes["sample_sequence"]
                sample = events[event.attributes["sample_event_id"]]
                self.assertTrue(all(event.attributes[k] == sample.attributes[k] for k in SAMPLE_FIELDS))
                saw_difference |= abs(event.attributes["sampled_temperature_anomaly_k"] - event.attributes["temperature_anomaly_before_k"]) > .01
                if stale:
                    self.assertFalse(any(c.fired_event_id and c.mechanism_id in (NS + "bulletin_raise", NS + "bulletin_clear")
                        and event.event_id in c.parent_event_ids for c in result.candidates))
        self.assertTrue(saw_stale and saw_difference)

    def test_tampered_energy_forcing_sample_and_topology_rejected(self):
        package, engine = self.engine()
        result = engine.run(0, 42)
        for kind, field in ((START, "forcing_duration_seconds"), (SAMPLE, "cumulative_forcing_energy_j_m2"),
            (REPORT, "sampled_temperature_anomaly_k"), (WARM, "threshold_k")):
            broken = copy.deepcopy(result)
            event = next(e for e in broken.events if e.event_type_id == kind)
            event.attributes[field] += 100
            self.assertTrue(package.validate_result(broken))
        broken = copy.deepcopy(result)
        unit = next(e for e in broken.entities if e.entity_type_id == NS + "climate_unit")
        unit.attributes["heat_capacity_j_m2_k"] = float("inf")
        self.assertTrue(package.validate_result(broken))
        broken = copy.deepcopy(result)
        broken.context_relations.pop()
        self.assertTrue(package.validate_result(broken))

    def test_short_window_and_shared_prefix_continuation(self):
        _, engine = self.engine(duration_seconds=.001)
        result = engine.run(0, 77)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertEqual(result.events, [])
        self.assertTrue(all(c.status.value == "right_censored" for c in result.candidates))
        _, engine = self.engine("opposing_forcings")
        snapshot = engine.run(0, 111, stop_after_events=2)
        left = engine.run(0, 111, snapshot=snapshot, continuation_seed=10)
        right = engine.run(0, 111, snapshot=snapshot, continuation_seed=11)
        self.assertTrue(left.validation["passed"], left.validation)
        self.assertTrue(right.validation["passed"], right.validation)
        self.assertEqual([e.to_dict() for e in left.events[:2]], [e.to_dict() for e in right.events[:2]])

    def test_cli_qa_and_training_export(self):
        output = ROOT / "data_generation" / "artifacts" / "test_runtime_output" / f"climate_{uuid.uuid4().hex}"
        process = subprocess.run([sys.executable, "data_generation/run_data_generation.py", "--domain", "climate",
            "--episode-count", "3", "--nodes-per-context", "40", "--judge-mode", "none", "--output-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["time_axis"]["unit"], "year")
        self.assertEqual(report["time_axis"]["kind"], "relative")
        self.assertGreater((output / "qa_pairs.jsonl").stat().st_size, 0)
        self.assertGreater((output / "training" / "qa_instruction.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
