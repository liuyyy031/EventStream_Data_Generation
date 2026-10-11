"""Water balance, directional lag routing, confluence and gauge evidence tests."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from data_generation.domain_packages.hydrology import HydrologyPackage, build_hydrology_temporal_models
from data_generation.domain_packages.hydrology.package import (
    NS, HOUR, SCENARIOS, RAIN, RAIN_END, RUNOFF, NO_RUNOFF, ARRIVAL, CLEAR,
    HIGH, SAMPLE, REPORT, WARNING, WITHDRAWN, runoff_plan,
)
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.temporal import ConditionalLogNormalModel, ExponentialArrivalModel
from data_generation.generation_core.validation import validate_text_alignment

ROOT = Path(__file__).resolve().parents[2]


class HydrologyTests(unittest.TestCase):
    def engine(self, scenario="isolated_storm", **config):
        package = HydrologyPackage({"nodes_per_context": 40, "scenario_weights": {scenario: 1}, **config})
        return package, SimulationEngine(package, build_hydrology_temporal_models(), max_events=2048)

    def test_configuration_and_registry(self):
        for config in ({"nodes_per_context": 10}, {"context_count": 0}, {"nodes_per_context": float("inf")},
            {"duration_seconds": float("nan")}, {"rainfall_intensity_range_mm_h": [-1, 3]},
            {"storm_duration_range_seconds": [0, 20]}, {"initial_loss_range_mm": [3, 2]},
            {"constant_loss_range_mm_h": [0]}, {"scenario_weights": {"unknown": 1}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                HydrologyPackage(config)
        package, _ = self.engine()
        self.assertEqual(len(package.mechanism_registry), 16)

    def test_analytic_excess_depth_volume_and_lag(self):
        plan = runoff_plan(20, 7200, 5, 3, 1, 600)
        self.assertEqual(plan["initial_fill_seconds"], 900)
        self.assertEqual(plan["pulse_duration_seconds"], 6300)
        self.assertAlmostEqual(plan["excess_flow_m3_s"], 17 / 3.6)
        self.assertAlmostEqual(plan["runoff_volume_m3"], 29750)
        self.assertAlmostEqual(plan["rain_depth_mm"], 40)
        self.assertAlmostEqual(plan["excess_depth_mm"], 29.75)
        self.assertAlmostEqual(plan["loss_depth_mm"] + plan["excess_depth_mm"], plan["rain_depth_mm"])
        for args in ((0, 3600, 5, 3, 1, 600), (2, 3600, 0, 3, 1, 600), (20, 3600, 25, 3, 1, 600)):
            self.assertEqual(runoff_plan(*args)["runoff_volume_m3"], 0)

    def test_three_scenarios_many_seeds_validate_and_align(self):
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario)
            for seed in range(20):
                result = engine.run(seed, seed + 100)
                self.assertTrue(result.validation["passed"], result.validation)
                self.assertEqual(package.validate_result(result), [])
                self.assertTrue(validate_text_alignment(result, package.render_episode_text(result), package.catalog())["passed"])
                self.assertFalse(result.episode_attributes["empirical_realism_claimed"])

    def test_thousand_nodes_sparse_and_conservative_travel(self):
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario, nodes_per_context=1000,
                rainfall_intensity_range_mm_h=[20, 20], initial_loss_range_mm=[0, 0], constant_loss_range_mm_h=[2, 2])
            result = engine.run(0, 42)
            self.assertEqual(len(result.entities), 1000)
            self.assertLess(len(result.context_relations), 1500)
            self.assertEqual({e.entity_type_id for e in result.entities},
                {NS + t for t in ("catchment", "reach", "gauge", "management_unit")})
            self.assertTrue(result.validation["passed"], result.validation)
            entities = {e.entity_id: e for e in result.entities}
            events = {e.event_id: e for e in result.events}
            downstream = {r.source_entity_id: r.target_entity_id for r in result.context_relations if r.relation_type_id == NS + "reach_flows_to_reach"}
            for candidate in result.candidates:
                if candidate.mechanism_id != NS + "downstream_routing" or not candidate.fired_event_id:
                    continue
                event, source = events[candidate.fired_event_id], events[candidate.parent_event_ids[0]]
                upstream = source.attributes["reach_id"]
                attrs = entities[upstream].attributes
                self.assertEqual(event.attributes["reach_id"], downstream[upstream])
                self.assertAlmostEqual(event.temporal.anchor("occurrence_start") - source.temporal.anchor("occurrence_start"),
                                       attrs["length_m"] / attrs["wave_celerity_m_s"])
                self.assertEqual(event.attributes["runoff_volume_m3"], source.attributes["runoff_volume_m3"])
                self.assertEqual(event.attributes["excess_flow_m3_s"], source.attributes["excess_flow_m3_s"])
            source_volume = sum(e.attributes["runoff_volume_m3"] for e in result.events if e.event_type_id == RUNOFF)
            outlet_volume = sum(e.attributes["runoff_volume_m3"] for e in result.events if e.event_type_id == CLEAR and e.attributes["reach_id"] not in downstream)
            self.assertAlmostEqual(outlet_volume, source_volume)

    def test_rain_does_not_force_runoff(self):
        _, engine = self.engine(rainfall_intensity_range_mm_h=[1, 1], constant_loss_range_mm_h=[5, 5])
        result = engine.run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertEqual([e.event_type_id for e in result.events], [RAIN, RAIN_END, NO_RUNOFF])

    def test_repeated_storm_retains_initial_loss_history(self):
        _, engine = self.engine("repeated_storms", rainfall_intensity_range_mm_h=[10, 10],
            storm_duration_range_seconds=[3600, 3600], initial_loss_range_mm=[15, 15], constant_loss_range_mm_h=[2, 2])
        result = engine.run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        rain = [e for e in result.events if e.event_type_id == RAIN]
        self.assertEqual(len(rain), 2)
        self.assertEqual(rain[0].attributes["initial_loss_before_mm"], 15)
        self.assertEqual(rain[0].attributes["runoff_volume_m3"], 0)
        self.assertEqual(rain[1].attributes["initial_loss_before_mm"], 5)
        self.assertAlmostEqual(rain[1].attributes["excess_depth_mm"], 4)
        self.assertTrue(any(r.relation_class == "statistical_influence" for r in result.event_relations))

    def test_confluence_parents_are_positive_overlapping_contributions(self):
        package = HydrologyPackage({"nodes_per_context": 40, "scenario_weights": {"tributary_confluence": 1},
            "rainfall_intensity_range_mm_h": [10, 10], "storm_duration_range_seconds": [7200, 7200],
            "initial_loss_range_mm": [0, 0], "constant_loss_range_mm_h": [0, 0]})
        # Controlled small fixture: neither tributary alone crosses the threshold.
        context = package.create_episode_context(0, 42)
        for entity in context.entities:
            if entity.entity_type_id == NS + "catchment":
                entity.attributes["area_km2"] = 1.0
            if entity.entity_type_id == NS + "reach":
                entity.attributes["threshold_flow_m3_s"] = entity.attributes["baseflow_m3_s"] + 1.5 * 10 / 3.6
        models = build_hydrology_temporal_models()
        models.replace(ExponentialArrivalModel(NS + "storm_arrival.prior_v1", 100), parameter_status="synthetic_test_fixture")
        result = SimulationEngine(package, models, max_events=2048).run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        events = {e.event_id: e for e in result.events}
        multi = [c for c in result.candidates if c.mechanism_id == NS + "flow_threshold_up" and c.fired_event_id]
        self.assertTrue(multi)
        for candidate in multi:
            event = events[candidate.fired_event_id]
            self.assertEqual(len(candidate.parent_event_ids), 2)
            self.assertEqual(set(candidate.parent_event_ids), set(event.attributes["active_contribution_event_ids"]))
            self.assertAlmostEqual(sum(event.attributes["active_contributions"].values()), 2 * 10 / 3.6)
            self.assertGreater(event.attributes["aggregate_flow_m3_s"], event.attributes["threshold_flow_m3_s"])

    def test_delayed_reports_freeze_samples_and_supersede_old_warnings(self):
        package = HydrologyPackage({"nodes_per_context": 40, "scenario_weights": {"isolated_storm": 1},
            "rainfall_intensity_range_mm_h": [100, 100], "storm_duration_range_seconds": [60, 60],
            "initial_loss_range_mm": [0, 0], "constant_loss_range_mm_h": [0, 0]})
        models = build_hydrology_temporal_models()
        models.replace(ConditionalLogNormalModel(NS + "telemetry_delay.prior_v1", math.log(7200), .01, {}, minimum_seconds=1), parameter_status="synthetic_test_fixture")
        models.replace(ConditionalLogNormalModel(NS + "response_delay.prior_v1", math.log(3600), .01, {}, minimum_seconds=1), parameter_status="synthetic_test_fixture")
        result = SimulationEngine(package, models, max_events=2048).run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        events = {e.event_id: e for e in result.events}
        reports = [e for e in result.events if e.event_type_id == REPORT]
        self.assertTrue(reports)
        for report in reports:
            sample = events[report.attributes["sample_event_id"]]
            self.assertEqual(report.attributes["sampled_flow_m3_s"], sample.attributes["sampled_flow_m3_s"])
            self.assertEqual(report.attributes["sampled_above_threshold"], sample.attributes["sampled_above_threshold"])
            self.assertGreater(report.temporal.anchor("occurrence_start") - sample.temporal.anchor("occurrence_start"), 6000)
        self.assertTrue(any(c.status.value == "cancelled" and c.mechanism_id == NS + "warning_raise" for c in result.candidates))
        self.assertFalse(any(e.event_type_id == WARNING for e in result.events))

    def test_out_of_order_reports_do_not_replace_newer_samples(self):
        package = HydrologyPackage({"nodes_per_context": 40, "scenario_weights": {"isolated_storm": 1},
            "rainfall_intensity_range_mm_h": [100, 100], "storm_duration_range_seconds": [60, 60],
            "initial_loss_range_mm": [0, 0], "constant_loss_range_mm_h": [0, 0]})
        models = build_hydrology_temporal_models()
        models.replace(ConditionalLogNormalModel(NS + "telemetry_delay.prior_v1", math.log(600), 2., {}, minimum_seconds=1), parameter_status="synthetic_test_fixture")
        engine = SimulationEngine(package, models, max_events=2048)
        saw_stale = False
        for seed in range(8):
            result = engine.run(seed, seed + 100)
            self.assertTrue(result.validation["passed"], result.validation)
            sequences = {}
            for event in result.events:
                if event.event_type_id == REPORT:
                    gauge = event.attributes["gauge_id"]
                    stale = event.attributes["sample_sequence"] <= sequences.get(gauge, 0)
                    self.assertEqual(event.attributes["stale_report"], stale)
                    saw_stale |= stale
                    if not stale:
                        sequences[gauge] = event.attributes["sample_sequence"]
        self.assertTrue(saw_stale)

    def test_tampered_volume_frozen_sample_and_topology_rejected(self):
        package, engine = self.engine(rainfall_intensity_range_mm_h=[100, 100], initial_loss_range_mm=[0, 0])
        result = engine.run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        for kind, field in ((ARRIVAL, "runoff_volume_m3"), (REPORT, "sampled_flow_m3_s")):
            broken = copy.deepcopy(result)
            event = next(e for e in broken.events if e.event_type_id == kind)
            event.attributes[field] += 1
            self.assertTrue(package.validate_result(broken))
        broken = copy.deepcopy(result)
        reach = next(e for e in broken.entities if e.entity_type_id == NS + "reach")
        reach.attributes["wave_celerity_m_s"] = 0
        self.assertTrue(package.validate_result(broken))

    def test_short_window_censoring_and_continuation(self):
        _, engine = self.engine(duration_seconds=.001)
        result = engine.run(0, 77)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertEqual(result.events, [])
        self.assertTrue(any(c.status.value == "right_censored" for c in result.candidates))
        _, engine = self.engine("tributary_confluence")
        snapshot = engine.run(0, 111, stop_after_events=2)
        left = engine.run(0, 111, snapshot=snapshot, continuation_seed=10)
        right = engine.run(0, 111, snapshot=snapshot, continuation_seed=11)
        self.assertTrue(left.validation["passed"], left.validation)
        self.assertTrue(right.validation["passed"], right.validation)
        self.assertEqual([e.to_dict() for e in left.events[:2]], [e.to_dict() for e in right.events[:2]])

    def test_cli_qa_and_training_export(self):
        output = ROOT / "data_generation" / "artifacts" / "test_runtime_output" / f"hydrology_{uuid.uuid4().hex}"
        process = subprocess.run([sys.executable, "data_generation/run_data_generation.py", "--domain", "hydrology",
            "--episode-count", "3", "--nodes-per-context", "40", "--judge-mode", "none", "--output-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["time_axis"]["unit"], "hour")
        self.assertGreater((output / "qa_pairs.jsonl").stat().st_size, 0)
        self.assertGreater((output / "training" / "qa_instruction.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
