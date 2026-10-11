"""Analytic water/thermal clocks, supply interlocks and frozen sensor evidence."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from data_generation.domain_packages.agriculture import AgriculturePackage, build_agriculture_temporal_models
from data_generation.domain_packages.agriculture.package import (
    NS, DAY, SCENARIOS, ESTABLISHED, MILESTONE, RAIN, FAILED, REPAIRED,
    DRY, WET, SAMPLE, REPORT, REQUEST, START, END, INTERRUPTED, DEFERRED, SAMPLE_FIELDS,
)
from data_generation.domain_packages.agriculture.physics import (
    advance_water, time_to_depletion, stress_coefficient, thermal_rate,
)
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.temporal import ConditionalLogNormalModel, ExponentialArrivalModel
from data_generation.generation_core.validation import validate_text_alignment

ROOT = Path(__file__).resolve().parents[2]


class AgricultureTests(unittest.TestCase):
    def engine(self, scenario="irrigation_response", **config):
        package = AgriculturePackage({"nodes_per_context": 40, "scenario_weights": {scenario: 1}, **config})
        return package, SimulationEngine(package, build_agriculture_temporal_models(), max_events=2048)

    def test_configuration_and_registry(self):
        for config in ({"nodes_per_context": 10}, {"context_count": 0}, {"nodes_per_context": float("inf")},
            {"duration_seconds": float("nan")}, {"effective_rain_range_mm": [-1, 2]},
            {"mean_temperature_range_c": [30, 20]}, {"reference_et_range_mm_day": [0]},
            {"initial_depletion_fraction": .4}, {"irrigation_target_fraction": -1},
            {"scenario_weights": {"unknown": 1}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                AgriculturePackage(config)
        package, _ = self.engine()
        self.assertEqual(len(package.mechanism_registry), 15)

    def test_water_balance_piecewise_analytic_and_thermal_rate(self):
        self.assertEqual(stress_coefficient(20, 100, 40), 1)
        self.assertAlmostEqual(stress_coefficient(70, 100, 40), .5)
        self.assertEqual(stress_coefficient(100, 100, 40), 0)
        self.assertEqual(time_to_depletion(10, 40, 5, 0, 100, 40), 6)
        depletion, et, drainage = advance_water(10, 16, 5, 0, 100, 40)
        self.assertAlmostEqual(depletion, 100 - 60 * math.exp(-5 * 10 / 60))
        self.assertAlmostEqual(et, depletion - 10)
        self.assertEqual(drainage, 0)
        self.assertEqual(thermal_rate(5, 8, 30), 0)
        self.assertEqual(thermal_rate(18, 8, 30), 10)
        self.assertEqual(thermal_rate(40, 8, 30), 22)

    def test_irrigated_crossing_drainage_and_no_timebin_dependence(self):
        for d, r, j, duration in ((80, 5, 20, 10), (10, 5, 0, 20), (90, 0, 20, 7),
            (40, 5, 5, 2), (0, 5, 20, 10), (100, 5, 0, 1000)):
            whole = advance_water(d, duration, r, j, 100, 40)
            first = advance_water(d, duration * .37, r, j, 100, 40)
            second = advance_water(first[0], duration * .63, r, j, 100, 40)
            self.assertAlmostEqual(whole[0], second[0])
            self.assertAlmostEqual(whole[1], first[1] + second[1])
            self.assertAlmostEqual(whole[2], first[2] + second[2])
            self.assertAlmostEqual(whole[0], d + whole[1] - j * duration + whole[2])
        delay = time_to_depletion(80, 20, 5, 20, 100, 40)
        self.assertAlmostEqual(advance_water(80, delay, 5, 20, 100, 40)[0], 20)
        self.assertIsNone(time_to_depletion(80, 20, 5, 0, 100, 40))
        self.assertIsNone(time_to_depletion(80, 40, 5, 1, 100, 40))
        for args in ((10, 1, float("inf"), 0, 100, 40), (10, -1, 5, 0, 100, 40)):
            with self.assertRaises(ValueError):
                advance_water(*args)

    def test_three_scenarios_many_seeds_validate_and_align(self):
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario)
            for seed in range(20):
                result = engine.run(seed, seed + 100)
                self.assertTrue(result.validation["passed"], result.validation)
                self.assertEqual(package.validate_result(result), [])
                self.assertTrue(validate_text_alignment(result, package.render_episode_text(result), package.catalog())["passed"])
                self.assertFalse(result.episode_attributes["empirical_realism_claimed"])
                if scenario == "rainfed_drydown":
                    self.assertFalse(any(e.event_type_id == START for e in result.events))

    def test_thousand_nodes_functional_sparse_bindings(self):
        for scenario in SCENARIOS:
            _, engine = self.engine(scenario, nodes_per_context=1000)
            result = engine.run(0, 42)
            self.assertTrue(result.validation["passed"], result.validation)
            self.assertEqual(len(result.entities), 1000)
            self.assertEqual(len(result.context_relations), 1210)
            self.assertEqual({e.entity_type_id for e in result.entities}, {NS + t for t in
                ("field", "crop_cohort", "soil_water_sensor", "irrigator", "supply_unit", "weather_zone")})
            if scenario == "shared_supply_interruption":
                active_fields = result.episode_attributes["selected_fields"]
                irrigators = {r.target_entity_id: r.source_entity_id for r in result.context_relations if r.relation_type_id == NS + "irrigator_serves_field"}
                supplies = {r.source_entity_id: r.target_entity_id for r in result.context_relations if r.relation_type_id == NS + "irrigator_fed_by_supply"}
                self.assertEqual(len({supplies[irrigators[f]] for f in active_fields}), 1)

    def test_milestones_follow_temperature_not_calendar_or_irrigation(self):
        for temperature, expected_count in ((0, 0), (18, 2), (28, 3)):
            _, engine = self.engine(duration_seconds=30 * DAY, mean_temperature_range_c=[temperature, temperature])
            result = engine.run(0, 42)
            self.assertTrue(result.validation["passed"], result.validation)
            establishment = next(e for e in result.events if e.event_type_id == ESTABLISHED)
            milestones = [e for e in result.events if e.event_type_id == MILESTONE]
            self.assertEqual(len(milestones), expected_count)
            for event in milestones:
                self.assertAlmostEqual(event.attributes["gdd_before"], event.attributes["target_gdd"])
                self.assertAlmostEqual(event.temporal.anchor("occurrence_start") - establishment.temporal.anchor("occurrence_start"),
                    event.attributes["target_gdd"] / (temperature - 8) * DAY)

    def test_effective_rain_clips_storage_and_records_excess(self):
        _, engine = self.engine("rainfed_drydown", effective_rain_range_mm=[500, 500])
        result = engine.run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        rain = next(e for e in result.events if e.event_type_id == RAIN)
        self.assertEqual(rain.attributes["depletion_after_mm"], 0)
        self.assertAlmostEqual(rain.attributes["rain_deep_percolation_mm"], 500 - rain.attributes["depletion_before_mm"])
        later = [e for e in result.events if e.temporal.anchor("occurrence_start") > rain.temporal.anchor("occurrence_start")]
        self.assertTrue(later)
        self.assertTrue(any(e.attributes.get("cumulative_effective_rain_mm") == 500 for e in later))

    def test_supply_workflow_actual_parents_and_partial_delivery(self):
        _, engine = self.engine("shared_supply_interruption")
        saw_interrupt, saw_defer, saw_resume = False, False, False
        for seed in range(30):
            result = engine.run(seed, seed + 100)
            self.assertTrue(result.validation["passed"], result.validation)
            events = {e.event_id: e for e in result.events}
            for candidate in result.candidates:
                if not candidate.fired_event_id:
                    continue
                event = events[candidate.fired_event_id]
                if event.event_type_id == INTERRUPTED:
                    saw_interrupt = True
                    parents = [events[i] for i in candidate.parent_event_ids]
                    self.assertEqual([e.event_type_id for e in parents], [START, FAILED])
                    self.assertLess(event.attributes["delivered_gross_volume_m3"], parents[0].attributes["planned_gross_volume_m3"])
                    self.assertAlmostEqual(event.temporal.anchor("occurrence_start"), parents[1].temporal.anchor("occurrence_start"))
                    self.assertTrue(any(c.status.value == "cancelled" and c.mechanism_id == NS + "irrigation_complete"
                        and parents[0].event_id in c.parent_event_ids for c in result.candidates))
                if event.event_type_id == DEFERRED:
                    saw_defer = True
                    self.assertEqual({events[i].event_type_id for i in candidate.parent_event_ids}, {REQUEST, FAILED})
                if candidate.mechanism_id == NS + "irrigation_resume":
                    saw_resume = True
                    self.assertEqual({events[i].event_type_id for i in candidate.parent_event_ids}, {REQUEST, REPAIRED})
        self.assertTrue(saw_interrupt and saw_defer and saw_resume)

    def test_recovered_interruption_does_not_block_later_requests(self):
        package, engine = self.engine("shared_supply_interruption")
        saw_case = False
        for seed in range(50):
            result = engine.run(seed, seed + 100)
            for event in result.events:
                if event.event_type_id == INTERRUPTED and not event.attributes["low_water_before"]:
                    saw_case = True
                    later = [e for e in result.events if e.attributes.get("field_id") == event.attributes["field_id"]
                        and e.temporal.anchor("occurrence_start") > event.temporal.anchor("occurrence_start")]
                    self.assertTrue(any(e.event_type_id == REQUEST for e in later))
            self.assertEqual(package.validate_result(result), [])
        self.assertTrue(saw_case)

    def test_sensor_reports_freeze_samples_and_ignore_stale_controls(self):
        package, _ = self.engine()
        models = build_agriculture_temporal_models()
        models.replace(ConditionalLogNormalModel(NS + "telemetry_delay.prior_v1", math.log(2 * DAY), 2., {}, minimum_seconds=1),
            parameter_status="synthetic_test_fixture")
        engine = SimulationEngine(package, models, max_events=2048)
        saw_stale, saw_changed_state = False, False
        for seed in range(15):
            result = engine.run(seed, seed + 100)
            self.assertTrue(result.validation["passed"], result.validation)
            events, sequences = {e.event_id: e for e in result.events}, {}
            for event in result.events:
                if event.event_type_id != REPORT:
                    continue
                field, sequence = event.attributes["field_id"], event.attributes["sample_sequence"]
                stale = sequence <= sequences.get(field, 0)
                self.assertEqual(event.attributes["stale_report"], stale)
                saw_stale |= stale
                if not stale:
                    sequences[field] = sequence
                sample = events[event.attributes["sample_event_id"]]
                self.assertTrue(all(event.attributes[key] == sample.attributes[key] for key in SAMPLE_FIELDS))
                saw_changed_state |= abs(event.attributes["sampled_depletion_mm"] - event.attributes["depletion_before_mm"]) > .01
                if stale:
                    self.assertFalse(any(c.fired_event_id and c.mechanism_id == NS + "irrigation_request"
                        and event.event_id in c.parent_event_ids for c in result.candidates))
        self.assertTrue(saw_stale and saw_changed_state)

    def test_tampered_water_thermal_delivery_samples_and_bindings_rejected(self):
        package, engine = self.engine()
        result = engine.run(0, 42)
        for kind, field in ((MILESTONE, "target_gdd"), (START, "planned_gross_volume_m3"),
            (END, "delivered_gross_volume_m3"), (REPORT, "sampled_depletion_mm"), (DRY, "cumulative_et_mm")):
            broken = copy.deepcopy(result)
            event = next(e for e in broken.events if e.event_type_id == kind)
            event.attributes[field] += 1
            self.assertTrue(package.validate_result(broken))
        for kind, field in (("irrigator", "flow_m3_s"), ("supply_unit", "rated_capacity_m3_s")):
            broken = copy.deepcopy(result)
            entity = next(e for e in broken.entities if e.entity_type_id == NS + kind)
            entity.attributes[field] = float("inf")
            self.assertTrue(package.validate_result(broken))
        broken = copy.deepcopy(result)
        broken.context_relations.pop()
        self.assertTrue(package.validate_result(broken))
        broken = copy.deepcopy(result)
        event = next(e for e in broken.events if e.event_type_id == START)
        wrong_supply = next(e.entity_id for e in broken.entities if e.entity_type_id == NS + "supply_unit"
            and e.entity_id != event.attributes["supply_id"])
        event.attributes["supply_id"] = wrong_supply
        candidate = next(c for c in broken.candidates if c.fired_event_id == event.event_id)
        candidate.attributes["supply_id"] = wrong_supply
        self.assertTrue(package.validate_result(broken))

    def test_short_window_censoring_and_continuation(self):
        _, engine = self.engine(duration_seconds=.001)
        result = engine.run(0, 77)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertEqual(result.events, [])
        self.assertTrue(any(c.status.value == "right_censored" for c in result.candidates))
        _, engine = self.engine("shared_supply_interruption")
        snapshot = engine.run(0, 111, stop_after_events=2)
        left = engine.run(0, 111, snapshot=snapshot, continuation_seed=10)
        right = engine.run(0, 111, snapshot=snapshot, continuation_seed=11)
        self.assertTrue(left.validation["passed"], left.validation)
        self.assertTrue(right.validation["passed"], right.validation)
        self.assertEqual([e.to_dict() for e in left.events[:2]], [e.to_dict() for e in right.events[:2]])

    def test_cli_qa_and_training_export(self):
        output = ROOT / "data_generation" / "artifacts" / "test_runtime_output" / f"agriculture_{uuid.uuid4().hex}"
        process = subprocess.run([sys.executable, "data_generation/run_data_generation.py", "--domain", "agriculture",
            "--episode-count", "3", "--nodes-per-context", "40", "--judge-mode", "none", "--output-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["time_axis"]["unit"], "day")
        self.assertGreater((output / "qa_pairs.jsonl").stat().st_size, 0)
        self.assertGreater((output / "training" / "qa_instruction.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
