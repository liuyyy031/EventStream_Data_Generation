"""Public-health contact attribution, frozen specimens and restriction contracts."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from data_generation.domain_packages.public_health import PublicHealthPackage, build_public_health_temporal_models
from data_generation.domain_packages.public_health.package import (
    NS, DAY, SCENARIOS, CONTACT_TYPES, EXPOSED, INFECTIOUS, RECOVERED,
    SAMPLE, POSITIVE, NEGATIVE, NOTIFIED, ISOLATED, QUARANTINED,
)
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.temporal import ConditionalLogNormalModel
from data_generation.generation_core.validation import validate_text_alignment

ROOT = Path(__file__).resolve().parents[2]


class PublicHealthTests(unittest.TestCase):
    def engine(self, scenario="household_cluster", **config):
        package = PublicHealthPackage({"nodes_per_context": 40,
            "scenario_weights": {scenario: 1.0}, **config})
        return package, SimulationEngine(package, build_public_health_temporal_models(), max_events=2048)

    def test_configuration_and_registry(self):
        for config in ({"nodes_per_context": 10}, {"context_count": 0},
                       {"duration_seconds": float("nan")}, {"screening_day": -1},
                       {"transmission_probability_per_contact": 1.1},
                       {"trace_coverage": -.1}, {"assay_infectious_sensitivity": float("inf")},
                       {"scenario_weights": {"unknown": 1}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                PublicHealthPackage(config)
        package, _ = self.engine()
        self.assertEqual(len(package.mechanism_registry), 18)

    def test_three_scenarios_validate_and_text_align(self):
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario)
            for seed in range(12):
                result = engine.run(seed, seed + 100)
                self.assertTrue(result.validation["passed"], result.validation)
                self.assertEqual(package.validate_result(result), [])
                self.assertTrue(validate_text_alignment(
                    result, package.render_episode_text(result), package.catalog())["passed"])
                self.assertFalse(result.episode_attributes["empirical_realism_claimed"])

    def test_thousand_nodes_and_source_grounded_transmission(self):
        for scenario in SCENARIOS:
            package, engine = self.engine(scenario, nodes_per_context=1000)
            result = engine.run(0, 42)
            self.assertEqual(len(result.entities), 1000)
            self.assertLess(len(result.context_relations), 10000)
            self.assertEqual({e.entity_type_id for e in result.entities},
                {NS + t for t in ("person", "household", "workplace", "laboratory", "health_unit")})
            self.assertTrue(result.validation["passed"], result.validation)
            events = {e.event_id: e for e in result.events}
            contacts = {r.relation_id: r for r in result.context_relations if r.relation_type_id in CONTACT_TYPES}
            for candidate in result.candidates:
                if not candidate.fired_event_id or not candidate.mechanism_id.endswith("transmission"):
                    continue
                event = events[candidate.fired_event_id]
                self.assertEqual(len(candidate.parent_event_ids), 1)
                parent = events[candidate.parent_event_ids[0]]
                contact = contacts[event.attributes["contact_relation_id"]]
                self.assertEqual(parent.event_type_id, INFECTIOUS)
                self.assertEqual(parent.attributes["person_id"], contact.source_entity_id)
                self.assertEqual(event.attributes["person_id"], contact.target_entity_id)
                self.assertEqual(event.attributes["disease_state_before"], "S")
                self.assertFalse(event.attributes["isolated_before"])
                self.assertFalse(event.attributes["quarantined_before"])

    def test_zero_transmission_keeps_contacts_without_inventing_parents(self):
        package, engine = self.engine(transmission_probability_per_contact=0, symptom_probability=0)
        result = engine.run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertTrue(any(r.relation_type_id in CONTACT_TYPES for r in result.context_relations))
        self.assertEqual([e.event_type_id for e in result.events], [EXPOSED, INFECTIOUS, RECOVERED])
        self.assertEqual(package.validate_result(result), [])

    def test_multiple_sources_compete_but_do_not_jointly_infect(self):
        _, engine = self.engine("workplace_cluster", transmission_probability_per_contact=1,
                               trace_coverage=0, symptom_probability=0)
        saw_competing_target = False
        for seed in range(8):
            result = engine.run(seed, seed + 10)
            exposed_people = [e.attributes["person_id"] for e in result.events if e.event_type_id == EXPOSED]
            self.assertEqual(len(exposed_people), len(set(exposed_people)))
            sources = {}
            for candidate in result.candidates:
                if candidate.mechanism_id.endswith("transmission"):
                    sources.setdefault(candidate.attributes["person_id"], set()).add(candidate.attributes["source_person_id"])
                    if candidate.fired_event_id:
                        self.assertEqual(len(candidate.parent_event_ids), 1)
            saw_competing_target |= any(len(value) > 1 for value in sources.values())
            self.assertTrue(result.validation["passed"], result.validation)
            contacts = {r.relation_id: r for r in result.context_relations if r.relation_type_id in CONTACT_TYPES}
            for event in result.events:
                if event.event_type_id == EXPOSED and "contact_relation_id" in event.attributes:
                    self.assertLessEqual(event.attributes["transmission_hazard_per_day"],
                        contacts[event.attributes["contact_relation_id"]].attributes["effective_contacts_per_day"])
        self.assertTrue(saw_competing_target)

    def test_late_positive_uses_sample_state_not_result_state(self):
        package = PublicHealthPackage({"nodes_per_context": 40,
            "scenario_weights": {"household_cluster": 1}, "duration_seconds": 30 * DAY,
            "transmission_probability_per_contact": 0, "symptom_probability": 1,
            "trace_coverage": 0, "assay_infectious_sensitivity": 1,
            "assay_false_positive_probability": 0})
        models = build_public_health_temporal_models()
        models.replace(ConditionalLogNormalModel(NS + "lab_turnaround.prior_v1",
            math.log(14 * DAY), .01, {}, minimum_seconds=DAY), parameter_status="synthetic_test_fixture")
        result = SimulationEngine(package, models, max_events=2048).run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        positive = next(e for e in result.events if e.event_type_id == POSITIVE)
        self.assertEqual(positive.attributes["disease_state_at_sample"], "I")
        self.assertEqual(positive.attributes["disease_state_before"], "R")
        sample = next(e for e in result.events if e.event_id == positive.attributes["specimen_event_id"])
        self.assertEqual(positive.attributes["assay_uniform_draw"], sample.attributes["assay_uniform_draw"])

    def test_negative_does_not_clear_infection(self):
        _, engine = self.engine(transmission_probability_per_contact=0, symptom_probability=1,
            assay_infectious_sensitivity=0, assay_exposed_sensitivity=0, assay_false_positive_probability=0)
        result = engine.run(0, 42)
        self.assertTrue(result.validation["passed"], result.validation)
        negative = next(e for e in result.events if e.event_type_id == NEGATIVE)
        self.assertEqual(negative.attributes["disease_state_before"], "I")
        self.assertEqual(negative.attributes["disease_state_after"], "I")
        self.assertTrue(any(e.event_type_id == RECOVERED for e in result.events))
        self.assertFalse(any(e.event_type_id == ISOLATED for e in result.events))

    def test_restrictions_and_precautionary_notification(self):
        _, engine = self.engine("workplace_cluster", trace_coverage=1, quarantine_adherence=1,
            symptom_probability=1, assay_infectious_sensitivity=1, transmission_probability_per_contact=.3)
        saw_restriction = saw_cancel = saw_restriction_cancel = False
        for seed in range(6):
            result = engine.run(seed, seed + 42)
            self.assertTrue(result.validation["passed"], result.validation)
            for event in result.events:
                if event.event_type_id == NOTIFIED:
                    self.assertFalse(event.attributes["confirmed_exposure"])
                    self.assertEqual(event.attributes["notification_basis"], "precautionary_cohort_roster")
                saw_restriction |= event.event_type_id in (ISOLATED, QUARANTINED)
            saw_cancel |= any(c.status.value == "cancelled" and c.mechanism_id.endswith("transmission")
                              for c in result.candidates)
            saw_restriction_cancel |= any(c.status.value == "cancelled" and c.mechanism_id.endswith("transmission")
                and any(p.state_path == "contact_restricted" and p.actual is True
                        for p in c.context_evidence.state_predicates) for c in result.candidates)
        self.assertTrue(saw_restriction)
        self.assertTrue(saw_cancel)
        self.assertTrue(saw_restriction_cancel)

    def test_tampered_assay_contact_and_notification_rejected(self):
        package, engine = self.engine("workplace_cluster", trace_coverage=1,
            symptom_probability=1, assay_infectious_sensitivity=1, transmission_probability_per_contact=.3)
        result = engine.run(0, 42)
        for kind, field, value in ((POSITIVE, "disease_state_at_sample", "S"),
                                   (NOTIFIED, "confirmed_exposure", True)):
            broken = copy.deepcopy(result)
            event = next(e for e in broken.events if e.event_type_id == kind)
            event.attributes[field] = value
            self.assertTrue(package.validate_result(broken))
        broken = copy.deepcopy(result)
        contact = next(r for r in broken.context_relations if r.relation_type_id in CONTACT_TYPES)
        contact.attributes["setting_id"] = "unrelated_setting"
        self.assertTrue(package.validate_result(broken))

    def test_short_horizon_and_continuation(self):
        _, engine = self.engine(duration_seconds=.001)
        result = engine.run(0, 77)
        self.assertTrue(result.validation["passed"], result.validation)
        self.assertEqual(result.events, [])
        _, engine = self.engine("workplace_cluster")
        snapshot = engine.run(0, 111, stop_after_events=2)
        left = engine.run(0, 111, snapshot=snapshot, continuation_seed=10)
        right = engine.run(0, 111, snapshot=snapshot, continuation_seed=11)
        self.assertTrue(left.validation["passed"], left.validation)
        self.assertTrue(right.validation["passed"], right.validation)
        self.assertEqual([e.to_dict() for e in left.events[:2]], [e.to_dict() for e in right.events[:2]])

    def test_cli_qa_and_training_export(self):
        output = ROOT / "data_generation" / "artifacts" / "test_runtime_output" / f"public_health_{uuid.uuid4().hex}"
        process = subprocess.run([sys.executable, "data_generation/run_data_generation.py",
            "--domain", "public_health", "--episode-count", "3", "--nodes-per-context", "40",
            "--judge-mode", "none", "--output-dir", str(output)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"], report)
        self.assertGreater((output / "qa_pairs.jsonl").stat().st_size, 0)
        self.assertGreater((output / "training" / "qa_instruction.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
