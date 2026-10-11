"""Offline plumbing tests; fake responses are not LLM quality measurements."""
import copy
import json
import unittest
from pathlib import Path

from data_generation.domain_packages.registry import DOMAIN_BUILDERS
from data_generation.evaluation.domain_acceptance import (
    COMMON_FALSE_CLAIM, DOMAIN_FALSE_CLAIMS, STRUCTURAL_KINDS, build_report,
    deterministic_review, evaluate_episode, inject_claim, semantic_challenge,
    structural_mutation, summarize_domain, truth_hash,
)
from data_generation.generation_core.claim_contract import build_claim_contract
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.semantic_correction import (
    LLMSemanticCorrector, _build_correction_prompt, _required_literals_by_sentence, _validate_correction_object,
)
from data_generation.generation_core.semantic_judge import LLMSemanticJudge, SemanticJudgeResult, _build_prompt
from data_generation.generation_core.validation import validate_episode_result


def make_episode(domain="transportation", seed=20261010):
    folder = Path(__file__).resolve().parents[1] / "domain_packages" / domain
    config = json.loads((folder / "config/default.json").read_text(encoding="utf-8"))
    settings = {**config["domain_config"], "nodes_per_context": 60, "base_seed": seed}
    package_type, models = DOMAIN_BUILDERS[domain]
    package = package_type(settings)
    result = SimulationEngine(package, models(), max_events=2048, time_axis=config.get("time_axis")).run(0, seed)
    return package, result


class FakeReviewClient:
    """Deterministic exact-string oracle for exercising transport protocols only."""
    model = "offline-test-double"
    base_url = "not-an-api"

    def __init__(self, *, protocol_error=False, miss=False, corrupt_truth=None):
        self.protocol_error, self.miss, self.corrupt_truth = protocol_error, miss, corrupt_truth
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        if self.protocol_error:
            return "invalid-json"
        data = json.loads(prompt.split("\nDATA:\n", 1)[1].split("\nPROTOCOL RETRY:", 1)[0])
        if "correction_contract" in data:
            updates = []
            for item in data["sentences"]:
                sentence = item["current_sentence"]
                for false_claim in (*DOMAIN_FALSE_CLAIMS.values(), COMMON_FALSE_CLAIM):
                    sentence = sentence.replace(" " + false_claim, "")
                if sentence != item["current_sentence"]:
                    updates.append({"sentence_index": item["sentence_index"], "sentence": sentence})
            if self.corrupt_truth:
                self.corrupt_truth.events[0].attributes["unauthorized_change"] = True
            return json.dumps({"sentence_updates": updates})
        rejected = False
        ids = []
        for claim in data["grounded_text"]["claims"]:
            sentence = data["grounded_text"]["sentences"][claim["sentence_index"]]
            if any(c in sentence for c in (*DOMAIN_FALSE_CLAIMS.values(), COMMON_FALSE_CLAIM)) and not self.miss:
                rejected, ids = True, claim.get("event_ids", [])
                break
        return json.dumps({"passed": not rejected, "issue_scope": "text" if rejected else "none",
            "reasons": ["Injected claim violates the declaration." if rejected else "Test double pass."],
            "flagged_record_ids": ids})


class ClaimContractTests(unittest.TestCase):
    def test_generic_and_legacy_scopes_are_losslessly_combined(self):
        catalog = {"domain": "test", "scope": {"model": {"measurement": "sampled"}, "empirical": False},
            "clinical_safety_scope": {"diagnostic_claims": False}}
        contract = build_claim_contract(catalog)
        self.assertFalse(contract["declared_scope"]["empirical"])
        self.assertFalse(contract["declared_scope"]["diagnostic_claims"])
        contract["declared_scope"]["model"]["measurement"] = "mutated"
        self.assertEqual(catalog["scope"]["model"]["measurement"], "sampled")
        with self.assertRaises(ValueError):
            build_claim_contract({"scope": []})
        with self.assertRaises(ValueError):
            build_claim_contract({"scope": {"a": False}, "clinical_safety_scope": {"a": True}})

    def test_all_domains_send_the_same_contract_to_judge_and_corrector(self):
        package, result = make_episode()
        text = package.render_episode_text(result)
        feedback = SemanticJudgeResult("semantic_rejected", False, "text", ["claim"], [result.events[0].event_id])
        for domain, (factory, _) in DOMAIN_BUILDERS.items():
            with self.subTest(domain=domain):
                catalog = factory({"nodes_per_context": 60}).catalog()
                contract = build_claim_contract(catalog)
                self.assertTrue(contract["declared_scope"])
                judged = json.loads(_build_prompt(result, text, catalog, reason_limit=5).split("\nDATA:\n")[1])
                corrected = json.loads(_build_correction_prompt(result, text, catalog, feedback, {}).split("\nDATA:\n")[1])
                self.assertEqual(judged["review_contract"]["domain_claim_scope"], contract)
                self.assertEqual(corrected["domain_claim_scope"], contract)


class DomainAcceptanceTests(unittest.TestCase):
    def test_fact_literal_guards_match_every_domain_projection_and_block_deletion(self):
        for domain in DOMAIN_BUILDERS:
            with self.subTest(domain=domain):
                package, result = make_episode(domain)
                text = package.render_episode_text(result)
                required = _required_literals_by_sentence(result, text)
                self.assertTrue(required)
                for index, literals in required.items():
                    self.assertTrue(all(lit in text["sentences"][index] for lit in literals))
                index = next(iter(required))
                literal = max(required[index], key=len)
                update = text["sentences"][index].replace(literal, "REMOVED") + " Clarification."
                updates, error = _validate_correction_object(
                    {"sentence_updates": [{"sentence_index": index, "sentence": update}]}, text, required)
                self.assertIsNone(updates)
                self.assertIn("removed required literals", error)

    def test_numeric_literal_prefix_cannot_replace_a_protected_value(self):
        updates, error = _validate_correction_object(
            {"sentence_updates": [{"sentence_index": 0, "sentence": "At +1.0009 seconds. Updated."}]},
            {"sentences": ["At +1.000 seconds."]}, {0: ["+1.000"]})
        self.assertIsNone(updates)
        self.assertIn("removed required literals", error)

    def test_all_domains_baseline_and_structural_mutations(self):
        for domain in DOMAIN_BUILDERS:
            with self.subTest(domain=domain):
                package, result = make_episode(domain)
                baseline_hash = truth_hash(result)
                record = evaluate_episode(package, result)
                self.assertTrue(record["baseline_validation"]["passed"], record["baseline_validation"])
                self.assertTrue(record["complete_observation_window"])
                self.assertEqual(len(record["structural_challenges"]), len(STRUCTURAL_KINDS))
                self.assertTrue(all(c["status"] == "rejected" for c in record["structural_challenges"]), record)
                report = summarize_domain(domain, [record], requested_count=1, semantic_mode="none")
                self.assertTrue(report["deterministic_passed"], report)
                self.assertIsNone(report["passed"])
                self.assertIsNone(report["semantic_passed"])
                self.assertEqual(truth_hash(result), baseline_hash)

    def test_mock_semantic_detection_recovery_and_invariants(self):
        for domain in DOMAIN_BUILDERS:
            with self.subTest(domain=domain):
                package, result = make_episode(domain)
                client = FakeReviewClient()
                record = evaluate_episode(package, result, judge=LLMSemanticJudge(client),
                    corrector=LLMSemanticCorrector(client))
                report = summarize_domain(domain, [record], requested_count=1, semantic_mode="llm")
                self.assertTrue(report["passed"], [(c["kind"], c["status"], c.get("correction")) for c in record["semantic_challenges"]])
                self.assertEqual(client.calls, 7)
                for case in record["semantic_challenges"]:
                    self.assertTrue(case["checks"]["qa_labels_preserved"])
                    self.assertEqual(case["truth_sha256_before"], case["truth_sha256_after"])

    def test_protocol_failures_and_misses_never_count_as_success(self):
        package, result = make_episode()
        for client in (FakeReviewClient(protocol_error=True), FakeReviewClient(miss=True)):
            record = evaluate_episode(package, result, judge=LLMSemanticJudge(client, protocol_max_attempts=2),
                corrector=LLMSemanticCorrector(client))
            report = summarize_domain(package.domain_id, [record], requested_count=1, semantic_mode="llm")
            self.assertFalse(report["passed"])
            self.assertEqual(report["semantic_detection_count"], 0)
            if client.protocol_error:
                self.assertEqual(report["protocol_failure_count"], 1)
                self.assertEqual(report["protocol_retry_count"], 1)
                self.assertEqual(report["semantic_challenge_count"], 0)
            else:
                self.assertEqual(report["semantic_challenge_count"], 2)

    def test_baseline_rejection_is_retained_without_screening(self):
        package, result = make_episode()
        bad, _ = structural_mutation(result, "event_time")
        record = evaluate_episode(package, bad)
        self.assertFalse(record["baseline_validation"]["passed"])
        self.assertEqual(record["structural_challenges"], [])
        report = summarize_domain(package.domain_id, [record], requested_count=1, semantic_mode="none")
        self.assertFalse(report["deterministic_passed"])

    def test_wrong_unit_scale_and_missing_candidate_parent_are_rejected_without_crash(self):
        package, result = make_episode()
        mutated, _ = structural_mutation(result, "time_unit")
        self.assertFalse(validate_episode_result(mutated, package.catalog())["checks"]["time_axis_contract"])
        mutated = copy.deepcopy(result)
        candidate = next(c for c in mutated.candidates if c.parent_event_ids)
        candidate.parent_event_ids = ["__missing_parent_event__"]
        self.assertFalse(validate_episode_result(mutated, package.catalog())["passed"])

    def test_illegal_structural_change_during_correction_fails_preservation_gate(self):
        package, result = make_episode()
        text = package.render_episode_text(result)
        client = FakeReviewClient(corrupt_truth=result)
        case = semantic_challenge(package, result, text, "domain_scope", DOMAIN_FALSE_CLAIMS[package.domain_id],
            LLMSemanticJudge(client), LLMSemanticCorrector(client))
        self.assertFalse(case["checks"]["structured_truth_preserved"])
        self.assertEqual(case["status"], "failed")

    def test_empty_domain_report_and_missing_coverage_cannot_pass(self):
        self.assertFalse(build_report([], semantic_mode="llm")["passed"])
        report = summarize_domain("test", [], requested_count=1, semantic_mode="none")
        self.assertFalse(report["deterministic_passed"])
        self.assertIsNone(report["semantic_detection_rate"])


if __name__ == "__main__":
    unittest.main()
