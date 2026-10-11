"""Evaluation-only cross-domain baselines, mutations and semantic recovery.

No screening/regeneration hides baseline rejections. Protocol failures are not
semantic detections. Controlled challenge artifacts must never enter training.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, is_dataclass, replace

from data_generation.generation_core.models import EventParticipant
from data_generation.generation_core.qa_export import generate_episode_qa
from data_generation.generation_core.validation import validate_episode_result, validate_text_alignment


DOMAIN_FALSE_CLAIMS = {
    "transportation": "This synthetic record proves the measured congestion frequency of a real city and establishes weather influence as proven causality.",
    "healthcare": "This record by itself establishes a definitive diagnosis and proves that the prior intervention was clinically effective.",
    "distributed_systems": "This workflow guarantees exactly-once server execution, and every client timeout necessarily cancels the server operation.",
    "energy": "This episode is a validated AC power-flow solution and proves coordinated protection and successful alternate supply through tie switches.",
    "environment_pollution": "This synthetic dissolved constituent threshold proves a violation of an actual legal drinking-water standard.",
    "ecology": "These threshold events provide a numerically solved population ODE and establish measured real-world population sizes.",
    "public_health": "Every precautionary contact-tracing roster entry proves an actual transmission, and this episode establishes severe clinical outcomes.",
    "hydrology": "This synthetic discharge threshold conclusively proves observed flood inundation at the participant location.",
    "agriculture": "The thermal-time proxy conclusively proves that actual flowering and physiological maturity were observed in this crop.",
    "climate": "This one-box snapshot is a measured annual-average regional temperature and proves real-world climate attribution from a validated GCM.",
}
COMMON_FALSE_CLAIM = "Every context-adjacent entity automatically causes this event, whether or not an event-parent mechanism is declared."
STRUCTURAL_KINDS = ("event_time", "time_unit", "node_binding", "node_type", "parent_link", "relation_lag")


def deterministic_review(package, result, text=None):
    """Always recompute validators; a stored successful validation is not evidence."""
    catalog = package.catalog()
    review = validate_episode_result(result, catalog)
    validator = getattr(package, "validate_result", None)
    if validator:
        errors = validator(result)
        review["errors"].extend(errors)
        review["checks"]["domain_contracts"] = not errors
    review["passed"] = not review["errors"]
    if text is not None:
        aligned = validate_text_alignment(result, text, catalog)
        review["checks"]["text_alignment"] = aligned["passed"]
        review["errors"].extend(aligned["errors"])
        review["passed"] = not review["errors"]
    return review


def structural_mutation(result, kind):
    """Each copy contains one deliberate corruption; no original is changed."""
    if kind not in STRUCTURAL_KINDS:
        raise ValueError(f"Unknown structural challenge: {kind}")
    if kind in {"event_time", "node_binding", "node_type"} and not result.events:
        return None, "requires at least one realized event"
    if kind in {"parent_link", "relation_lag"} and not result.event_relations:
        return None, "requires at least one realized event relation"
    changed = copy.deepcopy(result)
    if kind == "event_time":
        changed.events[0].temporal.occurrence_start_offset_seconds = -1.0
    elif kind == "time_unit":
        if not changed.time_axis:
            return None, "requires an explicit time-axis descriptor"
        changed.time_axis["seconds_per_unit"] *= 1000.0
    elif kind == "node_binding":
        part = changed.events[0].participants[0]
        changed.events[0].participants[0] = EventParticipant("__missing_entity__", part.role)
    elif kind == "node_type":
        entity_id = changed.events[0].participants[0].entity_id
        index = next(i for i, e in enumerate(changed.entities) if e.entity_id == entity_id)
        changed.entities[index] = replace(changed.entities[index], entity_type_id="__unregistered_node_type__")
    elif kind == "parent_link":
        changed.event_relations[0].source_event_id = "__missing_parent_event__"
    elif kind == "relation_lag":
        link = changed.event_relations[0].temporal_link
        changed.event_relations[0].temporal_link = replace(link, lag_seconds=link.lag_seconds + 1.0)
    return changed, None


def truth_hash(result):
    """Freeze the whole episode, including final state, not just event timestamps."""
    payload = {
        "episode": result.episode_record(), "context_attributes": result.context_attributes,
        "final_state": result.final_state,
        **{key: [item.to_dict() for item in getattr(result, key)] for key in (
            "entities", "context_relations", "events", "event_relations", "candidates", "risk_sets")},
    }
    return _hash(payload)


def _hash(payload):
    return hashlib.sha256(json.dumps(_canonical(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _canonical(value):
    """Lossless hash encoding for internal state with tuple keys and sets.

    This is not a training serialization. Explicit tags avoid coercing a tuple
    key to a colliding string and keep internal infinite clock sentinels stable.
    """
    if is_dataclass(value):
        return {"$record_type": type(value).__qualname__, "$record": _canonical(asdict(value))}
    if isinstance(value, dict):
        pairs = [[_canonical(k), _canonical(v)] for k, v in value.items()]
        return {"$mapping": sorted(pairs, key=lambda p: json.dumps(p[0], sort_keys=True))}
    if isinstance(value, tuple):
        return {"$tuple": [_canonical(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"$set": sorted((_canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"$nonfinite_float": str(value)}
    return value


def qa_label_hash(result, text):
    return _hash([{key: qa.get(key) for key in ("qa_id", "qa_type", "answer", "answer_source")}
        for qa in generate_episode_qa(result, text)])


def inject_claim(text, claim_text):
    changed = copy.deepcopy(text)
    claims = changed.get("claims", [])
    selected = next((c for c in claims if c.get("event_ids")
        and isinstance(c.get("sentence_index"), int)
        and 0 <= c["sentence_index"] < len(changed.get("sentences", []))), None)
    if selected is None:
        return None, None
    index = selected["sentence_index"]
    original = changed["sentences"][index]
    changed["sentences"][index] += " " + claim_text
    changed["text"] = " ".join(changed["sentences"])
    changed["projection_mode"] = "controlled_evaluation_text_challenge"
    return changed, {"sentence_index": index, "grounded_record_ids": selected["event_ids"],
        "injected_claim": claim_text, "original_sentence": original,
        "challenged_sentence": changed["sentences"][index]}


def semantic_challenge(package, result, text, kind, claim_text, judge, corrector):
    challenged, injection = inject_claim(text, claim_text)
    if challenged is None:
        return {"kind": kind, "status": "not_applicable", "reason": "no event-grounded sentence"}
    before = truth_hash(result)
    labels = qa_label_hash(result, text)
    alignment = validate_text_alignment(result, challenged, package.catalog())
    judged = judge.evaluate(result, challenged, package.catalog())
    # Require localization to the injected sentence, not just an unrelated rejection.
    localized = (judged.outcome == "semantic_rejected" and judged.issue_scope == "text"
        and bool(judged.flagged_record_ids)
        and set(judged.flagged_record_ids) <= set(injection["grounded_record_ids"]))
    corrected, post, correction = challenged, None, None
    if localized:
        correction = corrector.correct(result, challenged, package.catalog(), judged, correction_round=1)
        if correction.succeeded:
            corrected = correction.text_alignment
            post = judge.evaluate(result, corrected, package.catalog())
    recovered = bool(correction and correction.succeeded and post and post.passed)
    checks = {
        "injection_preserved_deterministic_alignment": alignment["passed"],
        "detected_and_localized": localized,
        "correction_succeeded": bool(correction and correction.succeeded),
        "post_correction_passed": bool(post and post.passed),
        "injected_literal_removed": claim_text not in corrected["text"],
        "corrected_alignment_passed": validate_text_alignment(result, corrected, package.catalog())["passed"],
        "structured_truth_preserved": before == truth_hash(result),
        "qa_labels_preserved": labels == qa_label_hash(result, corrected),
    }
    return {"kind": kind, "status": "passed" if recovered and all(checks.values()) else "failed",
        "injection": injection, "checks": checks, "judge": judged.to_dict(),
        "correction": correction.audit_dict() if correction else None,
        "post_correction_judge": post.to_dict() if post else None,
        "corrected_sentence": corrected["sentences"][injection["sentence_index"]],
        "truth_sha256_before": before, "truth_sha256_after": truth_hash(result)}


def evaluate_episode(package, result, *, judge=None, corrector=None):
    if (judge is None) != (corrector is None):
        raise ValueError("Semantic evaluation requires both Judge and corrector")
    text = package.render_episode_text(result)
    baseline = deterministic_review(package, result, text)
    # Natural exhaustion is a complete realization, not administrative truncation.
    complete_window = result.termination_reason in {"observation_window_ended", "candidate_set_exhausted"}
    record = {"domain": result.domain, "episode_id": result.episode_id,
        "entity_count": len(result.entities), "time_axis": copy.deepcopy(result.time_axis),
        "event_count": len(result.events), "event_relation_count": len(result.event_relations),
        "termination_reason": result.termination_reason, "baseline_validation": baseline,
        "complete_observation_window": complete_window,
        "structural_challenges": [], "semantic_challenges": [],
        "baseline_judge": None, "semantic_status": "not_run" if judge is None else "not_eligible"}
    if not baseline["passed"] or not complete_window:
        return record
    original_hash = truth_hash(result)
    for kind in STRUCTURAL_KINDS:
        mutated, reason = structural_mutation(result, kind)
        if mutated is None:
            record["structural_challenges"].append({"kind": kind, "status": "not_applicable", "reason": reason})
            continue
        # Domain replay need not consume malformed common records. The common
        # validator is already sufficient to reject a missing parent or entity.
        review = validate_episode_result(mutated, package.catalog())
        if review["passed"]:
            review = deterministic_review(package, mutated)
        record["structural_challenges"].append({"kind": kind,
            "status": "rejected" if not review["passed"] else "missed", "validation": review})
    record["original_truth_preserved_after_mutations"] = original_hash == truth_hash(result)
    if judge:
        baseline_judge = judge.evaluate(result, text, package.catalog())
        record["baseline_judge"] = baseline_judge.to_dict()
        record["semantic_status"] = "baseline_rejected" if not baseline_judge.passed else "evaluated"
        # No replacement episode conceals a rejection or protocol failure.
        if baseline_judge.passed:
            for kind, claim in (("domain_scope", DOMAIN_FALSE_CLAIMS[result.domain]),
                                ("adjacency_as_parenthood", COMMON_FALSE_CLAIM)):
                record["semantic_challenges"].append(semantic_challenge(
                    package, result, text, kind, claim, judge, corrector))
    return record


def summarize_domain(domain, records, *, requested_count, semantic_mode):
    structural = [c for r in records for c in r["structural_challenges"]]
    semantic = [c for r in records for c in r["semantic_challenges"]]
    coverage = {kind: sum(c["kind"] == kind and c["status"] != "not_applicable" for c in structural)
        for kind in STRUCTURAL_KINDS}
    baseline_pass = sum(r["baseline_validation"]["passed"] and r["complete_observation_window"] for r in records)
    deterministic_passed = (len(records) == requested_count and baseline_pass == requested_count
        and all(coverage.values()) and not any(c["status"] == "missed" for c in structural)
        and all(r.get("original_truth_preserved_after_mutations", False) for r in records))
    protocols = []
    for r in records:
        if r["baseline_judge"]:
            protocols.append(r["baseline_judge"])
        for c in r["semantic_challenges"]:
            protocols.extend(item for item in (c.get("judge"), c.get("correction"),
                c.get("post_correction_judge")) if item)
    semantic_passed = None if semantic_mode == "none" else (
        len(semantic) == requested_count * 2 and all(c["status"] == "passed" for c in semantic)
        and all(r["baseline_judge"] and r["baseline_judge"]["passed"] for r in records))
    detected = sum(c.get("checks", {}).get("detected_and_localized", False) for c in semantic)
    recovered = sum(c["status"] == "passed" for c in semantic)
    return {"domain": domain, "requested_episode_count": requested_count,
        "evaluated_episode_count": len(records), "baseline_deterministic_pass_count": baseline_pass,
        "deterministic_passed": deterministic_passed,
        "structural_coverage_by_kind": coverage,
        "structural_rejection_count": sum(c["status"] == "rejected" for c in structural),
        "structural_miss_count": sum(c["status"] == "missed" for c in structural),
        "structural_not_applicable_count": sum(c["status"] == "not_applicable" for c in structural),
        "semantic_status": "not_run" if semantic_mode == "none" else "evaluated",
        "semantic_passed": semantic_passed,
        "baseline_judge_pass_count": sum(bool(r["baseline_judge"] and r["baseline_judge"]["passed"]) for r in records),
        "semantic_challenge_count": len(semantic), "semantic_detection_count": detected,
        "expected_semantic_challenge_count": requested_count * 2 if semantic_mode == "llm" else 0,
        "semantic_recovery_count": recovered,
        "semantic_detection_rate": detected / len(semantic) if semantic else None,
        "semantic_recovery_rate": recovered / len(semantic) if semantic else None,
        "end_to_end_semantic_recovery_rate": recovered / (requested_count * 2) if semantic_mode == "llm" and requested_count else None,
        "llm_completion_call_count": sum(p.get("protocol_attempt_count", 1) for p in protocols),
        "protocol_failure_count": sum(p["outcome"] == "protocol_error" for p in protocols),
        "protocol_retry_count": sum(max(0, p.get("protocol_attempt_count", 1) - 1) for p in protocols),
        "passed": (deterministic_passed and semantic_passed) if semantic_mode == "llm" else None}


def build_report(domain_reports, *, semantic_mode):
    return {"schema": "cross-domain-acceptance-v1", "evaluation_only": True,
        "must_not_be_used_for_training": True,
        "deterministic_passed": bool(domain_reports) and all(r["deterministic_passed"] for r in domain_reports),
        "semantic_status": "not_run" if semantic_mode == "none" else "evaluated",
        "passed": (bool(domain_reports) and all(r["passed"] for r in domain_reports)) if semantic_mode == "llm" else None,
        "domains": domain_reports,
        "acceptance_policy": "Strict smoke acceptance: all baselines and executed challenges pass; every structural kind has coverage per domain. Unexecuted semantics never counts as passed.",
        "limitations": ["Controlled corruption tests are not exhaustive domain validation.",
            "Same-model post-review is not an independent semantic oracle; paraphrased false claims may survive and require manual or independent review.",
            "Small-sample rates do not establish population performance or statistical realism.",
            "No baseline screening or regeneration; protocol errors are reported separately from semantic errors."]}
