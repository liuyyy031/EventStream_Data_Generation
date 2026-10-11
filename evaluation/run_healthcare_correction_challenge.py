"""Run an evaluation-only healthcare semantic-correction challenge.

This command deliberately appends an unsupported diagnostic and treatment-
effect claim to otherwise valid grounded text. The structured episode is never
changed. Resulting files are evaluation artifacts and must not be training data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_generation.domain_packages.healthcare import (  # noqa: E402
    HealthcarePackage,
    build_healthcare_temporal_models,
)
from data_generation.generation_core.scheduler import SimulationEngine  # noqa: E402
from data_generation.generation_core.semantic_correction import (  # noqa: E402
    LLMSemanticCorrector,
)
from data_generation.generation_core.semantic_judge import (  # noqa: E402
    LLMSemanticJudge,
)
from data_generation.generation_core.validation import (  # noqa: E402
    validate_text_alignment,
)


INJECTED_CLAIM = (
    " This record by itself establishes a definitive diagnosis and proves that "
    "the prior intervention was clinically effective."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether the LLM Judge detects an injected healthcare text "
            "claim and whether correction removes it without changing structure."
        )
    )
    parser.add_argument("--challenge-count", type=int, default=5)
    parser.add_argument("--nodes-per-context", type=int, default=1000)
    parser.add_argument("--max-events-per-episode", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-max-tokens", type=int, default=1200)
    parser.add_argument("--correction-max-tokens", type=int, default=1800)
    parser.add_argument("--protocol-max-attempts", type=int, default=3)
    parser.add_argument(
        "--max-screening-attempts",
        type=int,
        help="Maximum generated episodes screened to collect baseline passes.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_generation/artifacts/healthcare_correction_challenge",
    )
    args = parser.parse_args()

    if args.challenge_count <= 0:
        parser.error("--challenge-count must be positive")
    if args.nodes_per_context < 12:
        parser.error("--nodes-per-context must be at least 12")
    if args.max_events_per_episode <= 0:
        parser.error("--max-events-per-episode must be positive")
    max_screening_attempts = args.max_screening_attempts or args.challenge_count * 3
    if max_screening_attempts < args.challenge_count:
        parser.error("--max-screening-attempts cannot be smaller than --challenge-count")

    output_dir = _unique_output_directory(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    from data_generation.llm_client import LLMClient

    client = LLMClient(base_url=args.judge_base_url, model=args.judge_model)
    judge = LLMSemanticJudge(
        client,
        max_tokens=args.judge_max_tokens,
        protocol_max_attempts=args.protocol_max_attempts,
    )
    corrector = LLMSemanticCorrector(
        client,
        max_tokens=args.correction_max_tokens,
        protocol_max_attempts=args.protocol_max_attempts,
    )
    package = HealthcarePackage(
        {
            "base_seed": args.seed,
            "nodes_per_context": args.nodes_per_context,
        }
    )
    engine = SimulationEngine(
        package,
        build_healthcare_temporal_models(),
        max_events=args.max_events_per_episode,
    )
    catalog = package.catalog()

    cases = []
    screening_records = []
    attempt_index = 0
    while len(cases) < args.challenge_count and attempt_index < max_screening_attempts:
        episode_seed = args.seed + attempt_index * 104729
        result = engine.run(attempt_index, episode_seed)
        original_text = package.render_episode_text(result)
        original_alignment = validate_text_alignment(result, original_text, catalog)
        if not original_alignment["passed"]:
            raise RuntimeError(
                "Generated baseline text failed deterministic alignment: "
                + "; ".join(original_alignment["errors"])
            )

        baseline_judge = judge.evaluate(result, original_text, catalog)
        screening_records.append(
            {
                "episode_id": result.episode_id,
                "episode_seed": episode_seed,
                "baseline_judge": baseline_judge.to_dict(),
            }
        )
        attempt_index += 1
        if not baseline_judge.passed:
            continue

        challenged_text, injection = inject_unsupported_clinical_claim(original_text)
        challenged_alignment = validate_text_alignment(result, challenged_text, catalog)
        if not challenged_alignment["passed"]:
            raise RuntimeError(
                "The controlled text challenge broke deterministic alignment: "
                + "; ".join(challenged_alignment["errors"])
            )

        structured_hash_before = structured_truth_sha256(result)
        challenge_judge = judge.evaluate(result, challenged_text, catalog)
        grounded_ids = grounded_record_ids(challenged_text)
        addressable = (
            challenge_judge.outcome == "semantic_rejected"
            and challenge_judge.issue_scope == "text"
            and bool(challenge_judge.flagged_record_ids)
            and set(challenge_judge.flagged_record_ids) <= grounded_ids
        )

        correction = None
        corrected_text = challenged_text
        corrected_alignment = {"passed": False, "errors": ["Correction not run"]}
        post_correction_judge = None
        if addressable:
            correction = corrector.correct(
                result,
                challenged_text,
                catalog,
                challenge_judge,
                correction_round=1,
            )
            if correction.succeeded:
                corrected_text = correction.text_alignment
                corrected_alignment = validate_text_alignment(
                    result, corrected_text, catalog
                )
                if corrected_alignment["passed"]:
                    post_correction_judge = judge.evaluate(
                        result, corrected_text, catalog
                    )

        structured_hash_after = structured_truth_sha256(result)
        injected_claim_removed = INJECTED_CLAIM.strip() not in str(
            corrected_text.get("text", "")
        )
        cases.append(
            {
                "case_id": f"healthcare_correction_case_{len(cases) + 1:04d}",
                "episode_id": result.episode_id,
                "episode_seed": episode_seed,
                "evaluation_only": True,
                "must_not_be_used_for_training": True,
                "injection": injection,
                "baseline_judge": baseline_judge.to_dict(),
                "challenge_judge": challenge_judge.to_dict(),
                "correction": correction.audit_dict() if correction else None,
                "post_correction_judge": (
                    post_correction_judge.to_dict() if post_correction_judge else None
                ),
                "checks": {
                    "baseline_passed": baseline_judge.passed,
                    "challenge_preserved_deterministic_alignment": challenged_alignment[
                        "passed"
                    ],
                    "challenge_detected_as_addressable_text_issue": addressable,
                    "correction_succeeded": bool(correction and correction.succeeded),
                    "corrected_text_passed_deterministic_alignment": corrected_alignment[
                        "passed"
                    ],
                    "post_correction_judge_passed": bool(
                        post_correction_judge and post_correction_judge.passed
                    ),
                    "injected_claim_removed": injected_claim_removed,
                    "structured_truth_sha256_preserved": (
                        structured_hash_before == structured_hash_after
                    ),
                },
                "hashes": {
                    "structured_truth_before": structured_hash_before,
                    "structured_truth_after": structured_hash_after,
                    "original_text": text_sha256(original_text),
                    "challenged_text": text_sha256(challenged_text),
                    "corrected_text": text_sha256(corrected_text),
                },
                "text_evidence": {
                    "original_sentence": injection["original_sentence"],
                    "challenged_sentence": injection["challenged_sentence"],
                    "corrected_sentence": corrected_text["sentences"][
                        injection["sentence_index"]
                    ],
                },
            }
        )

    report = build_report(
        cases,
        requested_count=args.challenge_count,
        screening_records=screening_records,
        output_dir=output_dir,
        model=args.judge_model,
        base_url=client.base_url,
        nodes_per_context=args.nodes_per_context,
        seed=args.seed,
    )
    write_jsonl(output_dir / "screening_judgments.jsonl", screening_records)
    write_jsonl(output_dir / "correction_challenge_cases.jsonl", cases)
    write_json(output_dir / "correction_challenge_report.json", report)
    write_json(
        output_dir / "manifest.json",
        {
            "evaluation_type": "healthcare_semantic_correction_controlled_challenge",
            "evaluation_only": True,
            "must_not_be_used_for_training": True,
            "created_at": datetime.now().astimezone().isoformat(),
            "injected_claim": INJECTED_CLAIM.strip(),
            "configuration": {
                "challenge_count": args.challenge_count,
                "max_screening_attempts": max_screening_attempts,
                "nodes_per_context": args.nodes_per_context,
                "max_events_per_episode": args.max_events_per_episode,
                "seed": args.seed,
                "judge_model": args.judge_model,
                "judge_base_url": client.base_url,
                "judge_max_tokens": args.judge_max_tokens,
                "correction_max_tokens": args.correction_max_tokens,
                "protocol_max_attempts": args.protocol_max_attempts,
            },
        },
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir.resolve()), "report": report},
            ensure_ascii=False,
            indent=2,
        )
    )


def inject_unsupported_clinical_claim(
    text_alignment: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Append a deterministic, unsupported claim to one event-grounded sentence."""

    challenged = copy.deepcopy(text_alignment)
    sentences = list(challenged.get("sentences", []))
    selected_claim = next(
        (
            claim
            for claim in challenged.get("claims", [])
            if claim.get("event_ids")
            and isinstance(claim.get("sentence_index"), int)
            and 0 <= claim["sentence_index"] < len(sentences)
        ),
        None,
    )
    if selected_claim is None:
        raise ValueError("The episode has no event-grounded sentence to challenge")
    sentence_index = selected_claim["sentence_index"]
    original_sentence = sentences[sentence_index]
    challenged_sentence = original_sentence + INJECTED_CLAIM
    sentences[sentence_index] = challenged_sentence
    challenged["sentences"] = sentences
    challenged["text"] = " ".join(sentences)
    challenged["projection_mode"] = "controlled_evaluation_text_challenge"
    return challenged, {
        "challenge_type": "unsupported_diagnosis_and_treatment_effect_claim",
        "sentence_index": sentence_index,
        "grounded_record_ids": list(selected_claim.get("event_ids", [])),
        "original_sentence": original_sentence,
        "challenged_sentence": challenged_sentence,
        "injected_claim": INJECTED_CLAIM.strip(),
    }


def grounded_record_ids(text_alignment: Dict[str, Any]) -> set[str]:
    output: set[str] = set()
    for claim in text_alignment.get("claims", []):
        output.update(str(item) for item in claim.get("event_ids", []))
        output.update(str(item) for item in claim.get("relation_ids", []))
    return output


def structured_truth_sha256(result: Any) -> str:
    payload = {
        "episode": result.episode_record(),
        "entities": [item.to_dict() for item in result.entities],
        "context_relations": [item.to_dict() for item in result.context_relations],
        "events": [item.to_dict() for item in result.events],
        "event_relations": [item.to_dict() for item in result.event_relations],
        "candidates": [item.to_dict() for item in result.candidates],
        "risk_sets": [item.to_dict() for item in result.risk_sets],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(text_alignment: Dict[str, Any]) -> str:
    return hashlib.sha256(
        str(text_alignment.get("text", "")).encode("utf-8")
    ).hexdigest()


def build_report(
    cases: list[Dict[str, Any]],
    *,
    requested_count: int,
    screening_records: list[Dict[str, Any]],
    output_dir: Path,
    model: str,
    base_url: str,
    nodes_per_context: int,
    seed: int,
) -> Dict[str, Any]:
    def count(check: str) -> int:
        return sum(bool(case["checks"].get(check)) for case in cases)

    case_count = len(cases)
    detected = count("challenge_detected_as_addressable_text_issue")
    corrected = count("correction_succeeded")
    recovered = count("post_correction_judge_passed")
    preserved = count("structured_truth_sha256_preserved")
    aligned = count("corrected_text_passed_deterministic_alignment")
    removed = count("injected_claim_removed")

    def rate(value: int, denominator: int) -> float | None:
        return round(value / denominator, 6) if denominator else None

    detection_rate = rate(detected, case_count)
    correction_rate = rate(corrected, detected)
    recovery_rate = rate(recovered, detected)
    passed = bool(
        case_count == requested_count
        and detection_rate is not None
        and detection_rate >= 0.8
        and correction_rate is not None
        and correction_rate >= 0.8
        and recovery_rate is not None
        and recovery_rate >= 0.8
        and preserved == case_count
        and aligned == corrected
        and removed == corrected
    )
    return {
        "passed": passed,
        "evaluation_only": True,
        "must_not_be_used_for_training": True,
        "requested_challenge_count": requested_count,
        "completed_challenge_count": case_count,
        "screened_episode_count": len(screening_records),
        "baseline_pass_count": sum(
            bool(item["baseline_judge"].get("passed"))
            for item in screening_records
        ),
        "challenge_detection_count": detected,
        "challenge_detection_rate": detection_rate,
        "correction_success_count": corrected,
        "correction_success_rate_given_detection": correction_rate,
        "post_correction_pass_count": recovered,
        "post_correction_pass_rate_given_detection": recovery_rate,
        "structured_truth_preservation_count": preserved,
        "corrected_alignment_pass_count": aligned,
        "injected_claim_removed_count": removed,
        "thresholds": {
            "challenge_detection_rate": 0.8,
            "correction_success_rate_given_detection": 0.8,
            "post_correction_pass_rate_given_detection": 0.8,
            "structured_truth_preservation_rate": 1.0,
        },
        "configuration": {
            "domain": "healthcare",
            "nodes_per_context": nodes_per_context,
            "seed": seed,
            "model": model,
            "base_url": base_url,
            "output_dir": str(output_dir.resolve()),
        },
        "interpretation": (
            "This controlled test measures detection and text-only recovery for one "
            "known unsupported clinical claim. It does not establish clinical or "
            "statistical realism and does not replace evaluation on naturally "
            "occurring model errors."
        ),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _unique_output_directory(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return path.with_name(f"{path.name}_{timestamp}")


if __name__ == "__main__":
    main()
