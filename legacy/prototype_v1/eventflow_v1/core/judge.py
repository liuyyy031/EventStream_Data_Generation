"""Semantic plausibility judges with protocol-safe LLM routing.

Judge output is advisory to semantic coherence and cannot override a failed
deterministic contract or establish real-world statistical fidelity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple

from .models import EpisodeBundle
from .validation import compute_episode_metrics


@dataclass
class JudgeResult:
    outcome: str
    passed: bool
    reasons: List[str]
    mode: str
    protocol_attempt_count: int
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "passed": self.passed,
            "reasons": self.reasons,
            "mode": self.mode,
            "protocol_attempt_count": self.protocol_attempt_count,
            "raw": self.raw,
        }


class SemanticJudge(Protocol):
    def evaluate(self, bundle: EpisodeBundle) -> JudgeResult: ...


class HeuristicSemanticJudge:
    """Offline sanity judge used when no external LLM is configured."""

    def evaluate(self, bundle: EpisodeBundle) -> JudgeResult:
        failures: List[str] = []
        metrics = compute_episode_metrics(bundle)
        roots = [event for event in bundle.events if event.event_role == "root"]
        episode_texts = [
            text
            for text in bundle.texts
            if text.text_role == "episode" and text.variant == "canonical"
        ]
        if not roots:
            failures.append("No root event anchors the episode")
        if not episode_texts:
            failures.append("No canonical episode narrative exists")
        if any(
            relation.status == "ground_truth"
            and not relation.provenance.get("recorded_when_rule_executed")
            for relation in bundle.relations
        ):
            failures.append("A ground-truth relation lacks executed-rule provenance")
        max_children = int(
            bundle.program.get("parameters", {}).get("max_children_per_event", 3)
        )
        if metrics["truncated"]:
            failures.append("The episode was truncated by its generation limit")
        if int(metrics["max_propagation_children"]) > max_children:
            failures.append("One event exceeds the configured propagation-child limit")
        passed = not failures
        return JudgeResult(
            outcome="passed" if passed else "semantic_rejected",
            passed=passed,
            reasons=failures or ["Offline semantic sanity checks passed"],
            mode="heuristic",
            protocol_attempt_count=1,
            raw={
                "does_not_claim_statistical_realism": True,
                "episode_metrics": metrics,
            },
        )


class LLMSemanticJudge:
    def __init__(
        self,
        max_tokens: int = 1200,
        protocol_max_attempts: int = 3,
        reason_limit: int = 4,
        response_preview_chars: int = 2000,
        client: Any | None = None,
    ):
        if client is None:
            try:
                from llm_client import LLMClient
            except ImportError as exc:  # pragma: no cover - depends on launch path
                raise RuntimeError(
                    "Could not import data_generation/llm_client.py for LLM judge mode"
                ) from exc
            client = LLMClient()
        self.client = client
        self.max_tokens = max_tokens
        self.protocol_max_attempts = max(1, protocol_max_attempts)
        self.reason_limit = max(1, reason_limit)
        self.response_preview_chars = max(200, response_preview_chars)

    def evaluate(self, bundle: EpisodeBundle) -> JudgeResult:
        payload = {
            "episode": bundle.episode_record(),
            "episode_metrics": compute_episode_metrics(bundle),
            "events": [event.to_dict() for event in bundle.events],
            "relations": [relation.to_dict() for relation in bundle.relations],
            "mechanisms": [mechanism.to_dict() for mechanism in bundle.mechanisms],
            "canonical_texts": [
                text.to_dict() for text in bundle.texts if text.variant == "canonical"
            ],
        }
        base_prompt = (
            "You are the semantic-coherence judge for a synthetic transportation event flow. "
            "Check traffic plausibility, event/relation/text consistency, unsupported claims, "
            "causal-parent provenance, truncation, and quantitative branching limits. "
            "A derived congestion is supported by an incoming causal relation. A recovery "
            "event is correctly supported by an incoming transition whose type is "
            "transitions_to_recovery; do not require or invent a causal-class relation for it. "
            "Reject any direct-cause statement that is absent from the relation records. "
            "When rejecting, report only failures present in the supplied records and identify "
            "the affected event or relation ID. Treat recorded "
            "calibration provenance as auditable evidence, not proof of real-world fidelity. "
            "Do not infer relations from topology or temporal proximity and do not claim "
            "statistical realism.\n"
            "Return exactly one compact JSON object and nothing else, using this schema:\n"
            '{"passed":true_or_false,"reasons":["reason 1","reason 2"]}\n'
            f"Requirements: reasons must contain 1 to {self.reason_limit} non-empty strings; "
            "each reason must be one short sentence under 24 words. Never return an empty "
            "reasons array. No Markdown and no additional keys.\n\nDATA:\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

        protocol_errors: List[Dict[str, Any]] = []
        for attempt in range(1, self.protocol_max_attempts + 1):
            correction = ""
            if attempt > 1:
                correction = (
                    "\n\nPROTOCOL RETRY: The previous response was invalid. Return only the single-line "
                    "compact JSON object. Include at least one non-empty reason."
                )
            response = ""
            try:
                response = self.client.complete(
                    base_prompt + correction,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                parsed, parse_error = _parse_json_object(response)
                if parse_error:
                    raise ValueError(parse_error)
                assert parsed is not None
                normalized, validation_error = _validate_judge_object(
                    parsed, self.reason_limit
                )
                if validation_error:
                    raise ValueError(validation_error)
                assert normalized is not None
            except Exception as exc:  # noqa: BLE001 - protocol retry boundary
                protocol_errors.append(
                    {
                        "attempt": attempt,
                        "error": str(exc),
                        "response_preview": _response_preview(
                            response, self.response_preview_chars
                        ),
                    }
                )
                continue

            passed = bool(normalized["passed"])
            reasons = list(normalized["reasons"])
            return JudgeResult(
                outcome="passed" if passed else "semantic_rejected",
                passed=passed,
                reasons=reasons,
                mode="llm",
                protocol_attempt_count=attempt,
                raw={
                    "parsed_response": normalized,
                    "protocol_errors_before_success": protocol_errors,
                },
            )

        return JudgeResult(
            outcome="protocol_error",
            passed=False,
            reasons=[
                f"LLM Judge protocol failed after {self.protocol_max_attempts} attempts"
            ],
            mode="llm",
            protocol_attempt_count=self.protocol_max_attempts,
            raw={"protocol_errors": protocol_errors},
        )


def build_judge(
    mode: str,
    max_tokens: int = 1200,
    protocol_max_attempts: int = 3,
    reason_limit: int = 4,
    response_preview_chars: int = 2000,
) -> SemanticJudge:
    if mode == "heuristic":
        return HeuristicSemanticJudge()
    if mode == "llm":
        return LLMSemanticJudge(
            max_tokens=max_tokens,
            protocol_max_attempts=protocol_max_attempts,
            reason_limit=reason_limit,
            response_preview_chars=response_preview_chars,
        )
    raise ValueError("judge_mode must be 'heuristic' or 'llm'")


def _validate_judge_object(
    parsed: Dict[str, Any], reason_limit: int
) -> Tuple[Dict[str, Any] | None, str | None]:
    if not isinstance(parsed.get("passed"), bool):
        return None, "LLM Judge field 'passed' must be boolean"
    reasons = parsed.get("reasons")
    if not isinstance(reasons, list):
        return None, "LLM Judge field 'reasons' must be an array"
    if not reasons:
        return None, "LLM Judge returned an empty reasons array"
    if any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
        return None, "LLM Judge reasons must all be non-empty strings"
    normalized_reasons = [reason.strip() for reason in reasons]
    normalized_reasons = [reason[:300] for reason in normalized_reasons[:reason_limit]]
    return {
        "passed": parsed["passed"],
        "reasons": normalized_reasons,
    }, None


def _parse_json_object(text: str) -> Tuple[Dict[str, Any] | None, str | None]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as direct_error:
        candidate = _first_balanced_json_object(stripped)
        if candidate is None:
            return None, f"Invalid JSON from LLM Judge: {direct_error}"
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as candidate_error:
            return None, f"Invalid JSON from LLM Judge: {candidate_error}"
    if not isinstance(parsed, dict):
        return None, "LLM Judge output was not a JSON object"
    return parsed, None


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _response_preview(text: str, character_budget: int) -> Dict[str, Any]:
    half = max(100, character_budget // 2)
    return {
        "character_count": len(text),
        "head": text[:half],
        "tail": text[-half:] if len(text) > half else text,
    }
