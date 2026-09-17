"""Protocol-safe LLM review of generated event-sequence coherence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Protocol, Tuple

from .models import EpisodeResult


@dataclass
class SemanticJudgeResult:
    outcome: str
    passed: bool
    issue_scope: str
    reasons: List[str]
    flagged_record_ids: List[str]
    mode: str = "llm"
    protocol_attempt_count: int = 1
    model_id: str | None = None
    provider_base_url: str | None = None
    protocol_errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticJudge(Protocol):
    def evaluate(
        self,
        result: EpisodeResult,
        text_alignment: Dict[str, Any],
        domain_catalog: Dict[str, Any],
    ) -> SemanticJudgeResult: ...


class LLMSemanticJudge:
    """Ask a configured text model to review one already-validated episode."""

    def __init__(
        self,
        client: Any,
        *,
        max_tokens: int = 1000,
        protocol_max_attempts: int = 3,
        reason_limit: int = 5,
        response_preview_chars: int = 2000,
    ) -> None:
        self.client = client
        self.max_tokens = max(64, int(max_tokens))
        self.protocol_max_attempts = max(1, int(protocol_max_attempts))
        self.reason_limit = max(1, int(reason_limit))
        self.response_preview_chars = max(200, int(response_preview_chars))

    @property
    def model_id(self) -> str | None:
        value = getattr(self.client, "model", None)
        return str(value) if value is not None else None

    @property
    def provider_base_url(self) -> str | None:
        value = getattr(self.client, "base_url", None)
        return str(value) if value is not None else None

    def evaluate(
        self,
        result: EpisodeResult,
        text_alignment: Dict[str, Any],
        domain_catalog: Dict[str, Any],
    ) -> SemanticJudgeResult:
        prompt = _build_prompt(
            result,
            text_alignment,
            domain_catalog,
            reason_limit=self.reason_limit,
        )
        protocol_errors: List[Dict[str, Any]] = []
        for attempt in range(1, self.protocol_max_attempts + 1):
            correction = ""
            if attempt > 1:
                correction = (
                    "\nPROTOCOL RETRY: Return only one compact JSON object with "
                    "exactly the requested keys and no Markdown."
                )
            response = ""
            try:
                response = self.client.complete(
                    prompt + correction,
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
            return SemanticJudgeResult(
                outcome="passed" if passed else "semantic_rejected",
                passed=passed,
                issue_scope=str(normalized["issue_scope"]),
                reasons=list(normalized["reasons"]),
                flagged_record_ids=list(normalized["flagged_record_ids"]),
                protocol_attempt_count=attempt,
                model_id=self.model_id,
                provider_base_url=self.provider_base_url,
                protocol_errors=protocol_errors,
            )
        return SemanticJudgeResult(
            outcome="protocol_error",
            passed=False,
            issue_scope="protocol",
            reasons=[
                f"LLM judge protocol failed after {self.protocol_max_attempts} attempts"
            ],
            flagged_record_ids=[],
            protocol_attempt_count=self.protocol_max_attempts,
            model_id=self.model_id,
            provider_base_url=self.provider_base_url,
            protocol_errors=protocol_errors,
        )


def _build_prompt(
    result: EpisodeResult,
    text_alignment: Dict[str, Any],
    domain_catalog: Dict[str, Any],
    *,
    reason_limit: int,
) -> str:
    used_mechanisms = {item.mechanism_id for item in result.candidates}
    mechanism_catalog = [
        item
        for item in domain_catalog.get("mechanisms", [])
        if item.get("mechanism_id") in used_mechanisms
    ]
    payload = {
        "review_contract": {
            "domain_claim_scope": domain_catalog.get(
                "clinical_safety_scope", {}
            ),
            "root_event_semantics": {
                "definition": "an event emitted by a mechanism with no event parents",
                "entity_temporal_primacy": False,
                "must_explain_preexisting_entity_state": False,
                "prior_events_on_the_same_entity_are_allowed": True,
                "interpretation": (
                    "root means parentless in the generated event graph, not the "
                    "first event ever observed on its participant entity"
                ),
            },
            "cancelled_candidate_semantics": {
                "definition": (
                    "a potential derived event that did not occur because its "
                    "eligibility or state precondition no longer held"
                ),
                "does_not_invalidate_parent_event": True,
                "does_not_assert_target_event_occurred": True,
            },
            "review_boundary": (
                "Do not invent entity-state or temporal-precedence constraints "
                "that are absent from the mechanism catalog and supplied evidence. "
                "Do not infer a diagnosis, prescription, or treatment-effect claim "
                "when the domain catalog explicitly excludes it."
            ),
        },
        "episode": result.episode_record(),
        "context_summary": {
            "context_id": result.context_id,
            "topology_profile": result.context_attributes.get(
                "topology_profile", {}
            ),
        },
        "events": [
            {
                "event_id": item.event_id,
                "event_type_id": item.event_type_id,
                "event_role": item.event_role,
                "occurrence_start_offset_seconds": (
                    item.temporal.occurrence_start_offset_seconds
                ),
                "participants": [part.to_dict() for part in item.participants],
                "attributes": item.attributes,
                "risk_set_id": item.provenance.get("risk_set_id"),
            }
            for item in result.events
        ],
        "event_relations": [
            {
                "relation_id": item.relation_id,
                "source_event_id": item.source_event_id,
                "target_event_id": item.target_event_id,
                "relation_class": item.relation_class,
                "relation_type_id": item.relation_type_id,
                "rule_id": item.rule_id,
                "lag_seconds": item.temporal_link.lag_seconds,
                "mechanism_group_id": item.mechanism_group_id,
                "context_evidence": item.context_evidence.to_dict(),
            }
            for item in result.event_relations
        ],
        "candidate_outcomes": [
            {
                "candidate_id": item.candidate_id,
                "mechanism_id": item.mechanism_id,
                "target_event_type_id": item.target_event_type_id,
                "parent_event_ids": item.parent_event_ids,
                "combination": item.combination,
                "activation_time": item.activation_time,
                "scheduled_time": item.scheduled_time,
                "status": item.status.value,
                "terminal_reason": item.terminal_reason,
                "fired_event_id": item.fired_event_id,
            }
            for item in result.candidates
        ],
        "risk_set_summaries": [
            {
                "risk_set_id": item.risk_set_id,
                "evaluated_at_offset_seconds": item.evaluated_at_offset_seconds,
                "selected_candidate_id": item.selected_candidate_id,
                "selected_event_id": item.selected_event_id,
                "selection_mode": item.selection_mode,
                "alternative_count": len(item.alternatives),
                "factorization": item.factorization,
                "parent_attribution": item.parent_attribution,
            }
            for item in result.risk_sets
        ],
        "mechanism_catalog": mechanism_catalog,
        "grounded_text": {
            "sentences": text_alignment.get("sentences", []),
            "claims": text_alignment.get("claims", []),
        },
    }
    return (
        "You are reviewing one synthetic event-sequence episode for semantic and "
        "mechanism coherence. The deterministic contract has already been checked. "
        "Treat review_contract in DATA as binding definitions. In particular, a "
        "root event is parentless in the event graph; it is not necessarily the "
        "first event on its participant entity and need not explain an entity state "
        "that already existed. A later independent root event on an already affected "
        "entity is not a contradiction when its mechanism permits it. If a child "
        "candidate is cancelled because its target state is no longer eligible, that "
        "is a coherent non-occurrence and does not invalidate its parent event or an "
        "earlier independent chain. Do not invent preconditions that are absent from "
        "the mechanism catalog and supplied evidence. "
        "Reject only concrete contradictions in the supplied records: impossible "
        "event ordering, a child unsupported by its declared mechanism, incompatible "
        "participant roles, a relation whose meaning conflicts with its source and "
        "target events, or grounded text that changes structured facts. Context "
        "adjacency is only eligibility evidence and is not automatically event "
        "parenthood. Statistical-influence relations must not be treated as proven "
        "causes. A rule-derived threshold flag is not automatically a diagnosis, "
        "and a workflow-scheduled follow-up is not automatically evidence of "
        "treatment efficacy; respect the domain claim scope supplied in DATA. "
        "Do not reject merely because the episode is synthetic, contains no "
        "events, uses prior parameters, has censored candidates, or differs from an "
        "empirical frequency. Do not claim real-world statistical realism.\n"
        "Classify every rejection before returning it. Use issue_scope=text only "
        "when the structured episode is coherent and rewriting grounded text alone "
        "can fix the problem without changing any event, relation, time, entity, "
        "candidate, or risk set. Use issue_scope=structure when the supplied "
        "structured records themselves are contradictory. Use issue_scope=mixed "
        "when both are present. A text-scoped rejection must flag only event or "
        "event-relation IDs that are referenced by grounded_text claims; a candidate "
        "or risk-set ID is never text-correctable. A passed result must use "
        "issue_scope=none.\n"
        "Return exactly one compact JSON object with no Markdown and exactly these keys:\n"
        '{"passed":true_or_false,"issue_scope":"none|text|structure|mixed",'
        '"reasons":["short reason"],'
        '"flagged_record_ids":["event/relation/candidate ID"]}\n'
        f"Use 1 to {reason_limit} short reasons. When passed, give one concise pass "
        "reason and an empty flagged_record_ids list. When rejected, every claimed "
        "problem must name at least one supplied record ID.\nDATA:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _validate_judge_object(
    parsed: Dict[str, Any], reason_limit: int
) -> Tuple[Dict[str, Any] | None, str | None]:
    if set(parsed) != {
        "passed",
        "issue_scope",
        "reasons",
        "flagged_record_ids",
    }:
        return None, "LLM judge returned missing or additional keys"
    if not isinstance(parsed.get("passed"), bool):
        return None, "LLM judge field 'passed' must be boolean"
    issue_scope = parsed.get("issue_scope")
    if issue_scope not in {"none", "text", "structure", "mixed"}:
        return None, "LLM judge field 'issue_scope' is invalid"
    if parsed["passed"] and issue_scope != "none":
        return None, "A passed episode must use issue_scope=none"
    if not parsed["passed"] and issue_scope == "none":
        return None, "A rejected episode must identify a non-none issue_scope"
    reasons = parsed.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        return None, "LLM judge reasons must be a non-empty array"
    if len(reasons) > reason_limit or any(
        not isinstance(reason, str) or not reason.strip() for reason in reasons
    ):
        return None, "LLM judge reasons violate the response contract"
    record_ids = parsed.get("flagged_record_ids")
    if not isinstance(record_ids, list) or any(
        not isinstance(record_id, str) or not record_id.strip()
        for record_id in record_ids
    ):
        return None, "LLM judge flagged_record_ids must be a string array"
    if not parsed["passed"] and not record_ids:
        return None, "A rejected episode must identify at least one record ID"
    if parsed["passed"] and record_ids:
        return None, "A passed episode must not flag record IDs"
    return {
        "passed": parsed["passed"],
        "issue_scope": issue_scope,
        "reasons": [str(reason).strip()[:400] for reason in reasons],
        "flagged_record_ids": [str(record_id).strip()[:200] for record_id in record_ids],
    }, None


def _parse_json_object(text: str) -> Tuple[Dict[str, Any] | None, str | None]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as direct_error:
        candidate = _first_balanced_json_object(stripped)
        if candidate is None:
            return None, f"Invalid JSON from LLM judge: {direct_error}"
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as candidate_error:
            return None, f"Invalid JSON from LLM judge: {candidate_error}"
    if not isinstance(parsed, dict):
        return None, "LLM judge output was not a JSON object"
    return parsed, None


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
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
