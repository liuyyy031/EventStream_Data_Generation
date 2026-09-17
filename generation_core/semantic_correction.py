"""Judge-guided, structure-preserving correction of grounded episode text."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Protocol, Tuple

from .models import EpisodeResult
from .semantic_judge import SemanticJudgeResult, _parse_json_object, _response_preview
from .validation import validate_text_alignment


@dataclass
class SemanticCorrectionResult:
    outcome: str
    succeeded: bool
    text_alignment: Dict[str, Any]
    sentence_updates: List[Dict[str, Any]] = field(default_factory=list)
    protocol_attempt_count: int = 1
    model_id: str | None = None
    provider_base_url: str | None = None
    protocol_errors: List[Dict[str, Any]] = field(default_factory=list)
    before_text_sha256: str | None = None
    after_text_sha256: str | None = None

    def audit_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("text_alignment", None)
        return payload


class SemanticCorrector(Protocol):
    def correct(
        self,
        result: EpisodeResult,
        text_alignment: Dict[str, Any],
        domain_catalog: Dict[str, Any],
        judge_result: SemanticJudgeResult,
        *,
        correction_round: int,
    ) -> SemanticCorrectionResult: ...


class LLMSemanticCorrector:
    """Rewrite only grounded sentences while all structured truth stays frozen."""

    def __init__(
        self,
        client: Any,
        *,
        max_tokens: int = 1800,
        protocol_max_attempts: int = 3,
        response_preview_chars: int = 2000,
    ) -> None:
        self.client = client
        self.max_tokens = max(128, int(max_tokens))
        self.protocol_max_attempts = max(1, int(protocol_max_attempts))
        self.response_preview_chars = max(200, int(response_preview_chars))

    @property
    def model_id(self) -> str | None:
        value = getattr(self.client, "model", None)
        return str(value) if value is not None else None

    @property
    def provider_base_url(self) -> str | None:
        value = getattr(self.client, "base_url", None)
        return str(value) if value is not None else None

    def correct(
        self,
        result: EpisodeResult,
        text_alignment: Dict[str, Any],
        domain_catalog: Dict[str, Any],
        judge_result: SemanticJudgeResult,
        *,
        correction_round: int,
    ) -> SemanticCorrectionResult:
        before_hash = _text_hash(text_alignment)
        required_literals = _required_literals_by_sentence(result, text_alignment)
        prompt = _build_correction_prompt(
            result,
            text_alignment,
            domain_catalog,
            judge_result,
            required_literals,
        )
        protocol_errors: List[Dict[str, Any]] = []
        last_error = ""
        for attempt in range(1, self.protocol_max_attempts + 1):
            retry_note = ""
            if attempt > 1:
                retry_note = (
                    "\nPROTOCOL RETRY: The previous response was invalid: "
                    f"{last_error}. Return only the requested compact JSON object."
                )
            response = ""
            try:
                response = self.client.complete(
                    prompt + retry_note,
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                )
                parsed, parse_error = _parse_json_object(response)
                if parse_error:
                    raise ValueError(parse_error)
                assert parsed is not None
                updates, validation_error = _validate_correction_object(
                    parsed,
                    text_alignment,
                    required_literals,
                )
                if validation_error:
                    raise ValueError(validation_error)
                assert updates is not None
                corrected = _apply_updates(
                    text_alignment,
                    updates,
                    correction_round=correction_round,
                    model_id=self.model_id,
                    judge_result=judge_result,
                )
                alignment_validation = validate_text_alignment(
                    result,
                    corrected,
                    domain_catalog,
                )
                if not alignment_validation["passed"]:
                    raise ValueError(
                        "Corrected text failed deterministic alignment: "
                        + "; ".join(alignment_validation["errors"])
                    )
            except Exception as exc:  # noqa: BLE001 - protocol boundary
                last_error = str(exc)
                protocol_errors.append(
                    {
                        "attempt": attempt,
                        "error": last_error,
                        "response_preview": _response_preview(
                            response, self.response_preview_chars
                        ),
                    }
                )
                continue
            return SemanticCorrectionResult(
                outcome="corrected",
                succeeded=True,
                text_alignment=corrected,
                sentence_updates=updates,
                protocol_attempt_count=attempt,
                model_id=self.model_id,
                provider_base_url=self.provider_base_url,
                protocol_errors=protocol_errors,
                before_text_sha256=before_hash,
                after_text_sha256=_text_hash(corrected),
            )
        return SemanticCorrectionResult(
            outcome="protocol_error",
            succeeded=False,
            text_alignment=copy.deepcopy(text_alignment),
            protocol_attempt_count=self.protocol_max_attempts,
            model_id=self.model_id,
            provider_base_url=self.provider_base_url,
            protocol_errors=protocol_errors,
            before_text_sha256=before_hash,
            after_text_sha256=before_hash,
        )


def _build_correction_prompt(
    result: EpisodeResult,
    text_alignment: Dict[str, Any],
    domain_catalog: Dict[str, Any],
    judge_result: SemanticJudgeResult,
    required_literals: Dict[int, List[str]],
) -> str:
    sentences = list(text_alignment.get("sentences", []))
    claims_by_sentence: Dict[int, List[Dict[str, Any]]] = {}
    for claim in text_alignment.get("claims", []):
        index = claim.get("sentence_index")
        if isinstance(index, int):
            claims_by_sentence.setdefault(index, []).append(claim)
    payload = {
        "correction_contract": {
            "allowed_change": "sentence wording only",
            "immutable_fields": [
                "event IDs",
                "event types",
                "timestamps",
                "participants",
                "attributes",
                "relation IDs",
                "relation classes",
                "relation directions",
                "relation predicates",
                "lags",
                "candidate states",
                "risk sets",
            ],
            "required_literal_rule": (
                "Every listed required literal must appear verbatim in the updated "
                "sentence for that index."
            ),
            "no_new_claims": True,
            "partial_updates_allowed": True,
        },
        "judge_feedback": {
            "issue_scope": judge_result.issue_scope,
            "reasons": judge_result.reasons,
            "flagged_record_ids": judge_result.flagged_record_ids,
        },
        "episode": result.episode_record(),
        "events": [event.to_dict() for event in result.events],
        "event_relations": [relation.to_dict() for relation in result.event_relations],
        "domain_claim_scope": domain_catalog.get("clinical_safety_scope", {}),
        "sentences": [
            {
                "sentence_index": index,
                "current_sentence": sentence,
                "claims": claims_by_sentence.get(index, []),
                "required_literals": required_literals.get(index, []),
            }
            for index, sentence in enumerate(sentences)
        ],
    }
    return (
        "You correct grounded text for one synthetic event-sequence episode. "
        "The structured episode is immutable and has already passed deterministic "
        "validation. Address only the judge's text issue. Do not add explanations "
        "that are not supported by the supplied records. Preserve the distinction "
        "between causal, transition, statistical-influence, workflow, and derivation "
        "relations. Do not turn a threshold flag into a diagnosis, a workflow step "
        "into treatment efficacy, adjacency into parenthood, or statistical influence "
        "into proven causality. Update only sentences that need correction.\n"
        "Return exactly one compact JSON object with no Markdown and exactly this key:\n"
        '{"sentence_updates":[{"sentence_index":0,"sentence":"replacement"}]}\n'
        "The array must contain at least one changed sentence, indices must be unique, "
        "and all required_literals for an updated index must occur verbatim.\nDATA:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _validate_correction_object(
    parsed: Dict[str, Any],
    text_alignment: Dict[str, Any],
    required_literals: Dict[int, List[str]],
) -> Tuple[List[Dict[str, Any]] | None, str | None]:
    if set(parsed) != {"sentence_updates"}:
        return None, "Correction response has missing or additional keys"
    raw_updates = parsed.get("sentence_updates")
    if not isinstance(raw_updates, list) or not raw_updates:
        return None, "sentence_updates must be a non-empty array"
    sentences = text_alignment.get("sentences", [])
    updates: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for raw in raw_updates:
        if not isinstance(raw, dict) or set(raw) != {"sentence_index", "sentence"}:
            return None, "Each sentence update must contain exactly sentence_index and sentence"
        index = raw.get("sentence_index")
        sentence = raw.get("sentence")
        if not isinstance(index, int) or not 0 <= index < len(sentences):
            return None, "Correction response contains an invalid sentence_index"
        if index in seen:
            return None, "Correction response contains duplicate sentence indices"
        if not isinstance(sentence, str) or not sentence.strip():
            return None, "Correction response contains an empty sentence"
        sentence = sentence.strip()
        if len(sentence) > 2500:
            return None, "Correction response contains an excessively long sentence"
        if sentence == sentences[index]:
            return None, "Correction response did not change the selected sentence"
        missing = [
            literal
            for literal in required_literals.get(index, [])
            if literal not in sentence
        ]
        if missing:
            return None, f"Corrected sentence {index} removed required literals: {missing}"
        seen.add(index)
        updates.append({"sentence_index": index, "sentence": sentence})
    return sorted(updates, key=lambda item: item["sentence_index"]), None


def _apply_updates(
    text_alignment: Dict[str, Any],
    updates: List[Dict[str, Any]],
    *,
    correction_round: int,
    model_id: str | None,
    judge_result: SemanticJudgeResult,
) -> Dict[str, Any]:
    corrected = copy.deepcopy(text_alignment)
    sentences = list(corrected.get("sentences", []))
    for update in updates:
        sentences[update["sentence_index"]] = update["sentence"]
    corrected["sentences"] = sentences
    corrected["text"] = " ".join(sentences)
    corrected["projection_mode"] = "llm_corrected_grounded_projection"
    history = list(corrected.get("correction_history", []))
    history.append(
        {
            "correction_round": correction_round,
            "model_id": model_id,
            "judge_reasons": list(judge_result.reasons),
            "flagged_record_ids": list(judge_result.flagged_record_ids),
            "updated_sentence_indices": [
                update["sentence_index"] for update in updates
            ],
        }
    )
    corrected["correction_history"] = history
    return corrected


def _required_literals_by_sentence(
    result: EpisodeResult,
    text_alignment: Dict[str, Any],
) -> Dict[int, List[str]]:
    event_by_id = {event.event_id: event for event in result.events}
    relation_by_id = {
        relation.relation_id: relation for relation in result.event_relations
    }
    output: Dict[int, List[str]] = {}
    for claim in text_alignment.get("claims", []):
        index = claim.get("sentence_index")
        if not isinstance(index, int):
            continue
        literals = output.setdefault(index, [])
        claim_relation_ids = claim.get("relation_ids", [])
        if not claim_relation_ids:
            for event_id in claim.get("event_ids", []):
                event = event_by_id.get(event_id)
                if event is None:
                    continue
                literals.append(
                    f"+{event.temporal.occurrence_start_offset_seconds:.3f}"
                )
                literals.extend(
                    f"{participant.role}={participant.entity_id}"
                    for participant in event.participants
                )
                literals.extend(
                    f"{key}={value}"
                    for key, value in sorted(event.attributes.items())
                )
        assertions = {
            item.get("relation_id"): item
            for item in claim.get("relation_assertions", [])
            if isinstance(item, dict)
        }
        for relation_id in claim_relation_ids:
            relation = relation_by_id.get(relation_id)
            if relation is None:
                continue
            literals.extend(
                [
                    relation.rule_id,
                    relation.source_event_id,
                    relation.target_event_id,
                    f"{relation.temporal_link.lag_seconds:.3f}",
                ]
            )
            predicate = assertions.get(relation_id, {}).get("surface_predicate")
            if isinstance(predicate, str) and predicate:
                literals.append(predicate)
        output[index] = list(dict.fromkeys(literals))
    return output


def _text_hash(text_alignment: Dict[str, Any]) -> str:
    text = str(text_alignment.get("text", ""))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
