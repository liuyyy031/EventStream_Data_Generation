"""Grounded event, relation, mechanism, and episode text generation."""

from __future__ import annotations

import random
from typing import Dict, List

from ...core.models import EpisodeBundle, Event, Relation, TextRecord
from .network import TransportNetwork


class TransportationTextGenerator:
    def __init__(self, network: TransportNetwork, text_config: Dict[str, object]):
        self.network = network
        self.language = str(text_config.get("language", "en"))
        self.paraphrase_rate = float(text_config.get("paraphrase_rate", 0.0))
        if self.language != "en":
            raise ValueError("EventFlow v1 currently ships verified English templates only")

    def generate(self, bundle: EpisodeBundle, seed: int) -> List[TextRecord]:
        rng = random.Random(seed)
        records: List[TextRecord] = []
        event_by_id = {event.event_id: event for event in bundle.events}
        text_counter = 0

        def new_id() -> str:
            nonlocal text_counter
            text_counter += 1
            return f"{bundle.episode_id}_text_{text_counter:05d}"

        for event in bundle.events:
            canonical = TextRecord(
                text_id=new_id(),
                episode_id=bundle.episode_id,
                text_role="event",
                variant="canonical",
                content=_event_sentence(event),
                aligned_event_ids=[event.event_id],
                aligned_relation_ids=[],
                aligned_mechanism_ids=[],
                grounded_facts=_event_facts(event),
                verbalized_fact_keys=_event_verbalized_keys(event),
            )
            records.append(canonical)
            if rng.random() < self.paraphrase_rate:
                records.append(
                    TextRecord(
                        text_id=new_id(),
                        episode_id=bundle.episode_id,
                        text_role="event",
                        variant="paraphrase",
                        content=_event_paraphrase(event),
                        aligned_event_ids=[event.event_id],
                        aligned_relation_ids=[],
                        aligned_mechanism_ids=[],
                        grounded_facts=_event_facts(event),
                        verbalized_fact_keys=_event_verbalized_keys(event),
                        derived_from=canonical.text_id,
                    )
                )

        mechanism_relation_ids = {
            relation_id
            for mechanism in bundle.mechanisms
            for relation_id in mechanism.member_relation_ids
        }
        for relation in bundle.relations:
            source = event_by_id[relation.source_event_id]
            target = event_by_id[relation.target_event_id]
            canonical = TextRecord(
                text_id=new_id(),
                episode_id=bundle.episode_id,
                text_role="relation",
                variant="canonical",
                content=_relation_sentence(relation, source, target),
                aligned_event_ids=[source.event_id, target.event_id],
                aligned_relation_ids=[relation.relation_id],
                aligned_mechanism_ids=(
                    [relation.mechanism_group_id] if relation.mechanism_group_id else []
                ),
                grounded_facts=_relation_facts(relation, source, target),
                verbalized_fact_keys=[
                    "source_event_id",
                    "target_event_id",
                    "relation_type",
                    "time_lag_seconds",
                ],
            )
            records.append(canonical)
            if relation.relation_id not in mechanism_relation_ids and rng.random() < self.paraphrase_rate:
                records.append(
                    TextRecord(
                        text_id=new_id(),
                        episode_id=bundle.episode_id,
                        text_role="relation",
                        variant="paraphrase",
                        content=_relation_paraphrase(relation, source, target),
                        aligned_event_ids=[source.event_id, target.event_id],
                        aligned_relation_ids=[relation.relation_id],
                        aligned_mechanism_ids=[],
                        grounded_facts=_relation_facts(relation, source, target),
                        verbalized_fact_keys=[
                            "source_event_id",
                            "target_event_id",
                            "relation_type",
                            "time_lag_seconds",
                        ],
                        derived_from=canonical.text_id,
                    )
                )

        relation_by_id = {relation.relation_id: relation for relation in bundle.relations}
        for mechanism in bundle.mechanisms:
            sources = [
                event_by_id[relation_by_id[relation_id].source_event_id]
                for relation_id in mechanism.member_relation_ids
            ]
            target = event_by_id[mechanism.target_event_id]
            source_labels = " and ".join(_event_label(source) for source in sources)
            records.append(
                TextRecord(
                    text_id=new_id(),
                    episode_id=bundle.episode_id,
                    text_role="mechanism",
                    variant="canonical",
                    content=(
                        f"{source_labels} must act together under rule {mechanism.rule_id} "
                        f"to produce {_event_label(target)}; neither topology nor temporal proximity "
                        "alone establishes these parent relations."
                    ),
                    aligned_event_ids=[source.event_id for source in sources] + [target.event_id],
                    aligned_relation_ids=list(mechanism.member_relation_ids),
                    aligned_mechanism_ids=[mechanism.mechanism_group_id],
                    grounded_facts={
                        "mechanism_group_id": mechanism.mechanism_group_id,
                        "combination": mechanism.combination,
                        "source_event_ids": [source.event_id for source in sources],
                        "target_event_id": target.event_id,
                        "rule_id": mechanism.rule_id,
                    },
                    verbalized_fact_keys=[
                        "combination",
                        "source_event_ids",
                        "target_event_id",
                        "rule_id",
                    ],
                )
            )

        ordered_events = sorted(bundle.events, key=lambda event: event.time_offset_seconds)
        independent_ids = set(bundle.program.get("independent_root_event_ids", []))
        main_events = [event for event in ordered_events if event.event_id not in independent_ids]
        independent_events = [
            event for event in ordered_events if event.event_id in independent_ids
        ]
        narrative = " ".join(_event_sentence(event) for event in main_events)
        independent_narrative = ""
        if independent_events:
            independent_narrative = (
                " Independent concurrent root events: "
                + " ".join(_event_sentence(event) for event in independent_events)
                + " These events were generated independently; no ground-truth relation "
                "to the main event flow is recorded."
            )
        relation_summary = " ".join(
            _relation_sentence(
                relation,
                event_by_id[relation.source_event_id],
                event_by_id[relation.target_event_id],
            )
            for relation in bundle.relations
        )
        episode_text = TextRecord(
            text_id=new_id(),
            episode_id=bundle.episode_id,
            text_role="episode",
            variant="canonical",
            content=(
                f"Main event flow: {narrative}{independent_narrative} "
                f"Event-flow relations: {relation_summary}"
            ),
            aligned_event_ids=[event.event_id for event in ordered_events],
            aligned_relation_ids=[relation.relation_id for relation in bundle.relations],
            aligned_mechanism_ids=[
                mechanism.mechanism_group_id for mechanism in bundle.mechanisms
            ],
            grounded_facts={
                "chronological_event_ids": [event.event_id for event in ordered_events],
                "relation_ids": [relation.relation_id for relation in bundle.relations],
                "mechanism_group_ids": [
                    mechanism.mechanism_group_id for mechanism in bundle.mechanisms
                ],
                "independent_root_event_ids": sorted(independent_ids),
            },
            verbalized_fact_keys=[
                "chronological_event_ids",
                "relation_ids",
                "mechanism_group_ids",
                "independent_root_event_ids",
                "event_attributes",
                "relation_time_lags",
            ],
        )
        records.append(episode_text)
        return records


def _event_label(event: Event) -> str:
    participants = ", ".join(event.participant_ids)
    return f"{_natural_type(event.event_type_id)} ({event.event_id}) on {participants}"


def _event_sentence(event: Event) -> str:
    participants = ", ".join(event.participant_ids)
    independent_note = (
        " It is an independently generated concurrent root event."
        if event.provenance.get("independent_from_primary_flow")
        else ""
    )
    return (
        f"At {event.event_time}, {_natural_type(event.event_type_id)} ({event.event_id}) "
        f"occurred on {participants}{_event_attribute_clause(event)}.{independent_note}"
    )


def _event_paraphrase(event: Event) -> str:
    participants = ", ".join(event.participant_ids)
    independent_note = (
        " This event is an independent concurrent root."
        if event.provenance.get("independent_from_primary_flow")
        else ""
    )
    return (
        f"The entities {participants} experienced {_natural_type(event.event_type_id)} "
        f"({event.event_id}) at {event.event_time}{_event_attribute_clause(event)}."
        f"{independent_note}"
    )


def _relation_sentence(relation: Relation, source: Event, target: Event) -> str:
    if relation.relation_type == "jointly_causes_congestion":
        link = "was one required parent in the joint mechanism producing"
    elif relation.relation_type == "initiates_congestion":
        link = "directly caused"
    elif relation.relation_type == "weather_induces_congestion":
        link = "caused"
    elif relation.relation_type == "propagates_downstream":
        link = "propagated downstream and produced"
    elif relation.relation_type == "transitions_to_recovery":
        link = "was followed by its linked recovery event"
    else:
        link = relation.relation_type.replace("_", " ")
    lag = target.time_offset_seconds - source.time_offset_seconds
    return (
        f"{_event_label(source)} {link} {_event_label(target)} "
        f"after {lag:.3f} seconds."
    )


def _relation_paraphrase(relation: Relation, source: Event, target: Event) -> str:
    if relation.relation_type == "jointly_causes_congestion":
        semantics = "contributed as a required member of a joint cause"
    else:
        semantics = relation.relation_type.replace("_", " ")
    lag = target.time_offset_seconds - source.time_offset_seconds
    return (
        f"Event {source.event_id} is an explicit parent of event {target.event_id}: "
        f"it {semantics}, with a {lag:.3f}-second lag."
    )


def _natural_type(event_type_id: str) -> str:
    return event_type_id.rsplit(".", 1)[-1].replace("_", " ")


def _event_attribute_clause(event: Event) -> str:
    if not event.attributes:
        return ""
    facts = "; ".join(
        f"{key.replace('_', ' ')} {_format_attribute_value(value)}"
        for key, value in sorted(event.attributes.items())
    )
    return f", with {facts}"


def _format_attribute_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _event_verbalized_keys(event: Event) -> List[str]:
    return [
        "event_id",
        "event_type_id",
        "event_time",
        "participants",
        *[f"attributes.{key}" for key in sorted(event.attributes)],
    ]


def _event_facts(event: Event) -> Dict[str, object]:
    return {
        "event_id": event.event_id,
        "event_type_id": event.event_type_id,
        "event_time": event.event_time,
        "participants": [participant.to_dict() for participant in event.participants],
        "attributes": dict(event.attributes),
    }


def _relation_facts(
    relation: Relation, source: Event, target: Event
) -> Dict[str, object]:
    return {
        "relation_id": relation.relation_id,
        "source_event_id": relation.source_event_id,
        "target_event_id": relation.target_event_id,
        "relation_type": relation.relation_type,
        "rule_id": relation.rule_id,
        "status": relation.status,
        "mechanism_group_id": relation.mechanism_group_id,
        "time_lag_seconds": round(
            target.time_offset_seconds - source.time_offset_seconds, 3
        ),
    }
