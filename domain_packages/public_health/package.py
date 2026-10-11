"""Synthetic community SEIR events and sample-grounded public-health workflows.

Parameters describe a fictional acute infection. Contact channels are Poisson
effective-contact opportunities, not observed encounters or proof of infection.
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from data_generation.generation_core.domain import (
    CandidateSpec, CandidateUpdate, EpisodeContext, ObservationPlan, RelationSpec,
)
from data_generation.generation_core.domain_spec import load_domain_spec
from data_generation.generation_core.mechanisms import load_mechanism_registry
from data_generation.generation_core.models import (
    ContextEvidence, Entity, EventParticipant, EventRecord, StatePredicateEvidence, TemporalExtent,
)
from data_generation.generation_core.temporal import (
    ConditionalLogNormalModel, DeterministicDelayModel, ExponentialArrivalModel,
    PiecewiseExponentialHazardModel, build_standard_temporal_registry,
)
from data_generation.generation_core.topology import SparseHeterogeneousTopologyGenerator

NS = "public_health."
DAY = 86400.0
EXPOSED = NS + "infection.exposed"
INFECTIOUS = NS + "infection.infectious_started"
SYMPTOMS = NS + "symptoms.started"
RECOVERED = NS + "infection.recovered"
SAMPLE = NS + "specimen.collected"
POSITIVE = NS + "assay.positive_reported"
NEGATIVE = NS + "assay.negative_reported"
REPORTED = NS + "surveillance.lab_positive_notified"
ISOLATED, ISOLATION_END = NS + "isolation.started", NS + "isolation.ended"
NOTIFIED = NS + "cohort_risk.notified"
QUARANTINED, QUARANTINE_END = NS + "quarantine.started", NS + "quarantine.ended"
DECLINED = NS + "quarantine.declined"
EVENT_TYPES = (EXPOSED, INFECTIOUS, SYMPTOMS, RECOVERED, SAMPLE, POSITIVE,
               NEGATIVE, REPORTED, ISOLATED, ISOLATION_END, NOTIFIED,
               QUARANTINED, QUARANTINE_END, DECLINED)
SCENARIOS = ("household_cluster", "workplace_cluster", "surveillance_response")
CONTACT_TYPES = (NS + "household_contact", NS + "workplace_contact")
SAMPLE_REASONS = {"symptom_sample": "symptoms", "scheduled_screening": "screening",
                  "tracing_sample": "cohort_notification"}
PREDICATES = {
    "household_transmission": "produced a realized infectious contact through the household channel at",
    "workplace_transmission": "produced a realized infectious contact through the workplace channel at",
    "latent_progression": "progressed after its latent period to",
    "symptom_onset": "was followed by symptom onset at",
    "infectious_resolution": "ended its infectious period at",
    "symptom_sample": "prompted specimen collection at",
    "tracing_sample": "prompted precautionary specimen collection at",
    "positive_assay": "was assayed and reported positive at",
    "negative_assay": "was assayed and reported negative at",
    "positive_report": "was communicated to the health unit at",
    "case_isolation": "prompted the isolation workflow at",
    "isolation_release": "reached the configured isolation release time at",
    "cohort_notification": "prompted a precautionary cohort-roster notification at",
    "quarantine_accept": "was followed by acceptance of quarantine at",
    "quarantine_decline": "was followed by declining quarantine at",
    "quarantine_release": "reached the configured quarantine release time at",
}


class PublicHealthPackage:
    domain_id = "public_health"

    def __init__(self, config=None):
        self.config = {
            "base_seed": 20260930, "context_count": 4, "nodes_per_context": 1000,
            "duration_seconds": 21 * DAY, "start_time": "2026-09-30T00:00:00+08:00",
            "transmission_probability_per_contact": .07, "symptom_probability": .65,
            "trace_coverage": .7, "quarantine_adherence": .8,
            "assay_false_positive_probability": .01, "assay_exposed_sensitivity": .25,
            "assay_infectious_sensitivity": .85, "screening_day": 4.0,
            "scenario_weights": dict(zip(SCENARIOS, (.35, .3, .35))), **(config or {}),
        }
        for key in ("base_seed", "context_count", "nodes_per_context"):
            value = self.config[key]
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{key} must be an integer")
            self.config[key] = int(value)
        if self.config["context_count"] < 1 or self.config["nodes_per_context"] < 20:
            raise ValueError("Public Health needs at least one context and 20 entities")
        for key in ("duration_seconds", "screening_day"):
            value = float(self.config[key])
            if not math.isfinite(value) or value < 0 or (key == "duration_seconds" and value == 0):
                raise ValueError(f"Invalid {key}")
            self.config[key] = value
        for key in ("transmission_probability_per_contact", "symptom_probability", "trace_coverage",
                    "quarantine_adherence", "assay_false_positive_probability",
                    "assay_exposed_sensitivity", "assay_infectious_sensitivity"):
            value = float(self.config[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must lie in [0, 1]")
            self.config[key] = value
        weights = self.config["scenario_weights"]
        if not weights or set(weights) - set(SCENARIOS) or any(
            not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()
        ) or sum(weights.values()) <= 0:
            raise ValueError("Invalid Public Health scenario weights")
        self.domain_spec = load_domain_spec(Path(__file__).with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(
            Path(__file__).with_name("mechanism_specs.json"), known_event_type_ids=EVENT_TYPES,
            known_context_relation_type_ids=self.domain_spec.context_relation_type_ids)
        self._context_cache = {}

    def catalog(self):
        roles = {role: NS + kind for role, kind in (
            ("person", "person"), ("source_person", "person"), ("household", "household"),
            ("workplace", "workplace"), ("lab", "laboratory"),
            ("unit", "health_unit"), ("reporting_unit", "health_unit"))}
        return {
            "domain": self.domain_id, "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_not_empirically_fitted",
            "scope": {"disease": "fictional_acute_community_infection",
                      "dynamics": "individual_SEIR_no_reinfection_or_severe_outcomes",
                      "contacts": "Poisson_effective_contact_channels_not_observed_encounters",
                      "tracing": "precautionary_cohort_roster_not_proven_transmission",
                      "restriction": "complete_contact_removal_while_isolated_or_quarantined",
                      "statistical_realism_claimed": False},
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {"mode": "deterministic_grounded_projection",
                                "free_form_causal_completion": False},
            "entity_types": [{"entity_type_id": NS + kind, "parent_type_id": None,
                              "required_attributes": attributes} for kind, attributes in (
                ("person", ["susceptibility_multiplier", "infectiousness_multiplier"]),
                ("household", ["setting_kind"]), ("workplace", ["setting_kind"]),
                ("laboratory", ["assay_id"]), ("health_unit", ["policy_id"]))],
            "event_types": [{"event_type_id": kind, "parent_type_id": kind.rsplit(".", 1)[0],
                             "required_attributes": ["workflow_id", "scenario_family", "person_id"],
                             "participant_role_types": roles} for kind in EVENT_TYPES],
            "context_relation_types": [{"relation_type_id": kind, "directed": True,
                                        "does_not_imply_event_parenthood": True}
                                       for kind in self.domain_spec.context_relation_type_ids],
            "event_relation_types": [{"relation_type_id": NS + key,
                "relation_class": self.mechanism_registry.get(NS + key).relation_semantics,
                "text_rendering": {"surface_predicate": predicate,
                    "asserted_relation_class": self.mechanism_registry.get(NS + key).relation_semantics}}
                for key, predicate in PREDICATES.items()],
            "mechanisms": self.mechanism_registry.descriptors(),
        }

    @staticmethod
    def _graph_data(entities, relations):
        bindings, relation_ids, contacts = {}, {}, defaultdict(list)
        person_attributes = {entity.entity_id: entity.attributes for entity in entities
                             if entity.entity_type_id == NS + "person"}
        for relation in relations:
            triple = (relation.relation_type_id, relation.source_entity_id, relation.target_entity_id)
            relation_ids[triple] = relation.relation_id
            if relation.relation_type_id in CONTACT_TYPES:
                contacts[relation.source_entity_id].append(relation)
            else:
                bindings[(relation.relation_type_id, relation.source_entity_id)] = relation.target_entity_id
        return {"bindings": bindings, "relation_ids": relation_ids,
                "contacts": dict(contacts), "person_attributes": person_attributes,
                "contact_by_id": {relation.relation_id: relation for relation in relations
                                  if relation.relation_type_id in CONTACT_TYPES}}

    def _build_context(self, index):
        rng = random.Random(self.config["base_seed"] + 7919 * index)
        context_id = f"public_health_context_{index:03d}"
        n = self.config["nodes_per_context"]
        people = int(.7 * n)
        households = max(2, n // 6)
        labs = max(1, n // 500)
        workplaces = n - people - households - 2 * labs
        if workplaces < 1:
            raise ValueError("Insufficient node budget for public-health settings")
        ids = {kind: [f"{context_id}_{kind}_{i:05d}" for i in range(count)]
               for kind, count in (("person", people), ("household", households),
                   ("workplace", workplaces), ("laboratory", labs), ("health_unit", labs))}
        entities = []
        for person in ids["person"]:
            entities.append(Entity(person, self.domain_id, NS + "person",
                {"susceptibility_multiplier": round(rng.uniform(.6, 1.4), 6),
                 "infectiousness_multiplier": round(rng.uniform(.7, 1.3), 6)}, context_id))
        for kind in ("household", "workplace"):
            for setting in ids[kind]:
                entities.append(Entity(setting, self.domain_id, NS + kind,
                                       {"setting_kind": kind}, context_id))
        for lab in ids["laboratory"]:
            entities.append(Entity(lab, self.domain_id, NS + "laboratory",
                                   {"assay_id": "synthetic_active_infection_assay"}, context_id))
        for unit in ids["health_unit"]:
            entities.append(Entity(unit, self.domain_id, NS + "health_unit",
                                   {"policy_id": "synthetic_7_day_restriction"}, context_id))
        membership, groups = {}, {}
        for kind in ("household", "workplace"):
            shuffled = list(ids["person"])
            rng.shuffle(shuffled)
            membership[kind] = {person: ids[kind][i % len(ids[kind])]
                                for i, person in enumerate(shuffled)}
            groups[kind] = defaultdict(list)
            for person, setting in membership[kind].items():
                groups[kind][setting].append(person)
        person_lab = {person: ids["laboratory"][i % labs] for i, person in enumerate(ids["person"])}
        lab_unit = dict(zip(ids["laboratory"], ids["health_unit"]))
        required = {
            NS + "person_in_household": list(membership["household"].items()),
            NS + "person_in_workplace": list(membership["workplace"].items()),
            NS + "person_tested_by_lab": list(person_lab.items()),
            NS + "person_registered_with_unit": [(person, lab_unit[person_lab[person]]) for person in ids["person"]],
            NS + "lab_reports_to_unit": list(lab_unit.items()),
            NS + "household_contact": [], NS + "workplace_contact": [],
        }
        for group in groups["household"].values():
            required[NS + "household_contact"].extend((source, target)
                for source in group for target in group if source != target)
        for group in groups["workplace"].values():
            for i, source in enumerate(group):
                for target in {group[(i-1) % len(group)], group[(i+1) % len(group)]} - {source}:
                    required[NS + "workplace_contact"].append((source, target))
        def contact_builder(kind):
            def build(source, target, local_rng):
                del target
                baseline = 1.6 if kind == "household" else .7
                return {"setting_id": membership[kind][source.entity_id], "setting_kind": kind,
                        "effective_contacts_per_day": round(baseline * local_rng.uniform(.6, 1.4), 6),
                        "contact_semantics": "potential_Poisson_effective_contact_channel"}
            return build
        layers = self.domain_spec.compile_relation_layers(attribute_builders={
            "household_contact_attributes": contact_builder("household"),
            "workplace_contact_attributes": contact_builder("workplace")})
        layers = [replace(layer, required_pairs=tuple(sorted(required[layer.relation_type_id])))
                  for layer in layers]
        topology = SparseHeterogeneousTopologyGenerator().generate(
            context_id=context_id, domain=self.domain_id, entities=entities, layers=layers,
            seed=self.config["base_seed"] + 7919 * index,
            observation_window_seconds=self.config["duration_seconds"])
        return {"entities": entities, "relations": topology.relations,
                "statistics": topology.statistics, "groups": dict(groups),
                **self._graph_data(entities, topology.relations)}

    @staticmethod
    def _initial_state(data, scenario, workflow_id):
        return {**{key: data[key] for key in ("bindings", "relation_ids", "contacts",
                                            "contact_by_id", "person_attributes")},
                "scenario": scenario, "workflow_id": workflow_id,
                "scheduled_transmissions": set(), "result_specimens": set(),
                "local": {person: {"disease": "S", "course": None, "exposure": None,
                    "infectious": None, "symptoms": None, "reported": None,
                    "isolated": False, "quarantined": False, "epoch": 0,
                    "isolation_event": None, "quarantine_event": None,
                    "notified": None, "quarantine_decision": None, "samples": set()}
                    for person in data["person_attributes"]}}

    def create_episode_context(self, episode_index, seed):
        index = episode_index % self.config["context_count"]
        if index not in self._context_cache:
            self._context_cache[index] = self._build_context(index)
        data = self._context_cache[index]
        rng = random.Random(seed ^ 0x5055424845414c)
        weights = self.config["scenario_weights"]
        scenario = rng.choices(list(weights), weights=list(weights.values()))[0]
        setting_kind = "household" if scenario == "household_cluster" else "workplace"
        group = rng.choice(list(data["groups"][setting_kind].values()))
        roots = rng.sample(group, min(2 if scenario == "workplace_cluster" else 1, len(group)))
        episode_id = f"public_health_episode_{episode_index:07d}"
        state = self._initial_state(data, scenario, episode_id + "_workflow")
        state.update(root_people=roots, screening_people=list(group) if scenario == "surveillance_response" else [])
        return EpisodeContext(episode_id, f"public_health_context_{index:03d}", self.domain_id,
            self.config["start_time"], self.config["duration_seconds"], data["entities"], data["relations"],
            state=state, context_attributes={"topology_profile": data["statistics"]},
            episode_attributes={"scenario_family": scenario, "empirical_realism_claimed": False,
                "pathogen_id": "fictional_acute_infection", "root_people": roots,
                "notification_basis": "precautionary_cohort_roster_not_confirmed_exposure",
                "latent_period_is_not_incubation_period": True,
                "restriction_policy_days": 7.0})

    @staticmethod
    def _restricted(local):
        return local["isolated"] or local["quarantined"]

    def _force(self, state, contact):
        source, target = contact.source_entity_id, contact.target_entity_id
        source_attrs, target_attrs = state["person_attributes"][source], state["person_attributes"][target]
        setting_multiplier = (self._workplace_multiplier(state["scenario"])
                              if contact.relation_type_id == NS + "workplace_contact" else 1.0)
        course_multiplier = .6 if state["local"][source]["course"] == "asymptomatic" else 1.0
        effective_probability = min(1.0, self.config["transmission_probability_per_contact"] *
                source_attrs["infectiousness_multiplier"] * target_attrs["susceptibility_multiplier"] *
                setting_multiplier * course_multiplier)
        return contact.attributes["effective_contacts_per_day"] * effective_probability

    @staticmethod
    def _workplace_multiplier(scenario):
        return {"household_cluster": .35, "workplace_cluster": 1.5, "surveillance_response": 1.0}[scenario]

    def _spec(self, key, context, person, parents=(), now=0.0, *, contact=None, extra=None, inputs=None):
        state = context.state
        mechanism = self.mechanism_registry.get(NS + key)
        roles, pairs = {"person": person}, set()
        attrs = {"workflow_id": state["workflow_id"], "scenario_family": state["scenario"],
                 "person_id": person}
        if key in SAMPLE_REASONS or key in ("positive_assay", "negative_assay", "positive_report"):
            lab = state["bindings"][(NS + "person_tested_by_lab", person)]
            roles["lab"] = lab
            pairs.add((NS + "person_tested_by_lab", person, lab))
        if key in ("positive_report", "case_isolation", "isolation_release", "cohort_notification",
                   "quarantine_accept", "quarantine_decline", "quarantine_release"):
            unit = state["bindings"][(NS + "person_registered_with_unit", person)]
            roles["unit"] = unit
            pairs.add((NS + "person_registered_with_unit", person, unit))
            if key == "positive_report":
                pairs.add((NS + "lab_reports_to_unit", roles["lab"], unit))
        if contact:
            source = contact.source_entity_id
            roles["source_person"] = source
            kind = contact.attributes["setting_kind"]
            setting = contact.attributes["setting_id"]
            roles[kind] = setting
            pairs.update(((contact.relation_type_id, source, person),
                          (NS + f"person_in_{kind}", source, setting),
                          (NS + f"person_in_{kind}", person, setting)))
            attrs.update(source_person_id=source, contact_relation_id=contact.relation_id,
                         contact_setting_kind=kind, contact_setting_id=setting)
            if key.endswith("transmission"):
                force = self._force(state, contact)
                inputs = {"log_contact_force": math.log(force)}
                attrs.update(source_epoch=state["local"][source]["epoch"],
                             target_epoch=state["local"][person]["epoch"])
            elif key == "cohort_notification":
                source_unit = state["bindings"][(NS + "person_registered_with_unit", source)]
                roles["reporting_unit"] = source_unit
                pairs.add((NS + "person_registered_with_unit", source, source_unit))
        if key in SAMPLE_REASONS:
            attrs["sample_reason"] = SAMPLE_REASONS[key]
        if extra:
            attrs.update(extra)
        evidence_ids = [state["relation_ids"][pair] for pair in sorted(pairs)]
        return CandidateSpec(mechanism.mechanism_id, mechanism.target_event_type_id, list(parents),
            [EventParticipant(entity, role) for role, entity in roles.items()],
            mechanism.generation_temporal_model_ref, inputs or {}, activation_time=now,
            combination=mechanism.parent_combination,
            phase_priority=5 if key in ("case_isolation", "quarantine_accept", "isolation_release", "quarantine_release") else 20,
            context_evidence=ContextEvidence(context_relation_ids=evidence_ids,
                entity_ids=list(dict.fromkeys(roles.values())), evaluated_at_offset_seconds=now),
            attributes=attrs, provenance={"input_anchor": "occurrence_start",
                                         "parameter_status": mechanism.parameter_status})

    def seed_candidates(self, context, rng):
        del rng
        result = [self._spec("external_exposure", context, person) for person in context.state["root_people"]]
        for person in context.state["screening_people"]:
            result.append(self._spec("scheduled_screening", context, person,
                inputs={"scheduled_offset_seconds": self.config["screening_day"] * DAY}))
        return result

    def revalidate_candidate(self, candidate, context, event_by_id, now):
        state, attrs = context.state, candidate.attributes
        key = candidate.mechanism_id.removeprefix(NS)
        person = attrs["person_id"]
        local = state["local"][person]
        valid, reason, inputs = True, "eligible", None
        predicates = []
        if key.endswith("transmission"):
            source = attrs["source_person_id"]
            source_local = state["local"][source]
            contact = state["contact_by_id"].get(attrs["contact_relation_id"])
            checks = ((source, "disease", "I", source_local["disease"]),
                      (person, "disease", "S", local["disease"]),
                      (source, "contact_restricted", False, self._restricted(source_local)),
                      (person, "contact_restricted", False, self._restricted(local)),
                      (source, "contact_epoch", attrs["source_epoch"], source_local["epoch"]),
                      (person, "contact_epoch", attrs["target_epoch"], local["epoch"]))
            predicates = [StatePredicateEvidence(subject, name, "equals", expected, actual, expected == actual)
                          for subject, name, expected, actual in checks]
            valid = all(predicate.passed for predicate in predicates) and contact is not None
            reason = "source_or_target_no_longer_eligible_for_transmission"
            if contact:
                valid &= (contact.source_entity_id == source and contact.target_entity_id == person)
                force = self._force(state, contact)
                valid &= force > 0
                inputs = {"log_contact_force": math.log(force)} if force > 0 else None
        elif key == "external_exposure":
            valid = local["disease"] == "S" and not self._restricted(local)
        elif key == "latent_progression":
            valid = local["disease"] == "E"
        elif key == "symptom_onset":
            valid = local["disease"] == "I" and local["course"] == "symptomatic" and local["symptoms"] is None
        elif key == "infectious_resolution":
            valid = local["disease"] == "I"
        elif key in SAMPLE_REASONS:
            valid = attrs["sample_reason"] not in local["samples"]
        elif key in ("positive_assay", "negative_assay"):
            specimen = attrs["specimen_event_id"]
            valid = specimen in event_by_id and specimen not in state["result_specimens"]
        elif key == "positive_report":
            valid = local["reported"] is None
        elif key == "case_isolation":
            valid = local["reported"] is not None and not local["isolated"]
        elif key == "isolation_release":
            valid = local["isolated"] and local["isolation_event"] in candidate.parent_event_ids
        elif key == "cohort_notification":
            source = attrs["source_person_id"]
            valid = local["notified"] is None and state["local"][source]["reported"] in candidate.parent_event_ids
        elif key in ("quarantine_accept", "quarantine_decline"):
            valid = local["notified"] is not None and local["quarantine_decision"] is None and not self._restricted(local)
        elif key == "quarantine_release":
            valid = local["quarantined"] and local["quarantine_event"] in candidate.parent_event_ids
        else:
            raise ValueError(f"Unsupported Public Health mechanism {key}")
        evidence = copy.deepcopy(candidate.context_evidence)
        evidence.evaluated_at_offset_seconds = now
        evidence.state_predicates = predicates or [StatePredicateEvidence(
            person, key + "_precondition", "equals", True, bool(valid), bool(valid))]
        return CandidateUpdate(bool(valid), None if valid else reason if predicates else key + "_precondition_failed",
                               temporal_inputs=inputs, context_evidence=evidence)

    def _assay_probability(self, disease):
        if disease == "E":
            return self.config["assay_exposed_sensitivity"]
        if disease == "I":
            return self.config["assay_infectious_sensitivity"]
        return self.config["assay_false_positive_probability"]

    def materialize_event(self, event_id, candidate, context, rng):
        state, kind = context.state, candidate.target_event_type_id
        attrs = copy.deepcopy(candidate.attributes)
        person, now = attrs["person_id"], candidate.scheduled_time
        local = state["local"][person]
        attrs.update(disease_state_before=local["disease"],
                     disease_state_after={EXPOSED: "E", INFECTIOUS: "I", RECOVERED: "R"}.get(kind, local["disease"]),
                     isolated_before=local["isolated"], quarantined_before=local["quarantined"])
        if kind == EXPOSED:
            draw = rng.random()
            attrs.update(symptom_uniform_draw=draw, symptom_probability=self.config["symptom_probability"],
                         symptom_course="symptomatic" if draw < self.config["symptom_probability"] else "asymptomatic",
                         exposure_source="external" if not candidate.parent_event_ids else attrs["contact_setting_kind"])
            if candidate.parent_event_ids:
                attrs["transmission_hazard_per_day"] = self._force(state, state["contact_by_id"][attrs["contact_relation_id"]])
        elif kind == SYMPTOMS:
            attrs["symptom_course"] = local["course"]
        elif kind == SAMPLE:
            probability, draw = self._assay_probability(local["disease"]), rng.random()
            attrs.update(specimen_event_id=event_id, sampled_at_offset_seconds=now,
                         disease_state_at_sample=local["disease"], assay_positive_probability=probability,
                         assay_uniform_draw=draw, assay_positive=draw < probability)
        elif kind == REPORTED:
            eligible = {}
            for contact in state["contacts"].get(person, []):
                target = contact.target_entity_id
                if target not in eligible and state["local"][target]["notified"] is None:
                    eligible[target] = contact.relation_id
            chosen = {target: contact_id for target, contact_id in eligible.items()
                      if rng.random() < self.config["trace_coverage"]}
            attrs.update(notification_roster=chosen, selected_cohort_size=len(chosen),
                         case_definition="lab_positive_report_not_proof_of_infection")
        elif kind == NOTIFIED:
            draw = rng.random()
            attrs.update(notification_basis="precautionary_cohort_roster", confirmed_exposure=False,
                         quarantine_uniform_draw=draw, quarantine_adherence=self.config["quarantine_adherence"],
                         accepts_quarantine=draw < self.config["quarantine_adherence"])
        elif kind in (ISOLATED, QUARANTINED):
            attrs["restriction_duration_seconds"] = 7 * DAY
        return EventRecord(event_id, context.episode_id, self.domain_id, kind,
            TemporalExtent(now), list(candidate.participants), attrs,
            event_role="root" if not candidate.parent_event_ids else "derived",
            provenance={"generator": "synthetic_community_seir_surveillance",
                        "mechanism_id": candidate.mechanism_id})

    def observation_plan(self, event, candidate, context):
        del candidate, context
        source = ("offline_synthetic_ground_truth" if event.event_type_id in (EXPOSED, INFECTIOUS, RECOVERED)
                  else "synthetic_lab_report" if event.event_type_id in (POSITIVE, NEGATIVE)
                  else "synthetic_public_health_workflow")
        return ObservationPlan(recording_model_ref=NS + "log_ingestion_delay.prior_v1", source=source)

    def apply_event(self, event, candidate, context):
        del candidate
        state, kind, attrs = context.state, event.event_type_id, event.attributes
        local = state["local"][attrs["person_id"]]
        if kind == EXPOSED:
            local.update(disease="E", exposure=event.event_id, course=attrs["symptom_course"])
        elif kind == INFECTIOUS:
            local.update(disease="I", infectious=event.event_id)
        elif kind == RECOVERED:
            local["disease"] = "R"
        elif kind == SYMPTOMS:
            local["symptoms"] = event.event_id
        elif kind == SAMPLE:
            local["samples"].add(attrs["sample_reason"])
        elif kind in (POSITIVE, NEGATIVE):
            state["result_specimens"].add(attrs["specimen_event_id"])
        elif kind == REPORTED:
            local["reported"] = event.event_id
        elif kind == NOTIFIED:
            local["notified"] = event.event_id
        elif kind in (QUARANTINED, DECLINED):
            local["quarantine_decision"] = event.event_id
            if kind == QUARANTINED:
                local.update(quarantined=True, quarantine_event=event.event_id, epoch=local["epoch"] + 1)
        elif kind == ISOLATED:
            local.update(isolated=True, isolation_event=event.event_id, epoch=local["epoch"] + 1)
        elif kind == ISOLATION_END:
            local.update(isolated=False, epoch=local["epoch"] + 1)
        elif kind == QUARANTINE_END:
            local.update(quarantined=False, epoch=local["epoch"] + 1)

    def _transmissions(self, context, source, now):
        state = context.state
        local = state["local"][source]
        if local["disease"] != "I" or self._restricted(local):
            return []
        result = []
        for contact in state["contacts"].get(source, []):
            target = contact.target_entity_id
            target_local = state["local"][target]
            signature = (contact.relation_id, local["epoch"], target_local["epoch"])
            if (target_local["disease"] != "S" or self._restricted(target_local) or
                signature in state["scheduled_transmissions"] or self._force(state, contact) <= 0):
                continue
            state["scheduled_transmissions"].add(signature)
            key = contact.attributes["setting_kind"] + "_transmission"
            result.append(self._spec(key, context, target, [local["infectious"]], now, contact=contact))
        return result

    def spawn_candidates(self, event, candidate, context, event_by_id, rng):
        del candidate, rng
        state, kind, attrs = context.state, event.event_type_id, event.attributes
        person, now, parents = attrs["person_id"], event.temporal.anchor("occurrence_start"), [event.event_id]
        result = []
        if kind == EXPOSED:
            result.append(self._spec("latent_progression", context, person, parents, now))
        elif kind == INFECTIOUS:
            result.append(self._spec("infectious_resolution", context, person, parents, now))
            if state["local"][person]["course"] == "symptomatic":
                result.append(self._spec("symptom_onset", context, person, parents, now))
            result.extend(self._transmissions(context, person, now))
        elif kind == SYMPTOMS:
            result.append(self._spec("symptom_sample", context, person, parents, now))
        elif kind == SAMPLE:
            key = "positive_assay" if attrs["assay_positive"] else "negative_assay"
            frozen = {field: attrs[field] for field in ("specimen_event_id", "sampled_at_offset_seconds",
                "disease_state_at_sample", "assay_positive_probability", "assay_uniform_draw", "assay_positive", "sample_reason")}
            result.append(self._spec(key, context, person, parents, now, extra=frozen))
        elif kind == POSITIVE and state["local"][person]["reported"] is None:
            result.append(self._spec("positive_report", context, person, parents, now))
        elif kind == REPORTED:
            result.append(self._spec("case_isolation", context, person, parents, now))
            for target, contact_id in attrs["notification_roster"].items():
                result.append(self._spec("cohort_notification", context, target, parents, now,
                                         contact=state["contact_by_id"][contact_id]))
        elif kind == NOTIFIED:
            result.append(self._spec("tracing_sample", context, person, parents, now))
            result.append(self._spec("quarantine_accept" if attrs["accepts_quarantine"] else "quarantine_decline",
                                     context, person, parents, now))
        elif kind in (ISOLATED, QUARANTINED):
            result.append(self._spec("isolation_release" if kind == ISOLATED else "quarantine_release",
                                     context, person, parents, now))
        elif kind in (ISOLATION_END, QUARANTINE_END):
            result.extend(self._transmissions(context, person, now))
            for contact_type in CONTACT_TYPES:
                for incoming in context.neighbors(person, contact_type, direction="incoming"):
                    result.extend(self._transmissions(context, incoming.neighbor_entity_id, now))
        return result

    def relation_specs(self, event, candidate, context, event_by_id):
        del event, context, event_by_id
        mechanism = self.mechanism_registry.get(candidate.mechanism_id)
        return [RelationSpec(source_event_id=parent, relation_class=mechanism.relation_semantics,
            relation_type_id=mechanism.mechanism_id, rule_id=mechanism.mechanism_id,
            context_evidence=copy.deepcopy(candidate.context_evidence),
            provenance={"parent_basis": "fired_infectious_contact_or_explicit_observation_workflow",
                        "topology_edge_alone_sufficient": False}) for parent in candidate.parent_event_ids]

    def validate_result(self, result):
        """Replay susceptibility, source attribution, sample snapshots and restrictions."""
        errors = []
        entities = {entity.entity_id: entity for entity in result.entities}
        layers = {layer.relation_type_id: layer for layer in self.domain_spec.compile_relation_layers(
            attribute_builders={"household_contact_attributes": lambda *_: {},
                                "workplace_contact_attributes": lambda *_: {}})}
        functional = set(layers) - set(CONTACT_TYPES)
        counts = defaultdict(int)
        for relation in result.context_relations:
            layer = layers.get(relation.relation_type_id)
            source, target = entities.get(relation.source_entity_id), entities.get(relation.target_entity_id)
            if not layer or not source or not target or source.entity_type_id not in layer.source_type_ids or target.entity_type_id not in layer.target_type_ids:
                errors.append(f"Invalid typed public-health relation {relation.relation_id}")
            if relation.relation_type_id in functional:
                counts[(relation.relation_type_id, relation.source_entity_id)] += 1
        if errors:
            return errors
        data = self._graph_data(result.entities, result.context_relations)
        for kind in functional:
            for entity in result.entities:
                if entity.entity_type_id in layers[kind].source_type_ids and counts[(kind, entity.entity_id)] != 1:
                    errors.append(f"Missing/ambiguous public-health binding {entity.entity_id}")
        for contact in data["contact_by_id"].values():
            setting_kind = contact.attributes.get("setting_kind")
            setting = contact.attributes.get("setting_id")
            if (setting_kind not in ("household", "workplace") or
                contact.relation_type_id != NS + setting_kind + "_contact" or
                contact.source_entity_id == contact.target_entity_id or any(
                    data["bindings"].get((NS + f"person_in_{setting_kind}", person)) != setting
                    for person in (contact.source_entity_id, contact.target_entity_id)) or
                not math.isfinite(float(contact.attributes.get("effective_contacts_per_day", 0))) or
                contact.attributes.get("effective_contacts_per_day", 0) <= 0):
                errors.append(f"Invalid shared-setting contact {contact.relation_id}")
        for person in data["person_attributes"]:
            lab = data["bindings"].get((NS + "person_tested_by_lab", person))
            if data["bindings"].get((NS + "lab_reports_to_unit", lab)) != data["bindings"].get((NS + "person_registered_with_unit", person)):
                errors.append(f"Laboratory/unit binding mismatch {person}")
        if errors:
            return errors
        state = self._initial_state(data, result.episode_attributes["scenario_family"], result.episode_id + "_workflow")
        context = EpisodeContext(result.episode_id, result.context_id, self.domain_id, result.start_time,
            result.duration_seconds, result.entities, result.context_relations, state=state)
        candidates = {candidate.fired_event_id: candidate for candidate in result.candidates if candidate.fired_event_id}
        incoming = defaultdict(list)
        for relation in result.event_relations:
            incoming[relation.target_event_id].append(relation)
        seen = {}
        for event in result.events:
            attrs, kind = event.attributes, event.event_type_id
            person = attrs.get("person_id")
            candidate = candidates.get(event.event_id)
            if person not in state["local"] or candidate is None:
                errors.append(f"Unknown public-health event binding {event.event_id}")
                continue
            roles = {participant.role: participant.entity_id for participant in event.participants}
            if roles.get("person") != person:
                errors.append(f"Person participant mismatch {event.event_id}")
            for role, binding in (("lab", "person_tested_by_lab"), ("unit", "person_registered_with_unit")):
                if role in roles and roles[role] != state["bindings"].get((NS + binding, person)):
                    errors.append(f"Workflow participant mismatch {event.event_id}")
            key = candidate.mechanism_id.removeprefix(NS)
            mechanism = self.mechanism_registry.get(candidate.mechanism_id)
            parents = [relation.source_event_id for relation in incoming[event.event_id]]
            if (set(parents) != set(candidate.parent_event_ids) or
                event.provenance.get("mechanism_id") != candidate.mechanism_id or
                mechanism.target_event_type_id != kind or
                (not parents) != (mechanism.parent_combination == "none")):
                errors.append(f"Event mechanism/parent mismatch {event.event_id}")
            for parent_id in parents:
                parent = seen.get(parent_id)
                expected_person = attrs.get("source_person_id", person) if key.endswith("transmission") or key == "cohort_notification" else person
                if (not parent or parent.event_type_id not in mechanism.source_event_type_ids or
                    parent.attributes.get("person_id") != expected_person):
                    errors.append(f"Invalid source attribution {event.event_id}")
            for field in ("person_id", "source_person_id", "contact_relation_id", "source_epoch", "target_epoch", "sample_reason", "specimen_event_id"):
                if field in candidate.attributes and candidate.attributes[field] != attrs.get(field):
                    errors.append(f"Candidate/event identity mismatch {event.event_id}")
            now = event.temporal.anchor("occurrence_start")
            decision = self.revalidate_candidate(candidate, context, seen, now)
            if not decision.valid:
                errors.append(f"Invalid replay precondition {event.event_id}: {decision.reason}")
            local = state["local"][person]
            expected_after = {EXPOSED: "E", INFECTIOUS: "I", RECOVERED: "R"}.get(kind, local["disease"])
            if (attrs.get("disease_state_before") != local["disease"] or attrs.get("disease_state_after") != expected_after or
                attrs.get("isolated_before") != local["isolated"] or attrs.get("quarantined_before") != local["quarantined"]):
                errors.append(f"Disease/restriction replay mismatch {event.event_id}")
            if kind == EXPOSED:
                draw = attrs.get("symptom_uniform_draw", -1)
                expected_course = "symptomatic" if draw < self.config["symptom_probability"] else "asymptomatic"
                if not 0 <= draw < 1 or attrs.get("symptom_course") != expected_course or attrs.get("symptom_probability") != self.config["symptom_probability"]:
                    errors.append(f"Invalid symptom-course mark {event.event_id}")
                if key.endswith("transmission"):
                    contact = state["contact_by_id"][attrs["contact_relation_id"]]
                    if roles.get("source_person") != contact.source_entity_id or roles.get(contact.attributes["setting_kind"]) != contact.attributes["setting_id"]:
                        errors.append(f"Transmission setting/source mismatch {event.event_id}")
                    if abs(attrs.get("transmission_hazard_per_day", -1) - self._force(state, contact)) > 1e-10:
                        errors.append(f"Transmission hazard mismatch {event.event_id}")
            elif kind == SAMPLE:
                probability = self._assay_probability(local["disease"])
                draw = attrs.get("assay_uniform_draw", -1)
                if (attrs.get("disease_state_at_sample") != local["disease"] or
                    attrs.get("sampled_at_offset_seconds") != now or attrs.get("specimen_event_id") != event.event_id or
                    attrs.get("assay_positive_probability") != probability or not 0 <= draw < 1 or
                    attrs.get("assay_positive") != (draw < probability)):
                    errors.append(f"Assay sample-state mismatch {event.event_id}")
            elif kind in (POSITIVE, NEGATIVE):
                sample = seen.get(attrs.get("specimen_event_id"))
                frozen = ("specimen_event_id", "sampled_at_offset_seconds", "disease_state_at_sample",
                          "assay_positive_probability", "assay_uniform_draw", "assay_positive", "sample_reason")
                if not sample or any(attrs.get(field) != sample.attributes.get(field) for field in frozen) or attrs.get("assay_positive") != (kind == POSITIVE):
                    errors.append(f"Assay result changed frozen specimen {event.event_id}")
            elif kind == REPORTED:
                roster = attrs.get("notification_roster", {})
                if (not isinstance(roster, dict) or attrs.get("selected_cohort_size") != len(roster) or
                    attrs.get("case_definition") != "lab_positive_report_not_proof_of_infection"):
                    errors.append(f"Invalid cohort dispatch roster {event.event_id}")
                else:
                    for target, contact_id in roster.items():
                        contact = state["contact_by_id"].get(contact_id)
                        if (not contact or contact.source_entity_id != person or
                            contact.target_entity_id != target or target not in state["local"] or
                            state["local"][target]["notified"] is not None):
                            errors.append(f"Ungrounded cohort dispatch target {event.event_id}")
            elif kind == NOTIFIED:
                source = attrs.get("source_person_id")
                report = seen.get(parents[0]) if parents else None
                if (not report or report.attributes.get("notification_roster", {}).get(person) != attrs.get("contact_relation_id") or
                    roles.get("source_person") != source or attrs.get("confirmed_exposure") is not False):
                    errors.append(f"Unsupported cohort notification {event.event_id}")
                draw = attrs.get("quarantine_uniform_draw", -1)
                if (not 0 <= draw < 1 or attrs.get("quarantine_adherence") != self.config["quarantine_adherence"] or
                    attrs.get("accepts_quarantine") != (draw < self.config["quarantine_adherence"])):
                    errors.append(f"Invalid quarantine acceptance mark {event.event_id}")
            elif kind in (QUARANTINED, DECLINED):
                notice = seen.get(parents[0]) if parents else None
                if not notice or notice.attributes.get("accepts_quarantine") != (kind == QUARANTINED):
                    errors.append(f"Quarantine response disagrees with notice mark {event.event_id}")
            elif kind in (ISOLATION_END, QUARANTINE_END):
                start = seen.get(parents[0]) if parents else None
                if not start or abs(now - start.temporal.anchor("occurrence_start") - 7 * DAY) > 1e-6:
                    errors.append(f"Restriction release schedule mismatch {event.event_id}")
            self.apply_event(event, candidate, context)
            seen[event.event_id] = event
        return errors

    def render_episode_text(self, result):
        sentences, claims = [], []
        common = ("person_id", "source_person_id", "contact_setting_kind", "exposure_source",
                  "symptom_course", "transmission_hazard_per_day", "sample_reason",
                  "specimen_event_id", "disease_state_at_sample", "assay_positive",
                  "sampled_at_offset_seconds", "case_definition", "selected_cohort_size",
                  "notification_basis", "confirmed_exposure", "accepts_quarantine",
                  "restriction_duration_seconds")
        for event in result.events:
            selected = common + (("disease_state_before", "disease_state_after")
                                 if event.event_type_id in (EXPOSED, INFECTIOUS, RECOVERED) else ())
            attrs = {field: event.attributes[field] for field in selected if field in event.attributes}
            roles = ", ".join(f"{participant.role}={participant.entity_id}" for participant in event.participants)
            sentences.append(f"+{event.temporal.anchor('occurrence_start') / DAY:.6f} day "
                f"{event.event_id} {event.event_type_id} ({roles}); "
                f"{json.dumps(attrs, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}.")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                "event_ids": [event.event_id], "relation_ids": [],
                "evidence_fields": ["event_type_id", "temporal", "participants"] +
                                   [f"attributes.{field}" for field in attrs]})
        for relation in result.event_relations:
            predicate = PREDICATES[relation.relation_type_id.removeprefix(NS)]
            sentences.append(f"{relation.source_event_id} {predicate} {relation.target_event_id} "
                f"({relation.relation_class}, +{relation.temporal_link.lag_seconds / DAY:.6f} day; rule={relation.rule_id}).")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                "event_ids": [relation.source_event_id, relation.target_event_id],
                "relation_ids": [relation.relation_id],
                "relation_assertions": [{"relation_id": relation.relation_id,
                    "asserted_relation_class": relation.relation_class, "surface_predicate": predicate}],
                "evidence_fields": ["rule_id", "source_event_id", "target_event_id", "temporal_link"]})
        return {"alignment_id": result.episode_id + "_text_alignment", "episode_id": result.episode_id,
                "domain": self.domain_id, "projection_mode": "deterministic_grounded_projection",
                "sentences": sentences, "text": " ".join(sentences), "claims": claims}


def build_public_health_temporal_models():
    registry = build_standard_temporal_registry()
    models = [
        ExponentialArrivalModel(NS + "external_arrival.prior_v1", 2.0 / DAY),
        PiecewiseExponentialHazardModel(NS + "transmission_hazard.prior_v1", 1.0 / DAY, {"log_contact_force": 1.0}),
        DeterministicDelayModel(NS + "restriction_duration.policy_v1", 7 * DAY),
    ]
    for name, median_days, sigma, minimum_days in (
        ("latent_period", 2.0, .35, .1), ("symptom_delay", .7, .5, .02),
        ("infectious_duration", 7.0, .3, .2), ("collection_delay", .4, .4, .01),
        ("lab_turnaround", .7, .4, .02), ("reporting_delay", .15, .4, .005),
        ("response_delay", .2, .4, .005), ("notification_delay", .8, .5, .01),
        ("log_ingestion_delay", .02, .4, .0001)):
        models.append(ConditionalLogNormalModel(NS + name + ".prior_v1", math.log(median_days * DAY),
                                               sigma, {}, minimum_seconds=minimum_days * DAY))
    for model in models:
        registry.register(model)
    return registry
