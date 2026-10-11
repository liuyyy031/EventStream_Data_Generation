"""Conditional, synthetic patch-food-web events; not a calibrated ecosystem model.

Context links specify ecological eligibility, not realized event parenthood.  A
regional disturbance, local resource state and risk clock must all agree before
a population response is emitted.  Numbers are declared synthetic priors.
"""
from __future__ import annotations

import copy
import json
import math
import random
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
    ConditionalLogNormalModel, ExponentialArrivalModel, PiecewiseExponentialHazardModel,
    build_standard_temporal_registry,
)
from data_generation.generation_core.topology import SparseHeterogeneousTopologyGenerator

NS = "ecology."
DAY = 86400.0
DROUGHT, DROUGHT_END = NS + "climate.drought.started", NS + "climate.drought.ended"
LOW, RESTORED = NS + "resource.availability.decreased", NS + "resource.availability.restored"
HERB_LOW, PRED_LOW = NS + "population.herbivore.decreased", NS + "population.predator.decreased"
HERB_RESTORED, PRED_RESTORED = NS + "population.herbivore.restored", NS + "population.predator.restored"
PULSE, PULSE_END = NS + "resource.pulse.started", NS + "resource.pulse.ended"
HIGH, NORMAL = NS + "resource.availability.increased", NS + "resource.availability.normalized"
HERB_HIGH, PRED_HIGH = NS + "population.herbivore.increased", NS + "population.predator.increased"
ARRIVED, ESTABLISHED, FAILED = (NS + "population.invader." + name for name in
                                ("propagule_arrived", "established", "establishment_failed"))
EVENT_TYPES = (DROUGHT, DROUGHT_END, LOW, RESTORED, HERB_LOW, PRED_LOW,
               HERB_RESTORED, PRED_RESTORED, PULSE, PULSE_END, HIGH, NORMAL,
               HERB_HIGH, PRED_HIGH, ARRIVED, ESTABLISHED, FAILED)
SCENARIOS = ("regional_drought", "resource_pulse", "invasive_colonization")
PREDICATES = {
    "drought_end": "ended at", "drought_resource_loss": "reduced local resource availability at",
    "resource_recovery": "allowed local resource availability to recover at",
    "herbivore_decline": "was followed, conditional on resource deficit, by herbivore decline at",
    "predator_decline": "was followed, conditional on prey deficit, by predator decline at",
    "herbivore_recovery": "was followed, conditional on food availability, by herbivore recovery at",
    "predator_recovery": "was followed, conditional on prey recovery, by predator recovery at",
    "pulse_end": "ended at", "pulse_resource_gain": "increased local resource availability at",
    "pulse_resource_normalize": "allowed local resource availability to normalize at",
    "herbivore_increase": "was followed, conditional on resource surplus, by herbivore increase at",
    "predator_increase": "was followed, conditional on prey surplus, by predator increase at",
    "propagule_dispersal": "dispersed propagules through an eligible corridor to",
    "invasion_established": "was followed by local establishment at",
    "invasion_failed": "was followed by failed local establishment at",
    "invasion_resource_loss": "reduced resource availability per native consumer at",
}


class EcologyPackage:
    domain_id = "ecology"

    def __init__(self, config=None):
        self.config = {
            "base_seed": 20261002, "context_count": 4, "nodes_per_context": 1000,
            "duration_seconds": 180 * DAY, "start_time": "2026-10-02T00:00:00+08:00",
            "scenario_weights": dict(zip(SCENARIOS, (.35, .30, .35))), **(config or {}),
        }
        for key in ("base_seed", "context_count", "nodes_per_context"):
            value = self.config[key]
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{key} must be an integer")
            self.config[key] = int(value)
        if self.config["context_count"] < 1 or self.config["nodes_per_context"] < 20:
            raise ValueError("Ecology needs at least one context and 20 entities")
        duration = float(self.config["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        self.config["duration_seconds"] = duration
        weights = self.config["scenario_weights"]
        if not weights or set(weights) - set(SCENARIOS) or any(
            not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()
        ) or sum(weights.values()) <= 0:
            raise ValueError("Invalid ecology scenario weights")
        path = Path(__file__)
        self.domain_spec = load_domain_spec(path.with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(
            path.with_name("mechanism_specs.json"),
            known_event_type_ids=EVENT_TYPES,
            known_context_relation_type_ids=self.domain_spec.context_relation_type_ids,
        )
        self._context_cache = {}

    def catalog(self):
        roles = {name: NS + kind for name, kind in (
            ("region", "ecoregion"), ("patch", "habitat_patch"),
            ("source_patch", "habitat_patch"), ("resource", "resource_pool"),
            ("herbivore", "population"), ("predator", "population"),
            ("invader", "population"), ("source_invader", "population"),
            ("corridor", "corridor"))}
        return {
            "domain": self.domain_id, "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_not_empirically_fitted",
            "scope": {"ecosystem": "terrestrial_patch_resource_consumer_predator_only",
                      "dynamics": "conditional_thresholds_and_competing_risk_clocks_not_population_ode",
                      "dispersal": "undirected_synthetic_patch_corridors",
                      "observation": "offline_synthetic_ground_truth_with_later_log_ingestion",
                      "statistical_realism_claimed": False},
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {"mode": "deterministic_grounded_projection",
                                "free_form_causal_completion": False},
            "entity_types": [{"entity_type_id": NS + kind, "parent_type_id": None,
                              "required_attributes": attrs} for kind, attrs in (
                ("ecoregion", ["synthetic_climate_zone"]),
                ("habitat_patch", ["area_km2", "habitat_quality", "invader_growth_potential_ratio",
                                   "drought_resource_factor", "pulse_resource_factor"]),
                ("resource_pool", ["baseline_biomass_kg", "carrying_capacity_kg"]),
                ("population", ["guild", "baseline_abundance"]),
                ("corridor", ["distance_km", "permeability"]))],
            "event_types": [{"event_type_id": kind, "parent_type_id": kind.rsplit(".", 1)[0],
                             "required_attributes": ["workflow_id", "scenario_family"],
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

    def _build_context(self, index):
        cfg = self.config
        rng = random.Random(cfg["base_seed"] + 7919 * index)
        context_id = f"ecology_context_{index:03d}"
        n = cfg["nodes_per_context"]
        region_count = max(2, n // 100)
        patch_count = (n - region_count + 1) // 6
        corridor_count = patch_count - 1
        extra_resources = n - region_count - 5 * patch_count - corridor_count
        assert patch_count >= 3 and extra_resources >= 0
        ids = {kind: [f"{context_id}_{kind}_{i:05d}" for i in range(count)]
               for kind, count in (("region", region_count), ("patch", patch_count),
                    ("resource", patch_count + extra_resources), ("herbivore", patch_count),
                    ("predator", patch_count), ("invader", patch_count),
                    ("corridor", corridor_count))}
        region_of = {ids["patch"][i]: ids["region"][i % region_count] for i in range(patch_count)}
        patch_info = {}
        entities = []
        for region in ids["region"]:
            entities.append(Entity(region, self.domain_id, NS + "ecoregion",
                                   {"synthetic_climate_zone": "temperate"}, context_id))
        for i, patch in enumerate(ids["patch"]):
            herb = rng.randint(20, 60)
            pred = rng.randint(5, 15)
            biomass = round(rng.uniform(3.2, 5.8) * herb, 6)
            quality = round(rng.uniform(.45, .95), 6)
            growth_potential = round(rng.uniform(.15, 1.3), 6)
            drought_factor = round(rng.uniform(.35, .58), 6)
            pulse_factor = round(rng.uniform(1.4, 1.9), 6)
            patch_info[patch] = {"region": region_of[patch], "resource": ids["resource"][i],
                                 "herbivore": ids["herbivore"][i], "predator": ids["predator"][i],
                                 "invader": ids["invader"][i], "baseline_herb": herb,
                                 "baseline_pred": pred, "baseline_biomass": biomass,
                                 "quality": quality, "invader_growth_potential_ratio": growth_potential,
                                 "drought_factor": drought_factor,
                                 "pulse_factor": pulse_factor}
            entities.append(Entity(patch, self.domain_id, NS + "habitat_patch",
                                   {"area_km2": round(rng.uniform(.5, 5.0), 6),
                                    "habitat_quality": quality,
                                    "invader_growth_potential_ratio": growth_potential,
                                    "drought_resource_factor": drought_factor,
                                    "pulse_resource_factor": pulse_factor}, context_id))
            entities.append(Entity(ids["resource"][i], self.domain_id, NS + "resource_pool",
                                   {"baseline_biomass_kg": biomass,
                                    "carrying_capacity_kg": round(2.5 * biomass, 6)}, context_id))
            for kind, baseline in (("herbivore", herb), ("predator", pred), ("invader", 0)):
                entities.append(Entity(ids[kind][i], self.domain_id, NS + "population",
                                       {"guild": kind, "baseline_abundance": baseline}, context_id))
        for i in range(extra_resources):
            entities.append(Entity(ids["resource"][patch_count + i], self.domain_id,
                                   NS + "resource_pool",
                                   {"baseline_biomass_kg": 25.0,
                                    "carrying_capacity_kg": 62.5}, context_id))
        corridor_ends, adjacent = {}, {patch: [] for patch in ids["patch"]}
        for i, corridor in enumerate(ids["corridor"], start=1):
            parent_index = rng.randrange(max(0, i - 6), i)
            source, target = ids["patch"][parent_index], ids["patch"][i]
            corridor_ends[corridor] = (source, target)
            distance = round(rng.uniform(.5, 8.0), 6)
            permeability = round(rng.uniform(.15, .95), 6)
            entities.append(Entity(corridor, self.domain_id, NS + "corridor",
                                   {"distance_km": distance, "permeability": permeability}, context_id))
            adjacent[source].append((target, corridor))
            adjacent[target].append((source, corridor))
        required = {
            NS + "patch_in_region": [(patch, region_of[patch]) for patch in ids["patch"]],
            NS + "resource_in_patch": [(resource, ids["patch"][i % patch_count])
                                         for i, resource in enumerate(ids["resource"])],
            NS + "population_in_patch": [(ids[kind][i], patch)
                                           for i, patch in enumerate(ids["patch"])
                                           for kind in ("herbivore", "predator", "invader")],
            NS + "herbivore_consumes_resource": list(zip(ids["herbivore"], ids["resource"])),
            NS + "invader_uses_resource": list(zip(ids["invader"], ids["resource"])),
            NS + "predator_preys_on_herbivore": list(zip(ids["predator"], ids["herbivore"])),
            NS + "invader_competes_with_herbivore": list(zip(ids["invader"], ids["herbivore"])),
            NS + "corridor_from_patch": [(corridor, ends[0]) for corridor, ends in corridor_ends.items()],
            NS + "corridor_to_patch": [(corridor, ends[1]) for corridor, ends in corridor_ends.items()],
        }
        layers = [replace(layer, required_pairs=tuple(required[layer.relation_type_id]))
                  for layer in self.domain_spec.compile_relation_layers()]
        topology = SparseHeterogeneousTopologyGenerator().generate(
            context_id=context_id, domain=self.domain_id, entities=entities, layers=layers,
            seed=cfg["base_seed"] + 7919 * index,
            observation_window_seconds=cfg["duration_seconds"])
        return {"entities": entities, "relations": topology.relations,
                "statistics": topology.statistics, "ids": ids, "patch_info": patch_info,
                "adjacent": adjacent, "corridor_ends": corridor_ends}

    def create_episode_context(self, episode_index, seed):
        index = episode_index % self.config["context_count"]
        if index not in self._context_cache:
            self._context_cache[index] = self._build_context(index)
        data = self._context_cache[index]
        rng = random.Random(seed ^ 0x45434f4c4f4759)
        weights = self.config["scenario_weights"]
        scenario = rng.choices(list(weights), weights=list(weights.values()))[0]
        selected_patch = rng.choice(data["ids"]["patch"])
        selected_region = data["patch_info"][selected_patch]["region"]
        local = {patch: {"factor": 1.0, "herb": info["baseline_herb"],
                          "pred": info["baseline_pred"], "invader": 0,
                          "arrival": None, "establishment": None,
                          "resource_low": None, "resource_high": None,
                          "herb_low": False, "pred_low": False,
                          "herb_high": False, "pred_high": False}
                 for patch, info in data["patch_info"].items()}
        episode_id = f"ecology_episode_{episode_index:07d}"
        state = {"scenario": scenario, "workflow_id": episode_id + "_workflow",
                 "region": selected_region, "origin_patch": selected_patch,
                 "patch_info": data["patch_info"], "adjacent": data["adjacent"],
                 "local": local, "disturbance_active": False,
                 "root_started": False, "root_ended": False}
        return EpisodeContext(
            episode_id=episode_id, context_id=f"ecology_context_{index:03d}",
            domain=self.domain_id, start_time=self.config["start_time"],
            duration_seconds=self.config["duration_seconds"],
            entities=data["entities"], context_relations=data["relations"],
            context_attributes={"topology_profile": data["statistics"],
                "scope": "one_terrestrial_patch_food_web; synthetic_priors_not_calibrated"},
            episode_attributes={"scenario_family": scenario, "empirical_realism_claimed": False,
                "event_parenthood": "typed_binding_plus_state_precondition_plus_fired_clock",
                "resource_unit": "kg_synthetic_biomass_per_effective_consumer"},
            state=state)

    @staticmethod
    def _availability(state, patch):
        info, local = state["patch_info"][patch], state["local"][patch]
        return info["baseline_biomass"] * local["factor"] / max(
            1.0, local["herb"] + 1.2 * local["invader"])

    @staticmethod
    def _competition_after(state, patch):
        info, local = state["patch_info"][patch], state["local"][patch]
        invader_after = round(info["invader_growth_potential_ratio"] * info["baseline_herb"])
        availability_after = info["baseline_biomass"] * local["factor"] / max(
            1.0, local["herb"] + 1.2 * invader_after)
        return invader_after, availability_after

    def _spec(self, key, context, parents=(), now=0.0, *, patch=None,
              source_patch=None, corridor=None, inputs=None):
        state = context.state
        mechanism = self.mechanism_registry.get(NS + key)
        patch = patch or state["origin_patch"]
        info = state["patch_info"][patch]
        roles, pairs = {}, set()
        if key in ("drought_start", "drought_end", "pulse_start", "pulse_end"):
            roles["region"] = state["region"]
        elif key in ("drought_resource_loss", "pulse_resource_gain"):
            roles.update(region=info["region"], patch=patch, resource=info["resource"])
            pairs.update(((NS + "patch_in_region", patch, info["region"]),
                          (NS + "resource_in_patch", info["resource"], patch)))
        else:
            roles["patch"] = patch
            if key in ("resource_recovery", "pulse_resource_normalize",
                       "herbivore_decline", "herbivore_recovery", "herbivore_increase",
                       "invasion_resource_loss"):
                roles["resource"] = info["resource"]
                pairs.add((NS + "resource_in_patch", info["resource"], patch))
            if key in ("herbivore_decline", "herbivore_recovery", "herbivore_increase",
                       "predator_decline", "predator_recovery", "predator_increase",
                       "invasion_resource_loss"):
                roles["herbivore"] = info["herbivore"]
                pairs.add((NS + "population_in_patch", info["herbivore"], patch))
            if key in ("herbivore_decline", "herbivore_recovery", "herbivore_increase",
                       "invasion_resource_loss"):
                pairs.add((NS + "herbivore_consumes_resource", info["herbivore"], info["resource"]))
            if key in ("predator_decline", "predator_recovery", "predator_increase"):
                roles["predator"] = info["predator"]
                pairs.update(((NS + "population_in_patch", info["predator"], patch),
                              (NS + "predator_preys_on_herbivore", info["predator"], info["herbivore"])))
            if key in ("propagule_root", "propagule_dispersal", "invasion_established",
                       "invasion_failed", "invasion_resource_loss"):
                roles["invader"] = info["invader"]
                pairs.add((NS + "population_in_patch", info["invader"], patch))
            if key == "invasion_resource_loss":
                pairs.update(((NS + "invader_uses_resource", info["invader"], info["resource"]),
                              (NS + "invader_competes_with_herbivore", info["invader"], info["herbivore"])))
            if key == "propagule_dispersal":
                source_info = state["patch_info"][source_patch]
                roles.update(source_patch=source_patch, source_invader=source_info["invader"],
                             corridor=corridor)
                ends = next((entry for entry in context.context_relations if
                    entry.relation_type_id == NS + "corridor_from_patch" and
                    entry.source_entity_id == corridor), None)
                if ends is None:
                    raise ValueError("Missing corridor endpoint")
                first = ends.target_entity_id
                second = patch if first == source_patch else source_patch
                pairs.update(((NS + "corridor_from_patch", corridor, first),
                              (NS + "corridor_to_patch", corridor, second),
                              (NS + "population_in_patch", source_info["invader"], source_patch)))
        matched = [relation.relation_id for relation in context.context_relations
                   if (relation.relation_type_id, relation.source_entity_id,
                       relation.target_entity_id) in pairs]
        if len(matched) != len(pairs):
            raise ValueError(f"Ecology candidate {key} lacks a unique typed binding")
        attrs = {"workflow_id": state["workflow_id"], "scenario_family": state["scenario"],
                 "patch_id": patch, "region_id": info["region"]}
        if source_patch:
            attrs.update(source_patch_id=source_patch, corridor_id=corridor)
        if key == "propagule_dispersal":
            corridor_entity = context.entity_by_id[corridor]
            inputs = {"corridor_permeability": corridor_entity.attributes["permeability"],
                      "distance_km": corridor_entity.attributes["distance_km"]}
        if key in ("invasion_established", "invasion_failed"):
            inputs = {"habitat_quality": info["quality"]}
        if key.startswith("herbivore_"):
            availability = self._availability(state, patch)
            inputs = {"resource_deficit": max(0.0, 2.5 - availability),
                      "resource_surplus": max(0.0, availability - 4.0)}
        if key.startswith("predator_"):
            prey_ratio = state["local"][patch]["herb"] / info["baseline_herb"]
            inputs = {"prey_deficit": max(0.0, .8 - prey_ratio),
                      "prey_surplus": max(0.0, prey_ratio - 1.0)}
        return CandidateSpec(
            mechanism_id=mechanism.mechanism_id,
            target_event_type_id=mechanism.target_event_type_id,
            parent_event_ids=list(parents),
            participants=[EventParticipant(entity, role) for role, entity in roles.items()],
            temporal_model_ref=mechanism.generation_temporal_model_ref,
            temporal_inputs=inputs or {}, activation_time=now,
            combination=mechanism.parent_combination, phase_priority=10,
            attributes=attrs,
            context_evidence=ContextEvidence(context_relation_ids=matched,
                entity_ids=list(dict.fromkeys(roles.values())),
                evaluated_at_offset_seconds=now),
            provenance={"input_anchor": "occurrence_start",
                        "parameter_status": mechanism.parameter_status})

    def seed_candidates(self, context, rng):
        del rng
        key = {"regional_drought": "drought_start", "resource_pulse": "pulse_start",
               "invasive_colonization": "propagule_root"}[context.state["scenario"]]
        return [self._spec(key, context)]

    def revalidate_candidate(self, candidate, context, event_by_id, now):
        del event_by_id
        key = candidate.mechanism_id.removeprefix(NS)
        state, patch = context.state, candidate.attributes["patch_id"]
        local = state["local"][patch]
        info = state["patch_info"][patch]
        availability = self._availability(state, patch)
        checks = {
            "drought_start": (not state["root_started"], "root_not_started", state["root_started"]),
            "pulse_start": (not state["root_started"], "root_not_started", state["root_started"]),
            "drought_end": (state["disturbance_active"], "drought_active", state["disturbance_active"]),
            "pulse_end": (state["disturbance_active"], "pulse_active", state["disturbance_active"]),
            "drought_resource_loss": (state["disturbance_active"] and local["factor"] == 1.0,
                                      "active_drought_and_unaffected_patch", local["factor"]),
            "pulse_resource_gain": (state["disturbance_active"] and local["factor"] == 1.0,
                                    "active_pulse_and_unaffected_patch", local["factor"]),
            "resource_recovery": (not state["disturbance_active"] and local["factor"] < 1.0,
                                  "drought_ended_and_resource_reduced", local["factor"]),
            "pulse_resource_normalize": (not state["disturbance_active"] and local["factor"] > 1.0,
                                         "pulse_ended_and_resource_elevated", local["factor"]),
            "herbivore_decline": (availability < 2.5 and not local["herb_low"],
                                  "resource_per_consumer_below_2_5", availability),
            "predator_decline": (local["herb"] < .8 * info["baseline_herb"] and not local["pred_low"],
                                 "prey_below_80_percent_baseline", local["herb"]),
            "herbivore_recovery": (local["herb_low"] and availability >= 2.5,
                                   "food_available_after_decline", availability),
            "predator_recovery": (local["pred_low"] and local["herb"] >= .8 * info["baseline_herb"],
                                  "prey_recovered", local["herb"]),
            "herbivore_increase": (availability > 5.0 and not local["herb_high"],
                                   "resource_per_consumer_above_5", availability),
            "predator_increase": (local["herb"] > 1.2 * info["baseline_herb"] and not local["pred_high"],
                                  "prey_above_120_percent_baseline", local["herb"]),
            "propagule_root": (local["arrival"] is None, "no_prior_propagule", local["arrival"]),
            "propagule_dispersal": (local["arrival"] is None and
                                    state["local"][candidate.attributes.get("source_patch_id", patch)]["establishment"] == "established",
                                    "source_established_and_target_unvisited", local["arrival"]),
            "invasion_established": (local["arrival"] is not None and local["establishment"] is None,
                                     "arrived_and_unresolved", local["establishment"]),
            "invasion_failed": (local["arrival"] is not None and local["establishment"] is None,
                                "arrived_and_unresolved", local["establishment"]),
            "invasion_resource_loss": (local["establishment"] == "established" and
                                       self._competition_after(state, patch)[1] < 2.5 and
                                       local["resource_low"] is None,
                                       "growing_competitor_reduces_per_capita_resource",
                                       self._competition_after(state, patch)[1]),
        }
        valid, name, actual = checks[key]
        evidence = copy.deepcopy(candidate.context_evidence)
        evidence.evaluated_at_offset_seconds = now
        evidence.state_predicates.append(StatePredicateEvidence(
            patch, name, "is_true", True, valid, valid))
        return CandidateUpdate(valid, None if valid else name + "_no_longer_satisfied",
                               temporal_inputs=self._updated_inputs(key, state, patch, candidate),
                               context_evidence=evidence)

    def _updated_inputs(self, key, state, patch, candidate):
        if key.startswith("herbivore_"):
            availability = self._availability(state, patch)
            return {"resource_deficit": max(0.0, 2.5 - availability),
                    "resource_surplus": max(0.0, availability - 4.0)}
        if key.startswith("predator_"):
            ratio = state["local"][patch]["herb"] / state["patch_info"][patch]["baseline_herb"]
            return {"prey_deficit": max(0.0, .8 - ratio),
                    "prey_surplus": max(0.0, ratio - 1.0)}
        return dict(candidate.temporal_inputs)

    def materialize_event(self, event_id, candidate, context, rng):
        del rng
        state = context.state
        kind = candidate.target_event_type_id
        patch = candidate.attributes["patch_id"]
        local, info = state["local"][patch], state["patch_info"][patch]
        attrs = copy.deepcopy(candidate.attributes)
        attrs["availability_before_kg_per_effective_consumer"] = round(self._availability(state, patch), 6)
        if kind in (LOW, RESTORED, HIGH, NORMAL):
            factor_after = local["factor"]
            if kind == LOW and candidate.mechanism_id.endswith("drought_resource_loss"):
                factor_after = info["drought_factor"]
            elif kind in (RESTORED, NORMAL):
                factor_after = 1.0
            elif kind == HIGH:
                factor_after = info["pulse_factor"]
            attrs["resource_factor_before"] = local["factor"]
            attrs["resource_factor_after"] = factor_after
            invader_after = local["invader"]
            if candidate.mechanism_id.endswith("invasion_resource_loss"):
                invader_after = self._competition_after(state, patch)[0]
                attrs.update(invader_abundance_before=local["invader"],
                             invader_abundance_after=invader_after)
            attrs["availability_after_kg_per_effective_consumer"] = round(
                info["baseline_biomass"] * factor_after /
                max(1.0, local["herb"] + 1.2 * invader_after), 6)
            attrs["driver"] = ("invader_competition" if candidate.mechanism_id.endswith("invasion_resource_loss")
                               else "regional_drought" if kind in (LOW, RESTORED)
                               else "regional_resource_pulse")
        elif kind in (HERB_LOW, HERB_RESTORED, HERB_HIGH, PRED_LOW, PRED_RESTORED, PRED_HIGH):
            guild = "herb" if "herbivore" in kind else "pred"
            before = local[guild]
            target = info["baseline_herb" if guild == "herb" else "baseline_pred"]
            after = (max(1, round(.65 * before)) if kind in (HERB_LOW, PRED_LOW)
                     else target if kind in (HERB_RESTORED, PRED_RESTORED)
                     else round(1.4 * before))
            attrs.update(population_guild="herbivore" if guild == "herb" else "predator",
                         abundance_before=before, abundance_after=after,
                         response_threshold=(2.5 if kind == HERB_LOW else 5.0 if kind == HERB_HIGH
                                             else .8 if kind in (PRED_LOW, PRED_RESTORED)
                                             else 1.2 if kind == PRED_HIGH else 2.5))
        elif kind == ESTABLISHED:
            attrs.update(invader_abundance_before=local["invader"],
                         invader_abundance_after=max(1, round(.2 * info["baseline_herb"])),
                         habitat_quality=info["quality"])
        elif kind == FAILED:
            attrs.update(invader_abundance_before=local["invader"],
                         invader_abundance_after=local["invader"],
                         habitat_quality=info["quality"])
        return EventRecord(event_id, context.episode_id, self.domain_id, kind,
                           TemporalExtent(candidate.scheduled_time), list(candidate.participants), attrs,
                           event_role="root" if not candidate.parent_event_ids else "derived",
                           provenance={"generator": "synthetic_patch_food_web",
                                       "mechanism_id": candidate.mechanism_id})

    def observation_plan(self, event, candidate, context):
        del event, candidate, context
        return ObservationPlan(recording_model_ref=NS + "log_ingestion_delay.prior_v1",
                               source="offline_synthetic_ground_truth")

    def apply_event(self, event, candidate, context):
        del candidate
        state, kind = context.state, event.event_type_id
        local = state["local"][event.attributes["patch_id"]]
        if kind in (DROUGHT, PULSE):
            state["root_started"] = True
            state["disturbance_active"] = True
        elif kind in (DROUGHT_END, PULSE_END):
            state["root_ended"] = True
            state["disturbance_active"] = False
        elif kind in (LOW, RESTORED, HIGH, NORMAL):
            local["factor"] = event.attributes["resource_factor_after"]
            if event.attributes.get("driver") == "invader_competition":
                local["invader"] = event.attributes["invader_abundance_after"]
            if kind == LOW:
                local["resource_low"] = event.event_id
            elif kind == HIGH:
                local["resource_high"] = event.event_id
            elif kind == RESTORED:
                local["resource_low"] = None
            else:
                local["resource_high"] = None
        elif kind in (HERB_LOW, HERB_RESTORED, HERB_HIGH, PRED_LOW, PRED_RESTORED, PRED_HIGH):
            guild = "herb" if "herbivore" in kind else "pred"
            local[guild] = event.attributes["abundance_after"]
            flag = "herb_low" if kind == HERB_LOW else "pred_low" if kind == PRED_LOW else (
                   "herb_high" if kind == HERB_HIGH else "pred_high" if kind == PRED_HIGH else
                   "herb_low" if kind == HERB_RESTORED else "pred_low")
            local[flag] = kind in (HERB_LOW, PRED_LOW, HERB_HIGH, PRED_HIGH)
        elif kind == ARRIVED:
            local["arrival"] = event.event_id
        elif kind in (ESTABLISHED, FAILED):
            local["establishment"] = "established" if kind == ESTABLISHED else "failed"
            if kind == ESTABLISHED:
                local["invader"] = event.attributes["invader_abundance_after"]

    def spawn_candidates(self, event, candidate, context, event_by_id, rng):
        del candidate, event_by_id, rng
        state, kind = context.state, event.event_type_id
        patch = event.attributes["patch_id"]
        now = event.temporal.anchor("occurrence_start")
        parent = [event.event_id]
        result = []
        if kind in (DROUGHT, PULSE):
            result.append(self._spec("drought_end" if kind == DROUGHT else "pulse_end",
                                     context, parent, now))
            key = "drought_resource_loss" if kind == DROUGHT else "pulse_resource_gain"
            for member, info in state["patch_info"].items():
                if info["region"] == state["region"]:
                    result.append(self._spec(key, context, parent, now, patch=member))
        elif kind in (DROUGHT_END, PULSE_END):
            key = "resource_recovery" if kind == DROUGHT_END else "pulse_resource_normalize"
            for member, local in state["local"].items():
                if state["patch_info"][member]["region"] == state["region"] and (
                    local["factor"] < 1.0 if kind == DROUGHT_END else local["factor"] > 1.0
                ):
                    result.append(self._spec(key, context, parent, now, patch=member))
        elif kind == LOW and self._availability(state, patch) < 2.5:
            result.append(self._spec("herbivore_decline", context, parent, now, patch=patch))
        elif kind == RESTORED and state["local"][patch]["herb_low"]:
            result.append(self._spec("herbivore_recovery", context, parent, now, patch=patch))
        elif kind == HERB_LOW:
            result.append(self._spec("predator_decline", context, parent, now, patch=patch))
        elif kind == HERB_RESTORED and state["local"][patch]["pred_low"]:
            result.append(self._spec("predator_recovery", context, parent, now, patch=patch))
        elif kind == HIGH and self._availability(state, patch) > 5.0:
            result.append(self._spec("herbivore_increase", context, parent, now, patch=patch))
        elif kind == HERB_HIGH:
            result.append(self._spec("predator_increase", context, parent, now, patch=patch))
        elif kind == ARRIVED:
            result.extend((self._spec("invasion_established", context, parent, now, patch=patch),
                           self._spec("invasion_failed", context, parent, now, patch=patch)))
        elif kind == ESTABLISHED:
            if self._competition_after(state, patch)[1] < 2.5:
                result.append(self._spec("invasion_resource_loss", context, parent, now, patch=patch))
            for target, corridor in state["adjacent"][patch]:
                if state["local"][target]["arrival"] is None:
                    result.append(self._spec("propagule_dispersal", context, parent, now,
                                             patch=target, source_patch=patch, corridor=corridor))
        return result

    def relation_specs(self, event, candidate, context, event_by_id):
        del event, context, event_by_id
        mechanism = self.mechanism_registry.get(candidate.mechanism_id)
        return [RelationSpec(source_event_id=parent,
            relation_class=mechanism.relation_semantics,
            relation_type_id=mechanism.mechanism_id, rule_id=mechanism.mechanism_id,
            mechanism_group_id=candidate.candidate_id if len(candidate.parent_event_ids) > 1 else None,
            context_evidence=copy.deepcopy(candidate.context_evidence),
            provenance={"parent_basis": "typed_binding_state_precondition_and_fired_clock",
                        "topology_edge_alone_sufficient": False})
            for parent in candidate.parent_event_ids]

    def validate_result(self, result):
        errors = []
        entities = {entity.entity_id: entity for entity in result.entities}
        relation_triples = {(relation.relation_type_id, relation.source_entity_id,
                             relation.target_entity_id) for relation in result.context_relations}
        if len(relation_triples) != len(result.context_relations):
            errors.append("Duplicate ecological context relation")
        for relation in result.context_relations:
            layer = next((item for item in self.domain_spec.compile_relation_layers()
                          if item.relation_type_id == relation.relation_type_id), None)
            source, target = entities.get(relation.source_entity_id), entities.get(relation.target_entity_id)
            if not layer or not source or not target or source.entity_type_id not in layer.source_type_ids or target.entity_type_id not in layer.target_type_ids:
                errors.append(f"Invalid typed ecological relation {relation.relation_id}")
        guild_rule = {"herbivore_consumes_resource": "herbivore",
                      "invader_uses_resource": "invader",
                      "predator_preys_on_herbivore": "predator",
                      "invader_competes_with_herbivore": "invader"}
        for relation in result.context_relations:
            suffix = relation.relation_type_id.removeprefix(NS)
            if suffix in guild_rule and entities[relation.source_entity_id].attributes.get("guild") != guild_rule[suffix]:
                errors.append(f"Invalid food-web source guild {relation.relation_id}")
            if suffix in ("predator_preys_on_herbivore", "invader_competes_with_herbivore") and entities[relation.target_entity_id].attributes.get("guild") != "herbivore":
                errors.append(f"Invalid food-web target guild {relation.relation_id}")
        region_of = {source: target for kind, source, target in relation_triples
                     if kind == NS + "patch_in_region"}
        populations = {(target, entities[source].attributes.get("guild")): source
                       for kind, source, target in relation_triples
                       if kind == NS + "population_in_patch"}
        consumed_resource = {source: target for kind, source, target in relation_triples
                             if kind == NS + "herbivore_consumes_resource"}
        patch_resources = {patch: consumed_resource[population] for (patch, guild), population
                           in populations.items() if guild == "herbivore" and
                           population in consumed_resource}
        patch_ids = {entity.entity_id for entity in result.entities
                     if entity.entity_type_id == NS + "habitat_patch"}
        if patch_ids != set(region_of) or patch_ids != set(patch_resources) or any((patch, guild) not in populations
            for patch in patch_ids for guild in ("herbivore", "predator", "invader")):
            errors.append("Missing ecological patch/region/population binding")
        for patch, resource in patch_resources.items():
            herb = populations[(patch, "herbivore")]
            pred = populations[(patch, "predator")]
            invader = populations[(patch, "invader")]
            expected = ((NS + "resource_in_patch", resource, patch),
                        (NS + "herbivore_consumes_resource", herb, resource),
                        (NS + "invader_uses_resource", invader, resource),
                        (NS + "predator_preys_on_herbivore", pred, herb),
                        (NS + "invader_competes_with_herbivore", invader, herb))
            if any(triple not in relation_triples for triple in expected):
                errors.append(f"Inconsistent local food-web binding {patch}")
        corridor_ends = {}
        for kind, source, target in relation_triples:
            if kind in (NS + "corridor_from_patch", NS + "corridor_to_patch"):
                corridor_ends.setdefault(source, {})[kind] = target
        if any(len(ends) != 2 or len(set(ends.values())) != 2 for ends in corridor_ends.values()):
            errors.append("Invalid ecological corridor endpoints")
        event_by_id = {event.event_id: event for event in result.events}
        incoming = {}
        for relation in result.event_relations:
            incoming.setdefault(relation.target_event_id, []).append(relation)
        state = {patch: {"factor": 1.0,
                         "herb": entities[populations[(patch, "herbivore")]].attributes["baseline_abundance"],
                         "pred": entities[populations[(patch, "predator")]].attributes["baseline_abundance"],
                         "invader": 0, "arrival": False, "establishment": None}
                 for patch in patch_ids if all((patch, guild) in populations
                 for guild in ("herbivore", "predator", "invader"))}
        observed_types = {}
        active = False
        for event in result.events:
            kind, attrs = event.event_type_id, event.attributes
            patch = attrs.get("patch_id")
            if patch not in state:
                errors.append(f"Event on unknown patch {event.event_id}")
                continue
            participants = {item.role: item.entity_id for item in event.participants}
            if participants.get("patch", patch) != patch:
                errors.append(f"Event patch participant mismatch {event.event_id}")
            if attrs.get("region_id") != region_of.get(patch) or (
                "region" in participants and participants["region"] != region_of.get(patch)
            ):
                errors.append(f"Event region binding mismatch {event.event_id}")
            for role, guild in (("herbivore", "herbivore"), ("predator", "predator"),
                                ("invader", "invader")):
                if role in participants and participants[role] != populations.get((patch, guild)):
                    errors.append(f"Event population binding mismatch {event.event_id}")
            if "resource" in participants and participants["resource"] != patch_resources.get(patch):
                errors.append(f"Event resource binding mismatch {event.event_id}")
            parents = incoming.get(event.event_id, [])
            mechanism_id = event.provenance.get("mechanism_id")
            try:
                mechanism = self.mechanism_registry.get(mechanism_id)
            except (KeyError, ValueError):
                errors.append(f"Unknown ecology mechanism {event.event_id}")
                continue
            if mechanism.target_event_type_id != kind or (not parents) != (mechanism.parent_combination == "none"):
                errors.append(f"Event mechanism/parent mismatch {event.event_id}")
            for relation in parents:
                parent = event_by_id.get(relation.source_event_id)
                if (not parent or parent.event_type_id not in mechanism.source_event_type_ids or
                    relation.relation_type_id != mechanism_id or
                    parent.temporal.anchor("occurrence_start") > event.temporal.anchor("occurrence_start")):
                    errors.append(f"Invalid ecology event parent {event.event_id}")
                elif mechanism_id == NS + "propagule_dispersal":
                    if parent.attributes.get("patch_id") != attrs.get("source_patch_id"):
                        errors.append(f"Dispersal parent patch mismatch {event.event_id}")
                elif mechanism_id in (NS + "drought_resource_loss", NS + "pulse_resource_gain",
                                      NS + "resource_recovery", NS + "pulse_resource_normalize"):
                    if parent.attributes.get("region_id") != attrs.get("region_id"):
                        errors.append(f"Disturbance parent region mismatch {event.event_id}")
                elif parent.attributes.get("patch_id") != patch:
                    errors.append(f"Local parent patch mismatch {event.event_id}")
            local = state[patch]
            info_herb = entities[populations[(patch, "herbivore")]].attributes["baseline_abundance"]
            info_pred = entities[populations[(patch, "predator")]].attributes["baseline_abundance"]
            biomass = entities[patch_resources[patch]].attributes["baseline_biomass_kg"]
            availability = biomass * local["factor"] / max(1.0, local["herb"] + 1.2 * local["invader"])
            if abs(attrs.get("availability_before_kg_per_effective_consumer", availability) - availability) > 1e-5:
                errors.append(f"Availability replay mismatch {event.event_id}")
            if kind in (DROUGHT, PULSE):
                active = True
            elif kind in (DROUGHT_END, PULSE_END):
                active = False
            elif kind in (LOW, RESTORED, HIGH, NORMAL):
                before = local["factor"]
                after = attrs.get("resource_factor_after")
                if attrs.get("resource_factor_before") != before or after is None:
                    errors.append(f"Resource factor mismatch {event.event_id}")
                else:
                    if kind == LOW and attrs.get("driver") == "regional_drought" and not (active and after < before):
                        errors.append(f"Unjustified drought resource loss {event.event_id}")
                    if kind == LOW and attrs.get("driver") == "regional_drought" and after != entities[patch].attributes["drought_resource_factor"]:
                        errors.append(f"Drought resource parameter mismatch {event.event_id}")
                    if kind == LOW and attrs.get("driver") == "invader_competition" and not (
                        local["establishment"] == "established" and after == before and
                        attrs.get("invader_abundance_before") == local["invader"] and
                        attrs.get("invader_abundance_after", 0) > local["invader"]):
                        errors.append(f"Unjustified invasion resource loss {event.event_id}")
                    if kind == HIGH and not (active and after > before):
                        errors.append(f"Unjustified resource gain {event.event_id}")
                    if kind == HIGH and after != entities[patch].attributes["pulse_resource_factor"]:
                        errors.append(f"Pulse resource parameter mismatch {event.event_id}")
                    if kind in (RESTORED, NORMAL) and after != 1.0:
                        errors.append(f"Resource baseline restoration mismatch {event.event_id}")
                    invader_after = attrs.get("invader_abundance_after", local["invader"])
                    expected_availability = biomass * after / max(
                        1.0, local["herb"] + 1.2 * invader_after)
                    if abs(attrs.get("availability_after_kg_per_effective_consumer", expected_availability)
                           - expected_availability) > 1e-5:
                        errors.append(f"Resource availability after-state mismatch {event.event_id}")
                    if kind == LOW and attrs.get("driver") == "invader_competition":
                        expected_invader = round(
                            entities[patch].attributes["invader_growth_potential_ratio"] * info_herb)
                        if invader_after != expected_invader:
                            errors.append(f"Invader growth parameter mismatch {event.event_id}")
                        if not (availability >= 2.5 and expected_availability < 2.5):
                            errors.append(f"Competition did not cross resource threshold {event.event_id}")
                        local["invader"] = invader_after
                    local["factor"] = after
            elif kind in (HERB_LOW, HERB_RESTORED, HERB_HIGH, PRED_LOW, PRED_RESTORED, PRED_HIGH):
                guild = "herb" if "herbivore" in kind else "pred"
                before, after = local[guild], attrs.get("abundance_after")
                if attrs.get("abundance_before") != before or not isinstance(after, (int, float)) or after < 0:
                    errors.append(f"Abundance replay mismatch {event.event_id}")
                else:
                    if kind == HERB_LOW and not (availability < 2.5 and after < before):
                        errors.append(f"Unsupported herbivore decline {event.event_id}")
                    if kind == HERB_HIGH and not (availability > 5.0 and after > before):
                        errors.append(f"Unsupported herbivore increase {event.event_id}")
                    if kind == PRED_LOW and not (before > after and local["herb"] < .8 * info_herb):
                        errors.append(f"Unsupported predator decline {event.event_id}")
                    if kind == PRED_HIGH and not (after > before and local["herb"] > 1.2 * info_herb):
                        errors.append(f"Unsupported predator increase {event.event_id}")
                    if kind == HERB_RESTORED and not (after == info_herb and availability >= 2.5):
                        errors.append(f"Unsupported herbivore restoration {event.event_id}")
                    if kind == PRED_RESTORED and not (after == info_pred and local["herb"] >= .8 * info_herb):
                        errors.append(f"Unsupported predator restoration {event.event_id}")
                    local[guild] = after
            elif kind == ARRIVED:
                if local["arrival"]:
                    errors.append(f"Repeated propagule arrival {event.event_id}")
                local["arrival"] = True
                if mechanism_id == NS + "propagule_dispersal":
                    source = attrs.get("source_patch_id")
                    corridor = attrs.get("corridor_id")
                    if not source or corridor not in corridor_ends or set(corridor_ends[corridor].values()) != {source, patch}:
                        errors.append(f"Unbound invasion dispersal {event.event_id}")
            elif kind in (ESTABLISHED, FAILED):
                if not local["arrival"] or local["establishment"] is not None:
                    errors.append(f"Invalid competing establishment outcome {event.event_id}")
                local["establishment"] = "established" if kind == ESTABLISHED else "failed"
                if kind == ESTABLISHED:
                    if attrs.get("invader_abundance_before") != local["invader"] or attrs.get("invader_abundance_after") != max(1, round(.2 * info_herb)):
                        errors.append(f"Invader establishment abundance mismatch {event.event_id}")
                    local["invader"] = attrs["invader_abundance_after"]
            observed_types[event.event_id] = kind
        return errors

    def render_episode_text(self, result):
        sentences, claims = [], []
        selected = ("patch_id", "region_id", "source_patch_id", "corridor_id", "driver",
                    "availability_before_kg_per_effective_consumer",
                    "availability_after_kg_per_effective_consumer", "resource_factor_after",
                    "population_guild", "abundance_before", "abundance_after",
                    "invader_abundance_after", "habitat_quality")
        for event in result.events:
            roles = ", ".join(f"{item.role}={item.entity_id}" for item in event.participants)
            attrs = {field: event.attributes[field] for field in selected if field in event.attributes}
            sentences.append(f"+{event.temporal.anchor('occurrence_start') / DAY:.3f} day "
                             f"{event.event_id} {event.event_type_id} ({roles}); "
                             f"{json.dumps(attrs, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}.")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                           "event_ids": [event.event_id], "relation_ids": [],
                           "evidence_fields": ["event_type_id", "temporal", "participants"] +
                           [f"attributes.{field}" for field in attrs]})
        for relation in result.event_relations:
            predicate = PREDICATES[relation.relation_type_id.removeprefix(NS)]
            sentences.append(f"{relation.source_event_id} {predicate} {relation.target_event_id} "
                             f"({relation.relation_class}, +{relation.temporal_link.lag_seconds / DAY:.3f} day; "
                             f"rule={relation.rule_id}).")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                           "event_ids": [relation.source_event_id, relation.target_event_id],
                           "relation_ids": [relation.relation_id],
                           "relation_assertions": [{"relation_id": relation.relation_id,
                               "asserted_relation_class": relation.relation_class,
                               "surface_predicate": predicate}],
                           "evidence_fields": ["rule_id", "source_event_id", "target_event_id", "temporal_link"]})
        return {"alignment_id": result.episode_id + "_text_alignment",
                "episode_id": result.episode_id, "domain": self.domain_id,
                "projection_mode": "deterministic_grounded_projection",
                "sentences": sentences, "text": " ".join(sentences), "claims": claims}


def build_ecology_temporal_models():
    registry = build_standard_temporal_registry()
    models = (
        ExponentialArrivalModel(NS + "root_arrival.prior_v1", default_rate_per_second=.25 / DAY),
        ConditionalLogNormalModel(NS + "disturbance_duration.prior_v1", math.log(35 * DAY), .45, {}, minimum_seconds=DAY),
        ConditionalLogNormalModel(NS + "pulse_duration.prior_v1", math.log(25 * DAY), .45, {}, minimum_seconds=DAY),
        ConditionalLogNormalModel(NS + "resource_response.prior_v1", math.log(2 * DAY), .55, {}, minimum_seconds=.1 * DAY),
        ConditionalLogNormalModel(NS + "resource_recovery.prior_v1", math.log(5 * DAY), .55, {}, minimum_seconds=.1 * DAY),
        PiecewiseExponentialHazardModel(NS + "herbivore_decline_hazard.prior_v1", .055 / DAY, {"resource_deficit": .5}),
        PiecewiseExponentialHazardModel(NS + "predator_decline_hazard.prior_v1", .035 / DAY, {"prey_deficit": 2.0}),
        PiecewiseExponentialHazardModel(NS + "herbivore_recovery_hazard.prior_v1", .045 / DAY, {"resource_surplus": .2}),
        PiecewiseExponentialHazardModel(NS + "predator_recovery_hazard.prior_v1", .035 / DAY, {"prey_surplus": .3}),
        PiecewiseExponentialHazardModel(NS + "herbivore_increase_hazard.prior_v1", .045 / DAY, {"resource_surplus": .25}),
        PiecewiseExponentialHazardModel(NS + "predator_increase_hazard.prior_v1", .03 / DAY, {"prey_surplus": .5}),
        PiecewiseExponentialHazardModel(NS + "dispersal_hazard.prior_v1", .006 / DAY,
                                        {"corridor_permeability": 1.0, "distance_km": -.15}),
        PiecewiseExponentialHazardModel(NS + "establishment_hazard.prior_v1", .04 / DAY, {"habitat_quality": .5}),
        PiecewiseExponentialHazardModel(NS + "failure_hazard.prior_v1", .02 / DAY, {"habitat_quality": -.5}),
        ConditionalLogNormalModel(NS + "log_ingestion_delay.prior_v1", math.log(.1 * DAY), .4, {}, minimum_seconds=.001 * DAY),
    )
    for model in models:
        registry.register(model)
    return registry
