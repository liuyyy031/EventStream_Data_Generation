"""Synthetic field water balances, thermal proxies and finite-rate irrigation.

Uses a fixed root zone, FAO-56-style single crop coefficient, constant weather
segments and ideal threshold sensors. This is not a crop-yield or pest model.
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
    build_standard_temporal_registry,
)
from data_generation.generation_core.topology import SparseHeterogeneousTopologyGenerator
from .physics import advance_water, stress_coefficient, thermal_rate, time_to_depletion

NS, DAY = "agriculture.", 86400.0
ESTABLISHED, MILESTONE = NS + "crop.established", NS + "phenology.thermal_milestone.reached"
RAIN, FAILED, REPAIRED = NS + "effective_rain.received", NS + "supply.failed", NS + "supply.restored"
DRY, WET = NS + "water.readily_available.exhausted", NS + "water.readily_available.restored"
SAMPLE, REPORT = NS + "sensor.sampled", NS + "sensor.reported"
REQUEST, START, END, INTERRUPTED, DEFERRED = (NS + "irrigation." + key for key in
    ("requested", "started", "completed", "interrupted", "deferred"))
EVENT_TYPES = (ESTABLISHED, MILESTONE, RAIN, FAILED, REPAIRED, DRY, WET,
               SAMPLE, REPORT, REQUEST, START, END, INTERRUPTED, DEFERRED)
SCENARIOS = ("rainfed_drydown", "irrigation_response", "shared_supply_interruption")
STAGES = ("established_cohort", "vegetative_thermal_proxy", "flowering_thermal_proxy", "maturity_thermal_proxy")
GDD_TARGETS = (80.0, 240.0, 480.0)
CROP_COEFFICIENTS = (.6, .9, 1.05, .8)
SAMPLE_FIELDS = ("sample_event_id", "sample_sequence", "sampled_at_offset_seconds",
    "sampled_depletion_mm", "sampled_water_content_m3_m3", "sampled_low_water", "sampled_raw_mm")
PREDICATES = {
    "supply_repair": "was repaired and restored at",
    "thermal_milestone": "accumulated the remaining thermal time to the proxy milestone at",
    "drydown_threshold": "anchored the water-balance segment deriving available-water exhaustion at",
    "water_restoration": "anchored the water-balance segment deriving available-water restoration at",
    "sensor_sample": "prompted the ideal root-zone sensor sample",
    "sensor_report": "was transmitted without changing its frozen sample as",
    "irrigation_request": "prompted the irrigation request",
    "irrigation_start": "was admitted to its available irrigation supply at",
    "irrigation_resume": "was required for the pending request to restart irrigation at",
    "irrigation_defer": "was required for the supply-unavailable deferral at",
    "irrigation_complete": "finished its planned finite-rate irrigation delivery at",
    "irrigation_interrupt": "was required for the actual irrigation interruption at",
}


class AgriculturePackage:
    domain_id = "agriculture"

    def __init__(self, config=None):
        self.config = {"base_seed": 20261008, "context_count": 4, "nodes_per_context": 1000,
            "duration_seconds": 45 * DAY, "start_time": "2026-10-08T00:00:00+08:00",
            "initial_depletion_fraction": .15, "irrigation_target_fraction": .2,
            "effective_rain_range_mm": [15., 60.], "mean_temperature_range_c": [15., 25.],
            "reference_et_range_mm_day": [3., 6.],
            "scenario_weights": dict(zip(SCENARIOS, (.35, .35, .3))), **(config or {})}
        for key in ("base_seed", "context_count", "nodes_per_context"):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
                raise ValueError(f"{key} must be a finite integer")
            self.config[key] = int(value)
        if self.config["context_count"] < 1 or self.config["nodes_per_context"] < 20:
            raise ValueError("Agriculture needs at least one context and 20 entities")
        duration = float(self.config["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        self.config["duration_seconds"] = duration
        for key in ("initial_depletion_fraction", "irrigation_target_fraction"):
            value = float(self.config[key])
            if not math.isfinite(value) or not 0 <= value < .35:
                raise ValueError(f"{key} must be in [0, .35), below all synthetic RAW fractions")
            self.config[key] = value
        for key in ("effective_rain_range_mm", "mean_temperature_range_c", "reference_et_range_mm_day"):
            values = self.config[key]
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"{key} must be a lower/upper pair")
            lo, hi = map(float, values)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi or key != "mean_temperature_range_c" and lo < 0:
                raise ValueError(f"Invalid {key}")
            self.config[key] = [lo, hi]
        weights = self.config["scenario_weights"]
        if not weights or set(weights) - set(SCENARIOS) or any(
            not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()
        ) or sum(weights.values()) <= 0:
            raise ValueError("Invalid Agriculture scenario weights")
        self.domain_spec = load_domain_spec(Path(__file__).with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(Path(__file__).with_name("mechanism_specs.json"),
            known_event_type_ids=EVENT_TYPES, known_context_relation_type_ids=self.domain_spec.context_relation_type_ids)
        self._context_cache = {}

    def catalog(self):
        roles = {role: NS + kind for role, kind in (("field", "field"), ("crop", "crop_cohort"),
            ("sensor", "soil_water_sensor"), ("irrigator", "irrigator"), ("supply", "supply_unit"), ("weather", "weather_zone"))}
        return {"domain": self.domain_id, "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_not_empirically_fitted",
            "scope": {"crop": "fictional_established_annual_cohort_fixed_root_zone",
                "water": "FAO56_style_single_coefficient_analytic_segments",
                "weather": "constant_temperature_and_reference_ET_plus_effective_rain_impulse",
                "phenology": "thermal_time_proxy_not_proof_of_actual_flowering_or_maturity",
                "sensor": "ideal_threshold_triggered_rootzone_estimate_no_noise",
                "supply": "externally_maintained_flow_capacity_not_a_finite_reservoir",
                "statistical_realism_claimed": False},
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {"mode": "deterministic_grounded_projection", "free_form_causal_completion": False},
            "entity_types": [{"entity_type_id": NS + kind, "parent_type_id": None, "required_attributes": attrs}
                for kind, attrs in (("field", ["area_ha", "field_capacity_m3_m3", "wilting_point_m3_m3", "root_depth_m", "depletion_fraction"]),
                    ("crop_cohort", ["base_temperature_c", "upper_temperature_c", "thermal_thresholds_degree_days", "crop_coefficients"]),
                    ("soil_water_sensor", ["measurement_semantics"]), ("irrigator", ["flow_m3_s", "application_efficiency"]),
                    ("supply_unit", ["rated_capacity_m3_s"]), ("weather_zone", ["mean_temperature_c", "reference_et_mm_day"]))],
            "event_types": [{"event_type_id": kind, "parent_type_id": kind.rsplit(".", 1)[0],
                "required_attributes": ["workflow_id", "scenario_family"], "participant_role_types": roles} for kind in EVENT_TYPES],
            "context_relation_types": [{"relation_type_id": kind, "directed": True,
                "does_not_imply_event_parenthood": True} for kind in self.domain_spec.context_relation_type_ids],
            "event_relation_types": [{"relation_type_id": NS + key,
                "relation_class": self.mechanism_registry.get(NS + key).relation_semantics,
                "text_rendering": {"surface_predicate": predicate,
                    "asserted_relation_class": self.mechanism_registry.get(NS + key).relation_semantics}}
                for key, predicate in PREDICATES.items()], "mechanisms": self.mechanism_registry.descriptors()}

    @staticmethod
    def _graph_data(entities, relations):
        bindings = {(r.relation_type_id, r.source_entity_id): r.target_entity_id for r in relations}
        inverse = {(r.relation_type_id, r.target_entity_id): r.source_entity_id for r in relations
                   if r.relation_type_id in (NS + "crop_in_field", NS + "sensor_observes_field", NS + "irrigator_serves_field")}
        fields = [e.entity_id for e in entities if e.entity_type_id == NS + "field"]
        field_links = {}
        for field in fields:
            irrigator = inverse[(NS + "irrigator_serves_field", field)]
            field_links[field] = {"crop": inverse[(NS + "crop_in_field", field)],
                "sensor": inverse[(NS + "sensor_observes_field", field)], "irrigator": irrigator,
                "supply": bindings[(NS + "irrigator_fed_by_supply", irrigator)],
                "weather": bindings[(NS + "field_in_weather_zone", field)]}
        return {"entity_attrs": {e.entity_id: e.attributes for e in entities}, "field_links": field_links,
            "relation_ids": {(r.relation_type_id, r.source_entity_id, r.target_entity_id): r.relation_id for r in relations},
            "field_ids": fields, "supply_ids": [e.entity_id for e in entities if e.entity_type_id == NS + "supply_unit"]}

    def _build_context(self, index):
        rng = random.Random(self.config["base_seed"] + 7919 * index)
        context_id, n = f"agriculture_context_{index:03d}", self.config["nodes_per_context"]
        nz, nu = max(2, n // 100), max(2, n // 50)
        nz += (n - nz - nu) % 4
        nf = (n - nz - nu) // 4
        ids = {kind: [f"{context_id}_{kind}_{i:05d}" for i in range(count)] for kind, count in
            (("field", nf), ("crop_cohort", nf), ("soil_water_sensor", nf), ("irrigator", nf), ("supply_unit", nu), ("weather_zone", nz))}
        entities = []
        for field in ids["field"]:
            wp = rng.uniform(.1, .18)
            entities.append(Entity(field, self.domain_id, NS + "field", {"area_ha": rng.uniform(.3, 1.5),
                "field_capacity_m3_m3": wp + rng.uniform(.12, .22), "wilting_point_m3_m3": wp,
                "root_depth_m": rng.uniform(.4, .8), "depletion_fraction": rng.uniform(.35, .55)}, context_id))
        for crop in ids["crop_cohort"]:
            entities.append(Entity(crop, self.domain_id, NS + "crop_cohort", {"crop_id": "fictional_annual_cohort",
                "base_temperature_c": 8.0, "upper_temperature_c": 30.0,
                "thermal_thresholds_degree_days": list(GDD_TARGETS), "crop_coefficients": list(CROP_COEFFICIENTS)}, context_id))
        for sensor in ids["soil_water_sensor"]:
            entities.append(Entity(sensor, self.domain_id, NS + "soil_water_sensor",
                {"measurement_semantics": "ideal_rootzone_water_estimate_at_threshold_transition"}, context_id))
        capacities = defaultdict(float)
        for i, irrigator in enumerate(ids["irrigator"]):
            flow = rng.uniform(.003, .008)
            capacities[ids["supply_unit"][i % nu]] += flow
            entities.append(Entity(irrigator, self.domain_id, NS + "irrigator", {
                "flow_m3_s": flow, "application_efficiency": rng.uniform(.7, .9)}, context_id))
        for supply in ids["supply_unit"]:
            entities.append(Entity(supply, self.domain_id, NS + "supply_unit", {
                "rated_capacity_m3_s": capacities[supply], "source_policy": "externally_maintained_supply_when_available"}, context_id))
        for weather in ids["weather_zone"]:
            entities.append(Entity(weather, self.domain_id, NS + "weather_zone", {
                "mean_temperature_c": rng.uniform(*self.config["mean_temperature_range_c"]),
                "reference_et_mm_day": rng.uniform(*self.config["reference_et_range_mm_day"])}, context_id))
        shuffled = list(ids["weather_zone"])
        rng.shuffle(shuffled)
        required = {NS + "crop_in_field": list(zip(ids["crop_cohort"], ids["field"])),
            NS + "sensor_observes_field": list(zip(ids["soil_water_sensor"], ids["field"])),
            NS + "irrigator_serves_field": list(zip(ids["irrigator"], ids["field"])),
            NS + "irrigator_fed_by_supply": [(p, ids["supply_unit"][i % nu]) for i, p in enumerate(ids["irrigator"])],
            NS + "field_in_weather_zone": [(f, shuffled[i % nz]) for i, f in enumerate(ids["field"])]}
        topology = SparseHeterogeneousTopologyGenerator().generate(context_id=context_id, domain=self.domain_id,
            entities=entities, layers=[replace(layer, required_pairs=tuple(required[layer.relation_type_id]))
                for layer in self.domain_spec.compile_relation_layers()], seed=self.config["base_seed"] + 7919 * index,
            observation_window_seconds=self.config["duration_seconds"])
        return {**self._graph_data(entities, topology.relations), "entities": entities,
                "relations": topology.relations, "statistics": topology.statistics}

    def _initial_state(self, data, scenario, workflow):
        local = {}
        for field in data["field_ids"]:
            f = data["entity_attrs"][field]
            taw = 1000 * (f["field_capacity_m3_m3"] - f["wilting_point_m3_m3"]) * f["root_depth_m"]
            initial = self.config["initial_depletion_fraction"] * taw
            local[field] = {"active": False, "stage": 0, "gdd": 0., "depletion": initial,
                "initial_depletion": initial, "taw": taw, "raw": f["depletion_fraction"] * taw,
                "last_time": 0., "et": 0., "net_irrigation": 0., "rain": 0., "drainage": 0.,
                "epoch": 0, "low_water": False, "irrigating": False, "start": None, "request": None,
                "cycle_gross_mm": 0., "sample_sequence": 0, "latest_sequence": 0,
                "latest_report": None, "latest_low": False}
        return {**{key: data[key] for key in ("entity_attrs", "field_links", "relation_ids")},
            "scenario": scenario, "workflow_id": workflow, "local": local,
            "supplies": {unit: {"available": True, "failure": None, "repair": None} for unit in data["supply_ids"]},
            "reported_samples": set(), "sampled_events": set()}

    def create_episode_context(self, episode_index, seed):
        index = episode_index % self.config["context_count"]
        if index not in self._context_cache:
            self._context_cache[index] = self._build_context(index)
        data, rng = self._context_cache[index], random.Random(seed ^ 0x41475249)
        weights = self.config["scenario_weights"]
        scenario = rng.choices(list(weights), weights=list(weights.values()))[0]
        if scenario == "shared_supply_interruption":
            groups = defaultdict(list)
            for field, links in data["field_links"].items():
                groups[links["supply"]].append(field)
            selected = rng.sample(rng.choice([group for group in groups.values() if len(group) >= 2]), 2)
        else:
            selected = [rng.choice(data["field_ids"])]
        episode_id = f"agriculture_episode_{episode_index:07d}"
        state = self._initial_state(data, scenario, episode_id + "_workflow")
        state["selected_fields"] = selected
        return EpisodeContext(episode_id, f"agriculture_context_{index:03d}", self.domain_id,
            self.config["start_time"], self.config["duration_seconds"], data["entities"], data["relations"], state=state,
            context_attributes={"topology_profile": data["statistics"]},
            episode_attributes={"scenario_family": scenario, "selected_fields": selected,
                "empirical_realism_claimed": False, "crop_id": "fictional_annual_cohort",
                "phenology_semantics": "thermal_milestones_are_proxies_not_verified_flowering_or_yield",
                "water_scope": "fixed_root_zone_no_groundwater_or_nutrients"})

    @staticmethod
    def _rates(state, field):
        local, links = state["local"][field], state["field_links"][field]
        crop, weather = (state["entity_attrs"][links[role]] for role in ("crop", "weather"))
        r = crop["crop_coefficients"][local["stage"]] * weather["reference_et_mm_day"] if local["active"] else 0.0
        j = 0.0
        if local["irrigating"] and state["supplies"][links["supply"]]["available"]:
            pump, f = state["entity_attrs"][links["irrigator"]], state["entity_attrs"][field]
            j = pump["flow_m3_s"] * DAY * pump["application_efficiency"] / (10 * f["area_ha"])
        gdd = thermal_rate(weather["mean_temperature_c"], crop["base_temperature_c"], crop["upper_temperature_c"]) if local["active"] else 0.0
        return r, j, gdd

    def _view(self, state, field, now):
        local = dict(state["local"][field])
        elapsed = max(0.0, now - local["last_time"]) / DAY
        r, j, gdd = self._rates(state, field)
        local["depletion"], et, drainage = advance_water(local["depletion"], elapsed, r, j, local["taw"], local["raw"])
        local["gdd"] += gdd * elapsed
        local["et"] += et
        local["net_irrigation"] += j * elapsed
        local["drainage"] += drainage
        if local["irrigating"]:
            eta = state["entity_attrs"][state["field_links"][field]["irrigator"]]["application_efficiency"]
            local["cycle_gross_mm"] += j * elapsed / eta
        local["last_time"] = now
        return local

    def _spec(self, key, context, parents=(), now=0.0, *, field=None, supply=None, delay=0.0, extra=None):
        state, pairs, roles = context.state, set(), {}
        mechanism = self.mechanism_registry.get(NS + key)
        attrs = {"workflow_id": state["workflow_id"], "scenario_family": state["scenario"]}
        if field:
            links = state["field_links"][field]
            roles.update(field=field, crop=links["crop"], weather=links["weather"])
            attrs["field_id"] = field
            pairs.update(((NS + "crop_in_field", links["crop"], field), (NS + "field_in_weather_zone", field, links["weather"])))
            if key in ("sensor_sample", "sensor_report", "irrigation_request"):
                roles["sensor"] = links["sensor"]
                pairs.add((NS + "sensor_observes_field", links["sensor"], field))
            if key.startswith("irrigation_"):
                roles.update(irrigator=links["irrigator"], supply=links["supply"])
                attrs["supply_id"] = links["supply"]
                pairs.update(((NS + "irrigator_serves_field", links["irrigator"], field),
                              (NS + "irrigator_fed_by_supply", links["irrigator"], links["supply"])))
        if supply:
            roles["supply"] = supply
            attrs["supply_id"] = supply
        if extra:
            attrs.update(extra)
        phase = (-30 if key in ("supply_failure", "supply_repair", "effective_rain") else
                 -10 if key.startswith("irrigation_") and key not in ("irrigation_request", "irrigation_defer") else
                 0 if key == "thermal_milestone" else 10 if key in ("drydown_threshold", "water_restoration") else
                 20 if key == "sensor_sample" else 30 if key == "sensor_report" else 40)
        return CandidateSpec(mechanism.mechanism_id, mechanism.target_event_type_id, list(parents),
            [EventParticipant(entity, role) for role, entity in roles.items()], mechanism.generation_temporal_model_ref,
            {"delay_seconds": delay} if mechanism.temporal_model_family == "deterministic_delay" else {},
            activation_time=now, combination=mechanism.parent_combination, phase_priority=phase, attributes=attrs,
            context_evidence=ContextEvidence(context_relation_ids=[state["relation_ids"][pair] for pair in sorted(pairs)],
                entity_ids=list(roles.values()), evaluated_at_offset_seconds=now),
            provenance={"input_anchor": "occurrence_start", "parameter_status": mechanism.parameter_status})

    def seed_candidates(self, context, rng):
        del rng
        selected, scenario = context.state["selected_fields"], context.state["scenario"]
        result = [self._spec("establishment", context, field=field) for field in selected]
        if scenario == "rainfed_drydown":
            result.append(self._spec("effective_rain", context, field=selected[0]))
        if scenario == "shared_supply_interruption":
            result.append(self._spec("supply_failure", context,
                supply=context.state["field_links"][selected[0]]["supply"]))
        return result

    def revalidate_candidate(self, candidate, context, event_by_id, now):
        del event_by_id
        state, attrs = context.state, candidate.attributes
        key, field, valid = candidate.mechanism_id.removeprefix(NS), attrs.get("field_id"), True
        local = self._view(state, field, now) if field else None
        if key == "establishment":
            valid = not local["active"]
        elif key == "supply_failure":
            valid = state["supplies"][attrs["supply_id"]]["failure"] is None
        elif key == "supply_repair":
            unit = state["supplies"][attrs["supply_id"]]
            valid = not unit["available"] and unit["failure"] in candidate.parent_event_ids
        elif key == "thermal_milestone":
            valid = local["active"] and local["stage"] == attrs["stage_before"]
            if now >= candidate.scheduled_time - 1e-7:
                valid &= local["gdd"] >= attrs["target_gdd"] - 1e-7
        elif key in ("drydown_threshold", "water_restoration"):
            r, j, _ = self._rates(state, field)
            derivative = r * stress_coefficient(local["depletion"], local["taw"], local["raw"]) - j
            valid = local["active"] and local["epoch"] == attrs["water_epoch"] and local["low_water"] == (key == "water_restoration")
            if now >= candidate.scheduled_time - 1e-7:
                valid &= (local["depletion"] >= local["raw"] - 1e-7 and derivative > 0) if key == "drydown_threshold" else (local["depletion"] <= local["raw"] + 1e-7 and derivative < 0 or local["depletion"] < local["raw"] - 1e-7)
        elif key == "sensor_sample":
            valid = candidate.parent_event_ids[0] not in state["sampled_events"]
        elif key == "sensor_report":
            valid = attrs["sample_event_id"] not in state["reported_samples"]
        elif key == "irrigation_request":
            valid = (state["scenario"] != "rainfed_drydown" and local["latest_report"] in candidate.parent_event_ids and
                     local["latest_low"] and local["low_water"] and not local["request"] and not local["irrigating"])
        elif key in ("irrigation_start", "irrigation_resume", "irrigation_defer"):
            unit = state["supplies"][attrs["supply_id"]]
            valid = (local["request"] in candidate.parent_event_ids and local["request"] is not None and
                not local["irrigating"] and local["low_water"] and unit["available"] != (key == "irrigation_defer"))
            if key == "irrigation_resume":
                valid &= unit["repair"] in candidate.parent_event_ids
            elif key == "irrigation_defer":
                valid &= unit["failure"] in candidate.parent_event_ids
        elif key in ("irrigation_complete", "irrigation_interrupt"):
            unit = state["supplies"][attrs["supply_id"]]
            valid = local["irrigating"] and local["start"] in candidate.parent_event_ids and unit["available"] == (key == "irrigation_complete")
            if key == "irrigation_interrupt":
                valid &= unit["failure"] in candidate.parent_event_ids
        elif key != "effective_rain":
            raise ValueError(f"Unsupported Agriculture mechanism {key}")
        evidence = copy.deepcopy(candidate.context_evidence)
        evidence.evaluated_at_offset_seconds = now
        evidence.state_predicates = [StatePredicateEvidence(field or attrs["supply_id"], key + "_precondition", "equals", True, bool(valid), bool(valid))]
        if field and "water_epoch" in attrs:
            evidence.state_predicates.append(StatePredicateEvidence(field, "water_epoch", "equals", attrs["water_epoch"], local["epoch"], attrs["water_epoch"] == local["epoch"]))
        return CandidateUpdate(bool(valid), None if valid else key + "_state_superseded", context_evidence=evidence)

    def _event_attributes(self, event_id, candidate, context):
        state, attrs = context.state, copy.deepcopy(candidate.attributes)
        kind, now, field = candidate.target_event_type_id, candidate.scheduled_time, attrs.get("field_id")
        if field:
            local, links = self._view(state, field, now), state["field_links"][field]
            f, crop, weather = (state["entity_attrs"][key] for key in (field, links["crop"], links["weather"]))
            attrs.update(depletion_before_mm=local["depletion"], taw_mm=local["taw"], raw_mm=local["raw"],
                water_content_before_m3_m3=f["field_capacity_m3_m3"] - local["depletion"] / (1000 * f["root_depth_m"]),
                ks_before=stress_coefficient(local["depletion"], local["taw"], local["raw"]),
                gdd_before=local["gdd"], stage_index_before=local["stage"], stage_before_name=STAGES[local["stage"]],
                crop_coefficient_before=crop["crop_coefficients"][local["stage"]], mean_temperature_c=weather["mean_temperature_c"],
                reference_et_mm_day=weather["reference_et_mm_day"], thermal_rate_degree_days_per_day=self._rates(state, field)[2],
                initial_depletion_mm=local["initial_depletion"], cumulative_et_mm=local["et"],
                cumulative_net_irrigation_mm=local["net_irrigation"], cumulative_effective_rain_mm=local["rain"],
                cumulative_deep_percolation_mm=local["drainage"], low_water_before=local["low_water"], irrigating_before=local["irrigating"])
            if kind == MILESTONE:
                attrs.update(stage_index_after=local["stage"] + 1, stage_after_name=STAGES[local["stage"] + 1],
                    milestone_semantics="thermal_proxy_not_verified_flowering_or_yield")
            elif kind == SAMPLE:
                attrs.update(sample_event_id=event_id, sample_sequence=local["sample_sequence"] + 1,
                    sampled_at_offset_seconds=now, sampled_depletion_mm=local["depletion"],
                    sampled_water_content_m3_m3=attrs["water_content_before_m3_m3"],
                    sampled_low_water=local["low_water"], sampled_raw_mm=local["raw"])
            elif kind == REPORT:
                attrs["stale_report"] = attrs["sample_sequence"] <= local["latest_sequence"]
            elif kind == START:
                pump = state["entity_attrs"][links["irrigator"]]
                target = self.config["irrigation_target_fraction"] * local["taw"]
                gross_depth = (local["depletion"] - target) / pump["application_efficiency"]
                volume = gross_depth * 10 * f["area_ha"]
                attrs.update(target_depletion_mm=target, planned_gross_depth_mm=gross_depth,
                    planned_gross_volume_m3=volume, equipment_flow_m3_s=pump["flow_m3_s"],
                    application_efficiency=pump["application_efficiency"], planned_duration_seconds=volume / pump["flow_m3_s"],
                    request_event_id=local["request"], irrigation_start_event_id=event_id,
                    dose_policy="replace_current_deficit_to_target_before_concurrent_ET_not_guaranteed_exact_target")
            elif kind in (END, INTERRUPTED):
                attrs.update(delivered_gross_depth_mm=local["cycle_gross_mm"],
                    delivered_gross_volume_m3=local["cycle_gross_mm"] * 10 * f["area_ha"],
                    irrigation_start_event_id=local["start"])
        else:
            unit = state["supplies"][attrs["supply_id"]]
            attrs["supply_available_before"] = unit["available"]
        return attrs

    def materialize_event(self, event_id, candidate, context, rng):
        attrs = self._event_attributes(event_id, candidate, context)
        if candidate.target_event_type_id == RAIN:
            depth = rng.uniform(*self.config["effective_rain_range_mm"])
            before = attrs["depletion_before_mm"]
            attrs.update(effective_rain_depth_mm=depth, depletion_after_mm=max(0., before - depth),
                rain_deep_percolation_mm=max(0., depth - before),
                rainfall_semantics="rootzone_effective_rain_input_not_gross_precipitation_or_infiltration_solver")
        return EventRecord(event_id, context.episode_id, self.domain_id, candidate.target_event_type_id,
            TemporalExtent(candidate.scheduled_time), list(candidate.participants), attrs,
            event_role="root" if not candidate.parent_event_ids else "derived",
            provenance={"generator": "synthetic_analytic_rootzone_thermal_irrigation", "mechanism_id": candidate.mechanism_id})

    def observation_plan(self, event, candidate, context):
        del candidate, context
        source = "synthetic_sensor_or_controller" if event.event_type_id in (SAMPLE, REPORT, REQUEST, DEFERRED) else "offline_synthetic_agronomic_ground_truth"
        return ObservationPlan(recording_model_ref=NS + "log_ingestion_delay.prior_v1", source=source)

    def apply_event(self, event, candidate, context):
        state, attrs, kind = context.state, event.attributes, event.event_type_id
        now = event.temporal.anchor("occurrence_start")
        for field in state["local"]:
            if state["local"][field]["active"] or field == attrs.get("field_id"):
                state["local"][field] = self._view(state, field, now)
        field = attrs.get("field_id")
        local = state["local"][field] if field else None
        if kind == ESTABLISHED:
            local["active"] = True
            local["epoch"] += 1
        elif kind == MILESTONE:
            local["stage"] += 1
            local["epoch"] += 1
        elif kind == RAIN:
            local["depletion"] = attrs["depletion_after_mm"]
            local["rain"] += attrs["effective_rain_depth_mm"]
            local["drainage"] += attrs["rain_deep_percolation_mm"]
            local["epoch"] += 1
        elif kind == DRY:
            local["low_water"] = True
        elif kind == WET:
            local["low_water"] = False
            if not local["irrigating"]:
                local["request"] = None
        elif kind == SAMPLE:
            local["sample_sequence"] = attrs["sample_sequence"]
            state["sampled_events"].update(candidate.parent_event_ids)
        elif kind == REPORT:
            state["reported_samples"].add(attrs["sample_event_id"])
            if not attrs["stale_report"]:
                local.update(latest_sequence=attrs["sample_sequence"], latest_report=event.event_id,
                             latest_low=attrs["sampled_low_water"])
        elif kind == REQUEST:
            local["request"] = event.event_id
        elif kind == START:
            local.update(irrigating=True, start=event.event_id, cycle_gross_mm=0., epoch=local["epoch"] + 1)
        elif kind in (END, INTERRUPTED):
            local.update(irrigating=False, start=None, epoch=local["epoch"] + 1)
            if kind == END or not local["low_water"]:
                local["request"] = None
        elif kind == FAILED:
            state["supplies"][attrs["supply_id"]].update(available=False, failure=event.event_id)
            for f, links in state["field_links"].items():
                if links["supply"] == attrs["supply_id"] and state["local"][f]["irrigating"]:
                    state["local"][f]["epoch"] += 1
        elif kind == REPAIRED:
            state["supplies"][attrs["supply_id"]].update(available=True, repair=event.event_id)

    def _water_predictions(self, context, event):
        state, field = context.state, event.attributes["field_id"]
        local = state["local"][field]
        if not local["active"]:
            return []
        r, j, _ = self._rates(state, field)
        derivative = r * stress_coefficient(local["depletion"], local["taw"], local["raw"]) - j
        recovery = local["low_water"]
        if (not recovery and derivative <= 0) or (recovery and derivative >= 0 and local["depletion"] >= local["raw"] - 1e-8):
            return []
        delay = 0.0 if (local["depletion"] >= local["raw"] and not recovery or local["depletion"] <= local["raw"] and recovery) else time_to_depletion(
            local["depletion"], local["raw"], r, j, local["taw"], local["raw"])
        if delay is None:
            return []
        return [self._spec("water_restoration" if recovery else "drydown_threshold", context,
            [event.event_id], event.temporal.anchor("occurrence_start"), field=field, delay=delay * DAY,
            extra={"water_epoch": local["epoch"], "predicted_threshold_depletion_mm": local["raw"]})]

    def _thermal_prediction(self, context, event):
        state, field = context.state, event.attributes["field_id"]
        local, links = state["local"][field], state["field_links"][field]
        targets = state["entity_attrs"][links["crop"]]["thermal_thresholds_degree_days"]
        rate = self._rates(state, field)[2]
        if local["stage"] >= len(targets) or rate <= 0:
            return []
        target = targets[local["stage"]]
        return [self._spec("thermal_milestone", context, [event.event_id], event.temporal.anchor("occurrence_start"),
            field=field, delay=max(0., target - local["gdd"]) / rate * DAY,
            extra={"stage_before": local["stage"], "target_gdd": target})]

    def spawn_candidates(self, event, candidate, context, event_by_id, rng):
        del candidate, event_by_id, rng
        state, attrs, kind = context.state, event.attributes, event.event_type_id
        now, parents, field = event.temporal.anchor("occurrence_start"), [event.event_id], attrs.get("field_id")
        result = []
        if kind in (ESTABLISHED, MILESTONE):
            result.extend(self._thermal_prediction(context, event))
        if kind in (ESTABLISHED, MILESTONE, RAIN, START, END, INTERRUPTED):
            result.extend(self._water_predictions(context, event))
        if kind in (DRY, WET) or kind == END and state["local"][field]["low_water"]:
            result.append(self._spec("sensor_sample", context, parents, now, field=field))
        elif kind == SAMPLE:
            result.append(self._spec("sensor_report", context, parents, now, field=field,
                extra={key: attrs[key] for key in SAMPLE_FIELDS}))
        elif kind == REPORT and not attrs["stale_report"] and attrs["sampled_low_water"] and state["scenario"] != "rainfed_drydown":
            result.append(self._spec("irrigation_request", context, parents, now, field=field))
        elif kind == REQUEST:
            unit = state["supplies"][attrs["supply_id"]]
            if unit["available"]:
                result.append(self._spec("irrigation_start", context, parents, now, field=field))
            else:
                result.append(self._spec("irrigation_defer", context, parents + [unit["failure"]], now, field=field))
        elif kind == START:
            result.append(self._spec("irrigation_complete", context, parents, now, field=field,
                delay=attrs["planned_duration_seconds"], extra={"irrigation_start_event_id": event.event_id}))
        elif kind == FAILED:
            supply = attrs["supply_id"]
            result.append(self._spec("supply_repair", context, parents, now, supply=supply))
            for f, links in state["field_links"].items():
                if links["supply"] == supply and state["local"][f]["irrigating"]:
                    result.append(self._spec("irrigation_interrupt", context,
                        [state["local"][f]["start"], event.event_id], now, field=f))
                elif links["supply"] == supply and state["local"][f]["request"] and state["local"][f]["low_water"]:
                    result.append(self._spec("irrigation_defer", context,
                        [state["local"][f]["request"], event.event_id], now, field=f))
        elif kind == REPAIRED:
            for f, links in state["field_links"].items():
                local = state["local"][f]
                if links["supply"] == attrs["supply_id"] and local["request"] and not local["irrigating"]:
                    result.append(self._spec("irrigation_resume", context, [local["request"], event.event_id], now, field=f))
        return result

    def relation_specs(self, event, candidate, context, event_by_id):
        del event, context, event_by_id
        mechanism = self.mechanism_registry.get(candidate.mechanism_id)
        return [RelationSpec(parent, mechanism.relation_semantics, mechanism.mechanism_id, mechanism.mechanism_id,
            context_evidence=copy.deepcopy(candidate.context_evidence),
            mechanism_group_id=candidate.candidate_id if len(candidate.parent_event_ids) > 1 else None,
            provenance={"parent_basis": "analytic_state_segment_or_frozen_observation_or_realized_supply_workflow",
                        "topology_edge_alone_sufficient": False}) for parent in candidate.parent_event_ids]

    @staticmethod
    def _same(left, right):
        if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-7)
        return left == right

    def validate_result(self, result):
        errors = []
        entities = {e.entity_id: e for e in result.entities}
        layers = {layer.relation_type_id: layer for layer in self.domain_spec.compile_relation_layers()}
        source_counts, target_counts = defaultdict(int), defaultdict(int)
        for rel in result.context_relations:
            layer, source, target = layers.get(rel.relation_type_id), entities.get(rel.source_entity_id), entities.get(rel.target_entity_id)
            if not layer or not source or not target or source.entity_type_id not in layer.source_type_ids or target.entity_type_id not in layer.target_type_ids:
                errors.append(f"Invalid typed Agriculture relation {rel.relation_id}")
            source_counts[(rel.relation_type_id, rel.source_entity_id)] += 1
            target_counts[(rel.relation_type_id, rel.target_entity_id)] += 1
        for kind, layer in layers.items():
            for entity in result.entities:
                if entity.entity_type_id in layer.source_type_ids and source_counts[(kind, entity.entity_id)] != 1:
                    errors.append(f"Missing/ambiguous Agriculture binding {entity.entity_id}")
                if layer.max_in_degree == 1 and entity.entity_type_id in layer.target_type_ids and target_counts[(kind, entity.entity_id)] != 1:
                    errors.append(f"Field must have a unique crop/sensor/irrigator {entity.entity_id}")
        for e in result.entities:
            a = e.attributes
            try:
                if e.entity_type_id == NS + "field":
                    valid = (0 < a["wilting_point_m3_m3"] < a["field_capacity_m3_m3"] < 1
                        and all(math.isfinite(a[k]) and a[k] > 0 for k in ("area_ha", "root_depth_m"))
                        and .35 <= a["depletion_fraction"] <= .55)
                elif e.entity_type_id == NS + "irrigator":
                    valid = math.isfinite(a["flow_m3_s"]) and a["flow_m3_s"] > 0 and 0 < a["application_efficiency"] <= 1
                elif e.entity_type_id == NS + "weather_zone":
                    valid = math.isfinite(a["mean_temperature_c"]) and math.isfinite(a["reference_et_mm_day"]) and a["reference_et_mm_day"] >= 0
                elif e.entity_type_id == NS + "crop_cohort":
                    valid = (math.isfinite(a["base_temperature_c"]) and math.isfinite(a["upper_temperature_c"])
                        and a["base_temperature_c"] < a["upper_temperature_c"]
                        and len(a["crop_coefficients"]) == len(a["thermal_thresholds_degree_days"]) + 1 == len(STAGES)
                        and all(v > 0 and math.isfinite(v) for v in a["crop_coefficients"] + a["thermal_thresholds_degree_days"])
                        and all(l < r for l, r in zip(a["thermal_thresholds_degree_days"], a["thermal_thresholds_degree_days"][1:])))
                elif e.entity_type_id == NS + "supply_unit":
                    valid = math.isfinite(a["rated_capacity_m3_s"]) and a["rated_capacity_m3_s"] > 0
                else:
                    valid = True
                if not valid:
                    errors.append(f"Invalid physical Agriculture attributes {e.entity_id}")
            except (KeyError, TypeError, ValueError):
                errors.append(f"Invalid physical Agriculture attributes {e.entity_id}")
        if errors:
            return errors
        data = self._graph_data(result.entities, result.context_relations)
        flow_demands = defaultdict(float)
        for links in data["field_links"].values():
            flow_demands[links["supply"]] += data["entity_attrs"][links["irrigator"]]["flow_m3_s"]
        if any(data["entity_attrs"][unit].get("rated_capacity_m3_s", -1) < demand - 1e-10 for unit, demand in flow_demands.items()):
            return ["Supply capacity cannot support its declared equipment connections"]
        state = self._initial_state(data, result.episode_attributes["scenario_family"], result.episode_id + "_workflow")
        context = EpisodeContext(result.episode_id, result.context_id, self.domain_id, result.start_time,
            result.duration_seconds, result.entities, result.context_relations, state=state)
        candidates = {c.fired_event_id: c for c in result.candidates if c.fired_event_id}
        incoming, seen = defaultdict(list), {}
        for rel in result.event_relations:
            incoming[rel.target_event_id].append(rel.source_event_id)
        for event in result.events:
            attrs, kind, now = event.attributes, event.event_type_id, event.temporal.anchor("occurrence_start")
            candidate = candidates.get(event.event_id)
            field, supply = attrs.get("field_id"), attrs.get("supply_id")
            if not candidate or field is not None and field not in state["local"] or field is None and supply not in state["supplies"]:
                errors.append(f"Unknown Agriculture event/candidate {event.event_id}")
                continue
            mechanism = self.mechanism_registry.get(candidate.mechanism_id)
            parents = [seen.get(eid) for eid in candidate.parent_event_ids]
            if (set(incoming[event.event_id]) != set(candidate.parent_event_ids) or
                kind != mechanism.target_event_type_id or event.provenance.get("mechanism_id") != candidate.mechanism_id or
                any(not p or p.event_type_id not in mechanism.source_event_type_ids for p in parents)):
                errors.append(f"Agriculture mechanism/parent mismatch {event.event_id}")
            if any(parent is None for parent in parents):
                continue
            roles = {p.role: p.entity_id for p in event.participants}
            if event.participants != candidate.participants or field and (roles.get("field") != field or any(
                entity_id != state["field_links"][field].get(role) for role, entity_id in roles.items() if role != "field")) or not field and roles.get("supply") != supply:
                errors.append(f"Agriculture typed participant mismatch {event.event_id}")
            if field and supply is not None and supply != state["field_links"][field]["supply"]:
                errors.append(f"Irrigation event refers to a supply outside its actual binding {event.event_id}")
            for parent in parents:
                expected = parent.attributes.get("field_id") == field if "field_id" in parent.attributes else parent.attributes.get("supply_id") == supply
                if not expected:
                    errors.append(f"Cross-field/cross-supply parent {event.event_id}")
            if not self.revalidate_candidate(candidate, context, seen, now).valid:
                errors.append(f"Invalid Agriculture replay precondition {event.event_id}")
            expected = self._event_attributes(event.event_id, candidate, context)
            if any(not self._same(attrs.get(key), value) for key, value in expected.items()):
                errors.append(f"Agriculture water/thermal/snapshot derivation mismatch {event.event_id}")
            if field:
                balance = (attrs["initial_depletion_mm"] + attrs["cumulative_et_mm"] - attrs["cumulative_net_irrigation_mm"]
                    - attrs["cumulative_effective_rain_mm"] + attrs["cumulative_deep_percolation_mm"])
                if not self._same(attrs["depletion_before_mm"], balance):
                    errors.append(f"Root-zone cumulative water balance mismatch {event.event_id}")
            if kind == RAIN:
                depth, before = attrs.get("effective_rain_depth_mm", -1), attrs["depletion_before_mm"]
                lo, hi = self.config["effective_rain_range_mm"]
                if not lo <= depth <= hi or not self._same(attrs.get("depletion_after_mm"), max(0., before - depth)) or not self._same(attrs.get("rain_deep_percolation_mm"), max(0., depth - before)):
                    errors.append(f"Effective-rain water balance mismatch {event.event_id}")
            elif kind == MILESTONE:
                target = state["entity_attrs"][state["field_links"][field]["crop"]]["thermal_thresholds_degree_days"][state["local"][field]["stage"]]
                if not self._same(attrs.get("target_gdd"), target) or not self._same(attrs["gdd_before"], target):
                    errors.append(f"Thermal milestone does not match accumulated degree-days {event.event_id}")
            elif kind in (REPORT, REQUEST):
                if kind == REPORT and any(not self._same(attrs.get(key), parents[0].attributes.get(key)) for key in SAMPLE_FIELDS):
                    errors.append(f"Sensor report changed frozen sample {event.event_id}")
            elif kind == END:
                start = parents[0]
                if (not self._same(now - start.temporal.anchor("occurrence_start"), start.attributes["planned_duration_seconds"]) or
                    not self._same(attrs["delivered_gross_volume_m3"], start.attributes["planned_gross_volume_m3"])):
                    errors.append(f"Irrigation duration/delivery mismatch {event.event_id}")
            elif kind == INTERRUPTED:
                if len(parents) != 2 or parents[0].event_type_id != START or parents[1].event_type_id != FAILED or not self._same(now, parents[1].temporal.anchor("occurrence_start")):
                    errors.append(f"Irrigation interruption lacks its actual start and supply failure {event.event_id}")
            self.apply_event(event, candidate, context)
            seen[event.event_id] = event
        return errors

    def render_episode_text(self, result):
        sentences, claims = [], []
        fields = ("field_id", "supply_id", "depletion_before_mm", "depletion_after_mm", "taw_mm", "raw_mm",
            "ks_before", "initial_depletion_mm", "cumulative_et_mm", "cumulative_net_irrigation_mm",
            "cumulative_effective_rain_mm", "cumulative_deep_percolation_mm", "gdd_before", "target_gdd",
            "stage_before_name", "stage_after_name", "milestone_semantics", "mean_temperature_c", "reference_et_mm_day",
            "crop_coefficient_before", "thermal_rate_degree_days_per_day", "effective_rain_depth_mm", "rain_deep_percolation_mm",
            "sample_event_id", "sampled_at_offset_seconds", "sampled_depletion_mm", "sampled_water_content_m3_m3",
            "sampled_low_water", "stale_report", "target_depletion_mm", "planned_gross_volume_m3", "planned_duration_seconds",
            "equipment_flow_m3_s", "application_efficiency", "delivered_gross_volume_m3", "irrigation_start_event_id")
        for event in result.events:
            attrs = {key: event.attributes[key] for key in fields if key in event.attributes}
            roles = ", ".join(f"{p.role}={p.entity_id}" for p in event.participants)
            sentences.append(f"+{event.temporal.anchor('occurrence_start') / DAY:.6f} day {event.event_id} "
                f"{event.event_type_id} ({roles}); {json.dumps(attrs, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}.")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                "event_ids": [event.event_id], "relation_ids": [], "evidence_fields":
                ["event_type_id", "temporal", "participants"] + [f"attributes.{key}" for key in attrs]})
        for rel in result.event_relations:
            predicate = PREDICATES[rel.relation_type_id.removeprefix(NS)]
            sentences.append(f"{rel.source_event_id} {predicate} {rel.target_event_id} "
                f"({rel.relation_class}, +{rel.temporal_link.lag_seconds / DAY:.6f} day; rule={rel.rule_id}).")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                "event_ids": [rel.source_event_id, rel.target_event_id], "relation_ids": [rel.relation_id],
                "relation_assertions": [{"relation_id": rel.relation_id, "asserted_relation_class": rel.relation_class, "surface_predicate": predicate}],
                "evidence_fields": ["source_event_id", "target_event_id", "temporal_link", "rule_id"]})
        return {"alignment_id": result.episode_id + "_text_alignment", "episode_id": result.episode_id,
            "domain": self.domain_id, "projection_mode": "deterministic_grounded_projection",
            "sentences": sentences, "text": " ".join(sentences), "claims": claims}


def build_agriculture_temporal_models():
    registry = build_standard_temporal_registry()
    registry.register(ExponentialArrivalModel(NS + "establishment_arrival.prior_v1", 2.0 / DAY))
    registry.register(ExponentialArrivalModel(NS + "rain_arrival.prior_v1", 1.0 / (12 * DAY)))
    registry.register(ExponentialArrivalModel(NS + "supply_failure.prior_v1", 1.0 / (12 * DAY)))
    registry.register(DeterministicDelayModel(NS + "physical_delay.prior_v1"))
    for name, median_days, sigma, minimum in (("repair_delay", 4., .4, .05), ("telemetry_delay", .12, .5, .001),
        ("control_delay", .1, .4, .001), ("actuation_delay", .03, .4, .001), ("log_ingestion_delay", .005, .4, .0001)):
        registry.register(ConditionalLogNormalModel(NS + name + ".prior_v1", math.log(median_days * DAY), sigma, {}, minimum_seconds=minimum * DAY))
    return registry
