"""Idealized independent one-box climate anomaly units and grounded monitoring.

Not a spatial climate forecast, weather generator, attribution study or fitted
climate-normal dataset. Forcing pulses and parameters are synthetic priors.
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from data_generation.generation_core.domain import CandidateSpec, CandidateUpdate, EpisodeContext, ObservationPlan, RelationSpec
from data_generation.generation_core.domain_spec import load_domain_spec
from data_generation.generation_core.mechanisms import load_mechanism_registry
from data_generation.generation_core.models import ContextEvidence, Entity, EventParticipant, EventRecord, StatePredicateEvidence, TemporalExtent
from data_generation.generation_core.temporal import ConditionalLogNormalModel, DeterministicDelayModel, ExponentialArrivalModel, build_standard_temporal_registry
from data_generation.generation_core.topology import SparseHeterogeneousTopologyGenerator
from .physics import advance_temperature, time_to_temperature

NS, YEAR = "climate.", 365.25 * 86400.0
START, END, SUM = (NS + k for k in ("radiative_forcing.started", "radiative_forcing.ended", "forcing_superposition.evaluated"))
WARM, WARM_END, WARM_PERSIST = (NS + "temperature.warm_exceedance." + k for k in ("started", "ended", "persistence_confirmed"))
COLD, COLD_END, COLD_PERSIST = (NS + "temperature.cold_exceedance." + k for k in ("started", "ended", "persistence_confirmed"))
SAMPLE, REPORT = NS + "anomaly.sampled", NS + "anomaly.reported"
BULLETIN, WITHDRAWN = NS + "monitoring.bulletin.issued", NS + "monitoring.bulletin.withdrawn"
EVENT_TYPES = (START, END, SUM, WARM, WARM_END, WARM_PERSIST, COLD, COLD_END, COLD_PERSIST, SAMPLE, REPORT, BULLETIN, WITHDRAWN)
SCENARIOS = ("warming_pulse", "cooling_pulse", "combined_warming", "opposing_forcings")
SAMPLE_FIELDS = ("sample_event_id", "sample_sequence", "sampled_at_offset_seconds", "sampled_temperature_anomaly_k", "sampled_status")
PREDICATES = {
    "forcing_end": "ended its specified forcing pulse at",
    "forcing_superposition": "was an actual active forcing term in the arithmetic superposition at",
    "warm_onset": "anchored the energy-balance segment deriving warm threshold entry at",
    "warm_end": "anchored the energy-balance segment deriving warm threshold exit at",
    "cold_onset": "anchored the energy-balance segment deriving cold threshold entry at",
    "cold_end": "anchored the energy-balance segment deriving cold threshold exit at",
    "warm_persistence": "remained continuously in warm exceedance for the configured confirmation duration until",
    "cold_persistence": "remained continuously in cold exceedance for the configured confirmation duration until",
    "periodic_sample": "preceded the next observation-policy appointment at",
    "sample_report": "was transmitted with its original frozen snapshot as",
    "bulletin_raise": "was the latest available sample report used to issue",
    "bulletin_clear": "was required to withdraw the previous sample-based bulletin at",
}


class ClimatePackage:
    domain_id = "climate"

    def __init__(self, config=None):
        self.config = {"base_seed": 20261011, "context_count": 4, "nodes_per_context": 1000,
            "duration_seconds": 30 * YEAR, "start_time": "2000-01-01T00:00:00+00:00",
            "positive_forcing_range_w_m2": [1.2, 2.8], "negative_forcing_range_w_m2": [-2.5, -1.],
            "forcing_duration_range_years": [5., 14.], "sample_period_years": 1., "persistence_years": 1.,
            "scenario_weights": dict(zip(SCENARIOS, (.3, .25, .25, .2))), **(config or {})}
        for key in ("base_seed", "context_count", "nodes_per_context"):
            v = self.config[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or int(v) != v:
                raise ValueError(f"{key} must be a finite integer")
            self.config[key] = int(v)
        if self.config["context_count"] < 1 or self.config["nodes_per_context"] < 20:
            raise ValueError("Climate needs at least one context and 20 entities")
        for key in ("duration_seconds", "sample_period_years", "persistence_years"):
            v = float(self.config[key])
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{key} must be finite and positive")
            self.config[key] = v
        for key in ("positive_forcing_range_w_m2", "negative_forcing_range_w_m2", "forcing_duration_range_years"):
            values = self.config[key]
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"{key} must be a lower/upper pair")
            lo, hi = map(float, values)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi or (hi >= 0 if key == "negative_forcing_range_w_m2" else lo <= 0):
                raise ValueError(f"Invalid {key}")
            self.config[key] = [lo, hi]
        weights = self.config["scenario_weights"]
        if not weights or set(weights) - set(SCENARIOS) or any(not math.isfinite(float(v)) or float(v) < 0 for v in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("Invalid Climate scenario weights")
        self.domain_spec = load_domain_spec(Path(__file__).with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(Path(__file__).with_name("mechanism_specs.json"),
            known_event_type_ids=EVENT_TYPES, known_context_relation_type_ids=self.domain_spec.context_relation_type_ids)
        self._context_cache = {}

    def catalog(self):
        roles = {k: NS + v for k, v in (("unit", "climate_unit"), ("driver", "forcing_driver"),
            ("driver_a", "forcing_driver"), ("driver_b", "forcing_driver"), ("observer", "anomaly_observer"), ("service", "climate_service"))}
        return {"domain": self.domain_id, "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_not_empirically_fitted",
            "scope": {"physics": "deterministic_piecewise_constant_one_box_anomaly_response",
                "unit": "fictional_independent_area_mean_climate_box_not_a_real_region",
                "baseline": "synthetic_stationary_reference_not_a_measured_climate_normal",
                "monitoring": "scheduled_instantaneous_model_anomaly_snapshots_not_annual_averages",
                "persistence": "configured_continuous_threshold_duration_not_climate_attribution",
                "weather_or_GCM": False, "statistical_realism_claimed": False},
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {"mode": "deterministic_grounded_projection", "free_form_causal_completion": False},
            "entity_types": [{"entity_type_id": NS + k, "parent_type_id": None, "required_attributes": attrs}
                for k, attrs in (("climate_unit", ["heat_capacity_j_m2_k", "damping_w_m2_k", "warm_threshold_k", "cold_threshold_k"]),
                    ("forcing_driver", ["forcing_semantics"]), ("anomaly_observer", ["sampling_policy"]), ("climate_service", ["bulletin_policy"]))],
            "event_types": [{"event_type_id": k, "parent_type_id": k.rsplit(".", 1)[0],
                "required_attributes": ["unit_id", "workflow_id", "scenario_family"], "participant_role_types": roles} for k in EVENT_TYPES],
            "context_relation_types": [{"relation_type_id": k, "directed": True, "does_not_imply_event_parenthood": True} for k in self.domain_spec.context_relation_type_ids],
            "event_relation_types": [{"relation_type_id": NS + key, "relation_class": self.mechanism_registry.get(NS + key).relation_semantics,
                "text_rendering": {"surface_predicate": predicate, "asserted_relation_class": self.mechanism_registry.get(NS + key).relation_semantics}}
                for key, predicate in PREDICATES.items()], "mechanisms": self.mechanism_registry.descriptors()}

    @staticmethod
    def _graph_data(entities, relations):
        bindings = {(r.relation_type_id, r.source_entity_id): r.target_entity_id for r in relations}
        drivers, observers = defaultdict(list), {}
        for rel in relations:
            if rel.relation_type_id == NS + "driver_forces_unit":
                drivers[rel.target_entity_id].append(rel.source_entity_id)
            elif rel.relation_type_id == NS + "observer_samples_unit":
                observers[rel.target_entity_id] = rel.source_entity_id
        return {"entity_attrs": {e.entity_id: e.attributes for e in entities}, "bindings": bindings,
            "drivers": dict(drivers), "observers": observers,
            "unit_ids": [e.entity_id for e in entities if e.entity_type_id == NS + "climate_unit"],
            "relation_ids": {(r.relation_type_id, r.source_entity_id, r.target_entity_id): r.relation_id for r in relations}}

    def _build_context(self, index):
        rng, n = random.Random(self.config["base_seed"] + 7919 * index), self.config["nodes_per_context"]
        context_id = f"climate_context_{index:03d}"
        nu = (n - max(1, n // 50)) // 4
        ns = n - 4 * nu
        ids = {k: [f"{context_id}_{k}_{i:05d}" for i in range(count)] for k, count in
            (("climate_unit", nu), ("forcing_driver", 2 * nu), ("anomaly_observer", nu), ("climate_service", ns))}
        entities = []
        for unit in ids["climate_unit"]:
            damping, tau = rng.uniform(.8, 1.6), rng.uniform(3, 8) * YEAR
            entities.append(Entity(unit, self.domain_id, NS + "climate_unit", {
                "heat_capacity_j_m2_k": damping * tau, "damping_w_m2_k": damping,
                "warm_threshold_k": .7, "cold_threshold_k": -.7,
                "reference_semantics": "fictional_stationary_anomaly_zero_not_historical_normal",
                "response_scope": "independent_one_box_not_validated_real_regional_climate"}, context_id))
        for driver in ids["forcing_driver"]:
            entities.append(Entity(driver, self.domain_id, NS + "forcing_driver", {"forcing_semantics": "externally_prescribed_effective_radiative_anomaly_pulse_not_emissions"}, context_id))
        for observer in ids["anomaly_observer"]:
            entities.append(Entity(observer, self.domain_id, NS + "anomaly_observer", {"sampling_policy": "ideal_point_snapshot_of_model_anomaly_not_raw_weather_station"}, context_id))
        for service in ids["climate_service"]:
            entities.append(Entity(service, self.domain_id, NS + "climate_service", {"bulletin_policy": "latest_available_sample_status_not_current_state_or_attribution"}, context_id))
        required = {NS + "driver_forces_unit": [(d, ids["climate_unit"][i // 2]) for i, d in enumerate(ids["forcing_driver"])],
            NS + "observer_samples_unit": list(zip(ids["anomaly_observer"], ids["climate_unit"])),
            NS + "observer_reports_to_service": [(o, ids["climate_service"][i % ns]) for i, o in enumerate(ids["anomaly_observer"])]}
        topology = SparseHeterogeneousTopologyGenerator().generate(context_id=context_id, domain=self.domain_id,
            entities=entities, layers=[replace(layer, required_pairs=tuple(required[layer.relation_type_id])) for layer in self.domain_spec.compile_relation_layers()],
            seed=self.config["base_seed"] + 7919 * index, observation_window_seconds=self.config["duration_seconds"])
        return {**self._graph_data(entities, topology.relations), "entities": entities, "relations": topology.relations, "statistics": topology.statistics}

    @staticmethod
    def _initial_state(data, scenario, workflow):
        return {**{k: data[k] for k in ("entity_attrs", "bindings", "drivers", "observers", "relation_ids")},
            "scenario": scenario, "workflow_id": workflow,
            "local": {unit: {"temperature": 0., "last_time": 0., "forcing_energy": 0., "feedback_energy": 0.,
                "active": {}, "used_drivers": set(), "epoch": 0, "warm": False, "cold": False,
                "warm_start": None, "cold_start": None, "warm_confirmed": False, "cold_confirmed": False,
                "sample_sequence": 0, "last_sample": None, "latest_sequence": 0, "latest_report": None,
                "latest_status": "normal", "bulletin": None, "bulletin_status": None, "reported_samples": set()} for unit in data["unit_ids"]}}

    def create_episode_context(self, episode_index, seed):
        index = episode_index % self.config["context_count"]
        if index not in self._context_cache:
            self._context_cache[index] = self._build_context(index)
        data, rng = self._context_cache[index], random.Random(seed ^ 0x434C494D)
        weights = self.config["scenario_weights"]
        scenario = rng.choices(list(weights), weights=list(weights.values()))[0]
        unit = rng.choice(data["unit_ids"])
        episode = f"climate_episode_{episode_index:07d}"
        state = self._initial_state(data, scenario, episode + "_workflow")
        state["selected_unit"] = unit
        return EpisodeContext(episode, f"climate_context_{index:03d}", self.domain_id, self.config["start_time"],
            self.config["duration_seconds"], data["entities"], data["relations"], state=state,
            context_attributes={"topology_profile": data["statistics"]}, episode_attributes={
                "scenario_family": scenario, "selected_unit": unit, "empirical_realism_claimed": False,
                "baseline_semantics": "synthetic_stationary_reference_not_measured_climate_normal",
                "physics_scope": "independent_one_box_climate_anomaly_not_GCM_or_daily_weather"})

    def _view(self, state, unit, now):
        local = dict(state["local"][unit])
        a = state["entity_attrs"][unit]
        forcing = sum(v["amplitude"] for v in local["active"].values())
        local["temperature"], forced, feedback = advance_temperature(local["temperature"], max(0., now - local["last_time"]),
            forcing, a["damping_w_m2_k"], a["heat_capacity_j_m2_k"])
        local["forcing_energy"] += forced
        local["feedback_energy"] += feedback
        local["last_time"] = now
        return local

    def _spec(self, key, context, unit, parents=(), now=0., delay=0., extra=None, driver=None):
        state, roles, pairs = context.state, {"unit": unit}, set()
        if driver:
            roles["driver"] = driver
            pairs.add((NS + "driver_forces_unit", driver, unit))
        if key == "forcing_superposition":
            pairs.update((NS + "driver_forces_unit", v["driver"], unit) for v in state["local"][unit]["active"].values())
            for role, (_, value) in zip(("driver_a", "driver_b"), sorted(state["local"][unit]["active"].items())):
                roles[role] = value["driver"]
        if key in ("first_sample", "periodic_sample", "sample_report", "bulletin_raise", "bulletin_clear"):
            observer = state["observers"][unit]
            roles["observer"] = observer
            pairs.add((NS + "observer_samples_unit", observer, unit))
            if key in ("sample_report", "bulletin_raise", "bulletin_clear"):
                roles["service"] = state["bindings"][(NS + "observer_reports_to_service", observer)]
                pairs.add((NS + "observer_reports_to_service", observer, roles["service"]))
        mechanism = self.mechanism_registry.get(NS + key)
        attrs = {"unit_id": unit, "workflow_id": state["workflow_id"], "scenario_family": state["scenario"], **(extra or {})}
        if driver:
            attrs["driver_id"] = driver
        phase = -30 if key in ("forcing_arrival", "forcing_end") else -20 if key == "forcing_superposition" else (
            -10 if key in ("warm_onset", "warm_end", "cold_onset", "cold_end") else
            0 if key in ("warm_persistence", "cold_persistence") else 10 if key in ("first_sample", "periodic_sample") else 20 if key == "sample_report" else 30)
        return CandidateSpec(mechanism.mechanism_id, mechanism.target_event_type_id, list(parents),
            [EventParticipant(e, r) for r, e in roles.items()], mechanism.generation_temporal_model_ref,
            {"delay_seconds": delay} if mechanism.temporal_model_family == "deterministic_delay" else {},
            activation_time=now, combination=mechanism.parent_combination, phase_priority=phase, attributes=attrs,
            context_evidence=ContextEvidence(context_relation_ids=[state["relation_ids"][p] for p in sorted(pairs)], entity_ids=list(roles.values()), evaluated_at_offset_seconds=now),
            provenance={"parameter_status": mechanism.parameter_status, "input_anchor": "occurrence_start"})

    def seed_candidates(self, context, rng):
        del rng
        unit, scenario = context.state["selected_unit"], context.state["scenario"]
        signs = [-1] if scenario == "cooling_pulse" else [1, -1] if scenario == "opposing_forcings" else [1, 1] if scenario == "combined_warming" else [1]
        candidates = [self._spec("forcing_arrival", context, unit, driver=driver, extra={"forcing_sign": sign})
            for driver, sign in zip(context.state["drivers"][unit], signs)]
        candidates.append(self._spec("first_sample", context, unit, delay=self.config["sample_period_years"] * YEAR))
        return candidates

    def revalidate_candidate(self, candidate, context, event_by_id, now):
        del event_by_id
        state, attrs, key = context.state, candidate.attributes, candidate.mechanism_id.removeprefix(NS)
        unit = attrs["unit_id"]
        local, a = self._view(state, unit, now), state["entity_attrs"][unit]
        forcing = sum(v["amplitude"] for v in local["active"].values())
        derivative = (forcing - a["damping_w_m2_k"] * local["temperature"]) / a["heat_capacity_j_m2_k"]
        valid = True
        if key == "forcing_arrival":
            valid = attrs["driver_id"] not in local["used_drivers"]
        elif key == "forcing_end":
            valid = attrs["forcing_start_event_id"] in local["active"]
        elif key == "forcing_superposition":
            valid = local["epoch"] == attrs["state_epoch"] and set(candidate.parent_event_ids) == set(local["active"]) and len(local["active"]) >= 2
        elif key in ("warm_onset", "warm_end", "cold_onset", "cold_end"):
            warm, ending = key.startswith("warm"), key.endswith("end")
            flag, threshold = ("warm", a["warm_threshold_k"]) if warm else ("cold", a["cold_threshold_k"])
            valid = local["epoch"] == attrs["state_epoch"] and local[flag] == ending
            if now >= candidate.scheduled_time - 1e-5:
                direction = -1 if warm == ending else 1
                valid &= direction * derivative > 0 and abs(local["temperature"] - threshold) <= 1e-7
        elif key in ("warm_persistence", "cold_persistence"):
            flag = "warm" if key.startswith("warm") else "cold"
            valid = local[flag] and local[flag + "_start"] in candidate.parent_event_ids and not local[flag + "_confirmed"]
        elif key in ("first_sample", "periodic_sample"):
            valid = local["last_sample"] is None if key == "first_sample" else local["last_sample"] in candidate.parent_event_ids
        elif key == "sample_report":
            valid = attrs["sample_event_id"] not in local["reported_samples"]
        elif key == "bulletin_raise":
            valid = local["latest_report"] in candidate.parent_event_ids and local["latest_status"] != "normal" and local["bulletin"] is None
        elif key == "bulletin_clear":
            valid = (local["latest_report"] in candidate.parent_event_ids and local["bulletin"] in candidate.parent_event_ids
                and local["bulletin"] is not None and local["latest_status"] != local["bulletin_status"])
        else:
            raise ValueError(f"Unsupported Climate mechanism {key}")
        evidence = copy.deepcopy(candidate.context_evidence)
        evidence.evaluated_at_offset_seconds = now
        evidence.state_predicates = [StatePredicateEvidence(unit, key + "_precondition", "equals", True, bool(valid), bool(valid))]
        if "state_epoch" in attrs:
            evidence.state_predicates.append(StatePredicateEvidence(unit, "state_epoch", "equals", attrs["state_epoch"], local["epoch"], attrs["state_epoch"] == local["epoch"]))
        return CandidateUpdate(bool(valid), None if valid else key + "_state_superseded", context_evidence=evidence)

    def _attributes(self, event_id, candidate, context):
        state, attrs = context.state, copy.deepcopy(candidate.attributes)
        kind, now, unit = candidate.target_event_type_id, candidate.scheduled_time, attrs["unit_id"]
        local, a = self._view(state, unit, now), state["entity_attrs"][unit]
        components = {eid: v["amplitude"] for eid, v in sorted(local["active"].items())}
        forcing = sum(components.values())
        attrs.update(temperature_anomaly_before_k=local["temperature"], initial_temperature_anomaly_k=0.,
            active_forcing_components_w_m2=components, total_forcing_before_w_m2=forcing,
            equilibrium_anomaly_before_k=forcing / a["damping_w_m2_k"], heat_capacity_j_m2_k=a["heat_capacity_j_m2_k"],
            damping_w_m2_k=a["damping_w_m2_k"], response_time_seconds=a["heat_capacity_j_m2_k"] / a["damping_w_m2_k"],
            cumulative_forcing_energy_j_m2=local["forcing_energy"], cumulative_feedback_energy_j_m2=local["feedback_energy"],
            warm_threshold_k=a["warm_threshold_k"], cold_threshold_k=a["cold_threshold_k"],
            warm_exceedance_before=local["warm"], cold_exceedance_before=local["cold"],
            anomaly_reference="synthetic_stationary_reference_not_historical_climate_normal")
        if kind == END:
            source = local["active"][attrs["forcing_start_event_id"]]
            attrs.update(forcing_amplitude_w_m2=source["amplitude"], total_forcing_after_w_m2=forcing - source["amplitude"])
        elif kind == SUM:
            attrs["superposition_semantics"] = "arithmetic_sum_of_actual_active_pulses_not_causal_attribution"
        elif kind in (WARM_PERSIST, COLD_PERSIST):
            attrs.update(confirmed_duration_seconds=self.config["persistence_years"] * YEAR,
                persistence_semantics="configured_model_threshold_duration_not_WMO_normal_or_attribution")
        elif kind == SAMPLE:
            status = "warm" if local["temperature"] >= a["warm_threshold_k"] else "cold" if local["temperature"] <= a["cold_threshold_k"] else "normal"
            attrs.update(sample_event_id=event_id, sample_sequence=local["sample_sequence"] + 1,
                sampled_at_offset_seconds=now, sampled_temperature_anomaly_k=local["temperature"], sampled_status=status,
                sampling_semantics="scheduled_point_snapshot_of_low_frequency_model_not_annual_mean")
        elif kind == REPORT:
            attrs["stale_report"] = attrs["sample_sequence"] <= local["latest_sequence"]
        elif kind == BULLETIN:
            attrs.update(bulletin_status=local["latest_status"], information_basis="latest_available_frozen_sample_not_current_physical_state_or_attribution")
        elif kind == WITHDRAWN:
            attrs.update(withdrawn_status=local["bulletin_status"], replacement_sample_status=local["latest_status"])
        return attrs

    def materialize_event(self, event_id, candidate, context, rng):
        attrs = self._attributes(event_id, candidate, context)
        if candidate.target_event_type_id == START:
            interval = "positive_forcing_range_w_m2" if attrs["forcing_sign"] == 1 else "negative_forcing_range_w_m2"
            amplitude, duration = rng.uniform(*self.config[interval]), rng.uniform(*self.config["forcing_duration_range_years"]) * YEAR
            attrs.update(forcing_amplitude_w_m2=amplitude, forcing_duration_seconds=duration,
                total_forcing_after_w_m2=attrs["total_forcing_before_w_m2"] + amplitude,
                forcing_semantics="prescribed_effective_radiative_anomaly_not_CO2_concentration_or_emissions")
        return EventRecord(event_id, context.episode_id, self.domain_id, candidate.target_event_type_id,
            TemporalExtent(candidate.scheduled_time, precision="second"), list(candidate.participants), attrs,
            event_role="root" if not candidate.parent_event_ids else "derived",
            provenance={"generator": "idealized_one_box_climate_anomaly", "mechanism_id": candidate.mechanism_id})

    def observation_plan(self, event, candidate, context):
        del candidate, context
        return ObservationPlan(recording_model_ref=NS + "log_ingestion_delay.prior_v1",
            source="synthetic_climate_monitoring" if event.event_type_id in (SAMPLE, REPORT, BULLETIN, WITHDRAWN) else "offline_idealized_climate_model_truth")

    def apply_event(self, event, candidate, context):
        state, attrs, kind = context.state, event.attributes, event.event_type_id
        unit, now = attrs["unit_id"], event.temporal.anchor("occurrence_start")
        state["local"][unit] = self._view(state, unit, now)
        local = state["local"][unit]
        if kind == START:
            local["active"][event.event_id] = {"amplitude": attrs["forcing_amplitude_w_m2"], "driver": attrs["driver_id"]}
            local["used_drivers"].add(attrs["driver_id"])
            local["epoch"] += 1
        elif kind == END:
            local["active"].pop(attrs["forcing_start_event_id"])
            local["epoch"] += 1
        elif kind in (WARM, WARM_END, COLD, COLD_END):
            flag = "warm" if kind in (WARM, WARM_END) else "cold"
            local[flag] = kind in (WARM, COLD)
            local[flag + "_start"] = event.event_id if local[flag] else None
            local[flag + "_confirmed"] = False
            local["epoch"] += 1
        elif kind in (WARM_PERSIST, COLD_PERSIST):
            local[("warm" if kind == WARM_PERSIST else "cold") + "_confirmed"] = True
        elif kind == SAMPLE:
            local.update(sample_sequence=attrs["sample_sequence"], last_sample=event.event_id)
        elif kind == REPORT:
            local["reported_samples"].add(attrs["sample_event_id"])
            if not attrs["stale_report"]:
                local.update(latest_sequence=attrs["sample_sequence"], latest_report=event.event_id, latest_status=attrs["sampled_status"])
        elif kind == BULLETIN:
            local.update(bulletin=event.event_id, bulletin_status=attrs["bulletin_status"])
        elif kind == WITHDRAWN:
            local.update(bulletin=None, bulletin_status=None)

    def _predictions(self, context, event):
        unit, state = event.attributes["unit_id"], context.state
        local, a = state["local"][unit], state["entity_attrs"][unit]
        forcing = sum(v["amplitude"] for v in local["active"].values())
        derivative = forcing - a["damping_w_m2_k"] * local["temperature"]
        specs = []
        for flag in ("warm", "cold"):
            ending = local[flag]
            direction = -1 if (flag == "warm") == ending else 1
            if direction * derivative <= 0:
                continue
            target = a[flag + "_threshold_k"]
            delay = time_to_temperature(local["temperature"], target, forcing, a["damping_w_m2_k"], a["heat_capacity_j_m2_k"])
            if delay is not None:
                specs.append(self._spec(flag + ("_end" if ending else "_onset"), context, unit, [event.event_id],
                    event.temporal.anchor("occurrence_start"), delay=delay, extra={"state_epoch": local["epoch"], "threshold_k": target}))
        return specs

    def _bulletin_actions(self, context, unit, now):
        local = context.state["local"][unit]
        if local["bulletin"] and local["latest_status"] != local["bulletin_status"]:
            return [self._spec("bulletin_clear", context, unit, [local["bulletin"], local["latest_report"]], now)]
        if not local["bulletin"] and local["latest_status"] != "normal":
            return [self._spec("bulletin_raise", context, unit, [local["latest_report"]], now)]
        return []

    def spawn_candidates(self, event, candidate, context, event_by_id, rng):
        del candidate, event_by_id, rng
        kind, attrs, now = event.event_type_id, event.attributes, event.temporal.anchor("occurrence_start")
        unit, state, parents = attrs["unit_id"], context.state, [event.event_id]
        local, specs = state["local"][unit], []
        if kind == START:
            specs.append(self._spec("forcing_end", context, unit, parents, now, attrs["forcing_duration_seconds"],
                driver=attrs["driver_id"], extra={"forcing_start_event_id": event.event_id}))
            if len(local["active"]) >= 2:
                specs.append(self._spec("forcing_superposition", context, unit, sorted(local["active"]), now, extra={"state_epoch": local["epoch"]}))
            else:
                specs += self._predictions(context, event)
        elif kind in (END, SUM, WARM, WARM_END, COLD, COLD_END):
            specs += self._predictions(context, event)
            if kind in (WARM, COLD):
                specs.append(self._spec("warm_persistence" if kind == WARM else "cold_persistence", context,
                    unit, parents, now, self.config["persistence_years"] * YEAR))
        elif kind == SAMPLE:
            specs.append(self._spec("periodic_sample", context, unit, parents, now, self.config["sample_period_years"] * YEAR))
            specs.append(self._spec("sample_report", context, unit, parents, now, extra={key: attrs[key] for key in SAMPLE_FIELDS}))
        elif kind in (REPORT, WITHDRAWN):
            if kind == WITHDRAWN or not attrs["stale_report"]:
                specs += self._bulletin_actions(context, unit, now)
        return specs

    def relation_specs(self, event, candidate, context, event_by_id):
        del event, context, event_by_id
        mechanism = self.mechanism_registry.get(candidate.mechanism_id)
        return [RelationSpec(parent, mechanism.relation_semantics, mechanism.mechanism_id, mechanism.mechanism_id,
            context_evidence=copy.deepcopy(candidate.context_evidence), mechanism_group_id=candidate.candidate_id if len(candidate.parent_event_ids) > 1 else None,
            provenance={"parent_basis": "realized_forcing_or_state_segment_or_frozen_sample_or_policy", "topology_edge_alone_sufficient": False}) for parent in candidate.parent_event_ids]

    @staticmethod
    def _same(left, right):
        if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-7)
        return left == right

    def validate_result(self, result):
        errors, entities = [], {e.entity_id: e for e in result.entities}
        layers = {l.relation_type_id: l for l in self.domain_spec.compile_relation_layers()}
        source_counts, target_counts = defaultdict(int), defaultdict(int)
        for rel in result.context_relations:
            layer, source, target = layers.get(rel.relation_type_id), entities.get(rel.source_entity_id), entities.get(rel.target_entity_id)
            if not layer or not source or not target or source.entity_type_id not in layer.source_type_ids or target.entity_type_id not in layer.target_type_ids:
                errors.append(f"Invalid typed Climate relation {rel.relation_id}")
            source_counts[(rel.relation_type_id, rel.source_entity_id)] += 1
            target_counts[(rel.relation_type_id, rel.target_entity_id)] += 1
        for key, layer in layers.items():
            for entity in result.entities:
                if entity.entity_type_id in layer.source_type_ids and source_counts[(key, entity.entity_id)] != 1:
                    errors.append(f"Missing/ambiguous Climate binding {entity.entity_id}")
                if entity.entity_type_id in layer.target_type_ids and layer.max_in_degree == 1 and target_counts[(key, entity.entity_id)] != 1:
                    errors.append(f"Climate unit needs exactly one observer {entity.entity_id}")
        for entity in result.entities:
            if entity.entity_type_id != NS + "climate_unit":
                continue
            a = entity.attributes
            try:
                valid = (all(math.isfinite(a[k]) and a[k] > 0 for k in ("heat_capacity_j_m2_k", "damping_w_m2_k", "warm_threshold_k"))
                    and math.isfinite(a["cold_threshold_k"]) and a["cold_threshold_k"] < 0)
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(f"Invalid Climate physical parameters {entity.entity_id}")
        if errors:
            return errors
        data = self._graph_data(result.entities, result.context_relations)
        if any(len(data["drivers"].get(unit, [])) != 2 for unit in data["unit_ids"]):
            return ["Each idealized Climate unit requires its two declared forcing drivers"]
        state = self._initial_state(data, result.episode_attributes["scenario_family"], result.episode_id + "_workflow")
        context = EpisodeContext(result.episode_id, result.context_id, self.domain_id, result.start_time, result.duration_seconds,
            result.entities, result.context_relations, state=state)
        candidates = {c.fired_event_id: c for c in result.candidates if c.fired_event_id}
        incoming, seen = defaultdict(list), {}
        for rel in result.event_relations:
            incoming[rel.target_event_id].append(rel.source_event_id)
        for event in result.events:
            attrs, kind, now = event.attributes, event.event_type_id, event.temporal.anchor("occurrence_start")
            unit, candidate = attrs.get("unit_id"), candidates.get(event.event_id)
            if unit not in state["local"] or not candidate:
                errors.append(f"Unknown Climate event/candidate {event.event_id}")
                continue
            parents = [seen.get(eid) for eid in candidate.parent_event_ids]
            mechanism = self.mechanism_registry.get(candidate.mechanism_id)
            if (set(incoming[event.event_id]) != set(candidate.parent_event_ids) or kind != mechanism.target_event_type_id
                or event.provenance.get("mechanism_id") != candidate.mechanism_id
                or any(not p or p.event_type_id not in mechanism.source_event_type_ids or p.attributes.get("unit_id") != unit for p in parents)):
                errors.append(f"Climate mechanism/parent mismatch {event.event_id}")
            if any(p is None for p in parents):
                continue
            roles = {p.role: p.entity_id for p in event.participants}
            observer = data["observers"][unit]
            expected_roles = {"unit": unit, "observer": observer, "service": data["bindings"][(NS + "observer_reports_to_service", observer)]}
            if kind == SUM:
                expected_roles.update({role: parent.attributes["driver_id"] for role, parent in zip(("driver_a", "driver_b"), parents)})
            if "driver_id" in attrs:
                expected_roles["driver"] = attrs["driver_id"]
                if data["bindings"].get((NS + "driver_forces_unit", attrs["driver_id"])) != unit:
                    errors.append(f"Forcing driver does not bind to Climate unit {event.event_id}")
            if event.participants != candidate.participants or roles.get("unit") != unit or any(expected_roles.get(r) != eid for r, eid in roles.items()):
                errors.append(f"Climate participant binding mismatch {event.event_id}")
            if not self.revalidate_candidate(candidate, context, seen, now).valid:
                errors.append(f"Invalid Climate replay precondition {event.event_id}")
            expected = self._attributes(event.event_id, candidate, context)
            if any(not self._same(attrs.get(k), v) for k, v in expected.items()):
                errors.append(f"Climate energy/state/sample derivation mismatch {event.event_id}")
            energy = attrs["cumulative_forcing_energy_j_m2"] - attrs["cumulative_feedback_energy_j_m2"]
            if not math.isclose(attrs["heat_capacity_j_m2_k"] * attrs["temperature_anomaly_before_k"], energy, rel_tol=1e-8, abs_tol=1e-4):
                errors.append(f"Climate cumulative anomaly-energy budget mismatch {event.event_id}")
            if kind == START:
                interval = self.config["positive_forcing_range_w_m2"] if attrs["forcing_sign"] == 1 else self.config["negative_forcing_range_w_m2"]
                duration_lo, duration_hi = self.config["forcing_duration_range_years"]
                if (attrs["forcing_sign"] not in (-1, 1) or not interval[0] <= attrs.get("forcing_amplitude_w_m2", math.inf) <= interval[1]
                    or not duration_lo * YEAR <= attrs.get("forcing_duration_seconds", -1) <= duration_hi * YEAR
                    or not self._same(attrs.get("total_forcing_after_w_m2"), attrs["total_forcing_before_w_m2"] + attrs["forcing_amplitude_w_m2"])):
                    errors.append(f"Climate forcing mark outside declared support {event.event_id}")
            elif kind == END:
                start = parents[0]
                if (attrs["forcing_start_event_id"] != start.event_id or attrs["driver_id"] != start.attributes["driver_id"]
                    or not self._same(now - start.temporal.anchor("occurrence_start"), start.attributes["forcing_duration_seconds"])):
                    errors.append(f"Climate forcing pulse duration/identity mismatch {event.event_id}")
            elif kind in (WARM, WARM_END, COLD, COLD_END):
                target = state["entity_attrs"][unit]["warm_threshold_k" if kind in (WARM, WARM_END) else "cold_threshold_k"]
                if not self._same(attrs.get("threshold_k"), target) or not self._same(attrs["temperature_anomaly_before_k"], target):
                    errors.append(f"Climate threshold crossing mismatch {event.event_id}")
            elif kind in (WARM_PERSIST, COLD_PERSIST):
                if not self._same(now - parents[0].temporal.anchor("occurrence_start"), self.config["persistence_years"] * YEAR):
                    errors.append(f"Climate persistence duration mismatch {event.event_id}")
            elif kind == SAMPLE:
                previous = parents[0].temporal.anchor("occurrence_start") if parents else 0.
                if not self._same(now - previous, self.config["sample_period_years"] * YEAR):
                    errors.append(f"Climate observation-policy timing mismatch {event.event_id}")
            elif kind == REPORT:
                if any(not self._same(attrs.get(k), parents[0].attributes.get(k)) for k in SAMPLE_FIELDS):
                    errors.append(f"Climate report changed its frozen sample {event.event_id}")
            elif kind == WITHDRAWN:
                if len(parents) != 2 or parents[0].event_type_id != BULLETIN or parents[1].event_type_id != REPORT:
                    errors.append(f"Climate withdrawal lacks actual bulletin and new report {event.event_id}")
            self.apply_event(event, candidate, context)
            if state["local"][unit]["warm"] and state["local"][unit]["cold"]:
                errors.append(f"Climate warm/cold state overlap {event.event_id}")
            seen[event.event_id] = event
        return errors

    def render_episode_text(self, result):
        sentences, claims = [], []
        fields = ("unit_id", "driver_id", "temperature_anomaly_before_k", "total_forcing_before_w_m2", "total_forcing_after_w_m2",
            "active_forcing_components_w_m2", "heat_capacity_j_m2_k", "damping_w_m2_k", "response_time_seconds",
            "cumulative_forcing_energy_j_m2", "cumulative_feedback_energy_j_m2", "forcing_amplitude_w_m2", "forcing_duration_seconds",
            "forcing_start_event_id", "warm_threshold_k", "cold_threshold_k", "threshold_k", "confirmed_duration_seconds",
            "sample_event_id", "sample_sequence", "sampled_at_offset_seconds", "sampled_temperature_anomaly_k", "sampled_status",
            "stale_report", "sampling_semantics", "bulletin_status", "withdrawn_status", "replacement_sample_status",
            "information_basis", "superposition_semantics", "anomaly_reference", "persistence_semantics")
        for event in result.events:
            attrs = {key: event.attributes[key] for key in fields if key in event.attributes}
            roles = ", ".join(f"{p.role}={p.entity_id}" for p in event.participants)
            sentences.append(f"+{event.temporal.anchor('occurrence_start') / YEAR:.9f} Julian year {event.event_id} "
                f"{event.event_type_id} ({roles}); {json.dumps(attrs, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}.")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                "event_ids": [event.event_id], "relation_ids": [], "evidence_fields": ["event_type_id", "temporal", "participants"] + [f"attributes.{k}" for k in attrs]})
        for rel in result.event_relations:
            predicate = PREDICATES[rel.relation_type_id.removeprefix(NS)]
            sentences.append(f"{rel.source_event_id} {predicate} {rel.target_event_id} "
                f"({rel.relation_class}, +{rel.temporal_link.lag_seconds / YEAR:.9f} Julian year; rule={rel.rule_id}).")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                "event_ids": [rel.source_event_id, rel.target_event_id], "relation_ids": [rel.relation_id],
                "relation_assertions": [{"relation_id": rel.relation_id, "asserted_relation_class": rel.relation_class, "surface_predicate": predicate}],
                "evidence_fields": ["source_event_id", "target_event_id", "temporal_link", "rule_id"]})
        return {"alignment_id": result.episode_id + "_text_alignment", "episode_id": result.episode_id,
            "domain": self.domain_id, "projection_mode": "deterministic_grounded_projection",
            "sentences": sentences, "text": " ".join(sentences), "claims": claims}


def build_climate_temporal_models():
    registry = build_standard_temporal_registry()
    registry.register(ExponentialArrivalModel(NS + "forcing_arrival.prior_v1", 1.0 / (2 * YEAR)))
    registry.register(DeterministicDelayModel(NS + "physical_delay.prior_v1"))
    for name, median, sigma, minimum in (("reporting_delay", .06, .5, .0001), ("review_delay", .03, .4, .0001), ("log_ingestion_delay", .002, .4, .00001)):
        registry.register(ConditionalLogNormalModel(NS + name + ".prior_v1", math.log(median * YEAR), sigma, {}, minimum_seconds=minimum * YEAR))
    return registry
