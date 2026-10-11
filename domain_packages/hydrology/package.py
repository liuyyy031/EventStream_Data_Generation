"""Event-scale initial/constant loss and conservative, non-attenuating lag routing.

Rectangular, constant-intensity storms yield rectangular excess-flow pulses.
This is not a distributed hydraulic solver or a flood-inundation model.
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

NS, HOUR = "hydrology.", 3600.0
RAIN, RAIN_END = NS + "rainfall.started", NS + "rainfall.ended"
RUNOFF, RUNOFF_END, NO_RUNOFF = NS + "runoff.started", NS + "runoff.ended", NS + "runoff.not_generated"
ARRIVAL, CLEAR = NS + "flow.contribution.arrived", NS + "flow.contribution.cleared"
HIGH, NORMAL = NS + "discharge.threshold.exceeded", NS + "discharge.threshold.cleared"
SAMPLE, REPORT = NS + "gauge.sampled", NS + "gauge.reported"
WARNING, WITHDRAWN = NS + "warning.issued", NS + "warning.withdrawn"
EVENT_TYPES = (RAIN, RAIN_END, RUNOFF, RUNOFF_END, NO_RUNOFF, ARRIVAL, CLEAR,
               HIGH, NORMAL, SAMPLE, REPORT, WARNING, WITHDRAWN)
SCENARIOS = ("isolated_storm", "tributary_confluence", "repeated_storms")
PULSE_FIELDS = ("pulse_id", "runoff_event_id", "excess_flow_m3_s", "runoff_volume_m3", "pulse_duration_seconds")
RAIN_FIELDS = ("rain_event_id", "rainfall_intensity_mm_h", "storm_duration_seconds",
    "initial_loss_before_mm", "initial_loss_after_mm", "constant_loss_mm_h", "catchment_area_km2",
    "rain_depth_mm", "excess_depth_mm", "loss_depth_mm", "initial_fill_seconds", "response_lag_seconds",
    "excess_flow_m3_s", "runoff_volume_m3", "pulse_duration_seconds", "storm_ordinal")
SAMPLE_FIELDS = ("sample_event_id", "sampled_at_offset_seconds", "sample_sequence",
                "sampled_flow_m3_s", "sampled_above_threshold", "threshold_flow_m3_s")
PREDICATES = {
    "storm_recurrence": "was followed under the recurrent-weather sequence prior by",
    "rainfall_end": "ended its configured finite rainfall at",
    "rain_runoff": "generated positive rainfall excess and lagged runoff at",
    "no_runoff": "had no positive rainfall excess, as recorded by",
    "runoff_end": "finished its translated rectangular runoff pulse at",
    "catchment_entry": "entered its receiving river reach as",
    "downstream_routing": "was translated without volume loss to the next downstream reach as",
    "pulse_clear": "finished its contribution at that reach at",
    "flow_threshold_single": "provided the active flow for the single-contribution threshold derivation",
    "flow_threshold_up": "contributed positive active flow to the threshold derivation",
    "flow_threshold_down": "was required for the recorded discharge-threshold clearing at",
    "gauge_sample": "prompted a threshold-triggered gauge sample at",
    "gauge_report": "was transmitted as the frozen gauge report",
    "warning_raise": "prompted the latest-report-based warning at",
    "warning_clear": "was required for the latest-report-based warning withdrawal at",
}


def runoff_plan(intensity, duration, initial_loss, constant_loss, area, lag):
    """Analytic initial/constant-loss excess for a constant-intensity storm.

    Initial loss is filled first. No drying recovery is modeled in this event.
    A pure catchment translation preserves excess depth, volume and width.
    """
    fill = initial_loss / intensity * HOUR if intensity > 0 else duration
    width = max(0.0, duration - fill) if intensity > constant_loss else 0.0
    rate = max(0.0, intensity - constant_loss) * area / 3.6 if width > 0 else 0.0
    depth = rate * width / (1000.0 * area)
    rain_depth = intensity * duration / HOUR
    return {"initial_loss_before_mm": initial_loss,
        "initial_loss_after_mm": max(0.0, initial_loss - rain_depth),
        "constant_loss_mm_h": constant_loss, "catchment_area_km2": area,
        "rainfall_intensity_mm_h": intensity, "storm_duration_seconds": duration,
        "rain_depth_mm": rain_depth, "excess_depth_mm": depth, "loss_depth_mm": rain_depth - depth,
        "initial_fill_seconds": fill, "response_lag_seconds": lag,
        "excess_flow_m3_s": rate, "runoff_volume_m3": rate * width, "pulse_duration_seconds": width}


class HydrologyPackage:
    domain_id = "hydrology"

    def __init__(self, config=None):
        self.config = {"base_seed": 20261005, "context_count": 4, "nodes_per_context": 1000,
            "duration_seconds": 24 * HOUR, "start_time": "2026-10-05T00:00:00+08:00",
            "rainfall_intensity_range_mm_h": [4., 32.], "storm_duration_range_seconds": [1800., 7200.],
            "initial_loss_range_mm": [2., 12.], "constant_loss_range_mm_h": [2., 12.],
            "scenario_weights": dict(zip(SCENARIOS, (.35, .35, .3))), **(config or {})}
        for key in ("base_seed", "context_count", "nodes_per_context"):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
                raise ValueError(f"{key} must be a finite integer")
            self.config[key] = int(value)
        if self.config["context_count"] < 1 or self.config["nodes_per_context"] < 20:
            raise ValueError("Hydrology requires at least one context and 20 typed entities")
        duration = float(self.config["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_seconds must be positive and finite")
        self.config["duration_seconds"] = duration
        for key in ("rainfall_intensity_range_mm_h", "storm_duration_range_seconds",
                    "initial_loss_range_mm", "constant_loss_range_mm_h"):
            values = self.config[key]
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"{key} must be a lower/upper pair")
            lo, hi = map(float, values)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo < 0 or hi < lo or (key == "storm_duration_range_seconds" and lo <= 0):
                raise ValueError(f"Invalid {key}")
            self.config[key] = [lo, hi]
        weights = self.config["scenario_weights"]
        if not weights or set(weights) - set(SCENARIOS) or any(
            not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()
        ) or sum(weights.values()) <= 0:
            raise ValueError("Invalid Hydrology scenario weights")
        self.domain_spec = load_domain_spec(Path(__file__).with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(Path(__file__).with_name("mechanism_specs.json"),
            known_event_type_ids=EVENT_TYPES, known_context_relation_type_ids=self.domain_spec.context_relation_type_ids)
        self._context_cache = {}

    def catalog(self):
        roles = {role: NS + kind for role, kind in (
            ("catchment", "catchment"), ("reach", "reach"), ("upstream_reach", "reach"),
            ("gauge", "gauge"), ("unit", "management_unit"))}
        return {"domain": self.domain_id, "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_not_empirically_fitted",
            "scope": {"process": "event_scale_constant_rainfall_initial_constant_loss",
                "routing": "rectangular_excess_hydrographs_pure_translation_no_attenuation",
                "basin": "synthetic_dendritic_network_no_backwater_or_reservoir",
                "warning": "synthetic_discharge_threshold_not_proof_of_inundation",
                "observation": "threshold_triggered_samples_not_periodic_gauge_history",
                "statistical_realism_claimed": False},
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {"mode": "deterministic_grounded_projection", "free_form_causal_completion": False},
            "entity_types": [{"entity_type_id": NS + kind, "parent_type_id": None,
                "required_attributes": attrs} for kind, attrs in (
                ("catchment", ["area_km2", "initial_loss_mm", "constant_loss_mm_h", "response_lag_seconds"]),
                ("reach", ["length_m", "wave_celerity_m_s", "baseflow_m3_s", "threshold_flow_m3_s"]),
                ("gauge", ["sampling_policy"]), ("management_unit", ["policy_id"]))],
            "event_types": [{"event_type_id": kind, "parent_type_id": kind.rsplit(".", 1)[0],
                "required_attributes": ["workflow_id", "scenario_family", "reach_id"],
                "participant_role_types": roles} for kind in EVENT_TYPES],
            "context_relation_types": [{"relation_type_id": kind, "directed": True,
                "does_not_imply_event_parenthood": True} for kind in self.domain_spec.context_relation_type_ids],
            "event_relation_types": [{"relation_type_id": NS + key,
                "relation_class": self.mechanism_registry.get(NS + key).relation_semantics,
                "text_rendering": {"surface_predicate": predicate,
                    "asserted_relation_class": self.mechanism_registry.get(NS + key).relation_semantics}}
                for key, predicate in PREDICATES.items()],
            "mechanisms": self.mechanism_registry.descriptors()}

    @staticmethod
    def _graph_data(entities, relations):
        attrs = {entity.entity_id: entity.attributes for entity in entities}
        bindings = {(r.relation_type_id, r.source_entity_id): r.target_entity_id for r in relations}
        gauges = defaultdict(list)
        for r in relations:
            if r.relation_type_id == NS + "gauge_observes_reach":
                gauges[r.target_entity_id].append(r.source_entity_id)
        return {"entity_attrs": attrs, "bindings": bindings,
            "relation_ids": {(r.relation_type_id, r.source_entity_id, r.target_entity_id): r.relation_id for r in relations},
            "downstream": {r.source_entity_id: r.target_entity_id for r in relations if r.relation_type_id == NS + "reach_flows_to_reach"},
            "gauges": dict(gauges),
            "reach_ids": [e.entity_id for e in entities if e.entity_type_id == NS + "reach"],
            "catchment_ids": [e.entity_id for e in entities if e.entity_type_id == NS + "catchment"],
            "gauge_ids": [e.entity_id for e in entities if e.entity_type_id == NS + "gauge"]}

    def _build_context(self, index):
        rng = random.Random(self.config["base_seed"] + 7919 * index)
        context_id = f"hydrology_context_{index:03d}"
        n = self.config["nodes_per_context"]
        nr, nc, nu = max(8, int(n * .4)), max(6, int(n * .3)), max(1, int(n * .05))
        ng = n - nr - nc - nu
        if ng < 2:
            raise ValueError("Insufficient Hydrology entity budget")
        ids = {kind: [f"{context_id}_{kind}_{i:05d}" for i in range(count)]
            for kind, count in (("reach", nr), ("catchment", nc), ("gauge", ng), ("management_unit", nu))}
        parent, children = {0: None, 1: 0, 2: 0}, defaultdict(list)
        children[0] = [1, 2]
        available = [0, 1, 2]
        for i in range(3, nr):
            p = rng.choice(available)
            parent[i] = p
            children[p].append(i)
            if len(children[p]) == 3:
                available.remove(p)
            available.append(i)
        upstream_count = {i: 1 for i in range(nr)}
        for i in range(nr - 1, 0, -1):
            upstream_count[parent[i]] += upstream_count[i]
        entities = []
        for i, reach in enumerate(ids["reach"]):
            baseflow = .15 * upstream_count[i]
            entities.append(Entity(reach, self.domain_id, NS + "reach", {
                "length_m": round(rng.uniform(300, 1500), 6),
                "wave_celerity_m_s": round(rng.uniform(.8, 2.0), 6),
                "baseflow_m3_s": baseflow,
                "threshold_flow_m3_s": baseflow + rng.uniform(.8, 3.0) * math.sqrt(upstream_count[i]),
                "local_reference_baseflow_m3_s": .15}, context_id))
        for catchment in ids["catchment"]:
            entities.append(Entity(catchment, self.domain_id, NS + "catchment", {
                "area_km2": round(rng.uniform(.3, 2.0), 6),
                "initial_loss_mm": rng.uniform(*self.config["initial_loss_range_mm"]),
                "constant_loss_mm_h": rng.uniform(*self.config["constant_loss_range_mm_h"]),
                "response_lag_seconds": rng.uniform(300, 1800)}, context_id))
        for gauge in ids["gauge"]:
            entities.append(Entity(gauge, self.domain_id, NS + "gauge", {"sampling_policy": "threshold_transition_triggered"}, context_id))
        for unit in ids["management_unit"]:
            entities.append(Entity(unit, self.domain_id, NS + "management_unit", {"policy_id": "synthetic_discharge_warning_v1"}, context_id))
        reaches = ids["reach"]
        required = {
            NS + "reach_flows_to_reach": [(reaches[i], reaches[parent[i]]) for i in range(1, nr)],
            NS + "catchment_drains_to_reach": [(c, reaches[1 + i % (nr - 1)]) for i, c in enumerate(ids["catchment"])],
            NS + "gauge_observes_reach": [(g, reaches[i % nr]) for i, g in enumerate(ids["gauge"])],
            NS + "gauge_reports_to_unit": [(g, ids["management_unit"][i % nu]) for i, g in enumerate(ids["gauge"])],
        }
        topology = SparseHeterogeneousTopologyGenerator().generate(context_id=context_id, domain=self.domain_id,
            entities=entities, layers=[replace(layer, required_pairs=tuple(required[layer.relation_type_id]))
                for layer in self.domain_spec.compile_relation_layers()], seed=self.config["base_seed"] + 7919 * index,
            observation_window_seconds=self.config["duration_seconds"])
        data = self._graph_data(entities, topology.relations)
        catch_by_reach = defaultdict(list)
        for c, r in required[NS + "catchment_drains_to_reach"]:
            catch_by_reach[r].append(c)
        confluences = []
        for group in children.values():
            eligible = [reaches[i] for i in group if reaches[i] in catch_by_reach]
            for left_index, left in enumerate(eligible):
                for right in eligible[left_index + 1:]:
                    confluences.append((catch_by_reach[left][0], catch_by_reach[right][0]))
        return {**data, "entities": entities, "relations": topology.relations,
                "statistics": topology.statistics, "confluences": confluences}

    @staticmethod
    def _initial_state(data, scenario, workflow):
        return {**{key: data[key] for key in ("entity_attrs", "bindings", "relation_ids", "downstream", "gauges")},
            "scenario": scenario, "workflow_id": workflow,
            "rain_active": {}, "storm_counts": defaultdict(int),
            "initial_remaining": {c: data["entity_attrs"][c]["initial_loss_mm"] for c in data["catchment_ids"]},
            "pulse_seen": set(), "reported_samples": set(), "sampled_transitions": set(),
            "reaches": {r: {"active": {}, "epoch": 0, "above": False, "high_event_id": None} for r in data["reach_ids"]},
            "gauge_state": {g: {"sample_sequence": 0, "latest_sequence": 0, "latest_report": None,
                "latest_above": False, "warning_event_id": None} for g in data["gauge_ids"]}}

    def create_episode_context(self, episode_index, seed):
        index = episode_index % self.config["context_count"]
        if index not in self._context_cache:
            self._context_cache[index] = self._build_context(index)
        data, rng = self._context_cache[index], random.Random(seed ^ 0x485944524f)
        weights = self.config["scenario_weights"]
        scenario = rng.choices(list(weights), weights=list(weights.values()))[0]
        roots = list(rng.choice(data["confluences"])) if scenario == "tributary_confluence" else [rng.choice(data["catchment_ids"])]
        episode_id = f"hydrology_episode_{episode_index:07d}"
        state = self._initial_state(data, scenario, episode_id + "_workflow")
        state["root_catchments"] = roots
        return EpisodeContext(episode_id, f"hydrology_context_{index:03d}", self.domain_id,
            self.config["start_time"], self.config["duration_seconds"], data["entities"], data["relations"], state=state,
            context_attributes={"topology_profile": data["statistics"], "basin_scope": "one_synthetic_dendritic_basin"},
            episode_attributes={"scenario_family": scenario, "empirical_realism_claimed": False,
                "root_catchments": roots, "routing_assumption": "pure_lag_no_attenuation_or_channel_loss",
                "threshold_is_not_proof_of_inundation": True, "initial_loss_recovery_modeled": False})

    def _spec(self, key, context, parents=(), now=0.0, *, catchment=None, reach=None, upstream=None,
              gauge=None, delay=0.0, extra=None):
        state, pairs = context.state, set()
        mechanism = self.mechanism_registry.get(NS + key)
        if reach is None:
            reach = state["bindings"][(NS + "catchment_drains_to_reach", catchment)]
        roles = {"reach": reach}
        attrs = {"workflow_id": state["workflow_id"], "scenario_family": state["scenario"], "reach_id": reach}
        if catchment:
            roles["catchment"] = catchment
            attrs["catchment_id"] = catchment
            if key != "downstream_routing":
                pairs.add((NS + "catchment_drains_to_reach", catchment, reach))
        if upstream:
            roles["upstream_reach"] = upstream
            attrs["upstream_reach_id"] = upstream
            pairs.add((NS + "reach_flows_to_reach", upstream, reach))
        if gauge:
            roles["gauge"] = gauge
            attrs["gauge_id"] = gauge
            pairs.add((NS + "gauge_observes_reach", gauge, reach))
            if key in ("gauge_report", "warning_raise", "warning_clear"):
                unit = state["bindings"][(NS + "gauge_reports_to_unit", gauge)]
                roles["unit"] = unit
                attrs["unit_id"] = unit
                pairs.add((NS + "gauge_reports_to_unit", gauge, unit))
        if extra:
            attrs.update(extra)
        evidence = [state["relation_ids"][pair] for pair in sorted(pairs)]
        return CandidateSpec(mechanism.mechanism_id, mechanism.target_event_type_id, list(parents),
            [EventParticipant(entity, role) for role, entity in roles.items()], mechanism.generation_temporal_model_ref,
            {"delay_seconds": delay} if mechanism.temporal_model_family == "deterministic_delay" else {},
            activation_time=now, combination=mechanism.parent_combination,
            phase_priority=(-20 if key in ("storm_onset", "storm_recurrence", "rainfall_end") else
                -10 if key in ("rain_runoff", "runoff_end", "no_runoff") else
                0 if key == "pulse_clear" else 5 if key in ("catchment_entry", "downstream_routing") else
                30 if key == "gauge_sample" else 40 if key == "gauge_report" else
                50 if key in ("warning_raise", "warning_clear") else 20),
            attributes=attrs, context_evidence=ContextEvidence(context_relation_ids=evidence,
                entity_ids=list(roles.values()), evaluated_at_offset_seconds=now),
            provenance={"input_anchor": "occurrence_start", "parameter_status": mechanism.parameter_status})

    def seed_candidates(self, context, rng):
        del rng
        return [self._spec("storm_onset", context, catchment=c) for c in context.state["root_catchments"]]

    @staticmethod
    def _flow(state, reach):
        return state["entity_attrs"][reach]["baseflow_m3_s"] + math.fsum(
            pulse["excess_flow_m3_s"] for pulse in state["reaches"][reach]["active"].values())

    def revalidate_candidate(self, candidate, context, event_by_id, now):
        del event_by_id
        state, attrs = context.state, candidate.attributes
        key, reach = candidate.mechanism_id.removeprefix(NS), attrs["reach_id"]
        local, valid = state["reaches"][reach], True
        if key in ("storm_onset", "storm_recurrence"):
            catchment = attrs["catchment_id"]
            expected_count = 0 if key == "storm_onset" else 1
            valid = catchment not in state["rain_active"] and state["storm_counts"][catchment] == expected_count
        elif key == "rainfall_end":
            valid = state["rain_active"].get(attrs["catchment_id"]) in candidate.parent_event_ids
        elif key in ("catchment_entry", "downstream_routing"):
            valid = (reach, attrs["pulse_id"]) not in state["pulse_seen"]
        elif key == "pulse_clear":
            valid = attrs["arrival_event_id"] in local["active"]
        elif key in ("flow_threshold_single", "flow_threshold_up", "flow_threshold_down"):
            is_high = self._flow(state, reach) > state["entity_attrs"][reach]["threshold_flow_m3_s"]
            valid = (attrs["flow_epoch"] == local["epoch"] and is_high == (key != "flow_threshold_down")
                     and local["above"] != is_high)
        elif key == "gauge_sample":
            valid = candidate.parent_event_ids[0] not in state["sampled_transitions"]
        elif key == "gauge_report":
            valid = attrs["sample_event_id"] not in state["reported_samples"]
        elif key in ("warning_raise", "warning_clear"):
            gauge = state["gauge_state"][attrs["gauge_id"]]
            valid = (gauge["latest_report"] == attrs["basis_report_event_id"] and
                gauge["latest_above"] == (key == "warning_raise") and
                bool(gauge["warning_event_id"]) == (key == "warning_clear"))
        elif key not in ("rain_runoff", "runoff_end", "no_runoff"):
            raise ValueError(f"Unsupported Hydrology mechanism {key}")
        evidence = copy.deepcopy(candidate.context_evidence)
        evidence.evaluated_at_offset_seconds = now
        evidence.state_predicates = [StatePredicateEvidence(reach, key + "_precondition", "equals", True, bool(valid), bool(valid))]
        if key in ("flow_threshold_single", "flow_threshold_up", "flow_threshold_down"):
            evidence.state_predicates.extend((
                StatePredicateEvidence(reach, "aggregate_flow_m3_s", "greater_than" if key != "flow_threshold_down" else "less_or_equal",
                    state["entity_attrs"][reach]["threshold_flow_m3_s"], self._flow(state, reach), is_high == (key != "flow_threshold_down")),
                StatePredicateEvidence(reach, "flow_epoch", "equals", attrs["flow_epoch"], local["epoch"], attrs["flow_epoch"] == local["epoch"])))
        return CandidateUpdate(bool(valid), None if valid else key + "_state_superseded", context_evidence=evidence)

    def _event_attributes(self, event_id, candidate, context):
        state, kind, attrs = context.state, candidate.target_event_type_id, copy.deepcopy(candidate.attributes)
        reach, now = attrs["reach_id"], candidate.scheduled_time
        local = state["reaches"][reach]
        if kind == RUNOFF:
            attrs.update(pulse_id=event_id, runoff_event_id=event_id)
        elif kind in (ARRIVAL, CLEAR):
            before = self._flow(state, reach)
            delta = attrs["excess_flow_m3_s"] * (1 if kind == ARRIVAL else -1)
            attrs.update(flow_before_m3_s=before, flow_after_m3_s=before + delta,
                         flow_epoch_before=local["epoch"], active_contribution_count_before=len(local["active"]))
        elif kind in (HIGH, NORMAL):
            attrs.update(aggregate_flow_m3_s=self._flow(state, reach),
                threshold_flow_m3_s=state["entity_attrs"][reach]["threshold_flow_m3_s"],
                active_contribution_event_ids=list(local["active"]),
                active_contributions={eid: pulse["excess_flow_m3_s"] for eid, pulse in local["active"].items()},
                inference_scope="synthetic_discharge_threshold_not_observed_inundation")
        elif kind == SAMPLE:
            attrs.update(sample_event_id=event_id, sampled_at_offset_seconds=now,
                sample_sequence=state["gauge_state"][attrs["gauge_id"]]["sample_sequence"] + 1,
                sampled_flow_m3_s=self._flow(state, reach),
                sampled_above_threshold=self._flow(state, reach) > state["entity_attrs"][reach]["threshold_flow_m3_s"],
                threshold_flow_m3_s=state["entity_attrs"][reach]["threshold_flow_m3_s"])
        elif kind == REPORT:
            attrs["stale_report"] = attrs["sample_sequence"] <= state["gauge_state"][attrs["gauge_id"]]["latest_sequence"]
        elif kind in (WARNING, WITHDRAWN):
            attrs["warning_basis"] = "latest_delivered_gauge_report_not_current_physical_flow"
        return attrs

    def materialize_event(self, event_id, candidate, context, rng):
        attrs = self._event_attributes(event_id, candidate, context)
        kind = candidate.target_event_type_id
        if kind == RAIN:
            c = attrs["catchment_id"]
            source = context.state["entity_attrs"][c]
            attrs.update(runoff_plan(rng.uniform(*self.config["rainfall_intensity_range_mm_h"]),
                rng.uniform(*self.config["storm_duration_range_seconds"]), context.state["initial_remaining"][c],
                source["constant_loss_mm_h"], source["area_km2"], source["response_lag_seconds"]))
            attrs.update(rain_event_id=event_id, storm_ordinal=context.state["storm_counts"][c] + 1)
        return EventRecord(event_id, context.episode_id, self.domain_id, kind,
            TemporalExtent(candidate.scheduled_time), list(candidate.participants), attrs,
            event_role="root" if not candidate.parent_event_ids else "derived",
            provenance={"generator": "synthetic_event_scale_rainfall_runoff_lag_routing", "mechanism_id": candidate.mechanism_id})

    def observation_plan(self, event, candidate, context):
        del candidate, context
        source = ("synthetic_gauge_sample_or_telemetry" if event.event_type_id in (SAMPLE, REPORT)
                  else "synthetic_management_workflow" if event.event_type_id in (WARNING, WITHDRAWN)
                  else "offline_synthetic_hydrologic_ground_truth")
        return ObservationPlan(recording_model_ref=NS + "log_ingestion_delay.prior_v1", source=source)

    def apply_event(self, event, candidate, context):
        state, kind, attrs = context.state, event.event_type_id, event.attributes
        reach = attrs["reach_id"]
        local = state["reaches"][reach]
        if kind == RAIN:
            c = attrs["catchment_id"]
            state["rain_active"][c] = event.event_id
            state["storm_counts"][c] += 1
        elif kind == RAIN_END:
            c = attrs["catchment_id"]
            state["rain_active"].pop(c)
            state["initial_remaining"][c] = attrs["initial_loss_after_mm"]
        elif kind == ARRIVAL:
            local["active"][event.event_id] = attrs
            local["epoch"] += 1
            state["pulse_seen"].add((reach, attrs["pulse_id"]))
        elif kind == CLEAR:
            local["active"].pop(attrs["arrival_event_id"])
            local["epoch"] += 1
        elif kind == HIGH:
            local.update(above=True, high_event_id=event.event_id)
        elif kind == NORMAL:
            local.update(above=False, high_event_id=None)
        elif kind == SAMPLE:
            state["gauge_state"][attrs["gauge_id"]]["sample_sequence"] = attrs["sample_sequence"]
            state["sampled_transitions"].update(candidate.parent_event_ids)
        elif kind == REPORT:
            state["reported_samples"].add(attrs["sample_event_id"])
            if not attrs["stale_report"]:
                state["gauge_state"][attrs["gauge_id"]].update(latest_sequence=attrs["sample_sequence"],
                    latest_report=event.event_id, latest_above=attrs["sampled_above_threshold"])
        elif kind == WARNING:
            state["gauge_state"][attrs["gauge_id"]]["warning_event_id"] = event.event_id
        elif kind == WITHDRAWN:
            state["gauge_state"][attrs["gauge_id"]]["warning_event_id"] = None

    def _threshold_candidate(self, context, change_event):
        state, reach = context.state, change_event.attributes["reach_id"]
        local = state["reaches"][reach]
        high = self._flow(state, reach) > state["entity_attrs"][reach]["threshold_flow_m3_s"]
        if high == local["above"]:
            return []
        parents = list(local["active"]) if high else [change_event.event_id, local["high_event_id"]]
        key = ("flow_threshold_single" if len(parents) == 1 else "flow_threshold_up") if high else "flow_threshold_down"
        return [self._spec(key, context, parents,
            change_event.temporal.anchor("occurrence_start"), reach=reach, extra={"flow_epoch": local["epoch"]})]

    def spawn_candidates(self, event, candidate, context, event_by_id, rng):
        del candidate, event_by_id, rng
        state, kind, attrs = context.state, event.event_type_id, event.attributes
        now, parents, reach = event.temporal.anchor("occurrence_start"), [event.event_id], attrs["reach_id"]
        c = attrs.get("catchment_id")
        result = []
        if kind == RAIN:
            frozen = {field: attrs[field] for field in RAIN_FIELDS}
            result.append(self._spec("rainfall_end", context, parents, now, catchment=c,
                                    delay=attrs["storm_duration_seconds"], extra=frozen))
            if attrs["runoff_volume_m3"] > 0:
                result.append(self._spec("rain_runoff", context, parents, now, catchment=c,
                    delay=attrs["initial_fill_seconds"] + attrs["response_lag_seconds"], extra=frozen))
        elif kind == RAIN_END:
            if attrs["runoff_volume_m3"] == 0:
                result.append(self._spec("no_runoff", context, parents, now, catchment=c,
                                        extra={field: attrs[field] for field in RAIN_FIELDS}))
            if state["scenario"] == "repeated_storms" and attrs["storm_ordinal"] == 1:
                result.append(self._spec("storm_recurrence", context, parents, now, catchment=c))
        elif kind == RUNOFF:
            pulse = {field: attrs[field] for field in PULSE_FIELDS}
            result.append(self._spec("runoff_end", context, parents, now, catchment=c,
                                    delay=attrs["pulse_duration_seconds"], extra=pulse))
            result.append(self._spec("catchment_entry", context, parents, now, catchment=c, extra=pulse))
        elif kind == ARRIVAL:
            pulse = {field: attrs[field] for field in PULSE_FIELDS}
            result.append(self._spec("pulse_clear", context, parents, now, reach=reach,
                delay=attrs["pulse_duration_seconds"], extra={**pulse, "arrival_event_id": event.event_id}))
            if reach in state["downstream"]:
                r = state["entity_attrs"][reach]
                delay = r["length_m"] / r["wave_celerity_m_s"]
                result.append(self._spec("downstream_routing", context, parents, now,
                    reach=state["downstream"][reach], upstream=reach,
                    delay=delay, extra={**pulse, "origin_catchment_id": attrs.get("origin_catchment_id", c),
                                       "routing_lag_seconds": delay}))
            result.extend(self._threshold_candidate(context, event))
        elif kind == CLEAR:
            result.extend(self._threshold_candidate(context, event))
        elif kind in (HIGH, NORMAL):
            result.extend(self._spec("gauge_sample", context, parents, now, reach=reach, gauge=g)
                          for g in state["gauges"].get(reach, []))
        elif kind == SAMPLE:
            result.append(self._spec("gauge_report", context, parents, now, reach=reach, gauge=attrs["gauge_id"],
                                    extra={field: attrs[field] for field in SAMPLE_FIELDS}))
        elif kind == REPORT and not attrs["stale_report"]:
            gauge = state["gauge_state"][attrs["gauge_id"]]
            if attrs["sampled_above_threshold"] and not gauge["warning_event_id"]:
                key = "warning_raise"
            elif not attrs["sampled_above_threshold"] and gauge["warning_event_id"]:
                key = "warning_clear"
                parents.append(gauge["warning_event_id"])
            else:
                return []
            result.append(self._spec(key, context, parents, now, reach=reach, gauge=attrs["gauge_id"],
                extra={"basis_report_event_id": event.event_id, **{field: attrs[field] for field in SAMPLE_FIELDS}}))
        return result

    def relation_specs(self, event, candidate, context, event_by_id):
        del event, context, event_by_id
        mechanism = self.mechanism_registry.get(candidate.mechanism_id)
        return [RelationSpec(parent, mechanism.relation_semantics, mechanism.mechanism_id, mechanism.mechanism_id,
            context_evidence=copy.deepcopy(candidate.context_evidence),
            mechanism_group_id=candidate.candidate_id if len(candidate.parent_event_ids) > 1 else None,
            provenance={"parent_basis": "realized_runoff_transport_or_active_superposition_or_explicit_workflow",
                        "topology_edge_alone_sufficient": False}) for parent in candidate.parent_event_ids]

    @staticmethod
    def _same(left, right):
        if isinstance(left, (float, int)) and not isinstance(left, bool) and isinstance(right, (float, int)) and not isinstance(right, bool):
            return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7)
        return left == right

    def validate_result(self, result):
        errors = []
        entities = {entity.entity_id: entity for entity in result.entities}
        for entity in result.entities:
            attrs = entity.attributes
            required = {NS + "catchment": ("area_km2", "initial_loss_mm", "constant_loss_mm_h", "response_lag_seconds"),
                NS + "reach": ("length_m", "wave_celerity_m_s", "baseflow_m3_s", "threshold_flow_m3_s")}.get(entity.entity_type_id, ())
            if any(not isinstance(attrs.get(field), (int, float)) or not math.isfinite(attrs[field]) or attrs[field] < 0 for field in required):
                errors.append(f"Invalid physical Hydrology attributes {entity.entity_id}")
            elif entity.entity_type_id == NS + "catchment" and attrs["area_km2"] <= 0:
                errors.append(f"Non-positive catchment area {entity.entity_id}")
            elif entity.entity_type_id == NS + "reach" and (attrs["length_m"] <= 0 or attrs["wave_celerity_m_s"] <= 0 or attrs["threshold_flow_m3_s"] <= attrs["baseflow_m3_s"]):
                errors.append(f"Invalid river length/celerity/threshold {entity.entity_id}")
        layers = {layer.relation_type_id: layer for layer in self.domain_spec.compile_relation_layers()}
        counts, incoming_reaches = defaultdict(int), defaultdict(int)
        for relation in result.context_relations:
            layer = layers.get(relation.relation_type_id)
            source, target = entities.get(relation.source_entity_id), entities.get(relation.target_entity_id)
            if not layer or not source or not target or source.entity_type_id not in layer.source_type_ids or target.entity_type_id not in layer.target_type_ids:
                errors.append(f"Invalid typed Hydrology relation {relation.relation_id}")
            counts[(relation.relation_type_id, relation.source_entity_id)] += 1
            if relation.relation_type_id == NS + "reach_flows_to_reach":
                incoming_reaches[relation.target_entity_id] += 1
        if errors:
            return errors
        data = self._graph_data(result.entities, result.context_relations)
        for kind, layer in layers.items():
            for entity in result.entities:
                if entity.entity_type_id in layer.source_type_ids:
                    count = counts[(kind, entity.entity_id)]
                    if (count != 1 and kind != NS + "reach_flows_to_reach") or count > 1:
                        errors.append(f"Missing/ambiguous Hydrology binding {entity.entity_id}")
        if len(data["downstream"]) != len(data["reach_ids"]) - 1 or max(incoming_reaches.values(), default=0) > 3:
            errors.append("River is not a single-outlet bounded-degree tree")
        for reach in data["reach_ids"]:
            visited, node = set(), reach
            while node in data["downstream"]:
                if node in visited:
                    errors.append(f"River routing cycle at {reach}")
                    break
                visited.add(node)
                node = data["downstream"][node]
        if errors:
            return errors
        state = self._initial_state(data, result.episode_attributes["scenario_family"], result.episode_id + "_workflow")
        context = EpisodeContext(result.episode_id, result.context_id, self.domain_id, result.start_time,
            result.duration_seconds, result.entities, result.context_relations, state=state)
        candidates = {c.fired_event_id: c for c in result.candidates if c.fired_event_id}
        incoming, seen = defaultdict(list), {}
        planned = defaultdict(list)
        for candidate in result.candidates:
            for parent in candidate.parent_event_ids:
                planned[(parent, candidate.mechanism_id)].append(candidate)
        for relation in result.event_relations:
            incoming[relation.target_event_id].append(relation.source_event_id)
        for event in result.events:
            attrs, kind, now = event.attributes, event.event_type_id, event.temporal.anchor("occurrence_start")
            candidate = candidates.get(event.event_id)
            reach = attrs.get("reach_id")
            if not candidate or reach not in state["reaches"]:
                errors.append(f"Unknown Hydrology event/candidate {event.event_id}")
                continue
            key = candidate.mechanism_id.removeprefix(NS)
            mechanism = self.mechanism_registry.get(candidate.mechanism_id)
            parents = [seen.get(eid) for eid in candidate.parent_event_ids]
            if (set(incoming[event.event_id]) != set(candidate.parent_event_ids) or
                event.provenance.get("mechanism_id") != candidate.mechanism_id or kind != mechanism.target_event_type_id or
                any(not parent or parent.event_type_id not in mechanism.source_event_type_ids for parent in parents)):
                errors.append(f"Mechanism/parent mismatch {event.event_id}")
            if any(parent is None for parent in parents):
                continue
            if event.participants != candidate.participants:
                errors.append(f"Candidate/participant mismatch {event.event_id}")
            roles = {p.role: p.entity_id for p in event.participants}
            if roles.get("reach") != reach:
                errors.append(f"Reach participant mismatch {event.event_id}")
            catchment, gauge = attrs.get("catchment_id"), attrs.get("gauge_id")
            if catchment and (roles.get("catchment") != catchment or state["bindings"].get((NS + "catchment_drains_to_reach", catchment)) != reach):
                errors.append(f"Invalid catchment binding {event.event_id}")
            if gauge and (roles.get("gauge") != gauge or state["bindings"].get((NS + "gauge_observes_reach", gauge)) != reach):
                errors.append(f"Invalid gauge binding {event.event_id}")
            if "unit" in roles and roles["unit"] != state["bindings"].get((NS + "gauge_reports_to_unit", gauge)):
                errors.append(f"Invalid management-unit binding {event.event_id}")
            if not self.revalidate_candidate(candidate, context, seen, now).valid:
                errors.append(f"Invalid replay precondition {event.event_id}")
            expected = self._event_attributes(event.event_id, candidate, context)
            if any(not self._same(attrs.get(field), value) for field, value in expected.items()):
                errors.append(f"Derived/snapshot attribute mismatch {event.event_id}")
            if kind == RAIN:
                source = state["entity_attrs"].get(catchment, {})
                intensity, duration = attrs.get("rainfall_intensity_mm_h", -1), attrs.get("storm_duration_seconds", -1)
                lo, hi = self.config["rainfall_intensity_range_mm_h"]
                dlo, dhi = self.config["storm_duration_range_seconds"]
                if not lo <= intensity <= hi or not dlo <= duration <= dhi:
                    errors.append(f"Rainfall outside configured prior support {event.event_id}")
                plan = runoff_plan(intensity, duration, state["initial_remaining"][catchment],
                    source["constant_loss_mm_h"], source["area_km2"], source["response_lag_seconds"])
                if any(not self._same(attrs.get(field), value) for field, value in plan.items()) or attrs.get("storm_ordinal") != state["storm_counts"][catchment] + 1:
                    errors.append(f"Rainfall water-balance/history mismatch {event.event_id}")
                if key == "storm_recurrence" and (state["scenario"] != "repeated_storms" or
                    len(parents) != 1 or parents[0].attributes["catchment_id"] != catchment):
                    errors.append(f"Invalid recurrent-weather sequence {event.event_id}")
            elif key in ("rainfall_end", "rain_runoff", "no_runoff"):
                source = parents[0]
                if attrs.get("catchment_id") != source.attributes.get("catchment_id") or any(
                    not self._same(attrs.get(field), source.attributes.get(field)) for field in RAIN_FIELDS):
                    errors.append(f"Changed rainfall-excess basis {event.event_id}")
                delay = source.attributes["storm_duration_seconds"] if key == "rainfall_end" else (
                    source.attributes["initial_fill_seconds"] + source.attributes["response_lag_seconds"] if key == "rain_runoff" else 0.0)
                if not self._same(now - source.temporal.anchor("occurrence_start"), delay) or (key == "rain_runoff" and attrs["runoff_volume_m3"] <= 0) or (key == "no_runoff" and attrs["runoff_volume_m3"] != 0):
                    errors.append(f"Invalid runoff timing/outcome {event.event_id}")
            elif key in ("runoff_end", "catchment_entry", "downstream_routing", "pulse_clear"):
                source = parents[0]
                if any(not self._same(attrs.get(field), source.attributes.get(field)) for field in PULSE_FIELDS):
                    errors.append(f"Pulse water-volume/shape changed {event.event_id}")
                delay = 0.0
                if key in ("runoff_end", "pulse_clear"):
                    delay = source.attributes["pulse_duration_seconds"]
                    if reach != source.attributes["reach_id"]:
                        errors.append(f"Pulse clearing at wrong reach {event.event_id}")
                elif key == "downstream_routing":
                    upstream = source.attributes["reach_id"]
                    r = state["entity_attrs"][upstream]
                    delay = r["length_m"] / r["wave_celerity_m_s"]
                    if state["downstream"].get(upstream) != reach or roles.get("upstream_reach") != upstream or attrs.get("upstream_reach_id") != upstream:
                        errors.append(f"Non-adjacent/wrong-direction routing {event.event_id}")
                elif reach != source.attributes["reach_id"]:
                    errors.append(f"Runoff entered wrong receiving reach {event.event_id}")
                if not self._same(now - source.temporal.anchor("occurrence_start"), delay) or not self._same(attrs["runoff_volume_m3"], attrs["excess_flow_m3_s"] * attrs["pulse_duration_seconds"]):
                    errors.append(f"Pulse travel time/water volume mismatch {event.event_id}")
            if kind in (RUNOFF, ARRIVAL):
                required_successors = ("runoff_end", "catchment_entry") if kind == RUNOFF else (
                    ("pulse_clear", "downstream_routing") if reach in state["downstream"] else ("pulse_clear",))
                for successor in required_successors:
                    if len(planned[(event.event_id, NS + successor)]) != 1:
                        errors.append(f"Missing/duplicate conservative pulse successor {event.event_id}: {successor}")
            if kind == HIGH:
                if set(candidate.parent_event_ids) != set(state["reaches"][reach]["active"]):
                    errors.append(f"Threshold parents are not realized active contributions {event.event_id}")
            elif kind == NORMAL:
                if (len(parents) != 2 or parents[0].event_type_id != CLEAR or
                    candidate.parent_event_ids[1] != state["reaches"][reach]["high_event_id"] or
                    any(p.attributes["reach_id"] != reach for p in parents)):
                    errors.append(f"Invalid threshold-clearing predecessors {event.event_id}")
            elif kind == SAMPLE:
                if parents[0].attributes["reach_id"] != reach:
                    errors.append(f"Gauge sampled wrong threshold event {event.event_id}")
            elif kind in (REPORT, WARNING, WITHDRAWN):
                source = parents[0]
                if source.attributes.get("gauge_id") != gauge or any(not self._same(
                    attrs.get(field), source.attributes.get(field)) for field in SAMPLE_FIELDS):
                    errors.append(f"Changed frozen gauge sample {event.event_id}")
                if kind == WITHDRAWN and (len(parents) != 2 or candidate.parent_event_ids[1] != state["gauge_state"][gauge]["warning_event_id"]):
                    errors.append(f"Warning withdrawal lacks its issued warning {event.event_id}")
            try:
                self.apply_event(event, candidate, context)
            except (KeyError, ValueError, TypeError):
                errors.append(f"Malformed Hydrology event prevents state replay {event.event_id}")
                break
            seen[event.event_id] = event
        return errors

    def render_episode_text(self, result):
        sentences, claims = [], []
        fields = ("catchment_id", "origin_catchment_id", "reach_id", "upstream_reach_id", "gauge_id", "pulse_id", "runoff_event_id", "rain_event_id",
            "rainfall_intensity_mm_h", "storm_duration_seconds", "initial_loss_before_mm", "initial_loss_after_mm",
            "catchment_area_km2", "constant_loss_mm_h", "rain_depth_mm", "excess_depth_mm", "loss_depth_mm",
            "initial_fill_seconds", "response_lag_seconds", "routing_lag_seconds", "storm_ordinal",
            "runoff_volume_m3", "excess_flow_m3_s",
            "pulse_duration_seconds", "flow_before_m3_s", "flow_after_m3_s", "aggregate_flow_m3_s", "threshold_flow_m3_s",
            "active_contribution_event_ids", "active_contributions", "inference_scope", "sample_event_id",
            "sampled_at_offset_seconds", "sampled_flow_m3_s", "sampled_above_threshold", "stale_report", "warning_basis")
        for event in result.events:
            attrs = {field: event.attributes[field] for field in fields if field in event.attributes}
            roles = ", ".join(f"{p.role}={p.entity_id}" for p in event.participants)
            sentences.append(f"+{event.temporal.anchor('occurrence_start') / HOUR:.6f} hour "
                f"{event.event_id} {event.event_type_id} ({roles}); "
                f"{json.dumps(attrs, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}.")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                "event_ids": [event.event_id], "relation_ids": [], "evidence_fields":
                ["event_type_id", "temporal", "participants"] + [f"attributes.{field}" for field in attrs]})
        for relation in result.event_relations:
            predicate = PREDICATES[relation.relation_type_id.removeprefix(NS)]
            sentences.append(f"{relation.source_event_id} {predicate} {relation.target_event_id} "
                f"({relation.relation_class}, +{relation.temporal_link.lag_seconds / HOUR:.6f} hour; rule={relation.rule_id}).")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                "event_ids": [relation.source_event_id, relation.target_event_id], "relation_ids": [relation.relation_id],
                "relation_assertions": [{"relation_id": relation.relation_id,
                    "asserted_relation_class": relation.relation_class, "surface_predicate": predicate}],
                "evidence_fields": ["source_event_id", "target_event_id", "rule_id", "temporal_link"]})
        return {"alignment_id": result.episode_id + "_text_alignment", "episode_id": result.episode_id,
            "domain": self.domain_id, "projection_mode": "deterministic_grounded_projection",
            "sentences": sentences, "text": " ".join(sentences), "claims": claims}


def build_hydrology_temporal_models():
    registry = build_standard_temporal_registry()
    registry.register(ExponentialArrivalModel(NS + "storm_arrival.prior_v1", 1.0 / HOUR))
    registry.register(DeterministicDelayModel(NS + "physical_delay.prior_v1"))
    for name, median, sigma, minimum in (("interstorm_gap", 1800., .4, 60.),
        ("telemetry_delay", 72., .5, 1.), ("response_delay", 30., .4, 1.), ("log_ingestion_delay", 5., .4, .1)):
        registry.register(ConditionalLogNormalModel(NS + name + ".prior_v1", math.log(median), sigma, {}, minimum_seconds=minimum))
    return registry
