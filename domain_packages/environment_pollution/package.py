"""Grounded pollution-pulse events on a synthetic dendritic river network.

This intentionally uses a small, auditable plug-flow approximation. It is
not EPA WASP, an advection-dispersion PDE, or a calibrated water-quality model.
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
    ConditionalLogNormalModel, DeterministicDelayModel, ExponentialArrivalModel,
    build_standard_temporal_registry,
)
from data_generation.generation_core.topology import SparseHeterogeneousTopologyGenerator

NS = "environment_pollution."
SPILL = NS + "source.spill.started"
BYPASS = NS + "treatment.bypass.started"
RAIN = NS + "rainfall.started"
LOADING = NS + "pollutant.loading.started"
END = NS + "pollutant.loading.ended"
ARRIVAL = NS + "plume.arrived"
CLEAR = NS + "plume.cleared"
ALERT = NS + "monitor.alert.issued"
NORMAL = NS + "monitor.normal.issued"
INTAKE_CLOSED = NS + "intake.closed"
INTAKE_REOPENED = NS + "intake.reopened"
SETTLED = NS + "workflow.settled"
EVENT_TYPES = (SPILL, BYPASS, RAIN, LOADING, END, ARRIVAL, CLEAR, ALERT,
               NORMAL, INTAKE_CLOSED, INTAKE_REOPENED, SETTLED)
SCENARIOS = ("industrial_spill", "wastewater_bypass", "storm_runoff")
PREDICATES = {
    "spill_loading": "initiated the pollutant loading recorded as",
    "bypass_loading": "led to the pollutant loading recorded as",
    "runoff_loading": "produced the runoff loading recorded as",
    "loading_end": "ended its finite pollutant loading at",
    "outfall_entry": "entered the receiving river reach as",
    "runoff_entry": "entered the receiving river reach as",
    "downstream_transport": "transported a surviving pollutant pulse to the next downstream reach as",
    "plume_clear": "was required for the trailing edge to clear the reach at",
    "monitor_alert": "was sampled and later reported as the monitor alert",
    "monitor_normal": "was required for the subsequent normal monitor report",
    "intake_close": "triggered the intake closure recorded as",
    "intake_reopen": "was required for the intake reopening recorded as",
    "workflow_settled": "completed the physically observed plume path as",
}


class EnvironmentPollutionPackage:
    domain_id = "environment_pollution"

    def __init__(self, config=None):
        self.config = {
            "base_seed": 20261001, "context_count": 4, "nodes_per_context": 1000,
            "duration_seconds": 300.0, "start_time": "2026-10-01T08:00:00+08:00",
            "scenario_weights": dict(zip(SCENARIOS, (.35, .3, .35))), **(config or {}),
        }
        for key in ("base_seed", "context_count", "nodes_per_context"):
            value = self.config[key]
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{key} must be an integer")
            self.config[key] = int(value)
        if self.config["context_count"] < 1 or self.config["nodes_per_context"] < 20:
            raise ValueError("Environment/pollution needs at least one context and 20 entities")
        duration = float(self.config["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_seconds must be positive and finite")
        self.config["duration_seconds"] = duration
        weights = self.config["scenario_weights"]
        if not weights or set(weights) - set(SCENARIOS) or any(
            not math.isfinite(float(w)) or float(w) < 0 for w in weights.values()
        ) or sum(weights.values()) <= 0:
            raise ValueError("Invalid environment/pollution scenario weights")
        self.domain_spec = load_domain_spec(Path(__file__).with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(
            Path(__file__).with_name("mechanism_specs.json"),
            known_event_type_ids=EVENT_TYPES,
            known_context_relation_type_ids=self.domain_spec.context_relation_type_ids,
        )
        self._context_cache = {}

    def catalog(self):
        roles = {name: NS + kind for name, kind in (
            ("reach", "reach"), ("upstream_reach", "reach"),
            ("outfall", "outfall"), ("catchment", "catchment"),
            ("monitor", "monitor"), ("intake", "intake"))}
        return {
            "domain": self.domain_id, "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_not_empirically_fitted",
            "scope": {"medium": "freshwater_river_reach_only",
                      "pollutant": "synthetic_dissolved_constituent_not_regulatory_standard",
                      "transport": "one_pulse_plug_flow_first_order_loss_fixed_reference_flow",
                      "observation": "physical_events_are_offline_ground_truth; monitor_reports_after_delay",
                      "statistical_realism_claimed": False},
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {"mode": "deterministic_grounded_projection",
                                "free_form_causal_completion": False},
            "entity_types": [{"entity_type_id": NS + kind, "parent_type_id": None,
                              "required_attributes": attrs} for kind, attrs in (
                ("reach", ["length_m", "velocity_m_s", "reference_flow_m3_s", "loss_rate_per_second"]),
                ("outfall", ["source_kind", "effluent_flow_m3_s"]),
                ("catchment", ["area_km2", "runoff_coefficient", "washoff_concentration_mg_l"]),
                ("monitor", ["alert_threshold_mg_l"]),
                ("intake", ["response_policy_seconds"]))],
            "event_types": [{"event_type_id": event_type, "parent_type_id": event_type.rsplit(".", 1)[0],
                             "required_attributes": ["workflow_id", "scenario_family"],
                             "participant_role_types": roles} for event_type in EVENT_TYPES],
            "context_relation_types": [{"relation_type_id": kind, "directed": True,
                                        "does_not_imply_event_parenthood": True}
                                       for kind in self.domain_spec.context_relation_type_ids],
            "event_relation_types": [{"relation_type_id": NS + key,
                "relation_class": self.mechanism_registry.get(NS + key).relation_semantics,
                "text_rendering": {"surface_predicate": value,
                    "asserted_relation_class": self.mechanism_registry.get(NS + key).relation_semantics}}
                for key, value in PREDICATES.items()],
            "mechanisms": self.mechanism_registry.descriptors(),
        }

    def _build_context(self, index):
        cfg = self.config
        rng = random.Random(cfg["base_seed"] + index * 7919)
        context_id = f"pollution_context_{index:03d}"
        n = cfg["nodes_per_context"]
        reach_count = max(8, n // 3)
        monitor_count = max(2, n // 5)
        outfall_count = max(2, n // 5)
        catchment_count = max(2, n // 10)
        intake_count = n - reach_count - monitor_count - outfall_count - catchment_count
        if intake_count < 2 or intake_count > monitor_count:
            raise ValueError("Node budget cannot support matched monitors and intakes")
        ids = {kind: [f"{context_id}_{kind}_{i:05d}" for i in range(count)]
               for kind, count in (("reach", reach_count), ("monitor", monitor_count),
                                   ("outfall", outfall_count), ("catchment", catchment_count),
                                   ("intake", intake_count))}
        parent = {0: None}
        for i in range(1, reach_count):
            parent[i] = rng.randrange(max(0, i - 5), i)
        upstream_count = {i: 1 for i in range(reach_count)}
        for i in range(reach_count - 1, 0, -1):
            upstream_count[parent[i]] += upstream_count[i]
        entities = []
        reach_attrs = {}
        for i, reach in enumerate(ids["reach"]):
            attrs = {"length_m": round(rng.uniform(20.0, 60.0), 6),
                     "velocity_m_s": round(rng.uniform(.8, 1.8), 6),
                     "reference_flow_m3_s": round(.35 * upstream_count[i], 6),
                     "loss_rate_per_second": .001}
            reach_attrs[reach] = attrs
            entities.append(Entity(reach, self.domain_id, NS + "reach", attrs, context_id))
        outfall_kind = {}
        for i, outfall in enumerate(ids["outfall"]):
            kind = "industrial" if i % 2 == 0 else "wastewater_treatment"
            outfall_kind[outfall] = kind
            entities.append(Entity(outfall, self.domain_id, NS + "outfall",
                {"source_kind": kind, "effluent_flow_m3_s": round(rng.uniform(.15, .45), 6)}, context_id))
        for catchment in ids["catchment"]:
            entities.append(Entity(catchment, self.domain_id, NS + "catchment",
                {"area_km2": round(rng.uniform(.5, 2.0), 6),
                 "runoff_coefficient": round(rng.uniform(.4, .8), 6),
                 "washoff_concentration_mg_l": round(rng.uniform(5, 20), 6)}, context_id))
        for monitor in ids["monitor"]:
            entities.append(Entity(monitor, self.domain_id, NS + "monitor",
                                   {"alert_threshold_mg_l": 1.0}, context_id))
        for intake in ids["intake"]:
            entities.append(Entity(intake, self.domain_id, NS + "intake",
                                   {"response_policy_seconds": 2.0}, context_id))
        reaches = ids["reach"]
        anchor = lambda i: reaches[1 + i % (reach_count - 1)]
        required = {
            NS + "reach_flows_to_reach": [(reaches[i], reaches[parent[i]]) for i in range(1, reach_count)],
            NS + "outfall_discharge_to_reach": [(source, anchor(i)) for i, source in enumerate(ids["outfall"])],
            NS + "catchment_drains_to_reach": [(source, anchor(i)) for i, source in enumerate(ids["catchment"])],
            NS + "monitor_observes_reach": [(monitor, anchor(i)) for i, monitor in enumerate(ids["monitor"])],
            NS + "intake_uses_reach": [(intake, anchor(i)) for i, intake in enumerate(ids["intake"])],
            NS + "intake_receives_from_monitor": [(intake, ids["monitor"][i])
                                                  for i, intake in enumerate(ids["intake"])],
        }
        layers = [replace(layer, required_pairs=tuple(required[layer.relation_type_id]))
                  for layer in self.domain_spec.compile_relation_layers()]
        topology = SparseHeterogeneousTopologyGenerator().generate(
            context_id=context_id, domain=self.domain_id, entities=entities, layers=layers,
            seed=cfg["base_seed"] + index * 7919,
            observation_window_seconds=cfg["duration_seconds"])
        downstream = {reaches[i]: reaches[parent[i]] for i in range(1, reach_count)}
        return {"entities": entities, "relations": topology.relations,
                "statistics": topology.statistics, "ids": ids,
                "reach_attrs": reach_attrs, "downstream": downstream,
                "outfall_kind": outfall_kind,
                "source_reach": {source: reach for relation_type, pairs in required.items()
                                 if relation_type in (NS + "outfall_discharge_to_reach", NS + "catchment_drains_to_reach")
                                 for source, reach in pairs},
                "monitor_by_reach": {reach: monitor for monitor, reach in required[NS + "monitor_observes_reach"]},
                "intake_by_reach": {reach: intake for intake, reach in required[NS + "intake_uses_reach"]}}

    def create_episode_context(self, episode_index, seed):
        index = episode_index % self.config["context_count"]
        if index not in self._context_cache:
            self._context_cache[index] = self._build_context(index)
        data = self._context_cache[index]
        rng = random.Random(seed ^ 0x504f4c4c5554)
        weights = self.config["scenario_weights"]
        scenario = rng.choices(list(weights), weights=list(weights.values()))[0]
        max_anchor = min(len(data["ids"]["intake"]), len(data["ids"]["catchment"]),
                         len(data["ids"]["monitor"]), len(data["ids"]["reach"]) - 1)
        if scenario == "storm_runoff":
            source = rng.choice(data["ids"]["catchment"][:max_anchor])
        else:
            needed = "industrial" if scenario == "industrial_spill" else "wastewater_treatment"
            source = rng.choice([source for source in data["ids"]["outfall"][:max_anchor]
                                 if data["outfall_kind"][source] == needed])
        reach = data["source_reach"][source]
        entity_by_id = {entity.entity_id: entity for entity in data["entities"]}
        source_attrs = entity_by_id[source].attributes
        if scenario == "industrial_spill":
            target_increment = rng.uniform(2.0, 6.0)
            added_flow = 0.0
            mass_flux = data["reach_attrs"][reach]["reference_flow_m3_s"] * target_increment / 1000.0
            loading_basis = {"synthetic_release_target_mg_l": round(target_increment, 6)}
        elif scenario == "wastewater_bypass":
            added_flow = source_attrs["effluent_flow_m3_s"]
            effluent_concentration = rng.uniform(10.0, 50.0)
            mass_flux = added_flow * effluent_concentration / 1000.0
            loading_basis = {"effluent_concentration_mg_l": round(effluent_concentration, 6)}
        else:
            rainfall = rng.uniform(20.0, 60.0)
            added_flow = rainfall * source_attrs["area_km2"] * source_attrs["runoff_coefficient"] / 3.6
            mass_flux = added_flow * source_attrs["washoff_concentration_mg_l"] / 1000.0
            loading_basis = {"rainfall_intensity_mm_h": round(rainfall, 6),
                             "catchment_area_km2": source_attrs["area_km2"],
                             "runoff_coefficient": source_attrs["runoff_coefficient"],
                             "washoff_concentration_mg_l": source_attrs["washoff_concentration_mg_l"]}
        episode_id = f"pollution_episode_{episode_index:07d}"
        state = {"scenario": scenario, "workflow_id": episode_id + "_workflow",
                 "source": source, "anchor": reach,
                 "mass_flux_kg_s": mass_flux, "added_flow_m3_s": added_flow,
                 "loading_basis": loading_basis,
                 "reach_attrs": data["reach_attrs"], "downstream": data["downstream"],
                 "monitor_by_reach": data["monitor_by_reach"],
                 "intake_by_reach": data["intake_by_reach"],
                 "loading_start_event_id": None, "loading_end_event_id": None,
                 "arrivals": {}, "clears": {}, "alerts": {}, "normals": {},
                 "intake_closed": {}, "intake_reopened": {},
                 "terminal_reach": None, "settled_scheduled": False}
        return EpisodeContext(
            episode_id=episode_id, context_id=f"pollution_context_{index:03d}",
            domain=self.domain_id, start_time=self.config["start_time"],
            duration_seconds=self.config["duration_seconds"],
            entities=data["entities"], context_relations=data["relations"],
            context_attributes={"topology_profile": data["statistics"],
                "river_scope": "one_dendritic_network_one_pulse_per_episode",
                "reference_flow_policy": "0.35_m3_s_local_contribution_per_reach"},
            episode_attributes={"scenario_family": scenario,
                "empirical_realism_claimed": False,
                "pollutant_id": "synthetic_dissolved_constituent",
                "reporting_floor_mg_l": .05,
                "event_parenthood": "realized_mass_transport_or_explicit_observation_workflow"},
            state=state)

    def _spec(self, key, context, parents=(), now=0.0, *, reach=None, upstream=None,
              monitor=None, intake=None, inputs=None, extra=None):
        mechanism = self.mechanism_registry.get(NS + key)
        state = context.state
        reach = reach or state["anchor"]
        roles = {"reach": reach}
        pairs = set()
        if key in ("spill_onset", "bypass_onset", "rain_onset", "spill_loading",
                   "bypass_loading", "runoff_loading", "loading_end", "outfall_entry", "runoff_entry"):
            source_role = "catchment" if state["scenario"] == "storm_runoff" else "outfall"
            roles[source_role] = state["source"]
            relation_type = NS + ("catchment_drains_to_reach" if source_role == "catchment"
                                  else "outfall_discharge_to_reach")
            pairs.add((relation_type, state["source"], state["anchor"]))
        if upstream:
            roles["upstream_reach"] = upstream
            pairs.add((NS + "reach_flows_to_reach", upstream, reach))
        if monitor:
            roles["monitor"] = monitor
            pairs.add((NS + "monitor_observes_reach", monitor, reach))
        if intake:
            roles["intake"] = intake
            pairs.add((NS + "intake_uses_reach", intake, reach))
            pairs.add((NS + "intake_receives_from_monitor", intake, monitor))
        evidence_ids = [relation.relation_id for relation in context.context_relations
                        if (relation.relation_type_id, relation.source_entity_id,
                            relation.target_entity_id) in pairs]
        if len(evidence_ids) != len(pairs):
            raise ValueError("Pollution candidate lacks a unique typed context binding")
        attrs = {"workflow_id": state["workflow_id"], "scenario_family": state["scenario"],
                 "pollutant_id": "synthetic_dissolved_constituent", "source_entity_id": state["source"],
                 "reach_id": reach, "source_added_flow_m3_s": round(state["added_flow_m3_s"], 9)}
        if extra:
            attrs.update(extra)
        return CandidateSpec(
            mechanism_id=mechanism.mechanism_id,
            target_event_type_id=mechanism.target_event_type_id,
            parent_event_ids=list(parents),
            participants=[EventParticipant(entity_id, role) for role, entity_id in roles.items()],
            temporal_model_ref=mechanism.generation_temporal_model_ref,
            temporal_inputs=inputs or {}, activation_time=now,
            combination=mechanism.parent_combination, phase_priority=10,
            attributes=attrs,
            context_evidence=ContextEvidence(context_relation_ids=evidence_ids,
                entity_ids=list(dict.fromkeys(roles.values())),
                evaluated_at_offset_seconds=now),
            provenance={"input_anchor": "observed_at" if key in ("intake_close", "intake_reopen")
                        else "occurrence_start",
                        "parameter_status": mechanism.parameter_status})

    def seed_candidates(self, context, rng):
        del rng
        key = {"industrial_spill": "spill_onset", "wastewater_bypass": "bypass_onset",
               "storm_runoff": "rain_onset"}[context.state["scenario"]]
        return [self._spec(key, context)]

    def revalidate_candidate(self, candidate, context, event_by_id, now):
        del event_by_id
        key = candidate.mechanism_id.removeprefix(NS)
        state = context.state
        if key == "intake_close":
            reach = candidate.attributes["reach_id"]
            normal_reported = reach in state["normals"]
            already_closed = reach in state["intake_closed"]
            valid = not normal_reported and not already_closed
            evidence = copy.deepcopy(candidate.context_evidence)
            evidence.evaluated_at_offset_seconds = now
            evidence.state_predicates.append(StatePredicateEvidence(
                state["intake_by_reach"][reach], "monitor_normal_already_reported", "equals",
                False, normal_reported, not normal_reported))
            evidence.state_predicates.append(StatePredicateEvidence(
                state["intake_by_reach"][reach], "intake_already_closed", "equals",
                False, already_closed, not already_closed))
            return CandidateUpdate(valid, None if valid else "normal_report_superseded_closure",
                                   context_evidence=evidence)
        return CandidateUpdate(True)

    def materialize_event(self, event_id, candidate, context, rng):
        del rng
        state = context.state
        attrs = copy.deepcopy(candidate.attributes)
        kind = candidate.target_event_type_id
        if kind in (SPILL, BYPASS, RAIN, LOADING, END):
            attrs.update(source_mass_flux_kg_s=round(state["mass_flux_kg_s"], 9),
                         **state["loading_basis"])
        if kind == END:
            attrs["end_reason"] = ("bypass_repaired" if state["scenario"] == "wastewater_bypass"
                                   else "finite_loading_ended")
        if kind == ARRIVAL:
            flux = attrs["mass_flux_kg_s"]
            base_flow = state["reach_attrs"][attrs["reach_id"]]["reference_flow_m3_s"]
            attrs["reference_flow_m3_s"] = base_flow
            attrs["concentration_increment_mg_l"] = round(
                1000.0 * flux / (base_flow + state["added_flow_m3_s"]), 9)
            reach = attrs["reach_id"]
            downstream = state["downstream"].get(reach)
            if downstream:
                delay = (state["reach_attrs"][reach]["length_m"] /
                         state["reach_attrs"][reach]["velocity_m_s"])
                next_flux = flux * math.exp(-state["reach_attrs"][reach]["loss_rate_per_second"] * delay)
                next_flow = state["reach_attrs"][downstream]["reference_flow_m3_s"] + state["added_flow_m3_s"]
                next_concentration = 1000.0 * next_flux / next_flow
            else:
                next_concentration = None
            attrs["next_reach_predicted_mg_l"] = (round(next_concentration, 9)
                                                   if next_concentration is not None else None)
            attrs["terminal_reason"] = ("network_outlet" if downstream is None else
                                        "below_reporting_floor" if next_concentration < .05 else None)
        elif kind == CLEAR:
            attrs["concentration_increment_mg_l"] = 0.0
            attrs["trailing_edge_policy"] = "loading_end_plus_cumulative_advection_time"
        elif kind == ALERT:
            monitor = next(p.entity_id for p in candidate.participants if p.role == "monitor")
            attrs["monitor_id"] = monitor
            attrs["alert_threshold_mg_l"] = context.entity_by_id[monitor].attributes["alert_threshold_mg_l"]
        elif kind == NORMAL:
            attrs["monitor_id"] = next(p.entity_id for p in candidate.participants if p.role == "monitor")
            attrs["reported_increment_mg_l"] = 0.0
        elif kind in (INTAKE_CLOSED, INTAKE_REOPENED):
            attrs["intake_id"] = next(p.entity_id for p in candidate.participants if p.role == "intake")
            attrs["monitor_id"] = next(p.entity_id for p in candidate.participants if p.role == "monitor")
        elif kind == SETTLED:
            attrs["outcome"] = "finite_pulse_cleared_through_reported_path"
            attrs["reached_reach_count"] = len(state["arrivals"])
        return EventRecord(event_id, context.episode_id, self.domain_id, kind,
                           TemporalExtent(candidate.scheduled_time), list(candidate.participants), attrs,
                           event_role="root" if not candidate.parent_event_ids else "derived",
                           provenance={"generator": "synthetic_river_pulse",
                                       "mechanism_id": candidate.mechanism_id})

    def observation_plan(self, event, candidate, context):
        del candidate, context
        if event.event_type_id in (ALERT, NORMAL):
            source = "synthetic_monitor_report"
        elif event.event_type_id in (INTAKE_CLOSED, INTAKE_REOPENED):
            source = "synthetic_intake_operator"
        else:
            source = "offline_synthetic_ground_truth"
        return ObservationPlan(recording_model_ref=NS + "log_ingestion_delay.prior_v1",
                               source=source)

    def apply_event(self, event, candidate, context):
        del candidate
        state = context.state
        reach = event.attributes["reach_id"]
        kind = event.event_type_id
        if kind == LOADING:
            state["loading_start_event_id"] = event.event_id
        elif kind == END:
            state["loading_end_event_id"] = event.event_id
        elif kind == ARRIVAL:
            state["arrivals"][reach] = event.event_id
            if event.attributes["terminal_reason"]:
                state["terminal_reach"] = reach
        elif kind == CLEAR:
            state["clears"][reach] = event.event_id
        elif kind == ALERT:
            state["alerts"][reach] = event.event_id
        elif kind == NORMAL:
            state["normals"][reach] = event.event_id
        elif kind == INTAKE_CLOSED:
            state["intake_closed"][reach] = event.event_id
        elif kind == INTAKE_REOPENED:
            state["intake_reopened"][reach] = event.event_id

    def _clear_spec(self, context, reach, event_by_id, now):
        state = context.state
        if not state["loading_end_event_id"] or reach not in state["arrivals"]:
            return None
        arrival = event_by_id[state["arrivals"][reach]]
        ended = event_by_id[state["loading_end_event_id"]]
        due = ended.temporal.anchor("occurrence_start") + arrival.attributes["cumulative_travel_seconds"]
        return self._spec("plume_clear", context,
            [arrival.event_id, ended.event_id], now, reach=reach,
            inputs={"scheduled_offset_seconds": due},
            extra={"cumulative_travel_seconds": arrival.attributes["cumulative_travel_seconds"]})

    def _normal_spec(self, context, reach, event_by_id, now):
        state = context.state
        if reach not in state["alerts"] or reach not in state["clears"]:
            return None
        cleared = event_by_id[state["clears"][reach]]
        monitor = state["monitor_by_reach"][reach]
        return self._spec("monitor_normal", context,
            [state["alerts"][reach], cleared.event_id], now,
            reach=reach, monitor=monitor,
            extra={"sample_time_offset_seconds": cleared.temporal.anchor("occurrence_start")})

    def spawn_candidates(self, event, candidate, context, event_by_id, rng):
        del candidate, rng
        state, kind = context.state, event.event_type_id
        reach = event.attributes["reach_id"]
        now = event.temporal.anchor("occurrence_start")
        parent = [event.event_id]
        result = []
        if kind in (SPILL, BYPASS, RAIN):
            key = {SPILL: "spill_loading", BYPASS: "bypass_loading", RAIN: "runoff_loading"}[kind]
            result.append(self._spec(key, context, parent, now))
        elif kind == LOADING:
            result.append(self._spec("loading_end", context, parent, now))
            entry_key = "runoff_entry" if state["scenario"] == "storm_runoff" else "outfall_entry"
            result.append(self._spec(entry_key, context, parent, now,
                extra={"mass_flux_kg_s": round(state["mass_flux_kg_s"], 9),
                       "cumulative_travel_seconds": 0.0}))
        elif kind == END:
            for arrived_reach in state["arrivals"]:
                result.append(self._clear_spec(context, arrived_reach, event_by_id, now))
        elif kind == ARRIVAL:
            downstream = state["downstream"].get(reach)
            if downstream and not event.attributes["terminal_reason"]:
                props = state["reach_attrs"][reach]
                delay = props["length_m"] / props["velocity_m_s"]
                flux = event.attributes["mass_flux_kg_s"] * math.exp(-props["loss_rate_per_second"] * delay)
                result.append(self._spec("downstream_transport", context, parent, now,
                    reach=downstream, upstream=reach,
                    inputs={"delay_seconds": delay},
                    extra={"mass_flux_kg_s": round(flux, 9),
                           "cumulative_travel_seconds": event.attributes["cumulative_travel_seconds"] + delay}))
            monitor = state["monitor_by_reach"].get(reach)
            if monitor and event.attributes["concentration_increment_mg_l"] >= context.entity_by_id[monitor].attributes["alert_threshold_mg_l"]:
                result.append(self._spec("monitor_alert", context, parent, now,
                    reach=reach, monitor=monitor,
                    extra={"sample_concentration_mg_l": event.attributes["concentration_increment_mg_l"],
                           "sample_time_offset_seconds": now}))
            clear = self._clear_spec(context, reach, event_by_id, now)
            if clear:
                result.append(clear)
        elif kind in (CLEAR, ALERT):
            if kind == CLEAR and reach == state["terminal_reach"] and not state["settled_scheduled"]:
                state["settled_scheduled"] = True
                result.append(self._spec("workflow_settled", context, parent, now, reach=reach))
            normal = self._normal_spec(context, reach, event_by_id, now)
            if normal:
                result.append(normal)
            if kind == ALERT:
                intake = state["intake_by_reach"].get(reach)
                if intake:
                    result.append(self._spec("intake_close", context, parent, now,
                        reach=reach, monitor=state["monitor_by_reach"][reach], intake=intake))
        elif kind == NORMAL and reach in state["intake_closed"]:
            result.append(self._spec("intake_reopen", context,
                [state["intake_closed"][reach], event.event_id], now,
                reach=reach, monitor=state["monitor_by_reach"][reach],
                intake=state["intake_by_reach"][reach]))
        return result

    def relation_specs(self, event, candidate, context, event_by_id):
        del event, context, event_by_id
        mechanism = self.mechanism_registry.get(candidate.mechanism_id)
        anchor = "observed_at" if candidate.mechanism_id in (NS + "intake_close", NS + "intake_reopen") else "occurrence_start"
        return [RelationSpec(source_event_id=parent,
            relation_class=mechanism.relation_semantics,
            relation_type_id=mechanism.mechanism_id, rule_id=mechanism.mechanism_id,
            source_anchor=anchor, target_anchor="occurrence_start",
            mechanism_group_id=candidate.candidate_id if len(candidate.parent_event_ids) > 1 else None,
            context_evidence=copy.deepcopy(candidate.context_evidence),
            provenance={"parent_basis": "realized_mass_transport_or_observed_response",
                        "topology_edge_alone_sufficient": False})
            for parent in candidate.parent_event_ids]

    def validate_result(self, result):
        """Replay type, flow-path, mass, time and response claims from exports."""
        errors = []
        entities = {entity.entity_id: entity for entity in result.entities}
        bindings = {(relation.relation_type_id, relation.source_entity_id): relation.target_entity_id
                    for relation in result.context_relations}
        if len(bindings) != len(result.context_relations):
            errors.append("Ambiguous functional river relation")
        for layer in self.domain_spec.compile_relation_layers():
            for relation in result.context_relations:
                if relation.relation_type_id != layer.relation_type_id:
                    continue
                source, target = entities.get(relation.source_entity_id), entities.get(relation.target_entity_id)
                if not source or not target or source.entity_type_id not in layer.source_type_ids or target.entity_type_id not in layer.target_type_ids:
                    errors.append(f"Invalid typed river relation {relation.relation_id}")
            if layer.relation_type_id != NS + "reach_flows_to_reach":
                for entity in result.entities:
                    if entity.entity_type_id in layer.source_type_ids and (layer.relation_type_id, entity.entity_id) not in bindings:
                        errors.append(f"Missing functional river binding for {entity.entity_id}")
        reaches = {entity.entity_id for entity in result.entities if entity.entity_type_id == NS + "reach"}
        downstream = {source: target for (kind, source), target in bindings.items()
                      if kind == NS + "reach_flows_to_reach"}
        outlets = reaches - set(downstream)
        if len(outlets) != 1 or len(downstream) != len(reaches) - 1:
            errors.append("River must have exactly one outlet and one downstream link per other reach")
        for reach in reaches:
            path = set()
            node = reach
            while node in downstream:
                if node in path:
                    errors.append(f"Directed river cycle at {reach}")
                    break
                path.add(node)
                node = downstream[node]
            if node not in reaches:
                errors.append(f"River path leaves typed reach set at {reach}")
        candidates = {candidate.fired_event_id: candidate for candidate in result.candidates
                      if candidate.fired_event_id}
        seen = {}
        alerts, normals, closed, reopened = set(), set(), set(), set()
        settled_count = 0
        tolerance = 2e-6
        for event in result.events:
            attrs, kind = event.attributes, event.event_type_id
            candidate = candidates.get(event.event_id)
            if candidate is None:
                errors.append(f"Missing fired candidate for {event.event_id}")
                continue
            mechanism = self.mechanism_registry.get(candidate.mechanism_id)
            if mechanism.target_event_type_id != kind:
                errors.append(f"Mechanism target mismatch at {event.event_id}")
            parents = [seen.get(parent_id) for parent_id in candidate.parent_event_ids]
            for parent in parents:
                if parent is None or parent.event_type_id not in mechanism.source_event_type_ids:
                    errors.append(f"Invalid realized parent at {event.event_id}")
                elif parent.temporal.anchor("occurrence_start") > candidate.activation_time + 1e-9:
                    errors.append(f"Future parent activated {event.event_id}")
            roles = {participant.role: participant.entity_id for participant in event.participants}
            if len(roles) != len(event.participants):
                errors.append(f"Duplicate participant role at {event.event_id}")
            role_types = {"reach": "reach", "upstream_reach": "reach", "outfall": "outfall",
                          "catchment": "catchment", "monitor": "monitor", "intake": "intake"}
            for role, entity_id in roles.items():
                entity = entities.get(entity_id)
                if role not in role_types or not entity or entity.entity_type_id != NS + role_types[role]:
                    errors.append(f"Incorrect node meaning for {role} at {event.event_id}")
            reach = roles.get("reach")
            if reach != attrs.get("reach_id"):
                errors.append(f"Reach identity mismatch at {event.event_id}")
            if attrs.get("scenario_family") != result.episode_attributes.get("scenario_family"):
                errors.append(f"Scenario mismatch at {event.event_id}")
            if event.temporal.anchor("observed_at") != event.temporal.anchor("occurrence_start"):
                errors.append(f"Invalid offline ground-truth time at {event.event_id}")
            if "outfall" in roles and bindings.get((NS + "outfall_discharge_to_reach", roles["outfall"])) != reach:
                errors.append(f"Outfall bound to wrong reach at {event.event_id}")
            if "catchment" in roles and bindings.get((NS + "catchment_drains_to_reach", roles["catchment"])) != reach:
                errors.append(f"Catchment bound to wrong reach at {event.event_id}")
            if "monitor" in roles and bindings.get((NS + "monitor_observes_reach", roles["monitor"])) != reach:
                errors.append(f"Monitor bound to wrong reach at {event.event_id}")
            if "intake" in roles and (bindings.get((NS + "intake_uses_reach", roles["intake"])) != reach
                                      or bindings.get((NS + "intake_receives_from_monitor", roles["intake"])) != roles.get("monitor")):
                errors.append(f"Intake/monitor binding mismatch at {event.event_id}")
            if kind == LOADING and reach in entities:
                source_id = attrs.get("source_entity_id")
                source = entities.get(source_id)
                base_q = entities[reach].attributes["reference_flow_m3_s"]
                scenario = attrs["scenario_family"]
                if scenario == "industrial_spill":
                    expected_q = 0.0
                    expected_mass = base_q * attrs["synthetic_release_target_mg_l"] / 1000.0
                elif scenario == "wastewater_bypass" and source:
                    expected_q = source.attributes["effluent_flow_m3_s"]
                    expected_mass = expected_q * attrs["effluent_concentration_mg_l"] / 1000.0
                elif scenario == "storm_runoff":
                    expected_q = attrs["rainfall_intensity_mm_h"] * attrs["catchment_area_km2"] * attrs["runoff_coefficient"] / 3.6
                    expected_mass = expected_q * attrs["washoff_concentration_mg_l"] / 1000.0
                else:
                    expected_q, expected_mass = math.nan, math.nan
                if (not math.isfinite(expected_mass)
                    or abs(attrs.get("source_mass_flux_kg_s", -1) - expected_mass) > tolerance
                    or abs(attrs.get("source_added_flow_m3_s", -1) - expected_q) > tolerance):
                    errors.append(f"Loading violates dimensional mass/flow basis at {event.event_id}")
            if kind == ARRIVAL and reach in entities:
                q = entities[reach].attributes["reference_flow_m3_s"] + attrs.get("source_added_flow_m3_s", 0)
                flux = attrs.get("mass_flux_kg_s", -1)
                expected_concentration = 1000.0 * flux / q if q > 0 else math.nan
                if not math.isfinite(expected_concentration) or abs(attrs.get("concentration_increment_mg_l", -1) - expected_concentration) > tolerance:
                    errors.append(f"Plume concentration/flow mismatch at {event.event_id}")
                next_reach = downstream.get(reach)
                if next_reach:
                    props = entities[reach].attributes
                    travel = props["length_m"] / props["velocity_m_s"]
                    surviving_flux = flux * math.exp(-props["loss_rate_per_second"] * travel)
                    next_q = (entities[next_reach].attributes["reference_flow_m3_s"]
                              + attrs.get("source_added_flow_m3_s", 0))
                    next_concentration = 1000.0 * surviving_flux / next_q
                    terminal_reason = "below_reporting_floor" if next_concentration < .05 else None
                else:
                    next_concentration = None
                    terminal_reason = "network_outlet"
                predicted = attrs.get("next_reach_predicted_mg_l")
                if (attrs.get("terminal_reason") != terminal_reason
                    or (next_concentration is None and predicted is not None)
                    or (next_concentration is not None and (predicted is None or abs(predicted - next_concentration) > tolerance))):
                    errors.append(f"Incorrect plume propagation/terminal decision at {event.event_id}")
                if candidate.mechanism_id == NS + "downstream_transport" and parents and parents[0]:
                    upstream = parents[0]
                    upstream_reach = upstream.attributes["reach_id"]
                    props = entities[upstream_reach].attributes
                    travel = props["length_m"] / props["velocity_m_s"]
                    expected_flux = upstream.attributes["mass_flux_kg_s"] * math.exp(-props["loss_rate_per_second"] * travel)
                    if (downstream.get(upstream_reach) != reach or roles.get("upstream_reach") != upstream_reach
                        or abs(event.temporal.anchor("occurrence_start") - upstream.temporal.anchor("occurrence_start") - travel) > tolerance
                        or abs(flux - expected_flux) > tolerance
                        or abs(attrs.get("cumulative_travel_seconds", -1) - upstream.attributes["cumulative_travel_seconds"] - travel) > tolerance):
                        errors.append(f"Invalid downstream mass transport at {event.event_id}")
                elif candidate.mechanism_id in (NS + "outfall_entry", NS + "runoff_entry") and parents and parents[0]:
                    if (reach != parents[0].attributes["reach_id"]
                        or abs(flux - parents[0].attributes["source_mass_flux_kg_s"]) > tolerance
                        or abs(attrs.get("cumulative_travel_seconds", -1)) > tolerance):
                        errors.append(f"Invalid source entry at {event.event_id}")
            elif kind == CLEAR and len(parents) == 2 and all(parents):
                arrival = next((parent for parent in parents if parent.event_type_id == ARRIVAL), None)
                ended = next((parent for parent in parents if parent.event_type_id == END), None)
                if (arrival is None or ended is None or arrival.attributes["reach_id"] != reach
                    or abs(event.temporal.anchor("occurrence_start") - ended.temporal.anchor("occurrence_start")
                           - arrival.attributes["cumulative_travel_seconds"]) > tolerance):
                    errors.append(f"Invalid trailing-edge clearance at {event.event_id}")
            elif kind == ALERT and parents and parents[0]:
                threshold = entities[roles["monitor"]].attributes["alert_threshold_mg_l"]
                sample = parents[0].attributes.get("concentration_increment_mg_l", -1)
                if (sample < threshold or abs(attrs.get("sample_concentration_mg_l", -1) - sample) > tolerance
                    or attrs.get("sample_time_offset_seconds") != parents[0].temporal.anchor("occurrence_start")):
                    errors.append(f"Alert not supported by sampled plume at {event.event_id}")
                alerts.add(reach)
            elif kind == NORMAL:
                if reach not in alerts or len(parents) != 2 or not all(parents) or {p.event_type_id for p in parents} != {CLEAR, ALERT}:
                    errors.append(f"Normal report lacks clearance and prior alert at {event.event_id}")
                normals.add(reach)
            elif kind == INTAKE_CLOSED:
                if reach in normals or reach not in alerts or reach in closed:
                    errors.append(f"Invalid intake closure at {event.event_id}")
                closed.add(reach)
            elif kind == INTAKE_REOPENED:
                if (reach not in closed or reach not in normals or reach in reopened
                    or len(parents) != 2 or not all(parents)
                    or {p.event_type_id for p in parents} != {NORMAL, INTAKE_CLOSED}):
                    errors.append(f"Invalid intake reopening at {event.event_id}")
                reopened.add(reach)
            elif kind == SETTLED:
                settled_count += 1
                if (settled_count > 1 or not parents or not parents[0]
                    or parents[0].event_type_id != CLEAR):
                    errors.append(f"Invalid physical workflow settlement at {event.event_id}")
                else:
                    clear_candidate = candidates.get(parents[0].event_id)
                    arrival_ids = [parent_id for parent_id in clear_candidate.parent_event_ids
                                   if seen.get(parent_id) and seen[parent_id].event_type_id == ARRIVAL] if clear_candidate else []
                    if len(arrival_ids) != 1 or not seen[arrival_ids[0]].attributes.get("terminal_reason"):
                        errors.append(f"Workflow settled before plume reached a terminal condition at {event.event_id}")
            seen[event.event_id] = event
        return errors

    def render_episode_text(self, result):
        sentences, claims = [], []
        projected_fields = {
            SPILL: ("source_mass_flux_kg_s", "synthetic_release_target_mg_l"),
            BYPASS: ("source_mass_flux_kg_s", "effluent_concentration_mg_l"),
            RAIN: ("rainfall_intensity_mm_h", "catchment_area_km2", "runoff_coefficient"),
            LOADING: ("source_mass_flux_kg_s", "source_added_flow_m3_s"),
            END: ("end_reason",),
            ARRIVAL: ("mass_flux_kg_s", "concentration_increment_mg_l",
                      "cumulative_travel_seconds", "terminal_reason"),
            CLEAR: ("concentration_increment_mg_l", "cumulative_travel_seconds"),
            ALERT: ("sample_concentration_mg_l", "alert_threshold_mg_l",
                    "sample_time_offset_seconds"),
            NORMAL: ("reported_increment_mg_l", "sample_time_offset_seconds"),
            INTAKE_CLOSED: ("intake_id", "monitor_id"),
            INTAKE_REOPENED: ("intake_id", "monitor_id"),
            SETTLED: ("outcome", "reached_reach_count"),
        }
        for event in result.events:
            roles = ", ".join(f"{p.role}={p.entity_id}" for p in event.participants)
            selected = {key: event.attributes[key] for key in projected_fields[event.event_type_id]
                        if key in event.attributes and event.attributes[key] is not None}
            attrs = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            sentences.append(f"+{event.temporal.anchor('occurrence_start'):.3f}s {event.event_id} "
                             f"{event.event_type_id} ({roles}); {attrs}.")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                           "event_ids": [event.event_id], "relation_ids": [],
                           "evidence_fields": ["event_type_id", "temporal", "participants"]
                           + [f"attributes.{key}" for key in selected]})
        for relation in result.event_relations:
            predicate = PREDICATES[relation.relation_type_id.removeprefix(NS)]
            sentences.append(f"{relation.source_event_id} {predicate} {relation.target_event_id} "
                             f"({relation.relation_class}, {relation.temporal_link.lag_seconds:.3f}s after "
                             f"{relation.temporal_link.source_anchor}; rule={relation.rule_id}).")
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


def build_environment_pollution_temporal_models():
    registry = build_standard_temporal_registry()
    for model in (
        ExponentialArrivalModel(NS + "root_arrival.prior_v1", default_rate_per_second=.05),
        ConditionalLogNormalModel(NS + "bypass_delay.prior_v1", math.log(3.0), .4, {}, minimum_seconds=.2),
        ConditionalLogNormalModel(NS + "runoff_response.prior_v1", math.log(10.0), .5, {}, minimum_seconds=1.0),
        ConditionalLogNormalModel(NS + "loading_duration.prior_v1", math.log(20.0), .4, {}, minimum_seconds=2.0),
        DeterministicDelayModel(NS + "advection_delay", 0.0),
        ConditionalLogNormalModel(NS + "monitor_delay.prior_v1", math.log(1.2), .4, {}, minimum_seconds=.1),
        DeterministicDelayModel(NS + "intake_response_delay", 2.0),
        ConditionalLogNormalModel(NS + "log_ingestion_delay.prior_v1", math.log(.5), .4, {}, minimum_seconds=.01),
    ):
        registry.register(model)
    return registry
