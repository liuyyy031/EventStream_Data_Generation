"""Synthetic event streams on a typed radial distribution feeder.

This is an event-logic model, not an electrical power-flow or protection study.
Topology, thresholds and delays are visible synthetic priors. In particular,
an energized physical path is necessary but not sufficient for event parenthood.
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
    PiecewiseExponentialHazardModel, build_standard_temporal_registry,
)
from data_generation.generation_core.topology import SparseHeterogeneousTopologyGenerator

NS = "energy."
FAULT = NS + "line.fault.occurred"
OPEN = NS + "breaker.opened"
INTERRUPTED = NS + "load.supply.interrupted"
REPAIRED = NS + "line.repair.completed"
CLOSED = NS + "breaker.closed"
RESTORED = NS + "load.supply.restored"
SURGE = NS + "load.demand.increased"
HIGH = NS + "line.loading.high"
NORMAL = NS + "load.demand.normalized"
SHORTFALL = NS + "generator.output.reduced"
LOW = NS + "reserve.low.observed"
RECOVERED = NS + "generator.output.recovered"
DISPATCHED = NS + "generator.reserve.dispatched"
RESERVE_RESTORED = NS + "reserve.restored"
SETTLED = NS + "workflow.settled"
EVENT_TYPES = (FAULT, OPEN, INTERRUPTED, REPAIRED, CLOSED, RESTORED, SURGE,
               HIGH, NORMAL, SHORTFALL, LOW, RECOVERED, DISPATCHED,
               RESERVE_RESTORED, SETTLED)
SCENARIOS = ("line_fault", "demand_surge", "generation_shortfall")
PREDICATES = {
    "protection_trip": "triggered the protecting breaker opening recorded as",
    "supply_interruption": "interrupted this downstream load's supply as",
    "repair_completion": "preceded the line repair completed as",
    "fault_reclose": "permitted the breaker reclosing recorded as",
    "supply_restoration": "restored this downstream load's supply as",
    "outage_join": "contributed one restored load to the settled workflow",
    "overload_observed": "was followed by measured high line loading recorded as",
    "overload_trip": "triggered the overload-protection opening recorded as",
    "demand_normalization": "returned to baseline demand as",
    "surge_reclose": "was a required condition for the breaker reclosing recorded as",
    "surge_settled": "ended the surge without a breaker trip as",
    "reserve_low_observed": "was followed by the low-reserve observation recorded as",
    "primary_recovery": "was followed by primary output recovery recorded as",
    "reserve_dispatch": "prompted reserve-generator dispatch recorded as",
    "reserve_restoration": "restored the aggregate reserve state as",
    "shortfall_settled": "resolved the shortfall workflow as",
}


class EnergyPackage:
    domain_id = "energy"

    def __init__(self, config=None):
        self.config = {
            "base_seed": 20260930, "context_count": 4, "nodes_per_context": 1000,
            "duration_seconds": 180.0, "start_time": "2026-09-30T08:00:00+08:00",
            "scenario_weights": dict(zip(SCENARIOS, (.4, .3, .3))), **(config or {}),
        }
        for key in ("base_seed", "context_count", "nodes_per_context"):
            value = self.config[key]
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{key} must be an integer")
            self.config[key] = int(value)
        if self.config["context_count"] < 1 or self.config["nodes_per_context"] < 20:
            raise ValueError("Energy requires at least one context with at least 20 entities")
        duration = float(self.config["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_seconds must be positive and finite")
        self.config["duration_seconds"] = duration
        weights = self.config["scenario_weights"]
        if not weights or set(weights) - set(SCENARIOS) or any(
            not math.isfinite(float(w)) or float(w) < 0 for w in weights.values()
        ) or sum(weights.values()) <= 0:
            raise ValueError("Invalid energy scenario weights")
        self.domain_spec = load_domain_spec(Path(__file__).with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(
            Path(__file__).with_name("mechanism_specs.json"),
            known_event_type_ids=EVENT_TYPES,
            known_context_relation_type_ids=self.domain_spec.context_relation_type_ids,
        )
        self._context_cache = {}

    def catalog(self):
        roles = {role: NS + kind for role, kind in (
            ("line", "line"), ("breaker", "breaker"), ("load", "load"),
            ("primary_generator", "generator"), ("reserve_generator", "generator"),
            ("bus", "bus"), ("upstream_bus", "bus"),
            ("downstream_bus", "bus"), ("attached_bus", "bus"))}
        return {
            "domain": self.domain_id, "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_not_empirically_fitted",
            "scope": {"network": "single_radial_distribution_feeder_no_tie_switches",
                      "electrical_model": "aggregate_load_and_line_rating_thresholds_not_power_flow",
                      "protection_model": "one_breaker_per_line_no_coordination_study",
                      "clock": "synchronized_synthetic_seconds",
                      "observation": "device_local_at_occurrence; collector_ingestion_later",
                      "statistical_realism_claimed": False},
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {"mode": "deterministic_grounded_projection",
                                "free_form_causal_completion": False},
            "entity_types": [{"entity_type_id": NS + kind, "parent_type_id": None,
                              "required_attributes": attrs} for kind, attrs in (
                ("bus", ["depth", "nominal_voltage_kv"]),
                ("line", ["rating_mw", "baseline_downstream_mw"]),
                ("breaker", ["normally_closed", "protected_line_id"]),
                ("load", ["baseline_demand_mw"]),
                ("generator", ["role", "capacity_mw"]))],
            "event_types": [{"event_type_id": t, "parent_type_id": t.rsplit(".", 1)[0],
                             "required_attributes": ["workflow_id", "scenario_family"],
                             "participant_role_types": roles} for t in EVENT_TYPES],
            "context_relation_types": [{"relation_type_id": t, "directed": True,
                                        "does_not_imply_event_parenthood": True}
                                       for t in self.domain_spec.context_relation_type_ids],
            "event_relation_types": [{"relation_type_id": NS + key,
                "relation_class": self.mechanism_registry.get(NS + key).relation_semantics,
                "text_rendering": {"surface_predicate": value,
                    "asserted_relation_class": self.mechanism_registry.get(NS + key).relation_semantics}}
                for key, value in PREDICATES.items()],
            "mechanisms": self.mechanism_registry.descriptors(),
        }

    def _build_context(self, index):
        cfg = self.config
        rng = random.Random(cfg["base_seed"] + 7919 * index)
        context_id = f"energy_context_{index:03d}"
        n = cfg["nodes_per_context"]
        bus_count = max(5, n // 4)
        line_count = bus_count - 1
        load_count = n - bus_count - 2 * line_count - 2
        if load_count < bus_count:
            raise ValueError("Node allocation cannot attach at least one load to each bus")
        buses = [f"{context_id}_bus_{i:05d}" for i in range(bus_count)]
        lines = [f"{context_id}_line_{i:05d}" for i in range(line_count)]
        breakers = [f"{context_id}_breaker_{i:05d}" for i in range(line_count)]
        loads = [f"{context_id}_load_{i:05d}" for i in range(load_count)]
        generators = [f"{context_id}_generator_{i:02d}" for i in range(2)]
        parent = {0: None}
        depth = {0: 0}
        for child in range(1, bus_count):
            parent[child] = rng.randrange(max(0, child - 4), child)
            depth[child] = depth[parent[child]] + 1
        load_bus_index = {load: i % bus_count for i, load in enumerate(loads)}
        load_demand = {load: round(rng.uniform(.2, 1.5), 6) for load in loads}
        descendants = {i: {i} for i in range(bus_count)}
        for child in range(bus_count - 1, 0, -1):
            descendants[parent[child]].update(descendants[child])
        downstream = {lines[i - 1]: tuple(load for load in loads
                       if load_bus_index[load] in descendants[i]) for i in range(1, bus_count)}
        baseline = {line: round(sum(load_demand[load] for load in affected), 6)
                    for line, affected in downstream.items()}
        rating = {line: round(baseline[line] * rng.uniform(1.15, 1.4), 6)
                  for line in lines}
        total_demand = round(sum(load_demand.values()), 6)
        entities = []
        for i, bus in enumerate(buses):
            entities.append(Entity(bus, self.domain_id, NS + "bus",
                                   {"depth": depth[i], "nominal_voltage_kv": 11.0}, context_id))
        for i, line in enumerate(lines):
            entities.append(Entity(line, self.domain_id, NS + "line",
                {"rating_mw": rating[line], "baseline_downstream_mw": baseline[line]}, context_id))
            entities.append(Entity(breakers[i], self.domain_id, NS + "breaker",
                {"normally_closed": True, "protected_line_id": line}, context_id))
        for load in loads:
            entities.append(Entity(load, self.domain_id, NS + "load",
                {"baseline_demand_mw": load_demand[load]}, context_id))
        for generator, role, capacity in zip(generators, ("primary", "reserve"),
                                              (1.25 * total_demand, .45 * total_demand)):
            entities.append(Entity(generator, self.domain_id, NS + "generator",
                {"role": role, "capacity_mw": round(capacity, 6)}, context_id))
        required = {
            NS + "line_from_bus": [(lines[i - 1], buses[parent[i]]) for i in range(1, bus_count)],
            NS + "line_to_bus": [(lines[i - 1], buses[i]) for i in range(1, bus_count)],
            NS + "breaker_protects_line": list(zip(breakers, lines)),
            NS + "load_attached_to_bus": [(load, buses[load_bus_index[load]]) for load in loads],
            NS + "generator_connected_to_bus": [(gen, buses[0]) for gen in generators],
        }
        layers = [replace(layer, required_pairs=tuple(required[layer.relation_type_id]))
                  for layer in self.domain_spec.compile_relation_layers()]
        topology = SparseHeterogeneousTopologyGenerator().generate(
            context_id=context_id, domain=self.domain_id, entities=entities, layers=layers,
            seed=cfg["base_seed"] + 7919 * index,
            observation_window_seconds=cfg["duration_seconds"])
        return {"entities": entities, "relations": topology.relations,
                "statistics": topology.statistics, "buses": buses, "lines": lines,
                "breakers": breakers, "loads": loads, "generators": generators,
                "downstream": downstream, "rating": rating, "baseline": baseline,
                "load_demand": load_demand, "total_demand": total_demand}

    def create_episode_context(self, episode_index, seed):
        index = episode_index % self.config["context_count"]
        if index not in self._context_cache:
            self._context_cache[index] = self._build_context(index)
        data = self._context_cache[index]
        rng = random.Random(seed ^ 0x454e45524759)
        weights = self.config["scenario_weights"]
        scenario = rng.choices(list(weights), weights=list(weights.values()))[0]
        selectable = [line for line in data["lines"] if 2 <= len(data["downstream"][line]) <= 12]
        if not selectable:
            raise ValueError("Feeder needs a line serving 2..12 loads for an outage workflow")
        line = rng.choice(selectable)
        breaker = data["breakers"][data["lines"].index(line)]
        affected = list(data["downstream"][line])
        trigger_load = rng.choice(affected)
        delta = round((data["rating"][line] - data["baseline"][line]) * rng.uniform(1.2, 1.8), 6)
        episode_id = f"energy_episode_{episode_index:07d}"
        state = {"scenario": scenario, "workflow_id": episode_id + "_workflow",
                 "line": line, "breaker": breaker, "affected_loads": affected,
                 "trigger_load": trigger_load, "surge_delta_mw": delta,
                 "baseline_mw": data["baseline"][line], "rating_mw": data["rating"][line],
                 "total_demand_mw": data["total_demand"],
                 "primary_generator": data["generators"][0],
                 "reserve_generator": data["generators"][1],
                 "root_bus": data["buses"][0],
                 "breaker_open": False, "demand_high": False,
                 "demand_normalized_event_id": None, "opened_event_id": None,
                 "repaired": False, "closed": False,
                 "interrupted": {}, "restored": {}, "join_scheduled": False,
                 "primary_recovered": False, "reserve_dispatched": False,
                 "reserve_restored": False, "shortfall_active": False}
        return EpisodeContext(
            episode_id=episode_id, context_id=f"energy_context_{index:03d}",
            domain=self.domain_id, start_time=self.config["start_time"],
            duration_seconds=self.config["duration_seconds"],
            entities=data["entities"], context_relations=data["relations"],
            context_attributes={"topology_profile": data["statistics"],
                "network_scope": "single_radial_feeder_no_alternate_supply",
                "rating_policy": "synthetic_baseline_downstream_demand_times_1.15_to_1.4"},
            episode_attributes={"scenario_family": scenario,
                "empirical_realism_claimed": False,
                "event_parenthood": "explicit_realized_mechanism_only_not_node_connectivity"},
            state=state)

    def _spec(self, key, context, parents=(), now=0.0, load=None, inputs=None):
        mechanism = self.mechanism_registry.get(NS + key)
        state = context.state
        if state["scenario"] == "generation_shortfall":
            roles = {"primary_generator": state["primary_generator"],
                     "reserve_generator": state["reserve_generator"],
                     "bus": state["root_bus"]}
            pairs = {(NS + "generator_connected_to_bus", state["primary_generator"], state["root_bus"]),
                     (NS + "generator_connected_to_bus", state["reserve_generator"], state["root_bus"])}
        else:
            roles = {"line": state["line"], "breaker": state["breaker"]}
            if load:
                roles["load"] = load
            pairs = {(NS + "breaker_protects_line", state["breaker"], state["line"])}
            line_to = next(r.target_entity_id for r in context.context_relations
                           if r.relation_type_id == NS + "line_to_bus" and r.source_entity_id == state["line"])
            line_from = next(r.target_entity_id for r in context.context_relations
                             if r.relation_type_id == NS + "line_from_bus" and r.source_entity_id == state["line"])
            roles["upstream_bus"] = line_from
            roles["downstream_bus"] = line_to
            pairs.update({(NS + "line_to_bus", state["line"], line_to),
                          (NS + "line_from_bus", state["line"], line_from)})
            if load:
                attachment = next(r.target_entity_id for r in context.context_relations
                                  if r.relation_type_id == NS + "load_attached_to_bus" and r.source_entity_id == load)
                roles["attached_bus"] = attachment
                pairs.add((NS + "load_attached_to_bus", load, attachment))
        evidence = [r.relation_id for r in context.context_relations
                    if (r.relation_type_id, r.source_entity_id, r.target_entity_id) in pairs]
        if len(evidence) != len(pairs):
            raise ValueError("Energy event lacks a unique typed topology binding")
        attrs = {"workflow_id": state["workflow_id"], "scenario_family": state["scenario"],
                 "local_observation": "device_or_controller_at_occurrence"}
        if state["scenario"] == "generation_shortfall":
            attrs.update(total_demand_mw=state["total_demand_mw"])
        else:
            attrs.update(line_id=state["line"], breaker_id=state["breaker"],
                         rating_mw=state["rating_mw"],
                         baseline_downstream_mw=state["baseline_mw"])
            if load:
                attrs["load_id"] = load
        return CandidateSpec(
            mechanism_id=mechanism.mechanism_id,
            target_event_type_id=mechanism.target_event_type_id,
            parent_event_ids=list(parents),
            participants=[EventParticipant(entity, role) for role, entity in roles.items()],
            temporal_model_ref=mechanism.generation_temporal_model_ref,
            temporal_inputs=inputs or {}, activation_time=now,
            combination=mechanism.parent_combination,
            phase_priority=10, attributes=attrs,
            context_evidence=ContextEvidence(context_relation_ids=evidence,
                entity_ids=list(dict.fromkeys(roles.values())),
                evaluated_at_offset_seconds=now),
            provenance={"input_anchor": "observed_at",
                        "parameter_status": mechanism.parameter_status})

    def seed_candidates(self, context, rng):
        del rng
        scenario = context.state["scenario"]
        if scenario == "line_fault":
            return [self._spec("fault_onset", context)]
        if scenario == "demand_surge":
            return [self._spec("surge_onset", context, load=context.state["trigger_load"])]
        return [self._spec("shortfall_onset", context)]

    def revalidate_candidate(self, candidate, context, event_by_id, now):
        del event_by_id
        key = candidate.mechanism_id.removeprefix(NS)
        state = context.state
        conditions = {
            "overload_observed": (state["demand_high"] and not state["breaker_open"], "surge_ended_or_breaker_open"),
            "overload_trip": (state["demand_high"] and not state["breaker_open"], "overload_no_longer_active"),
            "surge_settled": (not state["breaker_open"], "breaker_opened_before_surge_settlement"),
            "surge_reclose": (state["breaker_open"] and not state["demand_high"], "unsafe_to_reclose"),
            "reserve_low_observed": (state["shortfall_active"] and not state["reserve_restored"], "reserve_already_restored"),
            "reserve_dispatch": (state["shortfall_active"] and not state["reserve_restored"], "reserve_no_longer_low"),
            "reserve_restoration": (not state["reserve_restored"], "reserve_already_restored"),
        }
        if key in conditions:
            valid, reason = conditions[key]
            evidence = copy.deepcopy(candidate.context_evidence)
            evidence.evaluated_at_offset_seconds = now
            evidence.state_predicates.append(StatePredicateEvidence(
                state["line"] if state["scenario"] != "generation_shortfall" else state["primary_generator"],
                "episode_state." + key, "equals", True, valid, valid))
            return CandidateUpdate(valid, None if valid else reason, context_evidence=evidence)
        return CandidateUpdate(True)

    def materialize_event(self, event_id, candidate, context, rng):
        del rng
        state = context.state
        attrs = copy.deepcopy(candidate.attributes)
        kind = candidate.target_event_type_id
        if kind == SURGE:
            attrs.update(demand_delta_mw=state["surge_delta_mw"],
                         resulting_downstream_mw=round(state["baseline_mw"] + state["surge_delta_mw"], 6))
        elif kind == HIGH:
            attrs.update(measured_downstream_mw=round(state["baseline_mw"] + state["surge_delta_mw"], 6),
                         overload_ratio=round((state["baseline_mw"] + state["surge_delta_mw"]) / state["rating_mw"], 6))
        elif kind == OPEN:
            attrs.update(open_reason="line_fault" if state["scenario"] == "line_fault" else "overload",
                         downstream_load_count=len(state["affected_loads"]))
        elif kind == CLOSED:
            attrs.update(reclose_condition="repair_completed" if state["scenario"] == "line_fault"
                         else "demand_normalized")
        elif kind == SHORTFALL:
            attrs.update(primary_available_mw=round(.7 * state["total_demand_mw"], 6),
                         reserve_available_mw=0.0)
        elif kind == LOW:
            attrs.update(available_mw=round(.7 * state["total_demand_mw"], 6),
                         reserve_margin_mw=round(-.3 * state["total_demand_mw"], 6))
        elif kind in (RECOVERED, DISPATCHED, RESERVE_RESTORED):
            primary = (1.25 if state["primary_recovered"] or kind == RECOVERED else .7) * state["total_demand_mw"]
            reserve = (.45 if state["reserve_dispatched"] or kind == DISPATCHED else 0) * state["total_demand_mw"]
            attrs.update(aggregate_available_mw=round(primary + reserve, 6))
        elif kind == SETTLED:
            attrs["outcome"] = ("supply_restored" if state["scenario"] == "line_fault" else
                                "surge_cleared" if state["scenario"] == "demand_surge" else
                                "reserve_restored")
            if state["scenario"] != "generation_shortfall":
                attrs["restored_load_count"] = len(state["restored"])
        return EventRecord(event_id, context.episode_id, self.domain_id, kind,
                           TemporalExtent(candidate.scheduled_time), list(candidate.participants), attrs,
                           event_role="root" if not candidate.parent_event_ids else "derived",
                           provenance={"generator": "typed_radial_feeder_event_logic",
                                       "mechanism_id": candidate.mechanism_id})

    def observation_plan(self, event, candidate, context):
        del event, candidate, context
        return ObservationPlan(recording_model_ref=NS + "log_ingestion_delay.prior_v1",
                               source="synthetic_device_or_controller_local")

    def apply_event(self, event, candidate, context):
        del candidate
        state = context.state
        kind = event.event_type_id
        if kind == SURGE:
            state["demand_high"] = True
        elif kind == NORMAL:
            state["demand_high"] = False
            state["demand_normalized_event_id"] = event.event_id
        elif kind == OPEN:
            state["breaker_open"] = True
            state["opened_event_id"] = event.event_id
        elif kind == INTERRUPTED:
            state["interrupted"][event.attributes["load_id"]] = event.event_id
        elif kind == REPAIRED:
            state["repaired"] = True
        elif kind == CLOSED:
            state["breaker_open"] = False
            state["closed"] = True
        elif kind == RESTORED:
            state["restored"][event.attributes["load_id"]] = event.event_id
        elif kind == SHORTFALL:
            state["shortfall_active"] = True
        elif kind == RECOVERED:
            state["primary_recovered"] = True
            state["shortfall_active"] = False
        elif kind == DISPATCHED:
            state["reserve_dispatched"] = True
            state["shortfall_active"] = False
        elif kind == RESERVE_RESTORED:
            state["reserve_restored"] = True

    def spawn_candidates(self, event, candidate, context, event_by_id, rng):
        del candidate, event_by_id, rng
        state, kind = context.state, event.event_type_id
        now = event.temporal.anchor("observed_at")
        parent = [event.event_id]
        result = []
        if kind == FAULT:
            result.append(self._spec("protection_trip", context, parent, now))
        elif kind == SURGE:
            result.extend((self._spec("overload_observed", context, parent, now, state["trigger_load"]),
                           self._spec("demand_normalization", context, parent, now, state["trigger_load"])))
        elif kind == HIGH:
            ratio = (state["baseline_mw"] + state["surge_delta_mw"]) / state["rating_mw"]
            result.append(self._spec("overload_trip", context, parent, now,
                                     inputs={"overload_ratio": ratio}))
        elif kind == OPEN:
            result.extend(self._spec("supply_interruption", context, parent, now, load)
                          for load in state["affected_loads"])
            if state["scenario"] == "line_fault":
                result.append(self._spec("repair_completion", context, parent, now))
            elif state["demand_normalized_event_id"]:
                result.append(self._spec("surge_reclose", context,
                    [event.event_id, state["demand_normalized_event_id"]], now))
        elif kind == REPAIRED:
            result.append(self._spec("fault_reclose", context, parent, now))
        elif kind == NORMAL:
            if state["opened_event_id"]:
                result.append(self._spec("surge_reclose", context,
                    [state["opened_event_id"], event.event_id], now))
            else:
                result.append(self._spec("surge_settled", context, parent, now))
        elif kind == CLOSED:
            result.extend(self._spec("supply_restoration", context, parent, now, load)
                          for load in state["affected_loads"])
        elif kind == RESTORED and len(state["restored"]) == len(state["affected_loads"]) and not state["join_scheduled"]:
            state["join_scheduled"] = True
            result.append(self._spec("outage_join", context,
                list(state["restored"].values()), now))
        elif kind == SHORTFALL:
            result.extend((self._spec("reserve_low_observed", context, parent, now),
                           self._spec("primary_recovery", context, parent, now)))
        elif kind == LOW:
            result.append(self._spec("reserve_dispatch", context, parent, now))
        elif kind in (RECOVERED, DISPATCHED) and not state["reserve_restored"]:
            result.append(self._spec("reserve_restoration", context, parent, now))
        elif kind == RESERVE_RESTORED:
            result.append(self._spec("shortfall_settled", context, parent, now))
        return result

    def relation_specs(self, event, candidate, context, event_by_id):
        del event, context, event_by_id
        mechanism = self.mechanism_registry.get(candidate.mechanism_id)
        return [RelationSpec(source_event_id=parent,
            relation_class=mechanism.relation_semantics,
            relation_type_id=mechanism.mechanism_id,
            rule_id=mechanism.mechanism_id,
            source_anchor="observed_at", target_anchor="occurrence_start",
            mechanism_group_id=candidate.candidate_id if len(candidate.parent_event_ids) > 1 else None,
            context_evidence=copy.deepcopy(candidate.context_evidence),
            provenance={"parent_basis": "explicit_realized_feeder_mechanism",
                        "static_connectivity_alone_sufficient": False})
            for parent in candidate.parent_event_ids]

    def validate_result(self, result):
        errors = []
        entities = {entity.entity_id: entity for entity in result.entities}
        relations = result.context_relations
        bindings = {(r.relation_type_id, r.source_entity_id): r.target_entity_id for r in relations}
        if len(bindings) != len(relations):
            errors.append("Ambiguous functional topology binding")
        for layer in self.domain_spec.compile_relation_layers():
            for entity in result.entities:
                if entity.entity_type_id in layer.source_type_ids and (layer.relation_type_id, entity.entity_id) not in bindings:
                    errors.append(f"Missing typed topology binding for {entity.entity_id}")
            for rel in relations:
                if rel.relation_type_id != layer.relation_type_id:
                    continue
                source, target = entities.get(rel.source_entity_id), entities.get(rel.target_entity_id)
                if not source or not target or source.entity_type_id not in layer.source_type_ids or target.entity_type_id not in layer.target_type_ids:
                    errors.append(f"Invalid typed topology edge {rel.relation_id}")
        line_from = {source: target for (kind, source), target in bindings.items() if kind == NS + "line_from_bus"}
        line_to = {source: target for (kind, source), target in bindings.items() if kind == NS + "line_to_bus"}
        load_bus = {source: target for (kind, source), target in bindings.items() if kind == NS + "load_attached_to_bus"}
        incoming = {bus: line for line, bus in line_to.items()}
        if len(incoming) != len(line_to) or len(line_to) != sum(e.entity_type_id == NS + "bus" for e in result.entities) - 1:
            errors.append("Feeder is not a one-root radial tree")
        for line, bus in line_to.items():
            seen_buses = {bus}
            node = bus
            while node in incoming:
                node = line_from.get(incoming[node])
                if node is None or node in seen_buses:
                    errors.append(f"Feeder cycle or missing upstream bus at {line}")
                    break
                seen_buses.add(node)
        candidates = {c.fired_event_id: c for c in result.candidates if c.fired_event_id}
        seen = {}
        opened = False
        repaired = False
        normalized = False
        closed = False
        fault_seen = False
        high_seen = False
        reserve_restored_count = 0
        shortfall_seen = False
        low_seen = False
        primary_recovered = False
        reserve_dispatched = False
        interrupted = set()
        restored = set()
        settled_count = 0
        for event in result.events:
            attrs, kind = event.attributes, event.event_type_id
            candidate = candidates.get(event.event_id)
            if candidate is None:
                errors.append(f"Missing fired candidate for {event.event_id}")
                continue
            mechanism = self.mechanism_registry.get(candidate.mechanism_id)
            if mechanism.target_event_type_id != kind:
                errors.append(f"Mechanism target mismatch at {event.event_id}")
            for parent_id in candidate.parent_event_ids:
                parent = seen.get(parent_id)
                if parent is None or parent.event_type_id not in mechanism.source_event_type_ids:
                    errors.append(f"Invalid realized event parent at {event.event_id}")
                elif parent.temporal.anchor("observed_at") > candidate.activation_time + 1e-9:
                    errors.append(f"Unavailable parent at {event.event_id}")
            roles = {p.role: p.entity_id for p in event.participants}
            if len(roles) != len(event.participants):
                errors.append(f"Duplicate participant role at {event.event_id}")
            for role, entity_id in roles.items():
                entity = entities.get(entity_id)
                expected = {"line": "line", "breaker": "breaker", "load": "load",
                            "bus": "bus", "upstream_bus": "bus", "downstream_bus": "bus",
                            "attached_bus": "bus", "primary_generator": "generator",
                            "reserve_generator": "generator"}.get(role)
                if not entity or expected is None or entity.entity_type_id != NS + expected:
                    errors.append(f"Incorrect node meaning for {role} at {event.event_id}")
            if event.temporal.anchor("observed_at") != event.temporal.anchor("occurrence_start"):
                errors.append(f"Device-local observation time mismatch at {event.event_id}")
            if attrs.get("scenario_family") != result.episode_attributes.get("scenario_family"):
                errors.append(f"Scenario mismatch at {event.event_id}")
            if kind in (FAULT, SURGE, OPEN, INTERRUPTED, REPAIRED, CLOSED, RESTORED, HIGH, NORMAL):
                line = roles.get("line")
                breaker = roles.get("breaker")
                if bindings.get((NS + "breaker_protects_line", breaker)) != line:
                    errors.append(f"Breaker protects wrong line at {event.event_id}")
                if attrs.get("line_id") != line or attrs.get("breaker_id") != breaker:
                    errors.append(f"Event line/breaker identity mismatch at {event.event_id}")
                load = roles.get("load")
                if load and attrs.get("load_id") != load:
                    errors.append(f"Event load identity mismatch at {event.event_id}")
                if load and not _load_is_downstream(line, load, load_bus, incoming, line_from, line_to):
                    errors.append(f"Load not downstream of selected line at {event.event_id}")
            if kind in (SHORTFALL, LOW, RECOVERED, DISPATCHED, RESERVE_RESTORED):
                if any(bindings.get((NS + "generator_connected_to_bus", roles.get(g))) != roles.get("bus")
                       for g in ("primary_generator", "reserve_generator")):
                    errors.append(f"Generator/bus binding mismatch at {event.event_id}")
                for role, expected_role in (("primary_generator", "primary"),
                                            ("reserve_generator", "reserve")):
                    generator = entities.get(roles.get(role))
                    if generator and generator.attributes.get("role") != expected_role:
                        errors.append(f"Generator role mismatch at {event.event_id}")
            if kind == HIGH and not (attrs.get("measured_downstream_mw", 0) > attrs.get("rating_mw", math.inf)):
                errors.append(f"High loading lacks overload at {event.event_id}")
            if kind == FAULT:
                fault_seen = True
            elif kind == HIGH:
                high_seen = True
            elif kind == OPEN:
                if opened or closed:
                    errors.append(f"Duplicate breaker opening at {event.event_id}")
                if attrs["scenario_family"] == "line_fault" and not fault_seen:
                    errors.append(f"Fault protection opened without fault at {event.event_id}")
                if attrs["scenario_family"] == "demand_surge" and (not high_seen or normalized):
                    errors.append(f"Overload protection opened without active overload at {event.event_id}")
                opened = True
            elif kind == INTERRUPTED:
                if not opened or attrs.get("load_id") in interrupted:
                    errors.append(f"Invalid supply interruption at {event.event_id}")
                interrupted.add(attrs.get("load_id"))
            elif kind == REPAIRED:
                repaired = True
            elif kind == NORMAL:
                normalized = True
            elif kind == SHORTFALL:
                shortfall_seen = True
            elif kind == LOW:
                low_seen = True
                if not shortfall_seen or primary_recovered or reserve_dispatched:
                    errors.append(f"Reserve-low observation has no active shortfall at {event.event_id}")
            elif kind == DISPATCHED:
                if not low_seen or primary_recovered or reserve_dispatched:
                    errors.append(f"Reserve dispatched without active low-reserve state at {event.event_id}")
                reserve_dispatched = True
            elif kind == RECOVERED:
                if not shortfall_seen or primary_recovered:
                    errors.append(f"Primary recovered without preceding shortfall at {event.event_id}")
                primary_recovered = True
            elif kind == CLOSED:
                if not opened or (attrs["scenario_family"] == "line_fault" and not repaired) or (attrs["scenario_family"] == "demand_surge" and not normalized):
                    errors.append(f"Unsafe breaker reclose at {event.event_id}")
                line = roles.get("line")
                expected_loads = {load for load in load_bus
                                  if _load_is_downstream(line, load, load_bus, incoming, line_from, line_to)}
                if interrupted != expected_loads:
                    errors.append(f"Interruption set does not match radial downstream scope at {event.event_id}")
                closed = True
            elif kind == RESTORED:
                if not closed or attrs.get("load_id") not in interrupted or attrs.get("load_id") in restored:
                    errors.append(f"Invalid supply restoration at {event.event_id}")
                restored.add(attrs.get("load_id"))
            elif kind == SETTLED:
                settled_count += 1
                if settled_count > 1:
                    errors.append("Duplicate workflow settlement")
                if attrs["scenario_family"] == "line_fault" and (not closed or restored != interrupted):
                    errors.append("Fault workflow settled before downstream restoration")
                if attrs["scenario_family"] == "demand_surge" and (not normalized or (opened and (not closed or restored != interrupted))):
                    errors.append("Surge workflow settled before safe resolution")
                if attrs["scenario_family"] == "generation_shortfall" and reserve_restored_count != 1:
                    errors.append("Shortfall workflow settled before reserve restoration")
            elif kind == RESERVE_RESTORED:
                reserve_restored_count += 1
                demand = attrs.get("total_demand_mw", math.inf)
                expected_capacity = (.7 + .55 * primary_recovered + .45 * reserve_dispatched) * demand
                if (reserve_restored_count > 1 or not (primary_recovered or reserve_dispatched)
                    or attrs.get("aggregate_available_mw", 0) < demand
                    or abs(attrs.get("aggregate_available_mw", 0) - expected_capacity) > 1e-4):
                    errors.append(f"Invalid reserve restoration at {event.event_id}")
            seen[event.event_id] = event
        return errors

    def render_episode_text(self, result):
        sentences, claims = [], []
        for event in result.events:
            participants = ", ".join(f"{p.role}={p.entity_id}" for p in event.participants)
            attrs = json.dumps(event.attributes, ensure_ascii=False, sort_keys=True)
            sentences.append(f"At +{event.temporal.anchor('occurrence_start'):.6f} seconds, event {event.event_id}: "
                             f"{event.event_type_id} ({participants}); grounded fields {attrs}.")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                           "event_ids": [event.event_id], "relation_ids": [],
                           "evidence_fields": ["event_type_id", "temporal", "participants", "attributes"]})
        for relation in result.event_relations:
            key = relation.relation_type_id.removeprefix(NS)
            predicate = PREDICATES[key]
            sentences.append(f"Under generated rule {relation.rule_id}, event {relation.source_event_id} {predicate} "
                             f"event {relation.target_event_id}; occurrence minus source observed_at is "
                             f"{relation.temporal_link.lag_seconds:.6f} seconds.")
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


def _load_is_downstream(line, load, load_bus, incoming, line_from, line_to):
    node = load_bus.get(load)
    target = line_to.get(line)
    while node is not None:
        if node == target:
            return True
        parent_line = incoming.get(node)
        node = line_from.get(parent_line)
    return False


def build_energy_temporal_models():
    registry = build_standard_temporal_registry()
    for model in (
        ExponentialArrivalModel(NS + "root_arrival.prior_v1", default_rate_per_second=.12),
        ConditionalLogNormalModel(NS + "protection_delay.prior_v1", math.log(.08), .35, {}, minimum_seconds=.005),
        ConditionalLogNormalModel(NS + "repair_delay.prior_v1", math.log(45.0), .45, {}, minimum_seconds=5.0),
        ConditionalLogNormalModel(NS + "measurement_delay.prior_v1", math.log(.25), .4, {}, minimum_seconds=.01),
        ConditionalLogNormalModel(NS + "surge_duration.prior_v1", math.log(3.0), .5, {}, minimum_seconds=.2),
        ConditionalLogNormalModel(NS + "primary_recovery.prior_v1", math.log(3.0), .7, {}, minimum_seconds=.5),
        ConditionalLogNormalModel(NS + "dispatch_delay.prior_v1", math.log(2.0), .7, {}, minimum_seconds=.1),
        PiecewiseExponentialHazardModel(NS + "overload_trip_hazard.prior_v1", .25, {"overload_ratio": .8}),
        DeterministicDelayModel(NS + "reclose_delay", 1.0),
        ConditionalLogNormalModel(NS + "log_ingestion_delay.prior_v1", math.log(.03), .35, {}, minimum_seconds=.001),
    ):
        registry.register(model)
    return registry
