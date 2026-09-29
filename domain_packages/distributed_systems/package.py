"""Typed RPC workflows on a sparse deployment graph, not a real system emulator.

All numerical distributions are synthetic priors. Local availability is explicit:
a receiver acts on a received message, not on the sender's unreceived knowledge.
Client timeout never implies server cancellation or absence of side effects.
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
    ExponentialBackoffModel, PiecewiseExponentialHazardModel,
    build_standard_temporal_registry,
)
from data_generation.generation_core.topology import SparseHeterogeneousTopologyGenerator

NS = "distributed_systems."
SENT = NS + "rpc.request.sent"
RECEIVED = NS + "rpc.request.received"
PROCESSED = NS + "rpc.processing.completed"
RESPONSE = NS + "rpc.response.received"
TIMEOUT = NS + "rpc.request.timed_out"
EXHAUSTED = NS + "rpc.retry.exhausted"
SETTLED = NS + "workflow.settled"
DEGRADED = NS + "service.degraded"
RESTORED = NS + "service.restored"
EVENT_TYPES = (SENT, RECEIVED, PROCESSED, RESPONSE, TIMEOUT, EXHAUSTED,
               SETTLED, DEGRADED, RESTORED)
SCENARIOS = ("normal_rpc", "slow_dependency", "retry_pressure")
PREDICATES = {
    "request_delivery": "was transmitted and received as",
    "server_processing": "entered the server processing that completed as",
    "response_delivery": "produced the response received as",
    "client_timeout": "started the client deadline that expired as",
    "retry_backoff": "scheduled a bounded backoff retry sent as",
    "retry_exhaustion": "exhausted the configured retry budget as",
    "workflow_join": "contributed one terminal operation outcome to",
    "service_recovery": "transitioned to restored service state at",
}


class DistributedSystemsPackage:
    domain_id = "distributed_systems"

    def __init__(self, config=None):
        self.config = {
            "base_seed": 20260929, "context_count": 4, "nodes_per_context": 1000,
            "duration_seconds": 15.0, "start_time": "2026-09-29T08:00:00+08:00",
            "fanout": 3, "timeout_seconds": 0.5, "max_retries": 2,
            "scenario_weights": dict(zip(SCENARIOS, (0.4, 0.35, 0.25))),
            **(config or {}),
        }
        for key in ("base_seed", "context_count", "nodes_per_context", "fanout", "max_retries"):
            value = self.config[key]
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{key} must be an integer")
            self.config[key] = int(value)
        if self.config["nodes_per_context"] < 20:
            raise ValueError("Distributed-system contexts require at least 20 entities")
        if not 2 <= self.config["fanout"] <= min(12, int(self.config["nodes_per_context"] * .3) - 1):
            raise ValueError("fanout must be 2..12 and leave a distinct caller service")
        if self.config["context_count"] < 1 or not 0 <= self.config["max_retries"] <= 10:
            raise ValueError("context_count must be positive; max_retries must be 0..10")
        for key in ("duration_seconds", "timeout_seconds"):
            self.config[key] = float(self.config[key])
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
        weights = self.config["scenario_weights"]
        if not weights or set(weights) - set(SCENARIOS):
            raise ValueError("Unknown or empty distributed-system scenario weights")
        if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("Scenario weights must be finite, nonnegative and not all zero")
        self.domain_spec = load_domain_spec(Path(__file__).with_name("domain_spec.json"))
        self.mechanism_registry = load_mechanism_registry(
            Path(__file__).with_name("mechanism_specs.json"),
            known_event_type_ids=EVENT_TYPES,
            known_context_relation_type_ids=self.domain_spec.context_relation_type_ids,
        )
        self._context_cache = {}

    def catalog(self):
        role_types = {"caller": NS + "service", "callee": NS + "service",
                      "service": NS + "service", "endpoint": NS + "endpoint",
                      "execution_container": NS + "container", "execution_host": NS + "host"}
        return {
            "domain": self.domain_id, "vocabulary_policy": "open_namespaced_hierarchy",
            "fit_status": "transparent_synthetic_prior_not_empirically_fitted",
            "scope": {
                "workflow": "parallel_read_only_rpc_with_bounded_retries",
                "clock": "synchronized_synthetic_seconds_no_clock_skew",
                "observation": "actor_local_availability; recording_is_collector_ingestion",
                "timeout": "client_abandons_attempt_not_server_cancellation",
                "exactly_once_execution_claimed": False,
                "system_performance_realism_claimed": False,
            },
            "topology_contract": copy.deepcopy(self.domain_spec.topology_contract),
            "text_projection": {"mode": "deterministic_grounded_projection",
                                "free_form_causal_completion": False},
            "entity_types": [{"entity_type_id": NS + k, "parent_type_id": None,
                              "required_attributes": attrs} for k, attrs in (
                ("service", ["service_kind", "timeout_seconds"]),
                ("host", ["host_kind", "capacity_units"]),
                ("container", ["runtime", "instance_index"]),
                ("endpoint", ["method", "idempotent", "operation_kind"]))],
            "event_types": [{"event_type_id": t, "parent_type_id": t.rsplit(".", 1)[0],
                             "required_attributes": ["workflow_id", "availability_scope"],
                             "participant_role_types": role_types} for t in EVENT_TYPES],
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
        config = self.config
        rng = random.Random(config["base_seed"] + index * 7919)
        context_id = f"distributed_context_{index:03d}"
        n = config["nodes_per_context"]
        count = int(n * .3)
        host_count = max(2, int(n * .1))
        groups = {}
        entities = []
        for kind, total in (("service", count), ("container", count),
                            ("host", host_count), ("endpoint", n - 2 * count - host_count)):
            groups[kind] = []
            for i in range(total):
                eid = f"{context_id}_{kind}_{i:05d}"
                attrs = {"service": {"service_kind": "read_only_rpc", "timeout_seconds": config["timeout_seconds"]},
                         "container": {"runtime": "synthetic_container", "instance_index": i},
                         "host": {"host_kind": "compute", "capacity_units": rng.choice((2, 4, 8))},
                         "endpoint": {"method": "GET", "idempotent": True, "operation_kind": "read"}}[kind]
                entities.append(Entity(eid, self.domain_id, NS + kind, attrs, context_id))
                groups[kind].append(eid)
        services, containers, hosts, endpoints = (groups[k] for k in ("service", "container", "host", "endpoint"))
        shuffled_containers = rng.sample(containers, len(containers))
        deployment = {s: {"container": c, "host": rng.choice(hosts)}
                      for s, c in zip(services, shuffled_containers)}
        endpoint_owner = {e: services[i % count] for i, e in enumerate(endpoints)}
        # Reserve ONE feasible workload anchor, not a fixed degree for every
        # service. Other calls use the sparse stochastic topology generator.
        anchor_index = rng.randrange(count)
        pairs = [(services[anchor_index], services[j if j < anchor_index else j + 1])
                 for j in rng.sample(range(count - 1), config["fanout"])]
        required = {
            NS + "service_calls_service": tuple(pairs),
            NS + "service_runs_in_container": tuple((s, d["container"]) for s, d in deployment.items()),
            NS + "service_deployed_on_host": tuple((s, d["host"]) for s, d in deployment.items()),
            NS + "container_runs_on_host": tuple((d["container"], d["host"]) for d in deployment.values()),
            NS + "endpoint_exposed_by_service": tuple(endpoint_owner.items()),
        }
        layers = [replace(layer, required_pairs=required[layer.relation_type_id])
                  for layer in self.domain_spec.compile_relation_layers()]
        topology = SparseHeterogeneousTopologyGenerator().generate(
            context_id=context_id, domain=self.domain_id, entities=entities, layers=layers,
            seed=config["base_seed"] + index * 7919, observation_window_seconds=config["duration_seconds"])
        return entities, topology.relations, deployment, endpoint_owner, topology.statistics

    def create_episode_context(self, episode_index, seed):
        index = episode_index % self.config["context_count"]
        if index not in self._context_cache:
            self._context_cache[index] = self._build_context(index)
        entities, relations, deployment, owners, statistics = self._context_cache[index]
        rng = random.Random(seed ^ 0x525043)
        scenario = rng.choices(list(self.config["scenario_weights"]),
                               weights=list(self.config["scenario_weights"].values()))[0]
        adjacency = {}
        for rel in relations:
            if rel.relation_type_id == NS + "service_calls_service":
                adjacency.setdefault(rel.source_entity_id, []).append(rel.target_entity_id)
        caller = rng.choice([s for s, neighbors in adjacency.items() if len(neighbors) >= self.config["fanout"]])
        targets = rng.sample(adjacency[caller], self.config["fanout"])
        episode_id = f"distributed_episode_{episode_index:07d}"
        operations = {}
        for i, callee in enumerate(targets):
            op_id = f"{episode_id}_op_{i}"
            endpoint = rng.choice([e for e, owner in owners.items() if owner == callee])
            operations[op_id] = {"caller": caller, "callee": callee, "endpoint": endpoint,
                                **deployment[callee], "terminal_event_id": None, "outcome": None}
        affected = [] if scenario == "normal_rpc" else targets[:1] if scenario == "slow_dependency" else targets
        return EpisodeContext(
            episode_id=episode_id, context_id=f"distributed_context_{index:03d}", domain=self.domain_id,
            start_time=self.config["start_time"], duration_seconds=self.config["duration_seconds"],
            entities=entities, context_relations=relations,
            context_attributes={"topology_profile": statistics,
                                "deployment_policy": "one_container_per_service; typed_static_bindings"},
            episode_attributes={"scenario_family": scenario, "empirical_realism_claimed": False,
                "availability_semantics": "actor_local; cross_actor_transfer_requires_receive_event",
                "recording_semantics": "collector_ingestion_not_actor_availability",
                "tie_policy": "response_before_timeout_if_exact_same_timestamp"},
            state={"operations": operations, "attempts": {}, "affected": affected,
                   "degraded": {}, "inflight": {}, "host_inflight": {}, "join_scheduled": False,
                   "host_capacity": {e.entity_id: e.attributes["capacity_units"] for e in entities if e.entity_type_id == NS + "host"},
                   "workflow_id": episode_id + "_workflow"})

    def _spec(self, key, context, parents=(), operation_id=None, attempt=0, service=None,
              now=0.0, inputs=None):
        mechanism = self.mechanism_registry.get(NS + key)
        attrs = {"workflow_id": context.state["workflow_id"], "availability_scope": "actor_local"}
        if operation_id is not None:
            op = context.state["operations"][operation_id]
            attrs.update(operation_id=operation_id, idempotency_key=operation_id,
                         attempt=attempt, attempt_id=f"{operation_id}_attempt_{attempt}")
            roles = {"caller": op["caller"], "callee": op["callee"], "endpoint": op["endpoint"],
                     "execution_container": op["container"], "execution_host": op["host"]}
            pairs = {(NS + "service_calls_service", op["caller"], op["callee"]),
                     (NS + "service_runs_in_container", op["callee"], op["container"]),
                     (NS + "container_runs_on_host", op["container"], op["host"]),
                     (NS + "service_deployed_on_host", op["callee"], op["host"]),
                     (NS + "endpoint_exposed_by_service", op["endpoint"], op["callee"])}
            evidence_ids = [r.relation_id for r in context.context_relations
                            if (r.relation_type_id, r.source_entity_id, r.target_entity_id) in pairs]
            if len(evidence_ids) != len(pairs):
                raise ValueError("RPC lacks a unique typed call/deployment/endpoint binding")
            actor = op["callee"] if key in ("request_delivery", "server_processing") else op["caller"]
        else:
            actor = service or next(iter(context.state["operations"].values()))["caller"]
            roles, evidence_ids = {"service": actor}, []
        attrs["local_actor_entity_id"] = actor
        return CandidateSpec(
            mechanism_id=mechanism.mechanism_id, target_event_type_id=mechanism.target_event_type_id,
            parent_event_ids=list(parents), participants=[EventParticipant(v, k) for k, v in roles.items()],
            temporal_model_ref=mechanism.generation_temporal_model_ref, temporal_inputs=inputs or {},
            activation_time=now, combination=mechanism.parent_combination,
            phase_priority=30 if key == "client_timeout" else 10,
            attributes=attrs, context_evidence=ContextEvidence(
                context_relation_ids=evidence_ids, entity_ids=list(dict.fromkeys(roles.values())),
                evaluated_at_offset_seconds=now),
            provenance={"input_anchor": "observed_at", "parameter_status": mechanism.parameter_status})

    def seed_candidates(self, context, rng):
        return [self._spec("degradation_onset", context, service=s) for s in context.state["affected"]] + [
            self._spec("request_arrival", context, operation_id=op) for op in context.state["operations"]]

    def _processing_inputs(self, context, op_id):
        operation = context.state["operations"][op_id]
        service, host = operation["callee"], operation["host"]
        return {"inflight": context.state["inflight"].get(service, 0),
                "host_inflight": context.state["host_inflight"].get(host, 0),
                "log_capacity": math.log(context.state["host_capacity"][host]),
                "degraded": float(context.state["degraded"].get(service, False))}

    def revalidate_candidate(self, candidate, context, event_by_id, now):
        key = candidate.mechanism_id.removeprefix(NS)
        a = candidate.attributes
        if "operation_id" in a:
            op = context.state["operations"][a["operation_id"]]
            attempt = context.state["attempts"].get(a["attempt_id"], {})
            evidence = copy.deepcopy(candidate.context_evidence)
            evidence.evaluated_at_offset_seconds = now
            evidence.state_predicates = []
            if key == "client_timeout":
                accepted = bool(attempt.get("response_accepted"))
                evidence.state_predicates.append(StatePredicateEvidence(op["caller"],
                    f"attempts.{a['attempt_id']}.response_accepted", "equals", False, accepted, not accepted))
                return CandidateUpdate(not accepted, "matching_attempt_response_arrived_before_deadline" if accepted else None,
                                       context_evidence=evidence)
            if key == "retry_backoff":
                unresolved = op["terminal_event_id"] is None
                evidence.state_predicates.append(StatePredicateEvidence(op["caller"],
                    f"operations.{a['operation_id']}.terminal_event_id", "equals", None, op["terminal_event_id"], unresolved))
                return CandidateUpdate(unresolved, "operation_already_terminal" if not unresolved else None,
                                       context_evidence=evidence)
            if key == "server_processing":
                return CandidateUpdate(True, temporal_inputs=self._processing_inputs(context, a["operation_id"]),
                                       context_evidence=evidence)
        return CandidateUpdate(True)

    def materialize_event(self, event_id, candidate, context, rng):
        attrs = copy.deepcopy(candidate.attributes)
        t = candidate.scheduled_time
        if candidate.target_event_type_id == SENT:
            attrs.update(timeout_seconds=self.config["timeout_seconds"],
                         deadline_offset_seconds=t + self.config["timeout_seconds"],
                         request_method="GET", idempotent=True)
        elif candidate.target_event_type_id == RESPONSE:
            attempt = context.state["attempts"][attrs["attempt_id"]]
            op = context.state["operations"][attrs["operation_id"]]
            accepted = not attempt["timed_out"] and op["terminal_event_id"] is None and t <= attempt["deadline"]
            attrs.update(response_accepted=accepted, late_response=not accepted)
        elif candidate.target_event_type_id == SETTLED:
            outcomes = {op: x["outcome"] for op, x in context.state["operations"].items()}
            attrs.update(operation_outcomes=outcomes,
                         outcome="success" if all(x == "success" for x in outcomes.values()) else "partial_or_total_failure")
        return EventRecord(event_id, context.episode_id, self.domain_id, candidate.target_event_type_id,
                           TemporalExtent(t), list(candidate.participants), attrs,
                           event_role="root" if not candidate.parent_event_ids else "derived",
                           provenance={"generator": "typed_rpc_workflow", "mechanism_id": candidate.mechanism_id})

    def observation_plan(self, event, candidate, context):
        # Actor knows its own local action immediately; collector sees it later.
        return ObservationPlan(recording_model_ref=NS + "log_ingestion_delay", source="actor_local_instrumentation")

    def apply_event(self, event, candidate, context):
        state, a, kind = context.state, event.attributes, event.event_type_id
        if kind in (DEGRADED, RESTORED):
            state["degraded"][a["local_actor_entity_id"]] = kind == DEGRADED
            return
        if "operation_id" not in a:
            return
        op = state["operations"][a["operation_id"]]
        if kind == SENT:
            state["attempts"][a["attempt_id"]] = {"deadline": a["deadline_offset_seconds"],
                "timed_out": False, "response_accepted": False, "send_event_id": event.event_id}
        elif kind == RECEIVED:
            state["inflight"][op["callee"]] = state["inflight"].get(op["callee"], 0) + 1
            state["host_inflight"][op["host"]] = state["host_inflight"].get(op["host"], 0) + 1
        elif kind == PROCESSED:
            state["inflight"][op["callee"]] -= 1
            state["host_inflight"][op["host"]] -= 1
        elif kind == TIMEOUT:
            state["attempts"][a["attempt_id"]]["timed_out"] = True
        elif kind == RESPONSE and a["response_accepted"]:
            state["attempts"][a["attempt_id"]]["response_accepted"] = True
            op.update(terminal_event_id=event.event_id, outcome="success")
        elif kind == EXHAUSTED:
            op.update(terminal_event_id=event.event_id, outcome="retry_exhausted")

    def spawn_candidates(self, event, candidate, context, event_by_id, rng):
        a, kind = event.attributes, event.event_type_id
        now = event.temporal.anchor("observed_at")
        parents = [event.event_id]
        args = dict(parents=parents, now=now)
        if "operation_id" in a:
            args.update(operation_id=a["operation_id"], attempt=a["attempt"])
        result = []
        if kind == DEGRADED:
            result.append(self._spec("service_recovery", context, service=a["local_actor_entity_id"], **args))
        elif kind == SENT:
            result.extend([self._spec("request_delivery", context, **args),
                           self._spec("client_timeout", context, inputs={"delay_seconds": self.config["timeout_seconds"]}, **args)])
        elif kind == RECEIVED:
            result.append(self._spec("server_processing", context,
                          inputs=self._processing_inputs(context, a["operation_id"]), **args))
        elif kind == PROCESSED:
            result.append(self._spec("response_delivery", context, **args))
        elif kind == TIMEOUT:
            if a["attempt"] < self.config["max_retries"]:
                args["attempt"] += 1
                result.append(self._spec("retry_backoff", context, inputs={"attempt": a["attempt"]}, **args))
            else:
                result.append(self._spec("retry_exhaustion", context, **args))
        terminals = [op["terminal_event_id"] for op in context.state["operations"].values()]
        if all(terminals) and not context.state["join_scheduled"]:
            context.state["join_scheduled"] = True
            result.append(self._spec("workflow_join", context, parents=terminals, now=now))
        return result

    def relation_specs(self, event, candidate, context, event_by_id):
        key = candidate.mechanism_id.removeprefix(NS)
        mechanism = self.mechanism_registry.get(candidate.mechanism_id)
        return [RelationSpec(source_event_id=parent, relation_class=mechanism.relation_semantics,
                    relation_type_id=NS + key, rule_id=candidate.mechanism_id,
                    source_anchor="observed_at", target_anchor="occurrence_start",
                    mechanism_group_id=candidate.candidate_id if len(candidate.parent_event_ids) > 1 else None,
                    context_evidence=copy.deepcopy(candidate.context_evidence),
                    provenance={"parent_basis": "explicit_operation_and_attempt_workflow",
                                "availability_scope": "actor_local"}) for parent in candidate.parent_event_ids]

    def validate_result(self, result):
        """Replay domain invariants from exported evidence, without hidden state."""
        errors = []
        entities = {e.entity_id: e for e in result.entities}
        bindings = {(r.relation_type_id, r.source_entity_id, r.target_entity_id)
                    for r in result.context_relations}
        for layer in self.domain_spec.compile_relation_layers():
            counts = {}
            for rel in result.context_relations:
                if rel.relation_type_id != layer.relation_type_id:
                    continue
                source, target = entities.get(rel.source_entity_id), entities.get(rel.target_entity_id)
                if not source or not target or source.entity_type_id not in layer.source_type_ids or target.entity_type_id not in layer.target_type_ids:
                    errors.append(f"Invalid typed deployment edge {rel.relation_id}")
                counts[rel.source_entity_id] = counts.get(rel.source_entity_id, 0) + 1
            if layer.max_out_degree == 1:
                for entity in entities.values():
                    if entity.entity_type_id in layer.source_type_ids and counts.get(entity.entity_id) != 1:
                        errors.append(f"Missing or ambiguous binding for {entity.entity_id} in {layer.relation_type_id}")
        owner_maps = {kind: {r.source_entity_id: r.target_entity_id for r in result.context_relations
                            if r.relation_type_id == NS + kind} for kind in (
            "service_runs_in_container", "container_runs_on_host", "service_deployed_on_host")}
        for service, container in owner_maps["service_runs_in_container"].items():
            if owner_maps["container_runs_on_host"].get(container) != owner_maps["service_deployed_on_host"].get(service):
                errors.append(f"Service/container disagree on execution host for {service}")
        candidates = {c.fired_event_id: c for c in result.candidates if c.fired_event_id}
        seen, sends, timed_out, terminals, operation_roles = {}, {}, set(), {}, {}
        join_count = 0
        expected_roles = {"caller": "service", "callee": "service", "endpoint": "endpoint",
                          "execution_container": "container", "execution_host": "host"}
        for event in result.events:
            a, kind = event.attributes, event.event_type_id
            t = event.temporal.anchor("occurrence_start")
            candidate = candidates.get(event.event_id)
            if candidate is None:
                errors.append(f"No fired candidate for {event.event_id}")
                continue
            mechanism = self.mechanism_registry.get(candidate.mechanism_id)
            if mechanism.target_event_type_id != kind:
                errors.append(f"Mechanism target mismatch at {event.event_id}")
            for parent_id in candidate.parent_event_ids:
                parent = seen.get(parent_id)
                if parent is None or parent.event_type_id not in mechanism.source_event_type_ids:
                    errors.append(f"Invalid realized parent of {event.event_id}")
                elif parent.temporal.anchor("observed_at") > candidate.activation_time + 1e-9:
                    errors.append(f"Input unavailable at activation of {event.event_id}")
                elif "operation_id" in a and parent.attributes.get("operation_id") != a["operation_id"]:
                    errors.append(f"Cross-operation parent at {event.event_id}")
                elif "attempt" in a:
                    delta = 1 if candidate.mechanism_id == NS + "retry_backoff" else 0
                    if parent.attributes.get("attempt") != a["attempt"] - delta:
                        errors.append(f"Cross-attempt parent at {event.event_id}")
            roles = {p.role: p.entity_id for p in event.participants}
            if len(roles) != len(event.participants):
                errors.append(f"Duplicate participant role at {event.event_id}")
            required_roles = expected_roles if "operation_id" in a else {"service": "service"}
            for role, entity_type in required_roles.items():
                if roles.get(role) not in entities or entities[roles[role]].entity_type_id != NS + entity_type:
                    errors.append(f"Incorrect node meaning for role {role} at {event.event_id}")
            if a.get("availability_scope") != "actor_local" or event.temporal.observed_at_offset_seconds != t:
                errors.append(f"Invalid local availability at {event.event_id}")
            actor_role = "callee" if kind in (RECEIVED, PROCESSED) else "caller" if "operation_id" in a else "service"
            if a.get("local_actor_entity_id") != roles.get(actor_role):
                errors.append(f"Incorrect local observer at {event.event_id}")
            if "operation_id" in a and set(expected_roles) <= roles.keys():
                required = {(NS + "service_calls_service", roles["caller"], roles["callee"]),
                    (NS + "service_runs_in_container", roles["callee"], roles["execution_container"]),
                    (NS + "service_deployed_on_host", roles["callee"], roles["execution_host"]),
                    (NS + "container_runs_on_host", roles["execution_container"], roles["execution_host"]),
                    (NS + "endpoint_exposed_by_service", roles["endpoint"], roles["callee"])}
                if not required <= bindings:
                    errors.append(f"RPC deployment mismatch at {event.event_id}")
                endpoint = entities.get(roles["endpoint"])
                if endpoint is None or endpoint.attributes.get("method") != "GET" or endpoint.attributes.get("idempotent") is not True:
                    errors.append(f"RPC violates read-only endpoint scope at {event.event_id}")
                first_roles = operation_roles.setdefault(a["operation_id"], roles)
                if roles != first_roles:
                    errors.append(f"Operation changed its participants at {event.event_id}")
                if a.get("idempotency_key") != a["operation_id"] or a.get("attempt_id") != f"{a['operation_id']}_attempt_{a.get('attempt')}":
                    errors.append(f"Invalid operation/attempt identity at {event.event_id}")
                attempt_id = a["attempt_id"]
                if kind == SENT:
                    if attempt_id in sends or a["operation_id"] in terminals:
                        errors.append(f"Duplicate or post-terminal send at {event.event_id}")
                    if abs(a.get("deadline_offset_seconds", -1) - t - self.config["timeout_seconds"]) > 1e-9:
                        errors.append(f"Invalid client deadline at {event.event_id}")
                    sends[attempt_id] = event
                elif attempt_id not in sends:
                    errors.append(f"Attempt has no preceding send at {event.event_id}")
                elif kind == TIMEOUT:
                    if abs(t - sends[attempt_id].attributes["deadline_offset_seconds"]) > 1e-9 or a["operation_id"] in terminals:
                        errors.append(f"Invalid timeout at {event.event_id}")
                    timed_out.add(attempt_id)
                elif kind == RESPONSE:
                    expected = attempt_id not in timed_out and a["operation_id"] not in terminals and t <= sends[attempt_id].attributes["deadline_offset_seconds"]
                    if a.get("response_accepted") != expected or a.get("late_response") != (not expected):
                        errors.append(f"Late response incorrectly resolved operation at {event.event_id}")
                    if expected:
                        terminals[a["operation_id"]] = event.event_id
                elif kind == EXHAUSTED:
                    if a["attempt"] != self.config["max_retries"] or attempt_id not in timed_out or a["operation_id"] in terminals:
                        errors.append(f"Invalid retry exhaustion at {event.event_id}")
                    terminals[a["operation_id"]] = event.event_id
            if kind == SETTLED:
                join_count += 1
                if len(terminals) != self.config["fanout"] or set(candidate.parent_event_ids) != set(terminals.values()) or join_count != 1:
                    errors.append(f"Workflow joined before all distinct operations terminated at {event.event_id}")
            seen[event.event_id] = event
        return errors

    def render_episode_text(self, result):
        sentences, claims = [], []
        for event in result.events:
            participants = ", ".join(f"{p.role}={p.entity_id}" for p in event.participants)
            attrs = json.dumps(event.attributes, ensure_ascii=False, sort_keys=True)
            sentences.append(f"At +{event.temporal.anchor('occurrence_start'):.6f} seconds, event {event.event_id}: "
                             f"{event.event_type_id} ({participants}); actor-local facts {attrs}.")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                           "event_ids": [event.event_id], "relation_ids": [],
                           "evidence_fields": ["event_type_id", "temporal", "participants", "attributes"]})
        for relation in result.event_relations:
            key = relation.relation_type_id.removeprefix(NS)
            predicate = PREDICATES[key]
            sentences.append(f"Under generated rule {relation.rule_id}, event {relation.source_event_id} {predicate} "
                             f"event {relation.target_event_id}; target occurrence minus source actor-local observed_at "
                             f"is {relation.temporal_link.lag_seconds:.6f} seconds.")
            claims.append({"claim_id": f"claim_{len(claims)}", "sentence_index": len(sentences)-1,
                "event_ids": [relation.source_event_id, relation.target_event_id], "relation_ids": [relation.relation_id],
                "relation_assertions": [{"relation_id": relation.relation_id,
                    "asserted_relation_class": relation.relation_class, "surface_predicate": predicate}],
                "evidence_fields": ["rule_id", "source_event_id", "target_event_id", "temporal_link"]})
        return {"alignment_id": result.episode_id + "_text_alignment", "episode_id": result.episode_id,
                "domain": self.domain_id, "projection_mode": "deterministic_grounded_projection",
                "sentences": sentences, "text": " ".join(sentences), "claims": claims}


def build_distributed_systems_temporal_models():
    registry = build_standard_temporal_registry()
    models = [
        ExponentialArrivalModel(NS + "root_arrival", default_rate_per_second=8.0),
        ConditionalLogNormalModel(NS + "network_delay", math.log(.025), .45, {}, minimum_seconds=.001),
        PiecewiseExponentialHazardModel(NS + "processing_hazard", 8.0,
            {"inflight": -.35, "host_inflight": -.1, "log_capacity": .3, "degraded": -2.5}),
        DeterministicDelayModel(NS + "timeout", .5),
        ExponentialBackoffModel(NS + "retry_backoff", .15, multiplier=2, maximum_seconds=2, jitter_fraction=.2),
        ConditionalLogNormalModel(NS + "recovery_delay", math.log(1.5), .4, {}, minimum_seconds=.05),
        ConditionalLogNormalModel(NS + "log_ingestion_delay", math.log(.02), .35, {}, minimum_seconds=.001),
    ]
    for model in models:
        registry.register(model)
    return registry
