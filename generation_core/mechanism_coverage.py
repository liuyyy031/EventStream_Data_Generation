"""Descriptive mechanism coverage from emitted records, never a realism score."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import groupby

CATEGORIES = ("physical_process", "observation", "management_action", "derived_assessment", "unknown")


class MechanismCoverageAccumulator:
    def __init__(self, catalog):
        self.catalog = deepcopy(catalog)
        self.profile = self.catalog.get("coverage_profile", {})
        self.categories = self.profile.get("event_categories", {})
        declared_types = {d["event_type_id"] for d in self.catalog.get("event_types", [])}
        unknown = set(self.categories) - declared_types
        if unknown or any(c not in CATEGORIES[:-1] for c in self.categories.values()):
            raise ValueError(f"Invalid coverage category declaration: {sorted(unknown)}")
        self.mechanisms = {m["mechanism_id"]: self._row(m) for m in self.catalog.get("mechanisms", [])}
        self.episodes = {}
        self.inventory = set()
        self.used_entities = set()
        self.context_inventory = defaultdict(set)
        self.context_used = defaultdict(set)
        self.category_counts = Counter()
        self.type_counts = Counter()
        self.relation_classes = Counter()
        self.incoming_causal = defaultdict(set)
        self.events = set()
        self.fired_events = set()
        self.mechanism_fired_events = defaultdict(set)
        self.mechanism_multi_parent_events = defaultdict(set)
        self.temporal_rows = defaultdict(list)
        self.report_rows = defaultdict(list)
        self.stale_report_count = 0

    @staticmethod
    def _row(declaration):
        return {"mechanism_id": declaration["mechanism_id"], "registered": True,
            "target_event_type_id": declaration.get("target_event_type_id"),
            "relation_semantics": declaration.get("relation_semantics"),
            "parent_combination": declaration.get("parent_combination"),
            "required_context_relation_type_ids": list(declaration.get("required_context_relation_type_ids", [])),
            "candidate_count": 0, "checked_candidate_count": 0, "fired_event_count": 0,
            "status_counts": Counter(), "terminal_reason_counts": Counter(),
            "temporal_model_usage_counts": Counter(), "precondition_decision_counts": Counter(),
            "predicate_checks": {}, "legacy_terminal_predicate_snapshots": {},
            "runtime_audit_candidate_count": 0, "legacy_or_unaudited_candidate_count": 0,
            "multi_parent_fired_count": 0}

    def add_episode(self, episode):
        self.episodes[episode["episode_id"]] = {
            "episode_id": episode["episode_id"], "context_id": episode["context_id"],
            "termination_reason": episode.get("termination_reason"),
            "category_counts": Counter(), "event_count": 0, "candidate_status_counts": Counter()}

    def add_entity(self, entity):
        context = entity.get("context_id")
        key = (context, entity["entity_id"])
        self.inventory.add(key)
        self.context_inventory[context].add(key)

    def add_event(self, event):
        episode_id = event["episode_id"]
        row = self.episodes[episode_id]
        event_key = (episode_id, event["event_id"])
        self.events.add(event_key)
        event_type = event["event_type_id"]
        category = self.categories.get(event_type, "unknown")
        self.category_counts[category] += 1
        self.type_counts[event_type] += 1
        row["category_counts"][category] += 1
        row["event_count"] += 1
        for p in event.get("participants", []):
            key = (row["context_id"], p["entity_id"])
            self.used_entities.add(key)
            self.context_used[row["context_id"]].add(key)
        temporal = event["temporal"]
        start = temporal["occurrence_start_offset_seconds"]
        self.temporal_rows[episode_id].append((start, temporal.get("observed_at_offset_seconds"),
            temporal.get("recorded_at_offset_seconds")))
        attrs = event.get("attributes", {})
        self.stale_report_count += int(attrs.get("stale_report") is True)
        sample_field = self.profile.get("report_sample_time_fields", {}).get(event_type)
        if sample_field and attrs.get(sample_field) is not None:
            scope_fields = self.profile.get("report_scope_fields", {}).get(event_type, [])
            scope = tuple((f, attrs[f]) for f in scope_fields) if scope_fields and all(f in attrs for f in scope_fields) else tuple(
                sorted((p["role"], p["entity_id"]) for p in event.get("participants", [])))
            self.report_rows[(episode_id, scope)].append((start, attrs[sample_field]))

    @staticmethod
    def _add_predicate(output, predicate, passed_count, failed_count):
        key = (predicate["state_path"], predicate["operator"])
        row = output.setdefault(key, {"state_path": key[0], "operator": key[1], "passed_count": 0, "failed_count": 0})
        row["passed_count"] += passed_count
        row["failed_count"] += failed_count

    def add_candidate(self, candidate):
        mechanism_id = candidate["mechanism_id"]
        if mechanism_id not in self.mechanisms:
            self.mechanisms[mechanism_id] = self._row({"mechanism_id": mechanism_id})
            self.mechanisms[mechanism_id]["registered"] = False
        row = self.mechanisms[mechanism_id]
        row["candidate_count"] += 1
        status = candidate["status"]
        row["status_counts"][status] += 1
        row["terminal_reason_counts"][candidate.get("terminal_reason") or "unspecified"] += 1
        row["temporal_model_usage_counts"][candidate["temporal_model_ref"]] += 1
        self.episodes[candidate["episode_id"]]["candidate_status_counts"][status] += 1
        if status == "fired":
            fired_key = (candidate["episode_id"], candidate.get("fired_event_id"))
            self.fired_events.add(fired_key)
            self.mechanism_fired_events[mechanism_id].add(fired_key)
            row["fired_event_count"] += 1
            row["multi_parent_fired_count"] += int(len(set(candidate.get("parent_event_ids", []))) > 1)
            if len(set(candidate.get("parent_event_ids", []))) > 1:
                self.mechanism_multi_parent_events[mechanism_id].add(fired_key)
        audit = candidate.get("precondition_audit", {})
        if audit.get("schema") == "candidate-precondition-audit-v1":
            row["runtime_audit_candidate_count"] += 1
            row["checked_candidate_count"] += int(audit.get("evaluation_count", 0) > 0)
            row["precondition_decision_counts"].update({k: audit.get(k, 0) for k in (
                "evaluation_count", "valid_count", "invalid_count", "without_predicate_evidence_count")})
            for predicate in audit.get("predicates", {}).values():
                self._add_predicate(row["predicate_checks"], predicate, predicate["passed_count"], predicate["failed_count"])
        else:
            row["legacy_or_unaudited_candidate_count"] += 1
        # Keep terminal snapshots separately. They are never counted as the
        # entire dynamic checking history or added again to fresh-audit totals.
        for predicate in candidate.get("context_evidence", {}).get("state_predicates", []):
            self._add_predicate(row["legacy_terminal_predicate_snapshots"], predicate,
                int(predicate["passed"]), int(not predicate["passed"]))

    def add_relation(self, relation):
        self.relation_classes[relation["relation_class"]] += 1
        if relation["relation_class"] == "causal":
            self.incoming_causal[(relation["episode_id"], relation["target_event_id"])].add(relation["source_event_id"])

    def add_result(self, result):
        self.add_episode(result.episode_record())
        for entity in result.entities:
            self.add_entity(entity.to_dict())
        for event in result.events:
            self.add_event(event.to_dict())
        for candidate in result.candidates:
            self.add_candidate(candidate.to_dict())
        for relation in result.event_relations:
            self.add_relation(relation.to_dict())

    @staticmethod
    def _inversions(pairs):
        """Count arrivals older than a strictly earlier arrival's latest origin.

        Equal arrival times do not establish ordering. This counts affected
        arrivals, not all inverted event pairs, and never compares episodes.
        """
        latest = float("-inf")
        count = 0
        for _, group in groupby(sorted(pairs), key=lambda p: p[0]):
            origins = [p[1] for p in group]
            count += sum(origin < latest for origin in origins)
            latest = max([latest] + origins)
        return count

    def report(self):
        rows = []
        for key, source in sorted(self.mechanisms.items()):
            row = deepcopy(source)
            row["declared_fired_candidate_count"] = row["fired_event_count"]
            row["fired_event_count"] = len(self.mechanism_fired_events[key] & self.events)
            row["multi_parent_fired_count"] = len(self.mechanism_multi_parent_events[key] & self.events)
            for field in ("status_counts", "terminal_reason_counts", "temporal_model_usage_counts", "precondition_decision_counts"):
                row[field] = dict(sorted(row[field].items()))
            for field in ("predicate_checks", "legacy_terminal_predicate_snapshots"):
                row[field] = [v for _, v in sorted(row[field].items())]
            row["coverage_state"] = ("fired" if row["fired_event_count"] else
                "candidate_only" if row["candidate_count"] else "not_observed")
            rows.append(row)
        registered = [r for r in rows if r["registered"]]
        total_events = sum(self.category_counts.values())
        statuses = Counter()
        for row in rows:
            statuses.update(row["status_counts"])
        preconditions = Counter()
        for row in rows:
            preconditions.update(row["precondition_decision_counts"])
        observed_inversions = recorded_inversions = 0
        for timing in self.temporal_rows.values():
            observed_inversions += self._inversions([(observed, start) for start, observed, _ in timing if observed is not None])
            recorded_inversions += self._inversions([(recorded, start) for start, _, recorded in timing if recorded is not None])
        recovery_types = self.profile.get("recovery_or_clearing_event_types", [])
        return {
            "schema": "mechanism-coverage-report-v1", "domain": self.catalog.get("domain"),
            "status": "descriptive_not_a_pass_fail_gate",
            "scope": "written_episodes_only_not_rejected_generation_attempts",
            "episode_count": len(self.episodes), "registered_mechanism_count": len(registered),
            "candidate_created_mechanism_count": sum(r["candidate_count"] > 0 for r in registered),
            "checked_mechanism_count": sum(r["checked_candidate_count"] > 0 for r in registered),
            "fired_mechanism_count": sum(r["fired_event_count"] > 0 for r in registered),
            "not_observed_mechanism_ids": [r["mechanism_id"] for r in registered if not r["candidate_count"]],
            "candidate_only_mechanism_ids": [r["mechanism_id"] for r in registered if r["candidate_count"] and not r["fired_event_count"]],
            "unknown_mechanism_ids": [r["mechanism_id"] for r in rows if not r["registered"]],
            "event_count": total_events,
            "event_categories": {k: {"count": self.category_counts[k], "fraction": self.category_counts[k] / total_events if total_events else None} for k in CATEGORIES},
            "category_profile": deepcopy(self.profile),
            "unclassified_event_type_ids": sorted(t for t in self.type_counts if t not in self.categories),
            "candidate_status_counts": dict(sorted(statuses.items())),
            "preconditions": {"scope": "checks_on_existing_candidates_only",
                "runtime_decision_counts": dict(sorted(preconditions.items())),
                "runtime_audit_candidate_count": sum(r["runtime_audit_candidate_count"] for r in rows),
                "legacy_or_unaudited_candidate_count": sum(r["legacy_or_unaudited_candidate_count"] for r in rows),
                "predicate_pass_observation_count": sum(p["passed_count"] for r in rows for p in r["predicate_checks"]),
                "predicate_fail_observation_count": sum(p["failed_count"] for r in rows for p in r["predicate_checks"])},
            "lifecycle_facets": {
                "cancelled_or_superseded_not_emitted_count": statuses["cancelled"] + statuses["superseded"],
                "right_censored_unresolved_count": statuses["right_censored"],
                "recovery_or_clearing": {"declared_event_types": recovery_types,
                    "event_count": sum(self.type_counts[t] for t in recovery_types),
                    "status": "observed" if any(self.type_counts[t] for t in recovery_types) else "not_observed" if recovery_types else "not_declared"},
                "multi_parent_emitted_event_count": sum(r["multi_parent_fired_count"] for r in rows),
                "multi_causal_parent_event_count": sum(len(parents) > 1 and key in self.events for key, parents in self.incoming_causal.items()),
                "explicit_stale_report_count": self.stale_report_count,
                "sample_time_report_inversion_count": sum(self._inversions(pairs) for pairs in self.report_rows.values()),
                "observed_time_order_inversion_count": observed_inversions,
                "recorded_time_order_inversion_count": recorded_inversions,
            },
            "node_participation": {"background_entity_count": len(self.inventory),
                "realized_event_participant_count": len(self.used_entities & self.inventory),
                "participation_fraction": len(self.used_entities & self.inventory) / len(self.inventory) if self.inventory else None,
                "unknown_participant_reference_count": len(self.used_entities - self.inventory),
                "per_context": [{"context_id": c, "background_entity_count": len(ids),
                    "realized_event_participant_count": len(ids & self.context_used[c])}
                    for c, ids in sorted(self.context_inventory.items(), key=lambda p: str(p[0]))]},
            "attribution_checks": {"fired_candidate_without_event_count": len(self.fired_events - self.events),
                "event_without_fired_candidate_count": len(self.events - self.fired_events)},
            "event_relation_class_counts": dict(sorted(self.relation_classes.items())),
            "mechanisms": rows,
            "episodes": [dict(row, category_counts=dict(row["category_counts"]), candidate_status_counts=dict(row["candidate_status_counts"]))
                for _, row in sorted(self.episodes.items())],
            "limitations": [
                "Denominator is this implemented catalog, not all mechanisms in the real-world domain.",
                "Not observed is not not implemented, impossible, or a failed generation.",
                "Precondition audit covers existing candidates; domain branches rejecting before candidate creation are unrecorded.",
                "No fresh predicate evidence means unspecified, not that every precondition passed.",
                "Legacy terminal snapshots cannot reconstruct repeated dynamic checks.",
                "Right censoring is unresolved after the observation boundary, not proven non-occurrence.",
                "Physical/process category includes modeled digital state transitions and is not a realism claim.",
                "Stale-report flags and sample-time inversions are separate from delayed collector recording.",
                "Conditional branches sharing a prefix must not be pooled as independent episodes.",
            ],
        }


def coverage_summary(report):
    return {key: report[key] for key in ("schema", "status", "scope", "registered_mechanism_count",
        "candidate_created_mechanism_count", "checked_mechanism_count", "fired_mechanism_count",
        "not_observed_mechanism_ids", "event_categories", "preconditions", "lifecycle_facets", "node_participation")}
