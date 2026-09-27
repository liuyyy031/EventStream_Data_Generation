"""Scoped, observable history features; no future events or implicit parents."""
import math
from collections import defaultdict


class HistoryIndex:
    def __init__(self, events=()):
        self.events = []
        self.by_entity = defaultdict(list)
        self.by_type = defaultdict(list)
        for event in events:
            self.add(event)

    def add(self, event):
        self.events.append(event)
        self.by_type[event.event_type_id].append(event)
        for entity in {p.entity_id for p in event.participants}:
            self.by_entity[entity].append(event)

    def offsets(self, candidate, now, spec):
        scope = spec.get("scope", "entity")
        if scope == "entity":
            pool = {e.event_id: e for p in candidate.participants
                    for e in self.by_entity[p.entity_id]}.values()
        elif scope == "global":
            pool = self.events
        elif scope == "parents":
            parents = set(candidate.parent_event_ids)
            pool = (e for e in self.events if e.event_id in parents)
        elif scope == "process":
            key = spec.get("process_key", "process_id")
            value = candidate.attributes.get(key)
            if value is None:
                raise ValueError("Process history requires a process identifier")
            pool = (e for e in self.events if e.attributes.get(key) == value)
        else:
            raise ValueError("History scope must be entity, global, parents, or process")
        types = set(spec.get("event_types", []))
        anchor = spec.get("available_at", "occurrence_start")
        if anchor not in {"occurrence_start", "observed_at", "recorded_at"}:
            raise ValueError("Invalid history availability anchor")
        result = []
        for e in pool:
            if types and e.event_type_id not in types:
                continue
            try:
                available = e.temporal.anchor(anchor)
            except ValueError:
                continue
            if available <= now:
                result.append(e.temporal.occurrence_start_offset_seconds)
        return sorted(result)

    def inputs(self, candidate, now):
        inputs = dict(candidate.temporal_inputs)
        spec = inputs.get("history_spec")
        if not spec:
            return inputs
        offsets = self.offsets(candidate, now, spec)
        inputs["history_event_offsets"] = offsets
        inputs["history_count"] = len(offsets)
        lookback = spec.get("lookback_seconds")
        inputs["history_recent_count"] = sum(
            lookback is None or now - value <= float(lookback) for value in offsets
        )
        if offsets:
            inputs["last_event_offset_seconds"] = offsets[-1]
            inputs["time_since_last_event"] = now - offsets[-1]
        decay = float(spec.get("decay_per_second", 0.0))
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("History decay must be finite and non-negative")
        inputs["history_decayed_count"] = sum(math.exp(-decay * (now-t)) for t in offsets)
        return inputs
