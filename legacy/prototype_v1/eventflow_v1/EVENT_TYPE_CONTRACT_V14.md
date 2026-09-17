# EventFlow v1.4 event-type contract

## Frozen architecture

EventFlow v1.4 separates three concerns:

1. The common event envelope stores occurrence identity, exact time, a single
   namespaced `event_type_id`, qualified entity participants, attributes,
   causal role, observation metadata, and provenance.
2. A versioned domain registry defines each event type's category, meaning,
   allowed participant roles/entity types, and attribute contract.
3. DomainSpec rules define which registered source and target types may form a
   relation. A compatible type pair is not itself evidence that a relation ran.

The common event record deliberately does **not** repeat `event_category`,
`event_subtype`, `ontology_id`, or `ontology_version`. These are either derived
from the one type ID or stored once in the registry/manifest. It also does not
add a universal `occurrence_id` or `lifecycle_phase`: those concepts have
different meanings across road incidents, process activities, and telemetry
contexts. Domains represent lifecycle milestones with stable event type IDs
and explicit relations; correlation contexts are represented by qualified
participant entities or domain attributes.

## Common event shape

```json
{
  "event_id": "...",
  "episode_id": "...",
  "domain": "transportation",
  "event_type_id": "transportation.road.incident.vehicle_collision",
  "event_time": "2026-07-16T08:06:56+08:00",
  "time_offset_seconds": 416.0,
  "participants": [
    {"entity_id": "road_00184", "role": "affected_road"}
  ],
  "attributes": {},
  "event_role": "root",
  "observation": {},
  "provenance": {}
}
```

Participant qualifiers are part of the contract because a cross-domain event
can refer to affected objects, actors, resources, locations, cases, services,
or execution contexts. A bare list of IDs cannot preserve those semantics.

## Versioning and self-contained release

- Schema version: `eventflow-v1`.
- Generator version: `eventflow-v1.4.0`.
- Transportation registry: `streasoner.transportation.road` version `2.0.0`.
- Every output copies the exact registry to `event_type_registry.json`.
- `manifest.json` records the registry ID, version, filename, and SHA-256 of
  the released registry file.
- v1.3.1 output remains historical data and is not rewritten or mixed with
  v1.4 records. A future migration tool must use explicit mappings rather than
  emitting both old and new fields as competing sources of truth.

## Transportation scope

The implemented registry is explicitly road traffic, not the whole transport
sector. It separates:

- environmental conditions;
- traffic incidents;
- physical obstructions;
- equipment faults;
- operator actions;
- traffic-state changes.

The implemented types cover collision, breakdown, road debris, signal failure,
heavy-rain onset, flooding onset, road-closure start, roadworks start,
congestion onset, congestion easing, and normal-flow restoration.

`congestion_easing` always has `recovery_fraction < 1`; a
`normal_flow_restored` event always has `recovery_fraction = 1`. This prevents
the old `traffic_recovery` label from claiming full restoration for a partial
improvement.

## External design checks

- CloudEvents uses stable event identifiers, sources, types, subjects and
  timestamps, and recommends namespaced event types:
  https://github.com/cloudevents/spec/blob/main/cloudevents/spec.md
- OpenTelemetry requires a stable, low-cardinality event name and places
  dynamic values in attributes:
  https://opentelemetry.io/docs/specs/semconv/general/events/
- OCEL 2.0 qualifies event-to-object links instead of storing ambiguous object
  IDs without roles:
  https://www.ocel-standard.org/2.0/ocel20_specification.pdf
- IEEE XES separates event identity, timestamp, lifecycle information and
  extensible attributes for event streams:
  https://standards.ieee.org/ieee/1849/10907/
- DATEX II separates traffic elements, operator actions, conditions,
  obstructions, equipment faults and abnormal traffic:
  https://docs.datex2.eu/v3.1/level2user/situationPublication.html
- GTFS Realtime separates disruption causes from effects:
  https://gtfs.org/documentation/realtime/feed-entities/service-alerts/

## Validation obligations

An accepted event must satisfy all of the following:

- its `event_type_id` exists in the released registry;
- its attributes satisfy the registered required/optional schema;
- every participant role is allowed for that type;
- every participant entity has an allowed entity type;
- transportation cross-field rules hold, such as blocked lanes not exceeding
  the road's lane count;
- every relation references a registered source/target type combination and a
  named executed rule;
- generated text reproduces the type ID, participant links, attributes, exact
  time, and explicit relations without inventing unsupported causes.

## Deferred items

- Numeric capacity-reduction ranges and the initial congestion-severity model
  are transparent synthetic priors, not empirical truths.
- Incident detection, responder dispatch/arrival, roadway clearance, and
  observation delays need a later response/observation extension.
- OpenTelemetry and business-process registries remain explicit, non-loadable
  placeholders until their own rules and validators are implemented.
