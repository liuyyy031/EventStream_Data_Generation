# Distributed-systems domain structure

This directory reserves the domain contract only. The first implementation
should cover request, downstream call, response or timeout, retry, terminal
outcome, and recovery. Parentage should come from trace context or an explicit
synthetic rule; topology alone must never create an event parent relation.

Its reserved topology uses the shared sparse heterogeneous-graph engine with
separate call, deployment, and exposure layers. Cycles may exist in call layers,
while workflow layers can select the directed-acyclic operator. No spatial grid
or whole-graph connectivity requirement is inherited from transportation.

`mechanism_specs.json` is intentionally empty. It verifies the shared mechanism
contract without manufacturing trace semantics or failure probabilities before
a reference trace source and validation protocol are selected.
