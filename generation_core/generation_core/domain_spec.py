"""Load and validate executable domain topology specifications.

The specification deliberately stops at context structure.  Event mechanisms,
temporal distributions, and domain truth claims remain in reviewed domain
packages until they have their own evidence and validation protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .topology import RelationAttributeBuilder, RelationLayerSpec


_LAYER_FIELDS = {
    "family",
    "directed",
    "expected_out_degree",
    "max_out_degree",
    "max_in_degree",
    "community_count",
    "within_community_bias",
    "degree_sigma",
    "hub_fraction",
    "hub_multiplier",
    "ensure_weak_connectivity",
    "allow_self_loops",
    "allow_cycles",
    "temporal_edge_fraction",
}


@dataclass(frozen=True)
class LoadedDomainSpec:
    source_path: Path
    domain: str
    entity_type_ids: tuple[str, ...]
    context_relation_type_ids: tuple[str, ...]
    topology_contract: Dict[str, Any]
    raw: Dict[str, Any]

    def compile_relation_layers(
        self,
        *,
        parameter_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        attribute_builders: Mapping[str, RelationAttributeBuilder] | None = None,
    ) -> list[RelationLayerSpec]:
        """Compile JSON layer declarations into generator inputs.

        Overrides are keyed by relation_type_id and may only change numeric or
        structural controls; they cannot change the declared endpoint types.
        """

        overrides = parameter_overrides or {}
        builders = attribute_builders or {}
        declared_ids = {
            item["relation_type_id"]
            for item in self.topology_contract["relation_layers"]
        }
        unknown_overrides = set(overrides) - declared_ids
        if unknown_overrides:
            raise ValueError(
                f"Overrides reference undeclared relation layers: {sorted(unknown_overrides)}"
            )

        compiled: list[RelationLayerSpec] = []
        for declaration in self.topology_contract["relation_layers"]:
            relation_type_id = declaration["relation_type_id"]
            merged = {
                key: value
                for key, value in declaration.items()
                if key in _LAYER_FIELDS
            }
            supplied_overrides = dict(overrides.get(relation_type_id, {}))
            unsupported = set(supplied_overrides) - _LAYER_FIELDS
            if unsupported:
                raise ValueError(
                    f"Unsupported overrides for {relation_type_id}: {sorted(unsupported)}"
                )
            merged.update(supplied_overrides)
            if (
                not declaration.get("temporal_validity_allowed", False)
                and float(merged.get("temporal_edge_fraction", 0.0)) > 0.0
            ):
                raise ValueError(
                    f"Layer {relation_type_id} does not permit temporal validity intervals"
                )
            builder_id = declaration.get("attribute_builder_id")
            builder = None
            if builder_id is not None:
                try:
                    builder = builders[builder_id]
                except KeyError as exc:
                    raise ValueError(
                        f"No attribute builder registered for {builder_id}"
                    ) from exc
            compiled.append(
                RelationLayerSpec(
                    relation_type_id=relation_type_id,
                    source_type_ids=tuple(declaration["source_type_ids"]),
                    target_type_ids=tuple(declaration["target_type_ids"]),
                    attribute_builder=builder,
                    **merged,
                )
            )
        return compiled


def load_domain_spec(path: str | Path) -> LoadedDomainSpec:
    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("A domain specification must be a JSON object")

    domain = _required_string(payload, "domain")
    entity_type_ids = _unique_string_tuple(payload, "entity_type_ids")
    relation_type_ids = _unique_string_tuple(payload, "context_relation_type_ids")
    topology = payload.get("topology_contract")
    if not isinstance(topology, dict):
        raise ValueError("topology_contract must be an object")
    if topology.get("representation") != "sparse_typed_relation_layers":
        raise ValueError("Only sparse_typed_relation_layers is supported")
    if topology.get("n_ary_relation_policy") != "reify_as_typed_entity":
        raise ValueError("N-ary context relations must be represented by typed entities")
    if not isinstance(topology.get("spatial_coordinates_required"), bool):
        raise ValueError("spatial_coordinates_required must be boolean")

    layers = topology.get("relation_layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("topology_contract.relation_layers must be a non-empty list")
    seen_layers: set[str] = set()
    for layer in layers:
        _validate_layer_declaration(
            layer,
            set(entity_type_ids),
            set(relation_type_ids),
        )
        layer_id = layer["relation_type_id"]
        if layer_id in seen_layers:
            raise ValueError(f"Duplicate relation layer: {layer_id}")
        seen_layers.add(layer_id)

    spec = LoadedDomainSpec(
        source_path=source_path,
        domain=domain,
        entity_type_ids=entity_type_ids,
        context_relation_type_ids=relation_type_ids,
        topology_contract=topology,
        raw=payload,
    )
    # Compiling here catches invalid defaults even for structure-only packages.
    declarations_requiring_builders = {
        layer.get("attribute_builder_id")
        for layer in layers
        if layer.get("attribute_builder_id")
    }
    placeholder_builders = {
        builder_id: (lambda _source, _target, _rng: {})
        for builder_id in declarations_requiring_builders
    }
    compiled = spec.compile_relation_layers(attribute_builders=placeholder_builders)
    _validate_compiled_layers(compiled)
    return spec


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _unique_string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = payload.get(key)
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"{key} must be a non-empty string list")
    if len(values) != len(set(values)):
        raise ValueError(f"{key} contains duplicates")
    return tuple(values)


def _validate_layer_declaration(
    layer: Any,
    entity_type_ids: set[str],
    relation_type_ids: set[str],
) -> None:
    if not isinstance(layer, dict):
        raise ValueError("Every relation layer must be an object")
    required = {
        "relation_type_id",
        "family",
        "source_type_ids",
        "target_type_ids",
        "directed",
        "ensure_weak_connectivity",
    }
    missing = required - set(layer)
    if missing:
        raise ValueError(f"Relation layer lacks required fields: {sorted(missing)}")
    relation_type_id = _required_string(layer, "relation_type_id")
    if relation_type_id not in relation_type_ids:
        raise ValueError(
            f"Layer {relation_type_id} is absent from context_relation_type_ids"
        )
    for endpoint_key in ("source_type_ids", "target_type_ids"):
        endpoints = _unique_string_tuple(layer, endpoint_key)
        unknown = set(endpoints) - entity_type_ids
        if unknown:
            raise ValueError(
                f"Layer {relation_type_id} references unknown entity types: {sorted(unknown)}"
            )
    for flag in ("directed", "ensure_weak_connectivity"):
        if not isinstance(layer.get(flag), bool):
            raise ValueError(f"Layer {relation_type_id} field {flag} must be boolean")


def _validate_compiled_layers(layers: list[RelationLayerSpec]) -> None:
    supported = {
        "constrained_degree_weighted_block",
        "typed_bipartite",
        "directed_acyclic",
    }
    for layer in layers:
        if layer.family not in supported:
            raise ValueError(f"Unsupported topology family: {layer.family}")
        if layer.expected_out_degree < 0.0 or layer.max_out_degree <= 0:
            raise ValueError(f"Layer {layer.relation_type_id} has invalid degree controls")
        if layer.max_in_degree is not None and layer.max_in_degree <= 0:
            raise ValueError(f"Layer {layer.relation_type_id} has invalid max_in_degree")
        if layer.community_count <= 0 or layer.within_community_bias < 1.0:
            raise ValueError(f"Layer {layer.relation_type_id} has invalid community controls")
        if layer.degree_sigma < 0.0 or not 0.0 <= layer.hub_fraction <= 1.0:
            raise ValueError(f"Layer {layer.relation_type_id} has invalid heterogeneity controls")
        if layer.hub_multiplier < 1.0:
            raise ValueError(f"Layer {layer.relation_type_id} has invalid hub multiplier")
        if not 0.0 <= layer.temporal_edge_fraction <= 1.0:
            raise ValueError(f"Layer {layer.relation_type_id} has invalid temporal edge fraction")
        if layer.family == "directed_acyclic" and (
            not layer.directed or layer.allow_cycles
        ):
            raise ValueError(
                f"Layer {layer.relation_type_id} must be directed and acyclic"
            )
