"""Sparse, typed, domain-neutral context-graph generation.

The generator owns graph mechanics only.  A domain package supplies entity
types, admissible relation layers, hard constraints, and optional relation
attribute builders.  No spatial coordinates or global-connectivity assumption
is embedded in the common core.
"""

from __future__ import annotations

import bisect
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .models import ContextRelation, Entity


RelationAttributeBuilder = Callable[
    [Entity, Entity, random.Random], Dict[str, Any]
]


def relation_is_active(relation: ContextRelation, at_time: float) -> bool:
    """Return whether a context relation is valid at one episode offset."""

    start = relation.valid_from_offset_seconds
    end = relation.valid_to_offset_seconds
    return (start is None or start <= at_time) and (end is None or at_time <= end)


@dataclass(frozen=True)
class RelationNeighbor:
    neighbor_entity_id: str
    relation: ContextRelation


class ContextRelationIndex:
    """Sparse direction-aware index with optional temporal filtering."""

    def __init__(self, relations: Sequence[ContextRelation]) -> None:
        self._by_id: Dict[str, ContextRelation] = {}
        self._outgoing: Dict[tuple[str, str], List[RelationNeighbor]] = defaultdict(list)
        self._incoming: Dict[tuple[str, str], List[RelationNeighbor]] = defaultdict(list)
        for relation in relations:
            if relation.relation_id in self._by_id:
                raise ValueError(f"Duplicate context relation ID: {relation.relation_id}")
            self._by_id[relation.relation_id] = relation
            self._outgoing[(relation.source_entity_id, relation.relation_type_id)].append(
                RelationNeighbor(relation.target_entity_id, relation)
            )
            self._incoming[(relation.target_entity_id, relation.relation_type_id)].append(
                RelationNeighbor(relation.source_entity_id, relation)
            )
            if not relation.directed:
                self._outgoing[(relation.target_entity_id, relation.relation_type_id)].append(
                    RelationNeighbor(relation.source_entity_id, relation)
                )
                self._incoming[(relation.source_entity_id, relation.relation_type_id)].append(
                    RelationNeighbor(relation.target_entity_id, relation)
                )

    def get(self, relation_id: str) -> ContextRelation | None:
        return self._by_id.get(relation_id)

    def neighbors(
        self,
        entity_id: str,
        relation_type_id: str | None = None,
        *,
        at_time: float | None = None,
        direction: str = "outgoing",
    ) -> List[RelationNeighbor]:
        if direction not in {"outgoing", "incoming"}:
            raise ValueError("direction must be 'outgoing' or 'incoming'")
        index = self._outgoing if direction == "outgoing" else self._incoming
        if relation_type_id is None:
            values = [
                neighbor
                for (indexed_entity, _), neighbors in index.items()
                if indexed_entity == entity_id
                for neighbor in neighbors
            ]
        else:
            values = list(index.get((entity_id, relation_type_id), ()))
        if at_time is not None:
            values = [
                neighbor
                for neighbor in values
                if relation_is_active(neighbor.relation, at_time)
            ]
        return sorted(
            values,
            key=lambda item: (item.relation.relation_id, item.neighbor_entity_id),
        )


@dataclass(frozen=True)
class RelationLayerSpec:
    relation_type_id: str
    source_type_ids: tuple[str, ...]
    target_type_ids: tuple[str, ...]
    family: str = "constrained_degree_weighted_block"
    directed: bool = True
    expected_out_degree: float = 1.5
    max_out_degree: int = 8
    max_in_degree: int | None = None
    community_count: int = 4
    within_community_bias: float = 2.0
    degree_sigma: float = 0.7
    hub_fraction: float = 0.05
    hub_multiplier: float = 3.0
    ensure_weak_connectivity: bool = False
    allow_self_loops: bool = False
    allow_cycles: bool = True
    temporal_edge_fraction: float = 0.0
    attribute_builder: RelationAttributeBuilder | None = field(
        default=None, compare=False, repr=False
    )

    def descriptor(self) -> Dict[str, Any]:
        return {
            "relation_type_id": self.relation_type_id,
            "source_type_ids": list(self.source_type_ids),
            "target_type_ids": list(self.target_type_ids),
            "family": self.family,
            "directed": self.directed,
            "expected_out_degree": self.expected_out_degree,
            "max_out_degree": self.max_out_degree,
            "max_in_degree": self.max_in_degree,
            "community_count": self.community_count,
            "within_community_bias": self.within_community_bias,
            "degree_sigma": self.degree_sigma,
            "hub_fraction": self.hub_fraction,
            "hub_multiplier": self.hub_multiplier,
            "ensure_weak_connectivity": self.ensure_weak_connectivity,
            "allow_self_loops": self.allow_self_loops,
            "allow_cycles": self.allow_cycles,
            "temporal_edge_fraction": self.temporal_edge_fraction,
        }


@dataclass
class GeneratedTopology:
    relations: List[ContextRelation]
    statistics: Dict[str, Any]
    layer_descriptors: List[Dict[str, Any]]

    def adjacency(
        self, relation_type_id: str
    ) -> Dict[str, List[ContextRelation]]:
        result: Dict[str, List[ContextRelation]] = defaultdict(list)
        for relation in self.relations:
            if relation.relation_type_id == relation_type_id:
                result[relation.source_entity_id].append(relation)
        return dict(result)


class SparseHeterogeneousTopologyGenerator:
    """Generate typed relation layers without constructing an N x N matrix."""

    SUPPORTED_FAMILIES = {
        "constrained_degree_weighted_block",
        "typed_bipartite",
        "directed_acyclic",
    }

    def generate(
        self,
        *,
        context_id: str,
        domain: str,
        entities: Sequence[Entity],
        layers: Sequence[RelationLayerSpec],
        seed: int,
        observation_window_seconds: float | None = None,
    ) -> GeneratedTopology:
        if not entities:
            raise ValueError("A context graph requires at least one entity")
        entity_by_id = {entity.entity_id: entity for entity in entities}
        if len(entity_by_id) != len(entities):
            raise ValueError("Context graph entity IDs must be unique")
        known_types = {entity.entity_type_id for entity in entities}
        rng = random.Random(seed)
        relations: List[ContextRelation] = []
        layer_statistics: List[Dict[str, Any]] = []
        relation_counter = 1

        for layer_index, layer in enumerate(layers):
            self._validate_layer(layer, known_types, observation_window_seconds)
            layer_rng = random.Random(rng.getrandbits(64) ^ (layer_index + 1) * 104729)
            pairs, communities = self._sample_layer(entities, layer, layer_rng)
            layer_relations: List[ContextRelation] = []
            for source_id, target_id in pairs:
                source = entity_by_id[source_id]
                target = entity_by_id[target_id]
                attributes = (
                    layer.attribute_builder(source, target, layer_rng)
                    if layer.attribute_builder
                    else {}
                )
                valid_from, valid_to = _sample_validity_interval(
                    layer,
                    layer_rng,
                    observation_window_seconds,
                )
                relation = ContextRelation(
                    relation_id=f"{context_id}_relation_{relation_counter:08d}",
                    context_id=context_id,
                    domain=domain,
                    relation_type_id=layer.relation_type_id,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    directed=layer.directed,
                    valid_from_offset_seconds=valid_from,
                    valid_to_offset_seconds=valid_to,
                    attributes=attributes,
                    provenance={
                        "generation_family": layer.family,
                        "topology_layer_index": layer_index,
                        "synthetic_structure": True,
                    },
                )
                relation_counter += 1
                relations.append(relation)
                layer_relations.append(relation)
            stats = _layer_statistics(
                entities,
                layer_relations,
                layer,
                communities,
            )
            _validate_layer_result(entities, layer_relations, layer, stats)
            layer_statistics.append(stats)

        return GeneratedTopology(
            relations=relations,
            statistics={
                "entity_count": len(entities),
                "relation_count": len(relations),
                "relation_layer_count": len(layers),
                "sparse_representation": True,
                "dense_matrix_materialized": False,
                "layers": layer_statistics,
            },
            layer_descriptors=[layer.descriptor() for layer in layers],
        )

    def _validate_layer(
        self,
        layer: RelationLayerSpec,
        known_types: set[str],
        observation_window_seconds: float | None,
    ) -> None:
        if layer.family not in self.SUPPORTED_FAMILIES:
            raise ValueError(f"Unsupported topology family: {layer.family}")
        if not layer.source_type_ids or not layer.target_type_ids:
            raise ValueError("Relation layers require source and target types")
        unknown = set(layer.source_type_ids + layer.target_type_ids) - known_types
        if unknown:
            raise ValueError(f"Relation layer references unknown entity types: {sorted(unknown)}")
        if layer.expected_out_degree < 0.0 or layer.max_out_degree <= 0:
            raise ValueError("Invalid relation-layer degree controls")
        if layer.max_in_degree is not None and layer.max_in_degree <= 0:
            raise ValueError("max_in_degree must be positive when supplied")
        if layer.community_count <= 0 or layer.within_community_bias < 1.0:
            raise ValueError("Invalid community controls")
        if layer.degree_sigma < 0.0 or not 0.0 <= layer.hub_fraction <= 1.0:
            raise ValueError("Invalid degree-heterogeneity controls")
        if layer.hub_multiplier < 1.0:
            raise ValueError("hub_multiplier must be at least one")
        if not 0.0 <= layer.temporal_edge_fraction <= 1.0:
            raise ValueError("temporal_edge_fraction must lie in [0, 1]")
        if layer.temporal_edge_fraction and (
            observation_window_seconds is None
            or observation_window_seconds <= 0.0
        ):
            raise ValueError("Temporal relation layers require a positive observation window")
        if layer.family == "directed_acyclic" and (
            not layer.directed or layer.allow_cycles
        ):
            raise ValueError("directed_acyclic layers must be directed with cycles disabled")

    def _sample_layer(
        self,
        entities: Sequence[Entity],
        layer: RelationLayerSpec,
        rng: random.Random,
    ) -> tuple[List[tuple[str, str]], Dict[str, int]]:
        sources = [
            entity for entity in entities
            if entity.entity_type_id in layer.source_type_ids
        ]
        targets = [
            entity for entity in entities
            if entity.entity_type_id in layer.target_type_ids
        ]
        if not sources or not targets:
            return [], {}
        source_ids = {entity.entity_id for entity in sources}
        target_ids = {entity.entity_id for entity in targets}
        if layer.ensure_weak_connectivity and source_ids != target_ids:
            raise ValueError(
                "Weak connectivity can only be required when a layer has the same source and target population"
            )

        population = sorted(source_ids | target_ids)
        shuffled = list(population)
        rng.shuffle(shuffled)
        communities = {
            entity_id: index % min(layer.community_count, len(population))
            for index, entity_id in enumerate(shuffled)
        }
        source_weights = _degree_weights(sources, layer, rng)
        target_weights = _degree_weights(targets, layer, rng)
        target_sampler = _CommunitySampler(targets, target_weights, communities)
        source_sampler = _WeightedSampler(
            [entity.entity_id for entity in sources],
            [source_weights[entity.entity_id] for entity in sources],
        )
        ranks = {
            entity_id: index for index, entity_id in enumerate(sorted(population))
        }
        pairs: List[tuple[str, str]] = []
        pair_keys: set[tuple[str, str]] = set()
        out_degree: Counter[str] = Counter()
        in_degree: Counter[str] = Counter()

        def can_add(source_id: str, target_id: str) -> bool:
            if not layer.allow_self_loops and source_id == target_id:
                return False
            if layer.family == "directed_acyclic" and ranks[source_id] >= ranks[target_id]:
                return False
            key = (
                (source_id, target_id)
                if layer.directed
                else tuple(sorted((source_id, target_id)))
            )
            if key in pair_keys:
                return False
            if out_degree[source_id] >= layer.max_out_degree:
                return False
            if layer.max_in_degree is not None and in_degree[target_id] >= layer.max_in_degree:
                return False
            if not layer.directed and out_degree[target_id] >= layer.max_out_degree:
                return False
            return True

        def add(source_id: str, target_id: str) -> bool:
            if not can_add(source_id, target_id):
                return False
            key = (
                (source_id, target_id)
                if layer.directed
                else tuple(sorted((source_id, target_id)))
            )
            pair_keys.add(key)
            pairs.append((source_id, target_id))
            out_degree[source_id] += 1
            in_degree[target_id] += 1
            if not layer.directed:
                out_degree[target_id] += 1
                in_degree[source_id] += 1
            return True

        if layer.ensure_weak_connectivity and len(population) > 1:
            order = sorted(population) if layer.family == "directed_acyclic" else shuffled
            connected = [order[0]]
            available_parents = [order[0]]
            for target_id in order[1:]:
                rng.shuffle(available_parents)
                added = False
                for source_id in list(available_parents):
                    if add(source_id, target_id):
                        added = True
                        break
                if not added:
                    raise ValueError(
                        f"Cannot build a connected backbone for {layer.relation_type_id} under its degree constraints"
                    )
                connected.append(target_id)
                available_parents.append(target_id)
                available_parents = [
                    item for item in available_parents
                    if out_degree[item] < layer.max_out_degree
                ]

        desired_edges = int(round(layer.expected_out_degree * len(sources)))
        desired_edges = max(desired_edges, len(pairs))
        desired_edges = min(desired_edges, len(sources) * layer.max_out_degree)
        max_attempts = max(100, desired_edges * 40)
        attempts = 0
        same_block_probability = (
            0.0
            if layer.within_community_bias <= 1.0
            else 1.0 - 1.0 / layer.within_community_bias
        )
        while len(pairs) < desired_edges and attempts < max_attempts:
            attempts += 1
            source_id = source_sampler.draw(rng)
            prefer_community = (
                communities[source_id]
                if rng.random() < same_block_probability
                else None
            )
            target_id = target_sampler.draw(rng, prefer_community)
            add(source_id, target_id)
        minimum_acceptable = min(desired_edges, max(0, int(desired_edges * 0.95)))
        if len(pairs) < minimum_acceptable:
            raise ValueError(
                f"Layer {layer.relation_type_id} could only create {len(pairs)} of {desired_edges} requested edges"
            )
        return pairs, communities


class _WeightedSampler:
    def __init__(self, items: Sequence[str], weights: Sequence[float]) -> None:
        self.items = list(items)
        self.cumulative = list(accumulate(max(float(weight), 1e-12) for weight in weights))

    def draw(self, rng: random.Random) -> str:
        point = rng.random() * self.cumulative[-1]
        index = bisect.bisect_left(self.cumulative, point)
        return self.items[min(index, len(self.items) - 1)]


class _CommunitySampler:
    def __init__(
        self,
        entities: Sequence[Entity],
        weights: Mapping[str, float],
        communities: Mapping[str, int],
    ) -> None:
        self.global_sampler = _WeightedSampler(
            [entity.entity_id for entity in entities],
            [weights[entity.entity_id] for entity in entities],
        )
        grouped: Dict[int, List[Entity]] = defaultdict(list)
        for entity in entities:
            grouped[communities[entity.entity_id]].append(entity)
        self.by_community = {
            community: _WeightedSampler(
                [entity.entity_id for entity in items],
                [weights[entity.entity_id] for entity in items],
            )
            for community, items in grouped.items()
        }

    def draw(self, rng: random.Random, community: int | None) -> str:
        sampler = self.by_community.get(community, self.global_sampler)
        return sampler.draw(rng)


def _degree_weights(
    entities: Sequence[Entity],
    layer: RelationLayerSpec,
    rng: random.Random,
) -> Dict[str, float]:
    weights = {
        entity.entity_id: rng.lognormvariate(0.0, layer.degree_sigma)
        for entity in entities
    }
    hub_count = min(
        len(entities),
        int(round(layer.hub_fraction * len(entities))),
    )
    for entity in rng.sample(list(entities), hub_count):
        weights[entity.entity_id] *= layer.hub_multiplier
    return weights


def _sample_validity_interval(
    layer: RelationLayerSpec,
    rng: random.Random,
    observation_window_seconds: float | None,
) -> tuple[float | None, float | None]:
    if (
        not observation_window_seconds
        or rng.random() >= layer.temporal_edge_fraction
    ):
        return None, None
    start = rng.uniform(0.0, observation_window_seconds * 0.75)
    end = rng.uniform(start, observation_window_seconds)
    return round(start, 9), round(end, 9)


def _layer_statistics(
    entities: Sequence[Entity],
    relations: Sequence[ContextRelation],
    layer: RelationLayerSpec,
    communities: Mapping[str, int],
) -> Dict[str, Any]:
    eligible_ids = {
        entity.entity_id for entity in entities
        if entity.entity_type_id in set(layer.source_type_ids + layer.target_type_ids)
    }
    out_degree: Counter[str] = Counter()
    in_degree: Counter[str] = Counter()
    undirected_neighbors: Dict[str, set[str]] = {
        entity_id: set() for entity_id in eligible_ids
    }
    same_community_edges = 0
    type_by_id = {entity.entity_id: entity.entity_type_id for entity in entities}
    type_pair_counts: Counter[str] = Counter()
    for relation in relations:
        out_degree[relation.source_entity_id] += 1
        in_degree[relation.target_entity_id] += 1
        undirected_neighbors[relation.source_entity_id].add(relation.target_entity_id)
        undirected_neighbors[relation.target_entity_id].add(relation.source_entity_id)
        if communities.get(relation.source_entity_id) == communities.get(
            relation.target_entity_id
        ):
            same_community_edges += 1
        type_pair_counts[
            f"{type_by_id[relation.source_entity_id]}->{type_by_id[relation.target_entity_id]}"
        ] += 1
    out_values = [out_degree[entity.entity_id] for entity in entities if entity.entity_id in eligible_ids]
    return {
        "relation_type_id": layer.relation_type_id,
        "family": layer.family,
        "eligible_entity_count": len(eligible_ids),
        "edge_count": len(relations),
        "mean_out_degree": (
            sum(out_values) / len(out_values) if out_values else 0.0
        ),
        "max_out_degree": max(out_values, default=0),
        "unique_out_degree_count": len(set(out_values)),
        "weak_component_count": _component_count(undirected_neighbors),
        "same_community_edge_fraction": (
            same_community_edges / len(relations) if relations else 0.0
        ),
        "type_pair_counts": dict(type_pair_counts),
        "temporally_bounded_edge_count": sum(
            relation.valid_from_offset_seconds is not None
            or relation.valid_to_offset_seconds is not None
            for relation in relations
        ),
    }


def _component_count(neighbors: Mapping[str, set[str]]) -> int:
    unseen = set(neighbors)
    components = 0
    while unseen:
        components += 1
        start = unseen.pop()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current] & unseen:
                unseen.remove(neighbor)
                queue.append(neighbor)
    return components


def _validate_layer_result(
    entities: Sequence[Entity],
    relations: Sequence[ContextRelation],
    layer: RelationLayerSpec,
    statistics: Mapping[str, Any],
) -> None:
    type_by_id = {entity.entity_id: entity.entity_type_id for entity in entities}
    allowed_sources = set(layer.source_type_ids)
    allowed_targets = set(layer.target_type_ids)
    for relation in relations:
        if type_by_id[relation.source_entity_id] not in allowed_sources:
            raise ValueError(f"Invalid source type in {relation.relation_id}")
        if type_by_id[relation.target_entity_id] not in allowed_targets:
            raise ValueError(f"Invalid target type in {relation.relation_id}")
        if not layer.allow_self_loops and relation.source_entity_id == relation.target_entity_id:
            raise ValueError(f"Unexpected self loop in {relation.relation_id}")
    if statistics["max_out_degree"] > layer.max_out_degree:
        raise ValueError(f"Layer {layer.relation_type_id} exceeds max_out_degree")
    if layer.ensure_weak_connectivity and statistics["weak_component_count"] != 1:
        raise ValueError(f"Layer {layer.relation_type_id} is not weakly connected")
    if layer.family == "directed_acyclic":
        ranks = {
            entity_id: index
            for index, entity_id in enumerate(sorted(type_by_id))
        }
        if any(
            ranks[relation.source_entity_id] >= ranks[relation.target_entity_id]
            for relation in relations
        ):
            raise ValueError(f"Layer {layer.relation_type_id} contains a directed cycle")
