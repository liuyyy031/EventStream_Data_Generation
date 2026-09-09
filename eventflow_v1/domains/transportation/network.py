"""Scalable transportation topology generation and external-network loading."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ...core.models import Entity, NetworkEdge


@dataclass
class TransportNetwork:
    network_id: str
    structure_family: str
    entities: List[Entity]
    edges: List[NetworkEdge]

    def __post_init__(self) -> None:
        self.entity_by_id: Dict[str, Entity] = {
            entity.entity_id: entity for entity in self.entities
        }
        self.outgoing: Dict[str, List[NetworkEdge]] = {
            entity.entity_id: [] for entity in self.entities
        }
        for edge in self.edges:
            self.outgoing.setdefault(edge.source_entity_id, []).append(edge)

    @property
    def road_ids(self) -> List[str]:
        return [
            entity.entity_id
            for entity in self.entities
            if entity.entity_type == "road_segment"
        ]

    @property
    def region_ids(self) -> List[str]:
        return [
            entity.entity_id
            for entity in self.entities
            if entity.entity_type == "region"
        ]

    def network_record(self) -> Dict[str, object]:
        max_out_degree = max((len(edges) for edges in self.outgoing.values()), default=0)
        return {
            "network_id": self.network_id,
            "domain": "transportation",
            "structure_family": self.structure_family,
            "entity_count": len(self.entities),
            "edge_count": len(self.edges),
            "max_out_degree": max_out_degree,
            "representation": "sparse_directed_edge_list",
        }


def build_network(
    config: Dict[str, object],
    seed: int,
    demand_profile: Dict[str, object] | None = None,
) -> TransportNetwork:
    external = config.get("external_network_file")
    if external:
        return load_external_network(Path(str(external)))
    return generate_network(
        network_id=str(config.get("network_id", "transport_network_001")),
        structure_family=str(config.get("structure_family", "mixed")),
        node_count=int(config.get("node_count", 100)),
        extra_edge_ratio=float(config.get("extra_edge_ratio", 0.15)),
        max_out_degree=int(config.get("max_topology_out_degree", 6)),
        seed=seed,
        baseline_vc_min=float(
            (demand_profile or {}).get("baseline_volume_capacity_ratio_min", 0.45)
        ),
        baseline_vc_max=float(
            (demand_profile or {}).get("baseline_volume_capacity_ratio_max", 1.05)
        ),
    )


def generate_network(
    network_id: str,
    structure_family: str,
    node_count: int,
    extra_edge_ratio: float,
    max_out_degree: int,
    seed: int,
    baseline_vc_min: float = 0.45,
    baseline_vc_max: float = 1.05,
) -> TransportNetwork:
    """Generate a sparse graph; never materialize an N x N adjacency matrix."""
    rng = random.Random(seed)
    selected_family = structure_family

    coordinates, pairs = _topology(
        selected_family, node_count, extra_edge_ratio, max_out_degree, rng
    )
    region = Entity(
        entity_id=f"{network_id}_region_0000",
        network_id=network_id,
        entity_type="region",
        attributes={"name": "synthetic_transport_region"},
    )
    entities = [region]
    for index, (x, y) in enumerate(coordinates):
        capacity = int(rng.uniform(700, 2400))
        baseline_vc_ratio = rng.uniform(baseline_vc_min, baseline_vc_max)
        entities.append(
            Entity(
                entity_id=f"{network_id}_road_{index:06d}",
                network_id=network_id,
                entity_type="road_segment",
                attributes={
                    "region_id": region.entity_id,
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "length_km": round(rng.uniform(0.2, 2.5), 4),
                    "capacity_vehicles_per_hour": capacity,
                    "baseline_volume_vehicles_per_hour": round(
                        capacity * baseline_vc_ratio, 3
                    ),
                    "baseline_volume_capacity_ratio": round(baseline_vc_ratio, 6),
                    "speed_limit_kph": rng.choice([30, 40, 50, 60, 80]),
                    "lanes": rng.choice([1, 2, 2, 3, 4]),
                },
            )
        )

    edges: List[NetworkEdge] = []
    for edge_index, (source, target) in enumerate(_deduplicate_pairs(pairs)):
        source_entity = entities[source + 1]
        target_entity = entities[target + 1]
        distance = _distance(coordinates[source], coordinates[target])
        source_speed = float(source_entity.attributes["speed_limit_kph"])
        travel_seconds = max(8.0, distance / max(source_speed, 1.0) * 3600.0)
        edges.append(
            NetworkEdge(
                edge_id=f"{network_id}_edge_{edge_index:07d}",
                network_id=network_id,
                source_entity_id=source_entity.entity_id,
                target_entity_id=target_entity.entity_id,
                edge_type="downstream_reachability",
                attributes={
                    "distance_km": round(distance, 5),
                    "free_flow_travel_seconds": round(travel_seconds, 3),
                    "propagation_eligible": True,
                },
            )
        )
    return TransportNetwork(network_id, selected_family, entities, edges)


def load_external_network(path: Path) -> TransportNetwork:
    """Load a sparse external network using the documented EventFlow format."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    network_id = str(payload["network_id"])
    entities = [Entity(**item) for item in payload["entities"]]
    edges = [NetworkEdge(**item) for item in payload["edges"]]
    entity_ids = {entity.entity_id for entity in entities}
    invalid = [
        edge.edge_id
        for edge in edges
        if edge.source_entity_id not in entity_ids or edge.target_entity_id not in entity_ids
    ]
    if invalid:
        raise ValueError(f"External network has dangling edges: {invalid[:5]}")
    if sum(entity.entity_type == "road_segment" for entity in entities) < 2:
        raise ValueError("External network needs at least two road_segment entities")
    if not any(entity.entity_type == "region" for entity in entities):
        raise ValueError("External network needs at least one region entity")
    return TransportNetwork(
        network_id=network_id,
        structure_family=str(payload.get("structure_family", "external")),
        entities=entities,
        edges=edges,
    )


def _topology(
    family: str,
    node_count: int,
    extra_edge_ratio: float,
    max_out_degree: int,
    rng: random.Random,
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    if family == "corridor":
        coordinates = [(float(i), rng.uniform(-0.05, 0.05)) for i in range(node_count)]
        pairs = [(i, i + 1) for i in range(node_count - 1)]
    elif family == "grid":
        width = max(2, math.ceil(math.sqrt(node_count)))
        coordinates = [(float(i % width), float(i // width)) for i in range(node_count)]
        pairs = []
        for i in range(node_count):
            right = i + 1
            down = i + width
            if right < node_count and right // width == i // width:
                pairs.append((i, right))
                if rng.random() < 0.35:
                    pairs.append((right, i))
            if down < node_count:
                pairs.append((i, down))
                if rng.random() < 0.35:
                    pairs.append((down, i))
    elif family == "hub":
        # A bounded-degree radial hierarchy avoids the unrealistic single road
        # segment connected directly to hundreds of downstream segments.
        branch_factor = max(2, min(4, max_out_degree - 1))
        coordinates = [(0.0, 0.0)]
        pairs = []
        for i in range(1, node_count):
            parent = (i - 1) // branch_factor
            depth = int(math.floor(math.log(i * (branch_factor - 1) + 1, branch_factor)))
            nodes_at_depth = max(1, branch_factor**depth)
            position = i - (branch_factor**depth - 1) // (branch_factor - 1)
            angle = 2.0 * math.pi * position / nodes_at_depth
            radius = float(depth + 1)
            coordinates.append((radius * math.cos(angle), radius * math.sin(angle)))
            pairs.append((parent, i))
            # Some spokes are bidirectional, but both directions remain bounded.
            if i % 3 != 0:
                pairs.append((i, parent))
    elif family == "random_connected":
        coordinates = [(rng.random() * 10.0, rng.random() * 10.0) for _ in range(node_count)]
        order = sorted(range(node_count), key=lambda idx: coordinates[idx][0])
        pairs = [(order[i], order[i + 1]) for i in range(node_count - 1)]
    else:
        raise ValueError(f"Unsupported network structure_family: {family}")

    extra_count = max(0, int(node_count * extra_edge_ratio))
    pairs = _add_bounded_extra_edges(
        pairs, node_count, extra_count, max_out_degree, rng
    )
    return coordinates, pairs


def _add_bounded_extra_edges(
    pairs: List[Tuple[int, int]],
    node_count: int,
    extra_count: int,
    max_out_degree: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    unique = list(dict.fromkeys(pairs))
    existing = set(unique)
    out_degree: Dict[int, int] = {index: 0 for index in range(node_count)}
    for source, _ in unique:
        out_degree[source] += 1
    added = 0
    attempts = 0
    max_attempts = max(100, extra_count * 30)
    while added < extra_count and attempts < max_attempts:
        attempts += 1
        source = rng.randrange(node_count)
        target = rng.randrange(node_count)
        pair = (source, target)
        if (
            source == target
            or pair in existing
            or out_degree[source] >= max_out_degree
        ):
            continue
        unique.append(pair)
        existing.add(pair)
        out_degree[source] += 1
        added += 1
    return unique


def _deduplicate_pairs(pairs: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return list(dict.fromkeys(pair for pair in pairs if pair[0] != pair[1]))


def _distance(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return max(0.05, math.hypot(left[0] - right[0], left[1] - right[1]))
