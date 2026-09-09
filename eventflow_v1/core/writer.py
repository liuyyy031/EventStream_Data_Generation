"""Normalized streaming JSONL output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, IO

from .models import EpisodeBundle


class EventFlowWriter:
    FILES = {
        "networks": "networks.jsonl",
        "entities": "entities.jsonl",
        "network_edges": "network_edges.jsonl",
        "episodes": "episodes.jsonl",
        "episode_entities": "episode_entities.jsonl",
        "events": "events.jsonl",
        "relations": "relations.jsonl",
        "mechanisms": "mechanisms.jsonl",
        "texts": "texts.jsonl",
        "validation": "validation.jsonl",
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.handles: Dict[str, IO[str]] = {
            key: (self.output_dir / filename).open("w", encoding="utf-8", newline="\n")
            for key, filename in self.FILES.items()
        }

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()

    def __enter__(self) -> "EventFlowWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def write_network(self, network: Any) -> None:
        self._write("networks", network.network_record())
        for entity in network.entities:
            self._write("entities", entity.to_dict())
        for edge in network.edges:
            self._write("network_edges", edge.to_dict())

    def write_episode(self, bundle: EpisodeBundle) -> None:
        self._write("episodes", bundle.episode_record())
        self._write(
            "episode_entities",
            {
                "episode_id": bundle.episode_id,
                "network_id": bundle.network_id,
                "entity_ids": bundle.involved_entity_ids,
            },
        )
        for event in bundle.events:
            self._write("events", event.to_dict())
        for relation in bundle.relations:
            self._write("relations", relation.to_dict())
        for mechanism in bundle.mechanisms:
            self._write("mechanisms", mechanism.to_dict())
        for text in bundle.texts:
            self._write("texts", text.to_dict())
        self._write(
            "validation",
            {"episode_id": bundle.episode_id, **bundle.validation},
        )

    def write_json(self, filename: str, payload: Dict[str, Any]) -> None:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _write(self, stream: str, payload: Dict[str, Any]) -> None:
        self.handles[stream].write(json.dumps(payload, ensure_ascii=False) + "\n")
