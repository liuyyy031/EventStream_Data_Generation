"""Normalized JSON/JSONL writer for generated datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, IO

from .models import EpisodeResult
from .temporal import TemporalModelRegistry


class DatasetWriter:
    FILES = {
        "contexts": "contexts.jsonl",
        "entities": "entities.jsonl",
        "context_relations": "context_relations.jsonl",
        "episodes": "episodes.jsonl",
        "events": "events.jsonl",
        "event_relations": "event_relations.jsonl",
        "candidates": "candidates.jsonl",
        "episode_texts": "episode_texts.jsonl",
        "validation": "validation.jsonl",
    }

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.handles: Dict[str, IO[str]] = {
            key: (self.output_dir / filename).open("w", encoding="utf-8", newline="\n")
            for key, filename in self.FILES.items()
        }
        self._written_contexts: set[str] = set()
        self._written_entities: set[str] = set()
        self._written_context_relations: set[str] = set()

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self.handles.values():
            handle.close()

    def write_manifest(
        self,
        payload: Dict[str, Any],
        temporal_models: TemporalModelRegistry,
        domain_catalog: Dict[str, Any],
    ) -> None:
        self.write_json("manifest.json", payload)
        self.write_json("domain_catalog.json", domain_catalog)
        self.write_json(
            "temporal_model_registry.json", {"models": temporal_models.descriptors()}
        )

    def write_episode(self, result: EpisodeResult) -> None:
        if result.context_id not in self._written_contexts:
            self._write(
                "contexts",
                {
                    "context_id": result.context_id,
                    "domain": result.domain,
                    "entity_count": len(result.entities),
                    "context_relation_count": len(result.context_relations),
                },
            )
            self._written_contexts.add(result.context_id)
        for entity in result.entities:
            if entity.entity_id not in self._written_entities:
                self._write("entities", entity.to_dict())
                self._written_entities.add(entity.entity_id)
        for relation in result.context_relations:
            if relation.relation_id not in self._written_context_relations:
                self._write("context_relations", relation.to_dict())
                self._written_context_relations.add(relation.relation_id)
        self._write("episodes", result.episode_record())
        for event in result.events:
            self._write("events", event.to_dict())
        for relation in result.event_relations:
            self._write("event_relations", relation.to_dict())
        for candidate in result.candidates:
            self._write("candidates", candidate.to_dict())
        self._write(
            "validation", {"episode_id": result.episode_id, **result.validation}
        )

    def write_episode_text(self, payload: Dict[str, Any]) -> None:
        self._write("episode_texts", payload)

    def write_json(self, filename: str, payload: Dict[str, Any]) -> None:
        with (self.output_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _write(self, stream: str, payload: Dict[str, Any]) -> None:
        self.handles[stream].write(json.dumps(payload, ensure_ascii=False) + "\n")
