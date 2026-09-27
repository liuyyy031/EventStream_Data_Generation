from __future__ import annotations

import csv
import json
import sys
import unittest
import uuid
from pathlib import Path


DATA_GENERATION = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = DATA_GENERATION.parent
TEST_OUTPUT_ROOT = DATA_GENERATION / "artifacts" / "test_runtime_output"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_generation.generation_core.reference_data import (  # noqa: E402
    NormalizedJsonlReferenceAdapter,
)
from data_generation.generation_core.tabular_reference_adapter import (  # noqa: E402
    GenericTabularReferenceAdapter,
    write_normalized_reference_dataset,
)


class GenericTabularReferenceAdapterTests(unittest.TestCase):
    def test_csv_is_normalized_and_round_trips_through_audit_snapshot(self) -> None:
        root = self._test_directory()
        with self.subTest(source="csv"):
            source = root / "events.csv"
            fieldnames = [
                "encounter_id",
                "patient_id",
                "event_id",
                "event_time",
                "observed_time",
                "recorded_time",
                "event_type",
                "parent_event_id",
                "trace_id",
                "split",
                "care_setting",
            ]
            rows = []
            for index in range(6):
                split = "train" if index < 4 else ("val" if index == 4 else "test")
                rows.extend(
                    [
                        {
                            "encounter_id": f"enc-{index}",
                            "patient_id": f"patient-{index}",
                            "event_id": "e1",
                            "event_time": f"2026-01-0{index + 1} 08:00:00",
                            "observed_time": f"2026-01-0{index + 1} 08:00:01",
                            "recorded_time": f"2026-01-0{index + 1} 08:00:02",
                            "event_type": "COLLECTION",
                            "parent_event_id": "",
                            "trace_id": f"trace-{index}",
                            "split": split,
                            "care_setting": "outpatient",
                        },
                        {
                            "encounter_id": f"enc-{index}",
                            "patient_id": f"patient-{index}",
                            "event_id": "e2",
                            "event_time": f"2026-01-0{index + 1} 08:05:00",
                            "observed_time": f"2026-01-0{index + 1} 08:05:01",
                            "recorded_time": f"2026-01-0{index + 1} 08:05:02",
                            "event_type": "RESULT",
                            "parent_event_id": "e1",
                            "trace_id": f"trace-{index}",
                            "split": split,
                            "care_setting": "outpatient",
                        },
                    ]
                )
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            adapter = GenericTabularReferenceAdapter(self._mapping(include_split=True))
            dataset = adapter.load(source)
            self.assertEqual(len(dataset.windows), 6)
            self.assertEqual(len(dataset.events), 12)
            self.assertEqual(
                {item.split for item in dataset.windows},
                {"train", "validation", "holdout"},
            )
            child = next(item for item in dataset.events if item.source_record_id == "enc-0::e2")
            self.assertEqual(child.explicit_parent_record_ids, ["enc-0::e1"])
            self.assertEqual(child.occurrence_offset_seconds, 300.0)
            self.assertEqual(child.observed_at_offset_seconds, 301.0)
            self.assertEqual(adapter.last_report["explicit_parent_link_count"], 6)
            self.assertEqual(adapter.last_report["adjacent_events_promoted_to_parents"], 0)

            snapshot = root / "normalized"
            write_normalized_reference_dataset(
                dataset, snapshot, normalization_report=adapter.last_report
            )
            reloaded = NormalizedJsonlReferenceAdapter().load(snapshot)
            self.assertEqual(reloaded.fingerprint(), dataset.fingerprint())
            self.assertTrue((snapshot / "normalization_report.json").is_file())

    def test_jsonl_uses_deterministic_context_level_splits(self) -> None:
        root = self._test_directory()
        with self.subTest(source="jsonl"):
            source = root / "events.jsonl"
            records = []
            for index in range(12):
                records.append(
                    {
                        "encounter_id": f"enc-{index}",
                        "patient_id": f"patient-{index}",
                        "event_id": "only",
                        "event_time": f"2026-02-{index + 1:02d}T08:00:00+00:00",
                        "event_type": "COLLECTION",
                    }
                )
            source.write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            mapping = self._mapping(include_split=False)
            mapping["format"] = "jsonl"
            first = GenericTabularReferenceAdapter(mapping).load(source)
            second = GenericTabularReferenceAdapter(mapping).load(source)
            first_splits = {item.context_id: item.split for item in first.windows}
            second_splits = {item.context_id: item.split for item in second.windows}
            self.assertEqual(first_splits, second_splits)
            self.assertEqual(set(first_splits.values()), {"train", "validation", "holdout"})
            self.assertTrue(all(not item.explicit_parent_record_ids for item in first.events))

    def test_unknown_mapped_event_type_fails_closed(self) -> None:
        root = self._test_directory()
        with self.subTest(source="unknown_event_type"):
            source = root / "events.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "encounter_id": "enc-1",
                        "patient_id": "patient-1",
                        "event_id": "e1",
                        "event_time": "2026-01-01T08:00:00+00:00",
                        "event_type": "UNKNOWN",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            mapping = self._mapping(include_split=False)
            mapping["format"] = "jsonl"
            with self.assertRaisesRegex(ValueError, "absent from event_type_mapping"):
                GenericTabularReferenceAdapter(mapping).load(source)

    @staticmethod
    def _mapping(*, include_split: bool) -> dict:
        columns = {
            "stream_id": "encounter_id",
            "context_id": "patient_id",
            "timestamp": "event_time",
            "observed_timestamp": "observed_time",
            "recorded_timestamp": "recorded_time",
            "event_type": "event_type",
            "event_id": "event_id",
            "entity_id": "patient_id",
            "parent_event_id": "parent_event_id",
            "trace_id": "trace_id",
        }
        if include_split:
            columns["split"] = "split"
        return {
            "schema_version": "generic-tabular-reference-mapping-v1",
            "domain": "healthcare",
            "source_dataset_id": "unit-test-dataset",
            "source_version": "1",
            "format": "csv",
            "timezone": "Asia/Shanghai",
            "columns": columns,
            "event_type_mapping": {
                "COLLECTION": "healthcare.specimen.collection.completed",
                "RESULT": "healthcare.observation.laboratory.result_recorded",
            },
            "unknown_event_type_policy": "error",
            "context_state_columns": ["care_setting"],
            "split": {
                "seed": 19,
                "fractions": {"train": 0.8, "validation": 0.1, "holdout": 0.1},
            },
        }

    @staticmethod
    def _test_directory() -> Path:
        root = TEST_OUTPUT_ROOT / f"tabular_adapter_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        return root


if __name__ == "__main__":
    unittest.main()
