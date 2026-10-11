"""Recompute coverage from structured exports without risk sets or any LLM API."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.generation_core.mechanism_coverage import MechanismCoverageAccumulator
from data_generation.domain_packages.coverage_profiles import get_coverage_profile


def records(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL: {path.name}:{line_number}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", help="New report path; never overwrites existing files")
    args = parser.parse_args()
    source = Path(args.dataset_dir)
    output = Path(args.output) if args.output else source / "mechanism_coverage_recomputed.json"
    if output.exists():
        parser.error("Report already exists; choose another --output path")
    filenames = ("domain_catalog.json", "episodes.jsonl", "entities.jsonl", "events.jsonl", "candidates.jsonl", "event_relations.jsonl")
    missing = [name for name in filenames if not (source / name).is_file()]
    if missing:
        parser.error(f"Missing required structured files: {missing}")
    catalog = json.loads((source / "domain_catalog.json").read_text(encoding="utf-8-sig"))
    snapshot = bool(catalog.get("coverage_profile"))
    if not snapshot:
        catalog["coverage_profile"] = get_coverage_profile(catalog["domain"])
    coverage = MechanismCoverageAccumulator(catalog)
    for filename, method in (("episodes.jsonl", coverage.add_episode), ("entities.jsonl", coverage.add_entity),
            ("events.jsonl", coverage.add_event), ("candidates.jsonl", coverage.add_candidate),
            ("event_relations.jsonl", coverage.add_relation)):
        for record in records(source / filename):
            method(record)
    report = coverage.report()
    report["recomputation"] = {"dataset_dir": str(source.resolve()), "llm_called": False,
        "category_profile_source": "dataset_catalog_snapshot" if snapshot else "current_local_domain_profile_fallback",
        "read_files": list(filenames), "risk_sets_required": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output.resolve()), "registered_mechanism_count": report["registered_mechanism_count"],
        "fired_mechanism_count": report["fired_mechanism_count"], "event_categories": report["event_categories"],
        "llm_called": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
