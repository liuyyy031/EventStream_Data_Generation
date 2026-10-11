"""Domain-owned explicit reporting taxonomy, not event-generation rules."""
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

CATEGORIES = {"physical_process", "observation", "management_action", "derived_assessment"}


@lru_cache(maxsize=1)
def _profiles():
    return json.loads(Path(__file__).with_name("coverage_profiles.json").read_text(encoding="utf-8"))


def get_coverage_profile(domain):
    raw = deepcopy(_profiles().get(domain, {}))
    categories = {}
    for category, suffixes in raw.get("event_category_groups", {}).items():
        if category not in CATEGORIES:
            raise ValueError(f"Unknown coverage category {category}")
        for suffix in suffixes:
            event_type = domain + "." + suffix
            if event_type in categories:
                raise ValueError(f"Duplicated coverage type {event_type}")
            categories[event_type] = category
    return {"schema": "domain-mechanism-coverage-profile-v1",
        "event_categories": categories,
        "recovery_or_clearing_event_types": [domain + "." + t for t in raw.get("recovery_or_clearing", [])],
        "report_sample_time_fields": raw.get("report_sample_time_fields", {}),
        "report_scope_fields": raw.get("report_scope_fields", {}),
        "source": "domain_packages/coverage_profiles.json"}
