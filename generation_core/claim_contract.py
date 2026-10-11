"""One lossless claim boundary shared by semantic review and text correction."""

from __future__ import annotations

import copy
from typing import Any, Dict


def build_claim_contract(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Merge generic and legacy scopes; never silently erase a restriction.

    Declarations describe the implemented model, not the whole real domain.
    False-valued declarations must reach the LLM unchanged. Conflicting
    declarations are configuration errors rather than last-write-wins overrides.
    """
    scope: Dict[str, Any] = {}
    sources = []
    for key in ("scope", "clinical_safety_scope"):
        if key not in catalog:
            continue
        declaration = catalog[key]
        if not isinstance(declaration, dict):
            raise ValueError(f"{key} must be a claim-scope object")
        sources.append(key)
        for name, value in declaration.items():
            if name in scope and scope[name] != value:
                raise ValueError(f"Conflicting domain claim boundary: {name}")
            scope[name] = copy.deepcopy(value)
    return {
        "schema": "domain-claim-contract-v1",
        "domain": catalog.get("domain"),
        "declared_scope": scope,
        "scope_sources": sources,
        "fit_status": catalog.get("fit_status"),
        "text_projection": copy.deepcopy(catalog.get("text_projection", {})),
        "common_boundaries": {
            "scope_is_model_boundary_not_complete_domain_coverage": True,
            "undeclared_facts_must_not_be_invented": True,
            "statistical_realism_not_established_by_semantic_review": True,
            "context_adjacency_does_not_imply_event_parenthood": True,
            "statistical_influence_is_not_proven_causality": True,
            "scheduled_measurement_is_not_continuous_observation": True,
            "threshold_or_proxy_is_not_proof_of_unmodeled_outcome": True,
            "structured_records_are_immutable_during_text_correction": True,
        },
    }
