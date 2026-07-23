#!/usr/bin/env python3
"""STPP v17 role-routed generation with two fail-closed Judges."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from demo_sts_sde import AgentInteractionLogger
from demo_sts_stpp_v2 import build_graph_metadata_v2
from demo_sts_stpp_v6 import repair_and_report_edge_lags_v6
from demo_sts_stpp_v8 import _canonicalise_agent2_json_v8
from demo_sts_stpp_v12 import _judge_contract_consistent
from demo_sts_stpp_v16 import parse_scenario_with_judge_audit_v16
from llm_client import LLMClient
from safe_judge_generator_v17 import RoleRoutedJudgeNetworkSTPPGenerator
from stpp_adapter_v13 import canonicalise_spatial_layout_v13
from stpp_adapter_v14 import canonicalise_temporal_contract_v14
from stpp_adapter_v16 import (
    STPPV16Config,
    audit_structured_contract_v16,
    simulate_stpp_for_streasoner_v16,
)


@dataclass(frozen=True)
class STPPV17Config(STPPV16Config):
    """v17 keeps direct-route sampling and raises only the thinning ceiling."""

    intensity_upper_bound: float = 192.0
    max_raw_edge_revisions: int = 4


_HEADING_AFTER_EDGES = re.compile(
    r"(?im)^\s*(?:TEMPORAL\s+PATTERNS|PROPAGATED\s+ARRIVALS|"
    r"EDGE\s+MODULATION|SPATIAL\s+LAYOUT)\s*:"
)
_ARROW_PAIR = re.compile(
    r"(?:NODE\s*)?(\d+)\s*(?:->|→)\s*(?:NODE\s*)?(\d+)",
    re.IGNORECASE,
)


def _pairs_from_chain(text: str) -> List[Tuple[int, int]]:
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    return list(zip(numbers, numbers[1:]))


def _explicit_edge_section(scenario: str) -> str:
    match = re.search(r"(?im)^\s*EDGES\s*:\s*", scenario)
    if not match:
        return ""
    remainder = scenario[match.end() :]
    following = _HEADING_AFTER_EDGES.search(remainder)
    return remainder[: following.start()] if following else remainder


def _raw_full_path_edges(scenario: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for line in scenario.splitlines():
        if re.search(r"EVENT\s+FULL\s+PATH", line, re.IGNORECASE):
            after_colon = line.split(":", 1)[-1]
            pairs.extend(_pairs_from_chain(after_colon))
    return pairs


def _json_edges(
    structured_scenario: Mapping[str, Any] | None,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    if structured_scenario is None:
        return [], []
    pairs: List[Tuple[int, int]] = []
    issues: List[Dict[str, Any]] = []
    for index, edge in enumerate(structured_scenario.get("edges", []) or []):
        try:
            pairs.append((int(edge["source"]), int(edge["target"])))
        except (KeyError, TypeError, ValueError):
            issues.append(
                {
                    "field": f"edges[{index}]",
                    "problem": "JSON edge lacks integer source/target",
                    "error_source": "agent2",
                }
            )
    return pairs, issues


def audit_raw_edge_fidelity_v17(
    scenario: str,
    structured_scenario: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compare explicit edges, event paths and optionally Agent 2 JSON edges."""
    explicit_list = [
        (int(source), int(target))
        for source, target in _ARROW_PAIR.findall(_explicit_edge_section(scenario))
    ]
    path_list = _raw_full_path_edges(scenario)
    parsed_list, issues = _json_edges(structured_scenario)
    explicit = set(explicit_list)
    path_edges = set(path_list)
    parsed = set(parsed_list)

    if not explicit:
        issues.append(
            {
                "field": "EDGES",
                "problem": "No explicit top-level edge could be extracted",
                "error_source": "agent1",
            }
        )
    for edge in sorted(path_edges - explicit):
        issues.append(
            {
                "field": f"EVENT FULL PATH edge {edge[0]}->{edge[1]}",
                "problem": "Path edge is absent from the top-level EDGES section",
                "error_source": "agent1",
            }
        )
    duplicate_explicit = sorted(
        edge for edge, count in Counter(explicit_list).items() if count > 1
    )
    for edge in duplicate_explicit:
        issues.append(
            {
                "field": f"EDGES {edge[0]}->{edge[1]}",
                "problem": "Top-level edge is declared more than once",
                "error_source": "agent1",
            }
        )

    if structured_scenario is not None:
        for edge in sorted(explicit - parsed):
            issues.append(
                {
                    "field": f"JSON edges {edge[0]}->{edge[1]}",
                    "problem": "Agent 2 omitted an explicit top-level edge",
                    "error_source": "agent2",
                }
            )
        for edge in sorted(parsed - explicit):
            issues.append(
                {
                    "field": f"JSON edges {edge[0]}->{edge[1]}",
                    "problem": "Agent 2 added an edge absent from top-level EDGES",
                    "error_source": "agent2",
                }
            )
        duplicate_parsed = sorted(
            edge for edge, count in Counter(parsed_list).items() if count > 1
        )
        for edge in duplicate_parsed:
            issues.append(
                {
                    "field": f"JSON edges {edge[0]}->{edge[1]}",
                    "problem": "Agent 2 emitted a duplicate edge",
                    "error_source": "agent2",
                }
            )

    return {
        "passed": not issues,
        "explicit_edges": [list(edge) for edge in sorted(explicit)],
        "event_full_path_edges": [list(edge) for edge in sorted(path_edges)],
        "agent2_json_edges": [list(edge) for edge in sorted(parsed)],
        "issues": issues,
    }


def _edge_feedback(audit: Mapping[str, Any]) -> str:
    lines = [
        "Revise the same scenario. Keep valid nodes, values, schedules and "
        "windows unchanged, but correct the explicit-edge contract."
    ]
    for issue in audit.get("issues", []) or []:
        lines.append(f"- {issue.get('field')}: {issue.get('problem')}")
    lines.append(
        "Every consecutive pair in EVENT FULL PATH must be present in the "
        "top-level EDGES section. Return only the revised scenario."
    )
    return "\n".join(lines)


def _repair_raw_agent1_edges(
    generator: RoleRoutedJudgeNetworkSTPPGenerator,
    scenario: str,
    max_revisions: int,
) -> Tuple[str, Dict[str, Any], int]:
    current = scenario
    for revision in range(max_revisions + 1):
        audit = audit_raw_edge_fidelity_v17(current)
        if audit["passed"]:
            return current, audit, revision
        if revision >= max_revisions:
            return current, audit, revision
        print("\n=== STPP v17 deterministic raw-edge revision ===")
        for issue in audit["issues"]:
            print(f"  - {issue['field']}: {issue['problem']}")
        current = generator.generate_scenario_description(
            previous_scenario=current,
            previous_feedback=_edge_feedback(audit),
            iteration=revision + 2,
        )
    raise AssertionError("unreachable")


def _event_summary(events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(events)
    type_counts = Counter(str(event.get("event_type")) for event in rows)
    node_counts = Counter(str(event.get("node_id")) for event in rows)
    generation_counts = Counter(str(event.get("generation")) for event in rows)
    scenario_ids = Counter(str(event.get("scenario_event_id")) for event in rows)
    schedule_ids = Counter(str(event.get("schedule_id")) for event in rows)
    missing_semantic_ids = sum(
        not event.get("scenario_event_id") or not event.get("schedule_id")
        for event in rows
    )
    return {
        "num_events": len(rows),
        "event_type_counts": dict(sorted(type_counts.items())),
        "node_counts": dict(sorted(node_counts.items())),
        "generation_counts": dict(sorted(generation_counts.items())),
        "scenario_event_id_counts": dict(sorted(scenario_ids.items())),
        "schedule_id_counts": dict(sorted(schedule_ids.items())),
        "missing_semantic_id_count": int(missing_semantic_ids),
    }


def _matrix_summary(ts_data: np.ndarray) -> Dict[str, Any]:
    per_node = []
    for node_id, values in enumerate(ts_data):
        per_node.append(
            {
                "node_id": node_id,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "sum": float(np.sum(values)),
                "nonzero_bins": int(np.count_nonzero(values)),
            }
        )
    return {
        "shape": list(ts_data.shape),
        "contains_nan": bool(np.isnan(ts_data).any()),
        "contains_infinite": bool(np.isinf(ts_data).any()),
        "per_node": per_node,
    }


def _write_outputs_v17(
    scenario: str,
    structured_scenario: Dict[str, Any],
    graph_metadata: Dict[str, Any],
    ts_data: np.ndarray,
    events: List[Dict[str, Any]],
    generation_info: Dict[str, Any],
    domain: str,
    output_dir: Path,
) -> Dict[str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    domain_clean = re.sub(r"[^A-Za-z0-9_-]+", "_", domain).strip("_")
    prefix = f"{domain_clean}_node{ts_data.shape[0]}_stpp_v17_{timestamp}"
    complete_data = {
        "timestamp": timestamp,
        "generator_family": "role_routed_two_judge_direct_route_stpp_v17",
        "agent1_scenario": scenario,
        "agent2_structured_scenario": structured_scenario,
        "agent3_point_process_parameters": generation_info[
            "point_process_parameters"
        ],
        "agent4_time_varying_adjacency": graph_metadata,
        "agent5_event_stream": events,
        "agent5_simulation_data": ts_data,
        "generation_info": generation_info,
        "seq_len": int(ts_data.shape[1]),
    }
    pickle_path = output_dir / f"{prefix}_data.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(complete_data, handle)
    json_path = output_dir / f"{prefix}_data.json"
    json_data = dict(complete_data)
    json_data["agent5_simulation_data"] = ts_data.tolist()
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2, ensure_ascii=False)

    description_path = output_dir / f"{prefix}_results.txt"
    quality = generation_info["quality_report"]
    judge1 = generation_info["judge1_validation"]
    judge2 = generation_info["judge2_validation"]
    raw_edge = generation_info["raw_edge_fidelity_audit"]
    route = generation_info["declared_route_audit"]
    with description_path.open("w", encoding="utf-8") as handle:
        handle.write("=== STReasoner Role-Routed Two-Judge STPP v17 ===\n\n")
        handle.write(f"Generation time: {dt.datetime.now().isoformat()}\n")
        handle.write(f"Domain: {domain}\n")
        handle.write(f"Nodes: {ts_data.shape[0]}\n")
        handle.write(f"Time windows: {ts_data.shape[1]}\n")
        handle.write(f"All events: {generation_info['num_events']}\n")
        handle.write(f"Root events: {generation_info['num_root_events']}\n")
        handle.write(
            f"Propagated events: {generation_info['num_propagated_events']}\n"
        )
        handle.write(
            "Role models: "
            + json.dumps(generation_info["role_models"], ensure_ascii=False)
            + "\n"
        )
        handle.write(f"Judge 1 approved: {judge1['approved']}\n")
        handle.write(f"Judge 2 approved: {judge2['approved']}\n")
        handle.write(f"Raw edge fidelity passed: {raw_edge['passed']}\n")
        handle.write(f"Quality gates passed: {quality['passed']}\n")
        handle.write(
            "Final route violations: "
            f"{len(route['propagated_event_violations'])}\n\n"
        )
        for title, value in (
            ("Raw edge fidelity audit", raw_edge),
            ("Quality report", quality),
            ("Judge 1 audit", judge1),
            ("Judge 2 audit", judge2),
            ("Declared route audit", route),
            ("Event stream summary", generation_info["event_stream_summary"]),
            ("Matrix summary", generation_info["matrix_summary"]),
        ):
            handle.write(f"{title}\n{'-' * len(title)}\n")
            handle.write(json.dumps(value, indent=2, ensure_ascii=False))
            handle.write("\n\n")
        handle.write("Scenario\n--------\n")
        handle.write(scenario)
        handle.write("\n")
    return {
        "pickle": str(pickle_path),
        "json": str(json_path),
        "description": str(description_path),
        "visualization": None,
    }


def _make_client(model: str, timeout: int) -> LLMClient:
    return LLMClient(model=model, timeout=timeout)


def demo_stpp_generation_v17(
    enabled_judges: List[int] | None = None,
    enable_logging: bool = True,
    num_nodes: int = 3,
    domain: str = "traffic",
    stpp_config: STPPV17Config | None = None,
    output_dir: str = "output_stpp_v17",
    allow_unapproved_parse: bool = False,
    agent1_model: str = "deepseek-v4-pro",
    agent2_model: str = "deepseek-v4-pro",
    judge1_model: str = "deepseek-v4-flash",
    judge2_model: str = "deepseek-v4-pro",
    llm_timeout: int = 300,
) -> Dict[str, Any]:
    enabled_judges = [1, 2] if enabled_judges is None else enabled_judges
    config = stpp_config or STPPV17Config()
    logger = AgentInteractionLogger() if enable_logging else None
    generator = RoleRoutedJudgeNetworkSTPPGenerator(
        num_nodes=num_nodes,
        logger=logger,
        agent1_client=_make_client(agent1_model, llm_timeout),
        agent2_client=_make_client(agent2_model, llm_timeout),
        judge1_client=_make_client(judge1_model, llm_timeout),
        judge2_client=_make_client(judge2_model, llm_timeout),
    )
    generator.domain = domain
    scenario, seq_len = generator.generate_scenario_with_length_validation()
    scenario, raw_agent1_audit, raw_revisions = _repair_raw_agent1_edges(
        generator, scenario, config.max_raw_edge_revisions
    )
    if not raw_agent1_audit["passed"] and not allow_unapproved_parse:
        if logger:
            logger.save_complete_log()
        raise RuntimeError(
            "v17 fail-closed: Agent 1 raw edge/path contract failed; no STPP "
            f"data files were written. issues={raw_agent1_audit['issues']}"
        )

    final_time_info = generator._extract_time_info_from_scenario(scenario)  # noqa: SLF001
    seq_len = int(final_time_info.get("calculated_seq_len") or seq_len)
    generator.seq_len = seq_len
    if 1 in enabled_judges:
        scenario, parsed, judge1_audit, agent2_audit, pre_agent_audit = (
            parse_scenario_with_judge_audit_v16(
                generator,
                scenario,
                seq_len,
                repeated_causal_failures_before_revision=(
                    config.deterministic_agent2_failures_before_agent1_revision
                ),
            )
        )
    else:
        raw = generator.parse_scenario_to_structured_json(scenario)
        parsed, agent2_audit = _canonicalise_agent2_json_v8(raw)
        pre_agent_audit = audit_structured_contract_v16(parsed, seq_len)
        judge1_audit = {
            "required": False,
            "approved": None,
            "attempts": 0,
            "used_unapproved_parse": False,
            "agent1_revisions": 0,
            "agent2_attempts_per_scenario": [],
            "termination_reason": "judge_not_required",
            "last_judgment": None,
        }

    final_time_info = generator._extract_time_info_from_scenario(scenario)  # noqa: SLF001
    seq_len = int(final_time_info.get("calculated_seq_len") or seq_len)
    generator.seq_len = seq_len
    raw_edge_audit = audit_raw_edge_fidelity_v17(scenario, parsed)
    parse_failed = (
        (judge1_audit["required"] and judge1_audit["approved"] is not True)
        or not pre_agent_audit["passed"]
        or not raw_edge_audit["passed"]
    )
    if parse_failed and not allow_unapproved_parse:
        if logger:
            logger.save_complete_log()
        raise RuntimeError(
            "v17 fail-closed: Agent/Judge parsing or raw-edge fidelity failed; "
            "no STPP data files were written. "
            f"judge1={judge1_audit.get('approved')}; "
            f"contract={pre_agent_audit.get('passed')}; "
            f"raw_edge_issues={raw_edge_audit['issues']}"
        )

    structured_scenario, temporal_audit = canonicalise_temporal_contract_v14(
        parsed, seq_len
    )
    structured_scenario, spatial_audit = canonicalise_spatial_layout_v13(
        structured_scenario
    )
    final_pre_audit = audit_structured_contract_v16(structured_scenario, seq_len)
    final_raw_edge_audit = audit_raw_edge_fidelity_v17(
        scenario, structured_scenario
    )
    if (
        not final_pre_audit["passed"] or not final_raw_edge_audit["passed"]
    ) and not allow_unapproved_parse:
        if logger:
            logger.save_complete_log()
        raise RuntimeError(
            "v17 fail-closed: final deterministic pre-simulation audit failed; "
            f"contract_issues={final_pre_audit['blocking_issues']}; "
            f"edge_issues={final_raw_edge_audit['issues']}"
        )

    sampling_minutes = float(getattr(generator, "sampling_minutes", 60))
    lag_repair = repair_and_report_edge_lags_v6(
        scenario,
        structured_scenario,
        sampling_minutes,
        config.immediate_lag_epsilon_steps,
    )
    graph_metadata = build_graph_metadata_v2(structured_scenario, seq_len)
    ts_data, events, generation_info = simulate_stpp_for_streasoner_v16(
        structured_scenario, seq_len, config
    )
    quality = generation_info["quality_report"]
    parsed_seq_len = int(structured_scenario.get("seq_len", seq_len))
    quality["checks"].update(
        {
            "judge1_approved_or_not_required": (
                not judge1_audit["required"]
                or judge1_audit["approved"] is True
            ),
            "judge1_response_contract_consistent": _judge_contract_consistent(
                judge1_audit
            ),
            "agent2_seq_len_matches_generator": parsed_seq_len == seq_len,
            "agent2_canonicalisation_valid": not agent2_audit.get(
                "invalid_changes", []
            ),
            "temporal_contract_valid": temporal_audit["passed"],
            "temporal_contract_repair_free": temporal_audit["repair_free"],
            "spatial_layout_has_separated_nodes": spatial_audit["final_valid"],
            "spatial_layout_repair_free": not spatial_audit["repaired"],
            "raw_explicit_edges_match_paths_and_json": final_raw_edge_audit[
                "passed"
            ],
        }
    )
    quality["passed"] = all(quality["checks"].values())

    event_summary = _event_summary(events)
    matrix_summary = _matrix_summary(ts_data)
    deterministic_evidence = {
        "all_required_checks_passed": bool(quality["passed"]),
        "quality_checks": quality["checks"],
        "raw_edge_fidelity_audit": final_raw_edge_audit,
        "pre_simulation_contract_audit": final_pre_audit,
        "declared_route_audit": generation_info["declared_route_audit"],
        "temporal_contract_audit": temporal_audit,
        "judge1_validation": judge1_audit,
    }
    if 2 in enabled_judges:
        judge2_approved, judge2_audit = generator.judge_stpp_data(
            scenario,
            structured_scenario,
            deterministic_evidence,
            event_summary,
            matrix_summary,
        )
        judge2_audit["required"] = True
    else:
        judge2_approved = True
        judge2_audit = {
            "required": False,
            "approved": None,
            "error_source": "none",
            "issues": [],
            "confidence": "not_run",
            "overall_comment": "Judge 2 not requested",
            "judge_contract_consistent": True,
        }

    quality["checks"]["judge2_approved_or_not_required"] = bool(
        judge2_approved
    )
    quality["checks"]["judge2_response_contract_consistent"] = bool(
        judge2_audit.get("judge_contract_consistent") is True
    )
    quality["passed"] = all(quality["checks"].values())
    generation_info.update(
        {
            "judge1_validation": judge1_audit,
            "judge2_validation": judge2_audit,
            "role_models": generator.role_model_names,
            "agent1_raw_edge_revisions": raw_revisions,
            "agent2_canonicalisation": agent2_audit,
            "pre_generation_agent_contract_audit": pre_agent_audit,
            "raw_edge_fidelity_audit": final_raw_edge_audit,
            "temporal_contract_audit": temporal_audit,
            "spatial_layout_audit": spatial_audit,
            "sampling_minutes": sampling_minutes,
            "dt": sampling_minutes / 60.0,
            "lag_repair": lag_repair,
            "event_stream_summary": event_summary,
            "matrix_summary": matrix_summary,
        }
    )

    if 2 in enabled_judges and not judge2_approved:
        if logger:
            logger.save_complete_log()
        raise RuntimeError(
            "v17 fail-closed: Judge 2 rejected the final STPP sample; no data "
            f"files were written. error_source={judge2_audit['error_source']}; "
            f"issues={judge2_audit['issues']}"
        )

    data_files = _write_outputs_v17(
        scenario,
        structured_scenario,
        graph_metadata,
        ts_data,
        events,
        generation_info,
        domain,
        Path(output_dir),
    )
    if logger:
        logger.save_complete_log()
    return {
        "generator_family": "role_routed_two_judge_direct_route_stpp_v17",
        "agent1_scenario": scenario,
        "agent2_structured_scenario": structured_scenario,
        "agent5_event_stream": events,
        "agent5_simulation_data": ts_data,
        "generation_info": generation_info,
        "data_files": data_files,
        "logger": logger,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_nodes", type=int, default=3)
    parser.add_argument("--domain", default="traffic")
    parser.add_argument("--judges", default="1,2")
    parser.add_argument("--output_dir", default="output_stpp_v17")
    parser.add_argument(
        "--aggregation",
        choices=["count", "magnitude", "rolling_count", "rolling_magnitude"],
        default="magnitude",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--edge_branching_ratio", type=float, default=0.35)
    parser.add_argument("--target_immigrant_rate", type=float, default=0.30)
    parser.add_argument(
        "--agent1_model",
        default=os.environ.get("STPP_AGENT1_MODEL", "deepseek-v4-pro"),
    )
    parser.add_argument(
        "--agent2_model",
        default=os.environ.get("STPP_AGENT2_MODEL", "deepseek-v4-pro"),
    )
    parser.add_argument(
        "--judge1_model",
        default=os.environ.get("STPP_JUDGE1_MODEL", "deepseek-v4-flash"),
    )
    parser.add_argument(
        "--judge2_model",
        default=os.environ.get("STPP_JUDGE2_MODEL", "deepseek-v4-pro"),
    )
    parser.add_argument(
        "--llm_timeout",
        type=int,
        default=300,
        help="per-request timeout in seconds; Pro reasoning may exceed 120s",
    )
    parser.add_argument(
        "--allow_unapproved_parse",
        action="store_true",
        help="diagnose parse failures; Judge 2 remains fail-closed",
    )
    args = parser.parse_args()
    judges = (
        []
        if args.judges.lower() == "none"
        else [int(item.strip()) for item in args.judges.split(",") if item.strip()]
    )
    unknown_judges = sorted(set(judges) - {1, 2})
    if unknown_judges:
        raise ValueError(f"Unsupported Judge IDs: {unknown_judges}; use 1,2 or none")
    config = STPPV17Config(
        aggregation=args.aggregation,
        seed=args.seed,
        edge_branching_ratio=args.edge_branching_ratio,
        target_immigrant_rate=args.target_immigrant_rate,
    )
    result = demo_stpp_generation_v17(
        enabled_judges=judges,
        num_nodes=args.num_nodes,
        domain=args.domain,
        stpp_config=config,
        output_dir=args.output_dir,
        allow_unapproved_parse=args.allow_unapproved_parse,
        agent1_model=args.agent1_model,
        agent2_model=args.agent2_model,
        judge1_model=args.judge1_model,
        judge2_model=args.judge2_model,
        llm_timeout=args.llm_timeout,
    )
    quality = result["generation_info"]["quality_report"]
    route = result["generation_info"]["declared_route_audit"]
    print(f"Generated {result['generation_info']['num_events']} events")
    print(f"Quality gates passed: {quality['passed']}")
    print(
        "Final declared-route violations: "
        f"{len(route['propagated_event_violations'])}"
    )
    print(f"Role models: {result['generation_info']['role_models']}")
    print(f"Pickle: {result['data_files']['pickle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
