"""Server/local CLI for evaluation-only cross-domain acceptance challenges."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_generation.domain_packages.registry import DOMAIN_BUILDERS
from data_generation.domain_packages.coverage_profiles import get_coverage_profile
from data_generation.generation_core.mechanism_coverage import MechanismCoverageAccumulator
from data_generation.evaluation.domain_acceptance import build_report, evaluate_episode, summarize_domain
from data_generation.generation_core.claim_contract import build_claim_contract
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.semantic_correction import LLMSemanticCorrector
from data_generation.generation_core.semantic_judge import LLMSemanticJudge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", default="all", help="all or comma-separated registered domains")
    parser.add_argument("--episodes-per-domain", type=int, default=3)
    parser.add_argument("--nodes-per-context", type=int, default=1000)
    parser.add_argument("--max-events-per-episode", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--judge-mode", choices=("none", "llm"), default="none")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-workers", type=int, default=1, help="Parallel domains; episodes within each domain remain sequential")
    parser.add_argument("--judge-max-tokens", type=int, default=3000)
    parser.add_argument("--correction-max-tokens", type=int, default=4000)
    parser.add_argument("--protocol-max-attempts", type=int, default=3)
    parser.add_argument("--output-dir", default="data_generation/artifacts/domain_acceptance")
    args = parser.parse_args()
    domains = list(DOMAIN_BUILDERS) if args.domains == "all" else args.domains.split(",")
    if not domains or len(set(domains)) != len(domains) or any(d not in DOMAIN_BUILDERS for d in domains):
        parser.error("--domains must contain distinct registered domain IDs")
    if min(args.episodes_per_domain, args.nodes_per_context, args.max_events_per_episode,
           args.judge_workers, args.judge_max_tokens, args.correction_max_tokens, args.protocol_max_attempts) <= 0:
        parser.error("Counts and limits must be positive")
    if args.judge_mode == "llm" and not args.judge_model:
        parser.error("--judge-model is required for LLM evaluation")
    output = Path(args.output_dir)
    if output.exists():
        output = output.with_name(output.name + datetime.now().strftime("_%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)

    def run_domain(domain):
        config_path = REPOSITORY_ROOT / "data_generation" / "domain_packages" / domain / "config" / "default.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        domain_config = dict(config["domain_config"])
        domain_config.update(nodes_per_context=args.nodes_per_context, base_seed=args.seed)
        package_type, build_models = DOMAIN_BUILDERS[domain]
        package = package_type(domain_config)
        catalog = package.catalog()
        catalog["coverage_profile"] = get_coverage_profile(domain)
        coverage = MechanismCoverageAccumulator(catalog)
        engine = SimulationEngine(package, build_models(), max_events=args.max_events_per_episode,
            time_axis=config.get("time_axis"), observation_duration=config.get("observation_duration"))
        judge = corrector = None
        if args.judge_mode == "llm":
            # Existing client loads the server .env; no manual export required.
            from data_generation.llm_client import LLMClient
            client = LLMClient(model=args.judge_model, base_url=args.judge_base_url)
            judge = LLMSemanticJudge(client, max_tokens=args.judge_max_tokens,
                protocol_max_attempts=args.protocol_max_attempts)
            corrector = LLMSemanticCorrector(client, max_tokens=args.correction_max_tokens,
                protocol_max_attempts=args.protocol_max_attempts)
        records = []
        for i in range(args.episodes_per_domain):
            seed = args.seed + i * 104729
            result = engine.run(i, seed)
            record = evaluate_episode(package, result, judge=judge, corrector=corrector)
            if record["baseline_validation"]["passed"]:
                coverage.add_result(result)
            record.update(seed=seed, evaluation_only=True, must_not_be_used_for_training=True)
            records.append(record)
        coverage_report = coverage.report()
        coverage_report["scope"] = "deterministically_valid_evaluation_baselines_before_semantic_review"
        return records, summarize_domain(domain, records, requested_count=args.episodes_per_domain,
            semantic_mode=args.judge_mode), build_claim_contract(package.catalog()), coverage_report

    completed = {}
    with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
        futures = {executor.submit(run_domain, d): d for d in domains}
        for future in as_completed(futures):
            domain = futures[future]
            completed[domain] = future.result()
            print(f"Completed {domain}", file=sys.stderr, flush=True)
    records = [r for d in domains for r in completed[d][0]]
    report = build_report([completed[d][1] for d in domains], semantic_mode=args.judge_mode)
    report["configuration"] = {**vars(args), "domains": domains, "output_dir": str(output.resolve())}
    report["claim_contracts"] = {d: completed[d][2] for d in domains}
    coverage_dir = output / "mechanism_coverage"
    coverage_dir.mkdir()
    for domain in domains:
        (coverage_dir / (domain + ".json")).write_text(
            json.dumps(completed[domain][3], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["mechanism_coverage_reports"] = {d: "mechanism_coverage/" + d + ".json" for d in domains}
    (output / "acceptance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "acceptance_cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(output.resolve()), "deterministic_passed": report["deterministic_passed"],
        "semantic_status": report["semantic_status"], "passed": report["passed"],
        "domains": report["domains"]}, ensure_ascii=False, indent=2))
    # Exit 2 means a failed executed gate, not an unrequested semantic gate.
    if not report["deterministic_passed"] or report["passed"] is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
