"""Generate independent stochastic suffixes from a shared in-memory prefix."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.domain_packages.registry import DOMAIN_BUILDERS
from data_generation.generation_core.scheduler import SimulationEngine
from data_generation.generation_core.writer import DatasetWriter
from data_generation.generation_core.validation import validate_text_alignment
from data_generation.generation_core.qa_export import generate_episode_qa, instruction_training_example, chat_training_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=tuple(DOMAIN_BUILDERS), default="healthcare")
    parser.add_argument("--scenario-family", help="Restrict to one scenario declared by the selected domain")
    parser.add_argument("--nodes-per-context", type=int, default=1000)
    parser.add_argument("--prefix-events", type=int, default=2)
    parser.add_argument("--continuation-count", type=int, default=5)
    parser.add_argument("--max-events-per-episode", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.continuation_count < 1:
        parser.error("--continuation-count must be positive")
    output = Path(args.output_dir)
    if output.exists():
        parser.error("Output directory already exists; choose a new directory")
    package_type, build_models = DOMAIN_BUILDERS[args.domain]
    domain_config = {"nodes_per_context": args.nodes_per_context, "base_seed": args.seed}
    if args.scenario_family:
        domain_config["scenario_weights"] = {args.scenario_family: 1.0}
    package = package_type(domain_config)
    models = build_models()
    engine = SimulationEngine(package, models, max_events=args.max_events_per_episode)
    snapshot = engine.run(0, args.seed, stop_after_events=args.prefix_events)
    branches = []
    for index in range(args.continuation_count):
        result = engine.run(0, args.seed, snapshot=snapshot, continuation_seed=args.seed + (index+1)*104729)
        text = package.render_episode_text(result)
        alignment = validate_text_alignment(result, text, package.catalog())
        branch_dir = output / f"continuation_{index:03d}"
        with DatasetWriter(branch_dir) as writer:
            writer.write_manifest({"schema_version": "conditional-continuation-v1", "domain": args.domain,
                "config": vars(args), "prefix_group_id": snapshot.state["prefix_group_id"],
                "continuation_seed": result.episode_attributes["continuation_seed"],
                "llm_called": False, "split_policy": "keep_entire_prefix_group_together"}, models, package.catalog())
            writer.write_episode(result)
            writer.write_episode_text(text)
            for qa in generate_episode_qa(result, text):
                qa["prefix_group_id"] = snapshot.state["prefix_group_id"]
                writer.write_qa_pair(qa)
                writer.write_training_instruction(instruction_training_example(qa))
                writer.write_training_chat(chat_training_example(qa))
        branches.append({"branch": branch_dir.name, "event_count": len(result.events), "validation": result.validation, "text_alignment": alignment})
    report = {"passed": all(b["validation"]["passed"] and b["text_alignment"]["passed"] for b in branches),
        "prefix_group_id": snapshot.state["prefix_group_id"], "prefix_event_count": args.prefix_events,
        "semantics": "conditional_stochastic_continuations_not_causal_counterfactuals",
        "split_policy": "all branches of this prefix must stay in one dataset split",
        "identity_scope": "IDs are local to each branch directory; namespace by branch when combining",
        "committed_clock_policy": "discrete/policy appointments are retained; continuous clocks are conditionally resampled",
        "llm_called": False, "branches": branches}
    with (output / "continuation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output_dir": str(output), **report}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
