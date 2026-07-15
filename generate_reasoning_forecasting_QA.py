#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import functools
import glob
import json
import os
import pickle
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple


def format_timeseries_data(
    ts_data: Optional[Sequence[Sequence[Any]]], end_index: Optional[int] = None
) -> List[List[float]]:
    """Convert raw time-series data to a rounded list of floats, optionally truncated."""
    if ts_data is None:
        return []

    formatted: List[List[float]] = []
    for series in ts_data:
        truncated = series[: max(0, end_index)] if end_index is not None else series
        formatted.append([round(float(v), 2) for v in truncated])
    return formatted


def build_graph_structure_description(dataset: Dict[str, Any]) -> str:
    structured = dataset.get("structured_scenario", {})
    relationships = dataset.get("relationships") or structured.get("edges", [])
    if relationships:
        edges = []
        for rel in relationships:
            src = rel.get("source")
            tgt = rel.get("target")
            if src is None or tgt is None:
                continue
            edges.append(f"Node {src}->Node {tgt}")
        if edges:
            return "; ".join(edges)

    graph_data = dataset.get("graph") or structured.get("graph")
    if graph_data:
        base_adjacency = graph_data.get("base_adjacency", [])
        edges: List[str] = []
        num_nodes = len(base_adjacency)
        for src in range(num_nodes):
            for tgt in range(num_nodes):
                try:
                    weight = base_adjacency[src][tgt]
                except (IndexError, TypeError):
                    weight = 0
                if weight and weight > 0:
                    edges.append(f"Node {src}->Node {tgt}")
        if edges:
            return "; ".join(edges)

    return "No graph structure available"


def build_input_prefix(dataset: Dict[str, Any], timeseries: List[List[float]]) -> str:
    num_nodes = len(timeseries)
    ts_descriptions = []
    for node_id in range(num_nodes):
        ts_len = len(timeseries[node_id]) if node_id < len(timeseries) else 0
        ts_descriptions.append(
            f"Node {node_id} time series with length of {ts_len}: <ts><ts/>"
        )

    ts_part = "; ".join(ts_descriptions)
    graph_structure = build_graph_structure_description(dataset)

    return (
        "You are a spatial temporal analysis expert. "
        f"{ts_part}; Graph Structure: {graph_structure}, "
        "please analyze the spatial temporal data and answer the following question: "
    )


class DirectForecastingGenerator:
    """Generate forecasting QA pairs without invoking any LLM."""

    def __init__(self):
        self.task_type = "In-context Spatio-temporal Forecasting"
        self.last_data_dir = "batch_output"

        templates_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "prompts",
            "qa_generation",
            "reasoning_templates.json",
        )
        with open(templates_path, "r", encoding="utf-8") as f:
            reasoning_templates = json.load(f)

        question_template = reasoning_templates.get("forecasting", {}).get(
            "question",
            "Given the context {context_description}, predict node {node_id} during {prediction_window}.",
        )

        self.forecasting_question_template = question_template

    # --- Data loading helpers ---

    def load_existing_data_files(self, data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        if data_dir is None:
            data_dir = getattr(self, "last_data_dir", "batch_output")
        self.last_data_dir = data_dir

        pickle_files = glob.glob(os.path.join(data_dir, "task_*.pkl"))
        pickle_files.sort()

        datasets: List[Dict[str, Any]] = []
        print(f"Found {len(pickle_files)} complete data files in '{data_dir}':")

        for file_path in pickle_files:
            try:
                parsed = self.load_pickle_data_file(file_path)
                if parsed:
                    datasets.append(parsed)
                    print(f"  ✓ Loaded: {os.path.basename(file_path)}")
                else:
                    print(f"  ✗ Failed to parse: {os.path.basename(file_path)}")
            except Exception as exc:
                print(f"  ✗ Error loading {os.path.basename(file_path)}: {exc}")

        print(f"Successfully loaded {len(datasets)} datasets")
        return datasets

    def load_pickle_data_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        with open(file_path, "rb") as fh:
            complete_data = pickle.load(fh)

        structured_scenario = complete_data.get("agent2_structured_scenario", {})

        dataset_id = os.path.basename(file_path).replace(".pkl", "")
        nodes = structured_scenario.get("nodes", [])

        data = {
            "dataset_id": dataset_id,
            "structured_scenario": structured_scenario,
            "nodes": nodes,
            "ts_data": complete_data.get("agent5_simulation_data"),
            "seq_len": complete_data.get("seq_len"),
            "graph": complete_data.get("agent4_time_varying_adjacency"),
        }

        if nodes and data["ts_data"] is not None:
            return data
        return None

    # --- Forecast preparation helpers ---

    @staticmethod
    def _parse_time_period(time_period: Any) -> Optional[Tuple[int, int]]:
        if isinstance(time_period, str):
            if "-" not in time_period:
                return None
            start_str, end_str = time_period.split("-", 1)
            try:
                return int(start_str), int(end_str)
            except ValueError:
                return None
        if isinstance(time_period, (list, tuple)) and len(time_period) == 2:
            try:
                start, end = int(time_period[0]), int(time_period[1])
                return start, end
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_edge_targets(applies_to: str) -> Optional[Tuple[int, int]]:
        if not applies_to or "->" not in applies_to:
            return None
        left, right = applies_to.split("->", 1)
        try:
            return int(left), int(right)
        except ValueError:
            return None

    @staticmethod
    def _compute_forecast_windows(
        start: int, end: int, seq_len: Optional[int]
    ) -> Optional[Tuple[int, int, int, int, int]]:
        event_len = max(1, end - start)
        max_history = max(0, start)
        if max_history < 2:
            return None

        pred_len = max(event_len, 1)
        obs_len = pred_len * 2

        if obs_len > max_history:
            obs_len = max_history - (max_history % 2)
            if obs_len < 2:
                return None
            pred_len = obs_len // 2
            if pred_len == 0:
                return None

        observation_start = start - obs_len
        observation_end = start
        prediction_start = start
        prediction_end = start + pred_len

        if seq_len is not None:
            prediction_end = min(prediction_end, seq_len)
            pred_len = prediction_end - prediction_start
            if pred_len <= 0:
                return None

        if prediction_end < end:
            shortfall = end - prediction_end
            prediction_end += shortfall
            pred_len = prediction_end - prediction_start
            obs_len = pred_len * 2
            observation_start = observation_end - obs_len
            if observation_start < 0:
                return None

        if obs_len % 2:
            observation_start += 1
            obs_len = observation_end - observation_start
            pred_len = obs_len // 2
            prediction_end = prediction_start + pred_len

        if pred_len <= 0 or obs_len != pred_len * 2:
            return None

        return observation_start, observation_end, prediction_start, prediction_end, pred_len

    @staticmethod
    def _extract_prediction_values(
        ts_data: Optional[Sequence[Sequence[Any]]], node_id: int, start: int, end: int
    ) -> List[float]:
        if ts_data is None or node_id >= len(ts_data) or start >= end:
            return []
        series = ts_data[node_id]
        return [round(float(v), 2) for v in series[start:end]]

    def _prepare_forecasting_example(self, dataset: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
        structured = dataset.get("structured_scenario", {})
        patterns = structured.get("adjacency_modulation", {}).get("patterns", [])
        if not patterns:
            return None

        pattern = rng.choice(patterns)
        period = self._parse_time_period(pattern.get("time_period"))
        edge = self._parse_edge_targets(pattern.get("applies_to", ""))
        description = pattern.get("description", "Edge weight increase detected")

        if period is None or edge is None:
            return None

        start, end_inclusive = period
        if start is None or end_inclusive is None or end_inclusive < start:
            return None

        end_exclusive = end_inclusive + 1

        source_id, target_id = edge
        nodes = dataset.get("nodes", [])
        nodes_by_id = {n.get("id"): n for n in nodes}
        if target_id not in nodes_by_id:
            return None

        seq_len = dataset.get("seq_len")
        windows = self._compute_forecast_windows(start, end_exclusive, seq_len)
        if windows is None:
            return None

        observation_start, observation_end, prediction_start, prediction_end, prediction_len = windows
        observation_window = f"{observation_start}-{observation_end - 1}"
        prediction_window = f"{prediction_start}-{prediction_end - 1}"

        values = self._extract_prediction_values(
            dataset.get("ts_data"), target_id, prediction_start, prediction_end
        )

        return {
            "dataset_id": dataset["dataset_id"],
            "context_description": description,
            "target_node_id": target_id,
            "source_node_id": source_id,
            "observation_window": observation_window,
            "prediction_window": prediction_window,
            "prediction_length": prediction_len,
            "values": values,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "prediction_start": prediction_start,
            "prediction_end": prediction_end,
        }

    # --- Public API ---

    def generate_forecasting_qas(self, datasets: List[Dict[str, Any]], qa_per_task: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        qa_pairs: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "dataset_ids": set(),
            "prediction_lengths": [],
            "values_lengths": [],
            "target_node_counts": {},
            "source_node_counts": {},
        }

        for dataset in datasets:
            raw_ts = dataset.get("ts_data")
            if raw_ts is None:
                continue

            dataset_id = dataset["dataset_id"]
            rng = random.Random(hash(dataset_id))

            patterns = dataset.get("structured_scenario", {}).get("adjacency_modulation", {}).get("patterns", [])
            if not patterns:
                continue

            target_count = min(qa_per_task, len(patterns))
            seen_combinations = set()

            generated = 0
            attempts = 0
            max_attempts = max(target_count * 5, 10)

            while generated < target_count and attempts < max_attempts:
                attempts += 1
                example = self._prepare_forecasting_example(dataset, rng)
                if not example or not example.get("values"):
                    continue

                combo = (
                    example["context_description"],
                    example["target_node_id"],
                    example["prediction_window"],
                )
                if combo in seen_combinations:
                    continue

                prediction_start = example.get("prediction_start")
                truncated_ts = format_timeseries_data(raw_ts, prediction_start)
                if not truncated_ts or all(len(series) == 0 for series in truncated_ts):
                    continue

                input_prefix = build_input_prefix(dataset, truncated_ts)

                seen_combinations.add(combo)

                stats["dataset_ids"].add(example["dataset_id"])
                stats["prediction_lengths"].append(example["prediction_length"])
                stats["values_lengths"].append(len(example["values"]))
                stats["target_node_counts"][example["target_node_id"]] = (
                    stats["target_node_counts"].get(example["target_node_id"], 0) + 1
                )
                stats["source_node_counts"][example["source_node_id"]] = (
                    stats["source_node_counts"].get(example["source_node_id"], 0) + 1
                )

                question_text = self.forecasting_question_template.format(
                    context_description=example["context_description"],
                    node_id=example["target_node_id"],
                    prediction_window=example["prediction_window"],
                    prediction_length=example["prediction_length"],
                )
                question_text += (
                    f" Historical observation window: {example['observation_window']}."
                )

                qa_pairs.append(
                    {
                        "input": input_prefix + question_text,
                        "timeseries": truncated_ts,
                        "output": json.dumps(example["values"], ensure_ascii=False),
                        "category": "forecasting",
                    }
                )
                generated += 1

        print(f"Generated {len(qa_pairs)} forecasting QA pairs from {len(datasets)} datasets.")
        return qa_pairs, stats

    @staticmethod
    def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def merge_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = {
            "dataset_ids": set(),
            "prediction_lengths": [],
            "values_lengths": [],
            "target_node_counts": {},
            "source_node_counts": {},
        }
        for stats in stats_list:
            merged["dataset_ids"].update(stats.get("dataset_ids", set()))
            merged["prediction_lengths"].extend(stats.get("prediction_lengths", []))
            merged["values_lengths"].extend(stats.get("values_lengths", []))
            for key in ["target_node_counts", "source_node_counts"]:
                for node_id, count in stats.get(key, {}).items():
                    merged[key][node_id] = merged[key].get(node_id, 0) + count
        return merged

    @staticmethod
    def print_statistics(qa_pairs: List[Dict[str, Any]], stats: Dict[str, Any], split_counts: Dict[str, int]) -> None:
        total_pairs = len(qa_pairs)
        unique_datasets = len(stats.get("dataset_ids", set()))
        prediction_lengths = stats.get("prediction_lengths", [])
        values_lengths = stats.get("values_lengths", [])

        print("\n--- Forecasting Dataset Statistics ---")
        print(f"Total QA pairs: {total_pairs}")
        print(f"Unique datasets covered: {unique_datasets}")
        print(
            f"Split counts -> finetune: {split_counts['finetune']}, rl: {split_counts['rl']}, test: {split_counts['test']}"
        )

        if prediction_lengths:
            mean_len = statistics.mean(prediction_lengths)
            median_len = statistics.median(prediction_lengths)
            min_len = min(prediction_lengths)
            max_len = max(prediction_lengths)
            print(
                "Prediction window length: "
                f"mean={mean_len:.2f}, median={median_len:.2f}, min={min_len}, max={max_len}"
            )

        if values_lengths and any(values_lengths):
            mean_vals = statistics.mean(values_lengths)
            median_vals = statistics.median(values_lengths)
            min_vals = min(values_lengths)
            max_vals = max(values_lengths)
            print(
                "Answer sequence length: "
                f"mean={mean_vals:.2f}, median={median_vals:.2f}, min={min_vals}, max={max_vals}"
            )

        def print_top_counts(label: str, counter: Dict[Any, int]) -> None:
            if not counter:
                return
            top_items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
            formatted = ", ".join(f"{k}: {v}" for k, v in top_items)
            print(f"Top {label}: {formatted}")

        print_top_counts("target nodes", stats.get("target_node_counts", {}))
        print_top_counts("source nodes", stats.get("source_node_counts", {}))
        print("--- End of Statistics ---\n")

    def run(self, data_dir: str, qa_per_task: int, output_dir: str) -> None:
        print("--- Starting Direct Forecasting QA Generation ---")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(data_dir):
            data_dir_path = os.path.join(script_dir, data_dir)
        else:
            data_dir_path = data_dir

        datasets = self.load_existing_data_files(data_dir_path)
        if not datasets:
            print("No datasets found. Aborting.")
            return

        rng = random.Random(42)
        rng.shuffle(datasets)

        total_datasets = len(datasets)
        finetune_end = int(total_datasets * 0.4)
        rl_end = finetune_end + int(total_datasets * 0.4)

        finetune_datasets = datasets[:finetune_end]
        rl_datasets = datasets[finetune_end:rl_end]
        test_datasets = datasets[rl_end:]

        print(
            f"\nSplitting {total_datasets} loaded datasets into: "
            f"finetune={len(finetune_datasets)}, rl={len(rl_datasets)}, test={len(test_datasets)}\n"
        )

        finetune_qa, finetune_stats = self.generate_forecasting_qas(finetune_datasets, qa_per_task)
        rl_qa, rl_stats = self.generate_forecasting_qas(rl_datasets, qa_per_task)
        test_qa, test_stats = self.generate_forecasting_qas(test_datasets, qa_per_task)

        repo_root = os.path.dirname(script_dir)
        if not os.path.isabs(output_dir):
            output_dir_path = os.path.join(repo_root, output_dir)
        else:
            output_dir_path = output_dir
        os.makedirs(output_dir_path, exist_ok=True)

        self.write_jsonl(os.path.join(output_dir_path, "forecasting_finetune.jsonl"), finetune_qa)
        self.write_jsonl(os.path.join(output_dir_path, "forecasting_rl.jsonl"), rl_qa)
        self.write_jsonl(os.path.join(output_dir_path, "forecasting_test.jsonl"), test_qa)

        all_qa = finetune_qa + rl_qa + test_qa
        combined_stats = self.merge_stats([finetune_stats, rl_stats, test_stats])
        split_counts = {"finetune": len(finetune_qa), "rl": len(rl_qa), "test": len(test_qa)}

        self.print_statistics(all_qa, combined_stats, split_counts)
        print("--- Direct Forecasting QA Generation Finished ---")


def main() -> None:
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser(description="Generate forecasting QA pairs without LLM usage")
    parser.add_argument("--data_dir", type=str, default="batch_output", help="Directory containing source .pkl files.")
    parser.add_argument("--qa_per_task", type=int, default=3, help="Number of QA pairs to generate per dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "reasoning"),
        help="Directory to store the split forecasting QA JSONL files.",
    )
    args = parser.parse_args()

    generator = DirectForecastingGenerator()
    generator.run(data_dir=args.data_dir, qa_per_task=args.qa_per_task, output_dir=args.output_dir)


if __name__ == "__main__":
    main() 