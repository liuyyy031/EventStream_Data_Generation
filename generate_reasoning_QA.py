#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import copy
import random
import os
import re
import glob
import pickle
import time
from datetime import datetime
from itertools import cycle
from typing import Dict, List, Any, Optional, Tuple, Sequence
import argparse
import functools

from llm_client import LLMClient

from prompts.qa_generation.entity_identification import ENTITY_IDENTIFICATION_GENERATION_PROMPT
from prompts.qa_generation.etiological_reasoning import ETIOLOGICAL_REASONING_PROMPT
from prompts.qa_generation.correlation_reasoning import (
    DIRECT_CAUSAL_FROM_PROPAGATION_PROMPT,
    MULTI_HOP_REASONING_PROMPT,
)


def format_timeseries_data(ts_data: Optional[Sequence[Sequence[Any]]]) -> List[List[float]]:
    """Convert raw time-series data to a rounded list of floats."""
    if ts_data is None:
        return []
    formatted: List[List[float]] = []
    for series in ts_data:
        formatted.append([round(float(v), 2) for v in series])
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

    graph_data = structured.get("graph") or dataset.get("graph")
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
    ts_descriptions: List[str] = []
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


# --- I/O Configuration ---
QA_OUTPUT_DIR = "output_qa"
BATCH_INPUT_DIR = "batch_inputs_qa"


class QABatchGenerator:
    """
    Concurrent Spatio-Temporal QA Generator using an OpenAI-compatible LLM API.

    Generates four types of reasoning tasks based on given context.
    Configure access via the ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL``
    environment variables.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, max_workers: int = 8):
        """Initialize the QA generator."""
        self.llm = llm_client or LLMClient()
        self.max_workers = max_workers

        # Create directories
        os.makedirs(QA_OUTPUT_DIR, exist_ok=True)
        os.makedirs(BATCH_INPUT_DIR, exist_ok=True)
        
        # Task type definitions
        self.task_types = {
            "entity_identification": "Spatio-Entity Identification",
            "etiological_reasoning": "Etiological Spatio Reasoning", 
            "correlation_reasoning": "Spatio-Correlation Reasoning",
            "forecasting": "In-context Spatio-temporal Forecasting"
        }

        # Persist the most recent qa_per_task for metadata regeneration
        self.last_qa_per_task = 1
        # Persist the most recent data directory used for dataset loading
        self.last_data_dir = "batch_output"
        self.dataset_map: Dict[str, Dict[str, Any]] = {}
        self.dataset_split_map: Dict[str, str] = {}
        self.categories = ["entity", "etiological", "correlation"]

    def load_existing_data_files(self, data_dir: Optional[str] = None) -> List[Dict]:
        """Load existing complete data files from batch_output."""
        if data_dir is None:
            data_dir = getattr(self, "last_data_dir", "batch_output")
        self.last_data_dir = data_dir
        pickle_files = glob.glob(os.path.join(data_dir, "task_*.pkl"))
        pickle_files.sort()
        
        loaded_data = []
        print(f"Found {len(pickle_files)} complete data files in '{data_dir}':")
        
        for file_path in pickle_files:
            try:
                parsed_data = self.load_pickle_data_file(file_path)
                if parsed_data:
                    loaded_data.append(parsed_data)
                    print(f"  ✓ Loaded: {os.path.basename(file_path)}")
                else:
                    print(f"  ✗ Failed to parse: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ✗ Error loading {os.path.basename(file_path)}: {e}")
        
        print(f"Successfully loaded {len(loaded_data)} datasets")
        return loaded_data
    
    def load_pickle_data_file(self, file_path: str) -> Optional[Dict]:
        """Load a pickle file from the batch_output directory."""
        with open(file_path, 'rb') as f:
            complete_data = pickle.load(f)
        
        structured_scenario = complete_data.get("agent2_structured_scenario", {})
        
        node_descriptions = {}
        if "nodes" in structured_scenario:
            for node in structured_scenario["nodes"]:
                node_descriptions[node["id"]] = {"description": node.get("description", "")}
                
        dataset_id = os.path.basename(file_path).replace('.pkl', '')
                
        data = {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "scenario": complete_data.get("agent1_scenario", ""),
            "nodes": structured_scenario.get("nodes", []),
            "relationships": structured_scenario.get("edges", []),
            "node_descriptions": node_descriptions,
            "ts_data": complete_data.get("agent5_simulation_data"),
            "seq_len": complete_data.get("seq_len", 0),
            "structured_scenario": structured_scenario,
        }
        
        if data["nodes"] and data["ts_data"] is not None:
            return data
        else:
            return None

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from LLM response"""
        if not text:
            raise ValueError("Empty LLM response")
        
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass
        
        try:
            json.loads(text)
            return text
        except Exception:
            pass
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            json.loads(candidate)
            return candidate
        
        raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}...")

    # --- Prompt Creation Methods ---

    def _create_entity_identification_prompt(self, context: Dict, target_node: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Creates the prompt for an entity identification QA task generation."""
        nodes = context["nodes"]
        if not nodes:
            return "", {}

        if target_node is None:
            target_node = random.choice(nodes)

        node_id = target_node["id"]
        node_name = target_node["name"]
        node_description = context.get("node_descriptions", {}).get(node_id, {}).get("description", "")

        prompt = ENTITY_IDENTIFICATION_GENERATION_PROMPT.format(
            node_id=node_id,
            node_name=node_name,
            node_description=node_description
        )

        meta = {
            "qa_type": "entity_identification",
            "context": context,
            "target_node_id": node_id
        }
        return prompt, meta

    def _create_etiological_reasoning_prompt(self, context: Dict) -> Tuple[str, Dict]:
        """Creates the prompt for an etiological reasoning QA task."""
        scenario = context["scenario"]
        
        prompt = ETIOLOGICAL_REASONING_PROMPT.format(
            scenario=scenario
        )

        meta = {
            "qa_type": "etiological_reasoning",
            "scenario": scenario,
            "context": context,
        }
        return prompt, meta

    def _create_correlation_reasoning_prompt(self, context: Dict, setting: Optional[str] = None,
                                             direct_candidate: Optional[Tuple[Dict, Dict]] = None,
                                             indirect_events: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Dict]:
        """Creates the prompt for a correlation reasoning QA task using ground truth evidence."""
        nodes = context["nodes"]
        structured_scenario = (
            context.get("structured_scenario")
            or context.get("agent2_structured_scenario", {})
        )
        sampling_frequency = structured_scenario.get("sampling_frequency", "unknown interval")

        if setting not in {"direct_causal", "indirect_causal", None}:
            setting = None

        available_settings = ["direct_causal", "indirect_causal"]
        if setting is None:
            setting = random.choice(available_settings)
        prompt, meta = None, None

        if setting == "direct_causal":
            if direct_candidate is not None:
                evidence, target_node_data = direct_candidate
            else:
                candidates = self._get_direct_causal_candidates(structured_scenario)
                if not candidates:
                    return None, None
                random.shuffle(candidates)
                evidence, target_node_data = candidates[0]

            source_id = evidence.get("source")
            target_id = target_node_data.get("id")
            if source_id is None or target_id is None or source_id == target_id:
                return None, None

            source_node = next((n for n in nodes if n["id"] == source_id), None)
            target_node = next((n for n in nodes if n["id"] == target_id), None)
            if not source_node or not target_node:
                return None, None

            prompt = DIRECT_CAUSAL_FROM_PROPAGATION_PROMPT.format(
                source_node_name=f"Node {source_node['id']}", source_node_id=source_node['id'],
                target_node_name=f"Node {target_node['id']}", target_node_id=target_node['id'],
                time_period=evidence.get("time", "N/A"),
                sampling_frequency=sampling_frequency,
                correct_description=evidence.get("description", "")
            )
            meta = {"setting": "direct_causal"}

        elif setting == "indirect_causal":
            events = indirect_events
            if events is None:
                events = self._select_indirect_causal_samples(context, count=1)
                events = events[0] if events else None
            if not events or len(events) < 2:
                return None, None

            adj_mod_str = json.dumps(events, indent=2)
            prompt_template = MULTI_HOP_REASONING_PROMPT
            prompt_template = prompt_template.replace("{adjacency_modulations}", adj_mod_str)
            prompt = prompt_template.replace("{sampling_frequency}", sampling_frequency)
            meta = {"setting": "indirect_causal"}

        if prompt:
            meta.update({
                "qa_type": "correlation_reasoning",
                "context": context,
            })

        return prompt, meta

    def _get_direct_causal_candidates(self, structured_scenario: Dict[str, Any]) -> List[Tuple[Dict, Dict]]:
        candidates: List[Tuple[Dict, Dict]] = []
        drift_nodes = structured_scenario.get("drift_patterns", {}).get("nodes", [])
        for node_data in drift_nodes:
            target_id = node_data.get("id")
            for variation in node_data.get("propagated_variations", []):
                source_id = variation.get("source")
                if source_id is None or target_id is None or source_id == target_id:
                    continue
                candidates.append((variation, node_data))
        return candidates

    def _get_indirect_causal_candidates(self, structured_scenario: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        patterns = structured_scenario.get("adjacency_modulation", {}).get("patterns", [])
        candidates: List[List[Dict[str, Any]]] = []
        for start in range(len(patterns)):
            for end in range(start + 1, len(patterns)):
                sequence = patterns[start:end + 1]
                if len(sequence) >= 2:
                    candidates.append(sequence)
        return candidates

    def _select_direct_causal_samples(self, dataset: Dict[str, Any], count: int) -> List[Tuple[Dict, Dict]]:
        structured = dataset.get("structured_scenario", {})
        candidates = self._get_direct_causal_candidates(structured)
        if not candidates or count <= 0:
            return []
        random.shuffle(candidates)
        return candidates[:count]

    def _select_indirect_causal_samples(self, dataset: Dict[str, Any], count: int) -> List[List[Dict[str, Any]]]:
        structured = dataset.get("structured_scenario", {})
        candidates = self._get_indirect_causal_candidates(structured)
        if not candidates or count <= 0:
            return []
        random.shuffle(candidates)
        return candidates[:count]

    # --- Batch Request Creation ---

    def create_batch_requests(self, datasets: List[Dict], qa_per_task: int) -> List[Dict]:
        """Create all batch requests for QA generation."""
        requests = []
        for dataset in datasets:
            dataset_id = dataset['dataset_id']
            random.seed(hash(dataset_id))

            nodes = dataset.get("nodes", [])
            node_count = len(nodes)

            # Entity Identification: one QA per node in the dataset
            for idx, node in enumerate(nodes):
                prompt, meta = self._create_entity_identification_prompt(dataset, target_node=node)
                if prompt:
                    requests.append(self._create_bedrock_request(
                        record_id=f"{dataset_id}_entity_identification_{idx}",
                        prompt=prompt, max_tokens=2048
                    ))

            # Etiological Reasoning: fixed one QA per dataset
            prompt, meta = self._create_etiological_reasoning_prompt(dataset)
            if prompt:
                requests.append(self._create_bedrock_request(
                    record_id=f"{dataset_id}_etiological_reasoning_0",
                    prompt=prompt, max_tokens=2048
                ))

            # Correlation Reasoning - Direct causal samples
            direct_count = max(0, node_count - 1)
            direct_samples = self._select_direct_causal_samples(dataset, direct_count)
            for idx, candidate in enumerate(direct_samples):
                prompt, meta = self._create_correlation_reasoning_prompt(
                    dataset, setting="direct_causal", direct_candidate=candidate
                )
                if prompt:
                    requests.append(self._create_bedrock_request(
                        record_id=f"{dataset_id}_correlation_reasoning_direct_causal_{idx}",
                        prompt=prompt, max_tokens=2048
                    ))

            # Correlation Reasoning - Indirect causal (multi-hop) samples
            indirect_count = max(0, node_count - 2)
            indirect_samples = self._select_indirect_causal_samples(dataset, indirect_count)
            for idx, events in enumerate(indirect_samples):
                prompt, meta = self._create_correlation_reasoning_prompt(
                    dataset, setting="indirect_causal", indirect_events=events
                )
                if prompt:
                    requests.append(self._create_bedrock_request(
                        record_id=f"{dataset_id}_correlation_reasoning_indirect_causal_{idx}",
                        prompt=prompt, max_tokens=2048
                    ))

        print(f"Created {len(requests)} total batch requests.")
        return requests

    def _create_bedrock_request(self, record_id: str, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        """Create a single LLM request entry (legacy method name preserved)."""
        return {
            "recordId": record_id,
            "modelInput": {
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            },
        }

    def run_llm_requests(self, requests: List[Dict], stage_name: str) -> List[Dict]:
        """
        Run all requests concurrently against the LLM API and persist a JSONL
        copy of both the requests and the responses.
        """
        if not requests:
            print(f"No requests to run for stage {stage_name}.")
            return []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(BATCH_INPUT_DIR, exist_ok=True)
        os.makedirs(QA_OUTPUT_DIR, exist_ok=True)

        batch_file = os.path.join(BATCH_INPUT_DIR, f"{stage_name}_{timestamp}.jsonl")
        with open(batch_file, "w", encoding="utf-8") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        print(f"Saved {len(requests)} requests to {batch_file}")

        print(f"Running stage '{stage_name}' against the LLM API ...")
        results = self.llm.run_batch(requests, max_workers=self.max_workers)

        results_file = os.path.join(QA_OUTPUT_DIR, f"{stage_name}_{timestamp}_results.jsonl")
        with open(results_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"✓ Stage '{stage_name}' produced {len(results)} results -> {results_file}")
        return results

    
    # --- Result Processing and Finalization ---

    def process_results(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Process batch results and convert them into alignment-style QA entries."""
        if not hasattr(self, "dataset_map") or not self.dataset_map:
            print("Dataset metadata missing; reloading for result processing.")
            datasets = self.load_existing_data_files(self.last_data_dir)
            self.dataset_map = {d["dataset_id"]: d for d in datasets}

        print("Re-generating prompts' metadata to process results...")
        meta_map: Dict[str, Dict[str, Any]] = {}

        for dataset_id, dataset in self.dataset_map.items():
            random.seed(hash(dataset_id))

            nodes = dataset.get("nodes", [])
            node_count = len(nodes)

            for idx, node in enumerate(nodes):
                prompt_meta = self._create_entity_identification_prompt(dataset, target_node=node)[1]
                if prompt_meta:
                    meta_map[f"{dataset_id}_entity_identification_{idx}"] = prompt_meta

            prompt_meta = self._create_etiological_reasoning_prompt(dataset)[1]
            if prompt_meta:
                meta_map[f"{dataset_id}_etiological_reasoning_0"] = prompt_meta

            direct_count = max(0, node_count - 1)
            direct_samples = self._select_direct_causal_samples(dataset, direct_count)
            for idx, candidate in enumerate(direct_samples):
                prompt_meta = self._create_correlation_reasoning_prompt(
                    dataset, setting="direct_causal", direct_candidate=candidate
                )[1]
                if prompt_meta:
                    meta_map[f"{dataset_id}_correlation_reasoning_direct_causal_{idx}"] = prompt_meta

            indirect_count = max(0, node_count - 2)
            indirect_samples = self._select_indirect_causal_samples(dataset, indirect_count)
            for idx, events in enumerate(indirect_samples):
                prompt_meta = self._create_correlation_reasoning_prompt(
                    dataset, setting="indirect_causal", indirect_events=events
                )[1]
                if prompt_meta:
                    meta_map[f"{dataset_id}_correlation_reasoning_indirect_causal_{idx}"] = prompt_meta

        category_map = {
            "entity_identification": "entity",
            "etiological_reasoning": "etiological",
            "correlation_reasoning": "correlation",
        }

        option_cycles = {
            "entity": cycle(range(4)),
            "etiological": cycle(range(4)),
            "correlation": cycle(range(4)),
        }

        def format_options(category: str, options: Any) -> Optional[Tuple[List[str], int]]:
            if not isinstance(options, list) or len(options) != 4:
                return None
            processed_options = [str(opt).strip() for opt in options]
            if any(not opt for opt in processed_options):
                return None
            correct_idx = 0

            target_idx = next(option_cycles[category])
            if target_idx != correct_idx:
                processed_options[target_idx], processed_options[correct_idx] = (
                    processed_options[correct_idx],
                    processed_options[target_idx],
                )
                correct_idx = target_idx

            labelled = [
                f"{chr(ord('A') + idx)}. {option}"
                for idx, option in enumerate(processed_options)
            ]
            return labelled, correct_idx

        processed_records: List[Dict[str, Any]] = []

        print(f"Processing {len(results)} results from batch job...")
        for result in results:
            record_id = result.get("recordId")
            if not record_id or record_id not in meta_map:
                print(f"Warning: Skipping result with unknown recordId: {record_id}")
                continue

            meta = meta_map[record_id]
            qa_type = meta.get("qa_type")
            if qa_type not in category_map:
                continue
            category = category_map[qa_type]

            dataset = meta.get("context", {})
            dataset_id = dataset.get("dataset_id")
            if not dataset_id:
                continue

            timeseries = format_timeseries_data(dataset.get("ts_data"))
            if not timeseries:
                continue

            input_prefix = build_input_prefix(dataset, timeseries)

            try:
                content = result["modelOutput"]["content"]
                response_text = content[0]["text"]
                response_json = json.loads(self._extract_json_from_text(response_text))
            except Exception as err:
                print(f"Error decoding result {record_id}: {err}")
                continue

            formatted = format_options(category, response_json.get("options"))
            if not formatted:
                continue
            labelled_options, correct_idx = formatted
            correct_letter = chr(ord("A") + correct_idx)

            question_text = ""
            output_text = ""

            if qa_type == "entity_identification":
                question_text = response_json.get("question", "").strip()
                if not question_text:
                    target_node_id = meta.get("target_node_id")
                    question_text = f"Which (name, description) pair should Node {target_node_id} correspond to?"
                question_text += " Options: " + " ".join(labelled_options)
                output_text = f"<answer>{correct_letter}</answer>"
            elif qa_type == "etiological_reasoning":
                question_text = "Which etiological scenario can be inferred from the spatio-temporal data?"
                question_text += " Options: " + " ".join(labelled_options)
                output_text = f"<answer>{correct_letter}</answer>"
            elif qa_type == "correlation_reasoning":
                question_text = response_json.get("question", "").strip()
                if not question_text:
                    question_text = "Which statement best describes the causal relationship in the specified time steps?"
                question_text += " Options: " + " ".join(labelled_options)
                output_text = f"<answer>{correct_letter}</answer>"
            else:
                continue

            question_text = question_text.strip()
            output_text = str(output_text).strip()
            if not question_text or not output_text:
                continue

            record = {
                "dataset_id": dataset_id,
                "category": category_map[qa_type],
                "record": {
                    "input": input_prefix + question_text,
                    "timeseries": timeseries,
                    "output": output_text,
                    "category": category_map[qa_type],
                },
            }
            processed_records.append(record)

        print(f"Successfully processed {len(processed_records)} QA records.")
        return processed_records

    def save_qa_pairs(self, qa_pairs: List[Dict], output_file: str):
        """Save final QA pairs to a JSON file with statistics."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        type_counts = {}
        for qa in qa_pairs:
            task_type = qa.get("meta", {}).get("task_type", "Unknown")
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        dataset = {
            "dataset_info": {
                "name": "Spatio-Temporal Reasoning QA Dataset (Batch Generated)",
                "version": "1.0-batch",
                "total_samples": len(qa_pairs),
                "task_type_distribution": type_counts,
            },
            "qa_pairs": qa_pairs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(qa_pairs)} QA pairs to {output_file}")
        for task_type, count in type_counts.items():
            print(f"  - {task_type}: {count} samples")

    def run(self, data_dir: str, qa_per_task: int, output_dir: str):
        """Main execution flow for batch QA generation."""
        print("--- Starting Batch QA Generation Pipeline ---")

        datasets = self.load_existing_data_files(data_dir)
        if not datasets:
            print("No datasets found. Aborting.")
            return

        self.dataset_map = {dataset["dataset_id"]: dataset for dataset in datasets}

        shuffled_datasets = list(datasets)
        rng = random.Random(42)
        rng.shuffle(shuffled_datasets)

        total_datasets = len(shuffled_datasets)
        finetune_end = int(total_datasets * 0.4)
        rl_end = finetune_end + int(total_datasets * 0.4)

        finetune_datasets = shuffled_datasets[:finetune_end]
        rl_datasets = shuffled_datasets[finetune_end:rl_end]
        test_datasets = shuffled_datasets[rl_end:]

        self.dataset_split_map = {}
        for dataset in finetune_datasets:
            self.dataset_split_map[dataset["dataset_id"]] = "finetune"
        for dataset in rl_datasets:
            self.dataset_split_map[dataset["dataset_id"]] = "rl"
        for dataset in test_datasets:
            self.dataset_split_map[dataset["dataset_id"]] = "test"

        print(
            f"\nSplitting {total_datasets} loaded datasets into: "
            f"finetune={len(finetune_datasets)}, rl={len(rl_datasets)}, test={len(test_datasets)}\n"
        )

        requests = self.create_batch_requests(datasets, qa_per_task)
        self.last_qa_per_task = qa_per_task
        if not requests:
            print("No requests created. Aborting.")
            return

        results = self.run_llm_requests(requests, "qa-generation")
        if not results:
            print("No results produced. Aborting.")
            return

        processed_records = self.process_results(results)
        if not processed_records:
            print("No valid QA records generated. Aborting.")
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        if not os.path.isabs(output_dir):
            output_dir_path = os.path.join(repo_root, output_dir)
        else:
            output_dir_path = output_dir
        os.makedirs(output_dir_path, exist_ok=True)

        category_split_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            cat: {"finetune": [], "rl": [], "test": []} for cat in self.categories
        }
        category_stats: Dict[str, Dict[str, Any]] = {
            cat: {"total": 0, "datasets": set(), "split_counts": {"finetune": 0, "rl": 0, "test": 0}}
            for cat in self.categories
        }

        for item in processed_records:
            dataset_id = item["dataset_id"]
            category = item["category"]
            record = item["record"]

            if category not in category_split_records:
                continue

            split = self.dataset_split_map.get(dataset_id, "finetune")
            category_split_records[category][split].append(record)
            category_stats[category]["total"] += 1
            category_stats[category]["datasets"].add(dataset_id)
            category_stats[category]["split_counts"][split] += 1

        for category, splits in category_split_records.items():
            for split_name, records in splits.items():
                file_path = os.path.join(output_dir_path, f"{category}_{split_name}.jsonl")
                with open(file_path, "w", encoding="utf-8") as fh:
                    for record in records:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Saved {len(records)} records to {file_path}")

        total_records = sum(stats["total"] for stats in category_stats.values())

        print("\n--- Reasoning QA Dataset Statistics ---")
        print(f"Total QA pairs: {total_records}")
        for category in self.categories:
            stats = category_stats[category]
            total = stats["total"]
            unique_datasets = len(stats["datasets"])
            print(f"[{category}] total={total}, unique_datasets={unique_datasets}")
            print(
                f"  finetune={stats['split_counts']['finetune']}, "
                f"rl={stats['split_counts']['rl']}, "
                f"test={stats['split_counts']['test']}"
            )
        print("--- End of Statistics ---\n")

        print("--- Batch QA Generation Pipeline Finished ---")
def main():
    """Main function"""
    # Ensure print statements are flushed immediately
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser(description="Batch QA Generator for Spatio-Temporal Data")
    parser.add_argument("--data_dir", type=str, default="data_generation/batch_output", help="Directory with source .pkl files.")
    parser.add_argument("--qa_per_task", type=int, default=3, help="Number of QA pairs to generate per task type for each dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "reasoning"),
        help="Directory to store the split reasoning QA JSONL files.",
    )
    args = parser.parse_args()
    
    generator = QABatchGenerator()
    generator.run(
        data_dir=args.data_dir,
        qa_per_task=args.qa_per_task,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main() 