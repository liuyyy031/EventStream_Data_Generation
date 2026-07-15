#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import glob
import pickle
import re
import random
from typing import Dict, List, Any, Tuple, Optional


def format_value(value: Any) -> Any:
    """Format numerical values to three decimal places."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [format_value(v) for v in value]
    return value

def parse_qa_template(template: str, data: Dict[str, Any]) -> Tuple[str, str]:
    """Parse a QA template string to extract question and answer."""
    question_part, answer_part = template.split(" Answer: ")
    
    question = question_part.replace("Question: ", "").format(**data)
    answer = answer_part.format(**data)
    
    return question, format_value(answer)

def load_complete_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Load complete data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_timeseries_data(complete_data: Dict[str, Any]) -> List[List[float]]:
    """Extract time series data for all nodes."""
    ts_data = complete_data.get("agent5_simulation_data", [])
    if ts_data is None:
        return []
    
    # Format to 2 decimal places
    formatted_ts = []
    for node_series in ts_data:
        formatted_ts.append([round(float(v), 2) for v in node_series])
    
    return formatted_ts

def build_graph_structure_description(complete_data: Dict[str, Any]) -> str:
    """Build graph structure description from adjacency matrix."""
    time_varying_adj = complete_data.get("agent4_time_varying_adjacency", {})
    base_adjacency = time_varying_adj.get("base_adjacency", [])
    
    if not base_adjacency:
        return "No graph structure available"
    
    num_nodes = len(base_adjacency)
    edges = []
    
    for src in range(num_nodes):
        for tgt in range(num_nodes):
            if base_adjacency[src][tgt] > 0:
                edges.append(f"Node {src}->Node {tgt}")
    
    if not edges:
        return "No edges in graph"
    
    return "; ".join(edges)

def build_input_prefix(complete_data: Dict[str, Any], timeseries: List[List[float]]) -> str:
    """Build the input prefix with time series and graph structure descriptions."""
    num_nodes = len(timeseries)
    
    # Build time series descriptions
    ts_descriptions = []
    for node_id in range(num_nodes):
        ts_len = len(timeseries[node_id]) if node_id < len(timeseries) else 0
        ts_descriptions.append(f"Node {node_id} time series with length of {ts_len}: <ts><ts/>")
    
    ts_part = "; ".join(ts_descriptions)
    
    # Build graph structure description
    graph_structure = build_graph_structure_description(complete_data)
    
    prefix = f"You are a spatial temporal analysis expert. {ts_part}; Graph Structure: {graph_structure}, please analyze the spatial temporal data and answer the following question: "
    
    return prefix

def generate_temporal_qa(data: Dict[str, Any], templates: Dict[str, str], dataset_id: str, file_name: str, 
                        input_prefix: str, timeseries: List[List[float]]) -> List[Dict]:
    """Generate Temporal QA pairs in new format."""
    qa_pairs = []
    sde_params = data.get("agent3_sde_parameters", {})
    node_overrides = sde_params.get("node_overrides", {})

    for node_id_str, override in node_overrides.items():
        node_id = int(node_id_str)
        for pattern in override.get("drift_patterns", []):
            time_range = pattern.get("time_range")
            template_data = {
                "node_id": node_id,
                "time_range": time_range
            }

            for key, value in pattern.items():
                if key in templates:
                    template_data[key] = value
                    question, answer = parse_qa_template(templates[key], template_data)
                    qa_pairs.append({
                        "input": input_prefix + question,
                        "timeseries": timeseries,
                        "output": str(answer)
                    })

                if pattern.get("drift_type") == "sinusoidal":
                    for sub_key in ["A", "omega", "phi"]:
                        if sub_key in pattern and f"sinusoidal_{sub_key}" in templates:
                            template_data[sub_key] = pattern[sub_key]
                            question, answer = parse_qa_template(templates[f"sinusoidal_{sub_key}"], template_data)
                            qa_pairs.append({
                                "input": input_prefix + question,
                                "timeseries": timeseries,
                                "output": str(answer)
                            })
    return qa_pairs


def generate_spatial_qa(data: Dict[str, Any], templates: Dict[str, str], dataset_id: str, file_name: str,
                       input_prefix: str, timeseries: List[List[float]]) -> List[Dict]:
    """Generate Spatial QA pairs in new format."""
    qa_pairs = []
    time_varying_adj = data.get("agent4_time_varying_adjacency", {})
    
    base_adjacency = time_varying_adj.get("base_adjacency", [])
    num_nodes = len(base_adjacency)

    if num_nodes == 0:
        return []

    # Calculate transitive closure (reachability) using Floyd-Warshall algorithm
    reach = [[(1 if base_adjacency[i][j] > 0 else 0) for j in range(num_nodes)] for i in range(num_nodes)]
    for i in range(num_nodes):
        reach[i][i] = 1

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if reach[i][k] and reach[k][j]:
                    reach[i][j] = 1

    for src in range(num_nodes):
        for tgt in range(num_nodes):
            # edge_relationship
            if "edge_relationship" in templates:
                is_connected = base_adjacency[src][tgt] > 0
                template_data = {
                    "src": src,
                    "tgt": tgt,
                    "answer": "yes" if is_connected else "no",
                }
                question, answer = parse_qa_template(templates["edge_relationship"], template_data)
                qa_pairs.append({
                    "input": input_prefix + question,
                    "timeseries": timeseries,
                    "output": str(answer)
                })

            # indirect_connection
            if "indirect_connection" in templates:
                # Real check for an indirect path using the pre-computed reachability matrix.
                has_indirect_path = False
                # Check for an intermediate node k for a path src -> k -> ... -> tgt
                for k in range(num_nodes):
                    # An indirect path exists if there is an edge src->k and a path k->...->tgt
                    if base_adjacency[src][k] > 0 and reach[k][tgt]:
                        # If k is the target, the total path is just src->tgt (length 1).
                        # An intermediate node requires the path from k to have length >= 1, so k != tgt.
                        if k != tgt:
                            has_indirect_path = True
                            break
                
                template_data = {
                    "src": src,
                    "tgt": tgt,
                    "answer": "yes" if has_indirect_path else "no",
                }
                question, answer = parse_qa_template(templates["indirect_connection"], template_data)
                qa_pairs.append({
                    "input": input_prefix + question,
                    "timeseries": timeseries,
                    "output": str(answer)
                })

    return qa_pairs


def generate_spatial_temporal_qa(data: Dict[str, Any], templates: Dict[str, str], dataset_id: str, file_name: str,
                                input_prefix: str, timeseries: List[List[float]]) -> List[Dict]:
    """Generate Spatio-Temporal QA pairs in new format."""
    qa_pairs = []
    structured_scenario = data.get("agent2_structured_scenario", {})
    time_varying_adj = data.get("agent4_time_varying_adjacency", {})
    
    nodes = structured_scenario.get("nodes", [])
    for node in nodes:
        if "node_type" in templates:
            node_id = node.get("id")
            node_type = node.get("type")
            if node_type:
                question, answer = parse_qa_template(templates["node_type"], {"node_id": node_id, "node_type": node_type})
                qa_pairs.append({
                    "input": input_prefix + question,
                    "timeseries": timeseries,
                    "output": str(answer)
                })

    edges = structured_scenario.get("edges", [])
    for edge in edges:
        if "edge_lag" in templates:
            src = edge.get("source")
            tgt = edge.get("target")
            lag = edge.get("time_lag")
            if src is not None and tgt is not None and lag is not None:
                question, answer = parse_qa_template(templates["edge_lag"], {"src": src, "tgt": tgt, "lag": lag})
                qa_pairs.append({
                    "input": input_prefix + question,
                    "timeseries": timeseries,
                    "output": str(answer)
                })

    base_adjacency = time_varying_adj.get("base_adjacency", [])
    for pattern in time_varying_adj.get("time_modulation", {}).get("patterns", []):
        time_range = pattern.get("time_range")
        edge_modulations = pattern.get("edge_modulations", {})
        
        for edge_str, modulation in edge_modulations.items():
            if "->" not in edge_str:
                continue
            
            src_str, tgt_str = edge_str.split("->")
            src, tgt = int(src_str), int(tgt_str)
            
            multiplier = modulation.get("multiplier")
            if multiplier is None:
                continue

            if "edge_modulation" in templates:
                question, answer = parse_qa_template(templates["edge_modulation"], {"edge": f"({src},{tgt})", "time_range": time_range, "multiplier": multiplier})
                qa_pairs.append({
                    "input": input_prefix + question,
                    "timeseries": timeseries,
                    "output": str(answer)
                })
            
            if "effective_coupling_strength" in templates:
                if src < len(base_adjacency) and tgt < len(base_adjacency[src]):
                    base_strength = base_adjacency[src][tgt]
                    effective_strength = format_value(multiplier * base_strength)
                    question, answer = parse_qa_template(templates["effective_coupling_strength"], {"edge": f"({src},{tgt})", "time_range": time_range, "multiplier * base_adjacency": effective_strength})
                    qa_pairs.append({
                        "input": input_prefix + question,
                        "timeseries": timeseries,
                        "output": str(answer)
                    })

    return qa_pairs


def process_files(file_list: List[str], templates: Dict[str, Any], split_name: str) -> Tuple[List[Dict], Dict[str, int]]:
    """Process a list of files and return QA pairs with statistics."""
    qa_pairs = []
    stats = {'temporal': 0, 'spatial': 0, 'spatial_temporal': 0}
    
    for file_path in file_list:
        complete_data = load_complete_data(file_path)
        if not complete_data:
            continue
        
        file_name = os.path.basename(file_path)
        dataset_id = file_name.replace('.pkl', '')
        
        timeseries = extract_timeseries_data(complete_data)
        if not timeseries:
            print(f"Skipping {file_name}: no time series data")
            continue
        
        input_prefix = build_input_prefix(complete_data, timeseries)
        
        temporal_qa = generate_temporal_qa(complete_data, templates["temporal"], dataset_id, file_name, 
                                          input_prefix, timeseries)
        spatial_qa = generate_spatial_qa(complete_data, templates["spatial"], dataset_id, file_name,
                                        input_prefix, timeseries)
        spatial_temporal_qa = generate_spatial_temporal_qa(complete_data, templates["spatial_temporal"], 
                                                          dataset_id, file_name, input_prefix, timeseries)
        
        for qa in temporal_qa:
            qa['category'] = 'temporal'
        for qa in spatial_qa:
            qa['category'] = 'spatial'
        for qa in spatial_temporal_qa:
            qa['category'] = 'spatial_temporal'

        qa_pairs.extend(temporal_qa)
        qa_pairs.extend(spatial_qa)
        qa_pairs.extend(spatial_temporal_qa)
        
        stats['temporal'] += len(temporal_qa)
        stats['spatial'] += len(spatial_qa)
        stats['spatial_temporal'] += len(spatial_temporal_qa)
        
        print(f"[{split_name}] Generated {len(temporal_qa)} temporal, {len(spatial_qa)} spatial, "
              f"{len(spatial_temporal_qa)} spatial-temporal QA pairs from {file_name}")
    
    return qa_pairs, stats


def main():
    """Main function to generate QA pairs from SDE simulation data in JSONL format."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    data_source_dir = os.path.join(repo_root, "data_generation", "batch_output")
    template_file = os.path.join(repo_root, "data_generation", "prompts", "qa_generation", "alignment_templates.json")
    output_dir = os.path.join(repo_root, "data", "alignment")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load templates
    with open(template_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)

    # Find pkl data files
    data_files = glob.glob(os.path.join(data_source_dir, "task_*.pkl"))
    print(f"Found {len(data_files)} pkl files to process")
    
    # Shuffle and split datasets FIRST (80% train, 20% test)
    random.seed(42)  # For reproducibility
    random.shuffle(data_files)
    
    split_index = int(len(data_files) * 0.8)
    train_files = data_files[:split_index]
    test_files = data_files[split_index:]
    
    print(f"Split datasets: {len(train_files)} train, {len(test_files)} test")

    # Process train and test splits separately
    train_pairs, train_stats = process_files(train_files, templates, "train")
    test_pairs, test_stats = process_files(test_files, templates, "test")

    # Save the training set
    train_output_file = os.path.join(output_dir, "alignment_train.jsonl")
    with open(train_output_file, 'w', encoding='utf-8') as f:
        for qa_pair in train_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')

    # Save the testing set
    test_output_file = os.path.join(output_dir, "alignment_test.jsonl")
    with open(test_output_file, 'w', encoding='utf-8') as f:
        for qa_pair in test_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')

    print(f"\n--- QA Generation Summary ---")
    print(f"Total datasets: {len(data_files)} ({len(train_files)} train, {len(test_files)} test)")
    print(f"Total QA pairs: {len(train_pairs) + len(test_pairs)}")
    print(f"-----------------------------")
    print(f"Training set: {len(train_pairs)} samples (from {len(train_files)} datasets)")
    print(f"  - Temporal QA: {train_stats['temporal']}")
    print(f"  - Spatial QA: {train_stats['spatial']}")
    print(f"  - Spatio-Temporal QA: {train_stats['spatial_temporal']}")
    print(f"-----------------------------")
    print(f"Testing set: {len(test_pairs)} samples (from {len(test_files)} datasets)")
    print(f"  - Temporal QA: {test_stats['temporal']}")
    print(f"  - Spatial QA: {test_stats['spatial']}")
    print(f"  - Spatio-Temporal QA: {test_stats['spatial_temporal']}")
    print(f"-----------------------------")

    print(f"Saved training set to {train_output_file}")
    print(f"Saved testing set to {test_output_file}")


if __name__ == "__main__":
    main()