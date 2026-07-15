ENTITY_IDENTIFICATION_GENERATION_PROMPT = """You are given a node ID and its correct name and description from a spatio-temporal network simulation.
Your task is to generate a multiple-choice question to identify which (name, description) pair corresponds to the target node.

Target Node:
- ID: {node_id}
- Correct Name: {node_name}
- Correct Description: {node_description}

Requirements:
1) "options": list of exactly four strings containing (name, description) pairs. Do NOT prefix with labels.
   - The FIRST entry must be the correct pair (verbatim match to the given name and description).
   - The remaining three must be fluent but !!!incorrect!!!.
       - They should describe plausible but different node roles or locations.
       - Maintain the same style and variable domain (e.g., “traffic flow”, “industrial output”, “water pressure”).
       - Avoid contradictions or unrealistic content.

Output JSON format:
{{
    "question": "Which (name, description) pair should Node {node_id} correspond to?",
    "options": ["Correct pair", "Distractor 1", "Distractor 2", "Distractor 3"]
}}

Strictly return only a valid JSON object. Do not add explanations outside JSON."""