ETIOLOGICAL_REASONING_PROMPT = """You are given spatio-temporal context. Produce a multiple-choice item where the correct option is a concise macro summary of the Scenario Context.

Scenario: {scenario}

Requirements:
1) "observation": A concise macro summary of the Scenario in 12–20 words.
   - It must describe the system at a high level (e.g., an interconnected hydroponics circulation system, a wastewater treatment facility, etc.). Facility names are not important; do not invent new names.
   - It must explicitly mention the key node variables provided.
2) "options": list of exactly four scenario summaries (each ≤20 words) without labels.
   - The FIRST entry must be identical to "observation" (verbatim match).
   - The other three must be fluent but incorrect (they must contradict the Scenario or mention entities/processes not present in the Scenario/Involved Nodes).

Output JSON format:
{{
    "observation": "observed phenomena",
    "options": ["Correct summary", "Distractor 1", "Distractor 2", "Distractor 3"]
}}

Strictly based on given context, do not introduce external knowledge. Return only a valid JSON object, without any explanations outside JSON."""