DIRECT_CAUSAL_FROM_PROPAGATION_PROMPT = """You are a question designer for a spatio-temporal reasoning test.
Your task is to create a multiple-choice question based on a given piece of **true evidence** of a causal influence.

**True Evidence:**
- **Source Node:** {source_node_name} (ID: {source_node_id})
- **Target Node:** {target_node_name} (ID: {target_node_id})
- **Time Steps:** {time_period} (1 time step = {sampling_frequency})
- **Correct Description of Event:** "{correct_description}"

**Your Task:**
1.  Create a "question" that asks which statement best describes the influence on Node {target_node_id} during the specified time steps.
    - Explicitly use the phrase "time steps {time_period}" in the question text and append "(1 time step = {sampling_frequency})".
2.  Create an "options" list containing exactly four strings. The FIRST entry MUST be the **Correct Description** provided above, verbatim.
3.  The remaining three entries must be plausible but **incorrect** distractors describing different sources or incorrect effects.

**Output Format:**
Return ONLY a valid JSON object.
{{
    "question": "Your generated question.",
    "options": ["The correct description verbatim", "Distractor 1", "Distractor 2", "Distractor 3"]
}}
"""

MULTI_HOP_REASONING_PROMPT = """You are an expert in spatio-temporal network analysis. Your task is to analyze a set of direct causal events and identify a multi-hop propagation path to create a multiple-choice question.

**Given Direct Causal Events (Adjacency Modulations):**
{adjacency_modulations}

**Time Step Note:** 1 time step = {sampling_frequency}

**Your Task:**
1.  **Analyze the events:** Find a sequence of events that form a logical multi-hop path. The time periods of the events should be overlapping or consecutive.
2.  **Synthesize a description:** Create a concise, high-level description for the entire multi-hop event. This will be your correct answer.
3.  **Identify Nodes and Time:** State the start node, end node, and the overall time window for the multi-hop event in terms of time steps.
4.  **Generate a Question:** Create a "question" asking for the most appropriate description of the relationship between the start and end nodes during those time steps.
    - Explicitly reference the interval as "time steps X-Y" and append "(1 time step = {sampling_frequency})" in the question text.
5.  **Generate Options:** Create an "options" list with exactly four strings. The FIRST entry must be your synthesized description. The other three should be plausible but incorrect distractors.

**Example:**
*Given the following events:*
```json
[
    {
        "time_period": "16-18",
        "applies_to": "0->1",
        "description": "Morning rush hour from suburb to highway junction"
    },
    {
        "time_period": "17-19",
        "applies_to": "1->2",
        "description": "Morning commute flow reaches CBD"
    }
]
```

*A good JSON output would be:*
```json
{
    "question": "Which statement best describes the relationship between Node 0 and Node 2 during time steps 16-19 (1 time step = 30 minutes)?",
    "options": [
        "Morning commute traffic flows from the suburb (Node 0) to the CBD (Node 2).",
        "Evening traffic returns from the CBD to the suburb.",
        "There is no significant traffic flow between the suburb and the CBD.",
        "Traffic flows directly from the CBD to the suburb."
    ]
}
```

**Output Format:**
Return ONLY a valid JSON object.
{{
    "question": "Your generated question about the multi-hop relationship.",
    "options": ["Your synthesized correct description", "Distractor 1", "Distractor 2", "Distractor 3"]
}}
"""