FORECASTING_PROMPT = """Analyze the provided context and generate a detailed forecast description.

**Scenario Details:**
- **Target Node:** {target_node_name} (ID: {target_node_id})
- **Variable:** {target_node_variable}
- **Observation Window:** Steps {history_window}
- **Prediction Window:** Steps {prediction_window}
- **Key Event:** {events}
- **Statistical Hints:** {referenced_stats}

**Task:**
Based *only* on the information above, provide a JSON object describing the forecast.

**Output Format:**
```json
{
  "observation_window": "{history_window}",
  "prediction_window": "{prediction_window}",
  "prediction_length": {prediction_length},
  "target_node_id": {target_node_id},
  "context_description": "{events}",
  "summary": "text describing the expected behaviour during the prediction window",
  "confidence": "low/medium/high"
}
```"""
