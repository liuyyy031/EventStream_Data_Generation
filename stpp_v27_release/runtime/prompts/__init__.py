"""
Prompts module for Spatial-Temporal Reasoning multi-agent pipeline.

This module provides standardized prompt templates for the 8-agent architecture:
1. Scenario Generation Agent - generates natural language scenarios
2. Scenario Parsing Agent - converts natural language to structured JSON
   - Judge Agent 1 - validates scenario parsing accuracy
3. SDE Parameters Agent - generates SDE parameters from structured JSON
4. Time-Varying Adjacency Agent - generates adjacency matrices from structured JSON
   - Judge Agent 2 - validates parameters with multimodal analysis
5. Simulation Agent - runs numerical simulation (implemented in main code)
6. Visualization Agent - produces visual outputs (implemented in main code)
"""

from .scenario_generation_agent_prompt import SCENARIO_GENERATION_PROMPT
from .scenario_parsing_agent_prompt import SCENARIO_PARSING_PROMPT
from .sde_parameters_generation_agent_prompt import SDE_PARAMETERS_PROMPT
from .time_varying_adjacency_agent_prompt import TIME_VARYING_ADJACENCY_PROMPT
from .judge_scenario_parsing_prompt import JUDGE_SCENARIO_PARSING_PROMPT
from .judge_parameter_validation_prompt import JUDGE_PARAMETER_VALIDATION_PROMPT

__all__ = [
    'SCENARIO_GENERATION_PROMPT',
    'SCENARIO_PARSING_PROMPT',
    'SDE_PARAMETERS_PROMPT',
    'TIME_VARYING_ADJACENCY_PROMPT',
    'JUDGE_SCENARIO_PARSING_PROMPT',
    'JUDGE_PARAMETER_VALIDATION_PROMPT'
]
