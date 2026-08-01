import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from typing import Dict, List, Tuple, Any
import os
import sys
import datetime
from pathlib import Path

# Add ChatTS path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import prompts for 8-agent pipeline (including judge agents)
from prompts import (
    SCENARIO_GENERATION_PROMPT,
    SCENARIO_PARSING_PROMPT,
    SDE_PARAMETERS_PROMPT,
    TIME_VARYING_ADJACENCY_PROMPT,
    JUDGE_SCENARIO_PARSING_PROMPT,
    JUDGE_PARAMETER_VALIDATION_PROMPT
)

from llm_client import LLMClient


class LLMClientWrapper:
    """
    Adapter exposing a ``generate_content`` API on top of :class:`LLMClient`.

    Accepts either a plain string or a list of ``str`` / ``PIL.Image`` items
    (matching the legacy interface used throughout this file) and returns an
    object with a ``.text`` attribute.
    """

    def __init__(self, client: LLMClient = None):
        self.client = client or LLMClient()

    def generate_content(self, contents):
        if not isinstance(contents, list):
            contents = [contents]

        text_chunks = []
        image_b64 = None
        image_media_type = "image/png"

        for item in contents:
            if isinstance(item, str):
                text_chunks.append(item)
            elif hasattr(item, "format") and hasattr(item, "save"):  # PIL.Image
                import base64
                import io
                buf = io.BytesIO()
                fmt = (item.format or "PNG").upper()
                item.save(buf, format=fmt)
                image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                image_media_type = "image/jpeg" if fmt == "JPEG" else "image/png"
            else:
                text_chunks.append(str(item))

        prompt = "\n".join(text_chunks)

        if image_b64 is not None:
            text = self.client.complete_with_image(
                prompt=prompt,
                image_base64=image_b64,
                image_media_type=image_media_type,
            )
        else:
            text = self.client.complete(prompt=prompt)

        class _Response:
            def __init__(self, text):
                self.text = text

        return _Response(text)

class AgentInteractionLogger:
    """记录所有Agent交互的日志系统"""
    
    def __init__(self, output_dir: str = "output/agent_logs"):
        """初始化日志记录器
        
        Args:
            output_dir: 日志输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建带时间戳的会话目录
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # 存储所有交互记录
        self.interactions = []
        
        print(f"📝 Agent交互日志系统已启动")
        print(f"📁 日志目录: {self.session_dir}")
    
    def log_agent_interaction(self, 
                             agent_name: str,
                             agent_type: str,
                             iteration: int,
                             input_data: Any,
                             output_data: Any,
                             metadata: Dict[str, Any] = None):
        """记录一次Agent交互
        
        Args:
            agent_name: Agent名称 (e.g., "Agent 1: Scenario Generation")
            agent_type: Agent类型 (e.g., "generator", "parser", "judge")
            iteration: 迭代次数（从1开始）
            input_data: 输入数据
            output_data: 输出数据
            metadata: 附加元数据（如错误信息、验证结果等）
        """
        timestamp = datetime.datetime.now().isoformat()
        
        # 处理特殊类型数据（如numpy数组）
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return {
                    "_type": "numpy.ndarray",
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "data": obj.tolist()
                }
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        interaction_record = {
            "session_id": self.session_id,
            "timestamp": timestamp,
            "agent_name": agent_name,
            "agent_type": agent_type,
            "iteration": iteration,
            "input": serialize(input_data) if input_data is not None else None,
            "output": serialize(output_data) if output_data is not None else None,
            "metadata": metadata or {}
        }
        
        self.interactions.append(interaction_record)
        
        # 立即保存单个交互到独立文件
        interaction_file = self.session_dir / f"{len(self.interactions):03d}_{agent_type}_{agent_name.replace(' ', '_').replace(':', '')}_iter{iteration}.json"
        with open(interaction_file, 'w', encoding='utf-8') as f:
            json.dump(interaction_record, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  💾 已保存: {interaction_file.name}")
    
    def save_complete_log(self):
        """保存完整的交互日志到一个文件"""
        complete_log_file = self.session_dir / "complete_interaction_log.json"
        
        complete_log = {
            "session_id": self.session_id,
            "total_interactions": len(self.interactions),
            "interactions": self.interactions
        }
        
        with open(complete_log_file, 'w', encoding='utf-8') as f:
            json.dump(complete_log, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📊 完整交互日志已保存: {complete_log_file}")
        print(f"   总交互次数: {len(self.interactions)}")
        
        # 创建摘要报告
        self._create_summary_report()
    
    def _create_summary_report(self):
        """创建交互摘要报告"""
        summary_file = self.session_dir / "interaction_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Agent交互摘要报告 - Session {self.session_id}\n")
            f.write("=" * 80 + "\n\n")
            
            # 按Agent类型分组统计
            agent_stats = {}
            for interaction in self.interactions:
                agent_name = interaction['agent_name']
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {
                        'count': 0,
                        'iterations': []
                    }
                agent_stats[agent_name]['count'] += 1
                agent_stats[agent_name]['iterations'].append(interaction['iteration'])
            
            f.write(f"总交互次数: {len(self.interactions)}\n")
            f.write(f"涉及Agent数量: {len(agent_stats)}\n\n")
            
            f.write("各Agent交互统计:\n")
            f.write("-" * 80 + "\n")
            for agent_name, stats in agent_stats.items():
                f.write(f"\n{agent_name}:\n")
                f.write(f"  - 交互次数: {stats['count']}\n")
                f.write(f"  - 迭代轮数: {stats['iterations']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("详细交互时间线:\n")
            f.write("=" * 80 + "\n\n")
            
            for i, interaction in enumerate(self.interactions, 1):
                f.write(f"{i}. [{interaction['timestamp']}] {interaction['agent_name']} (迭代 {interaction['iteration']})\n")
                if interaction['metadata']:
                    f.write(f"   元数据: {interaction['metadata']}\n")
                f.write("\n")
        
        print(f"📋 交互摘要报告已保存: {summary_file}")

class NetworkSDEGenerator:
    """Network SDE Generator for Spatial-Temporal Data"""
    
    # Default sequence length for time series generation
    DEFAULT_SEQ_LEN = 168
    # Maximum allowed sequence length
    MAX_SEQ_LEN = 365
    
    def __init__(self, num_nodes: int = 3, logger: AgentInteractionLogger = None,
                 llm_client: LLMClient = None):
        """
        Initialize Network SDE generator.

        Args:
            num_nodes: Number of nodes in graph (number of time series)
            logger: Agent interaction logger (optional)
            llm_client: Optional pre-configured :class:`LLMClient`. If omitted,
                a default one is constructed from environment variables
                (``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL``).
        """
        self.num_nodes = num_nodes
        self.seq_len = None  # Will be determined from scenario
        self.logger = logger
        self.sde_params = None  # Will be populated during generation
        self.model = LLMClientWrapper(llm_client)
    
    def generate_scenario_description(self, previous_scenario: str = None, previous_feedback: str = None, iteration: int = 1) -> str:
        """Generate or revise scenario description
        
        Args:
            previous_scenario: Previous scenario text to revise (if retry)
            previous_feedback: Feedback from Judge Agent about previous scenario (if retry)
            iteration: Current iteration number
        
        Returns:
            Generated or revised scenario text
        """
        
        # Get domain hint if available
        domain_hint = ""
        if hasattr(self, 'domain') and self.domain:
            domain_hints = {
                'Transportation': 'traffic flow, vehicle movement, or transportation networks',
                'Energy': 'power grid, energy consumption, or renewable energy generation',
                'Environment&Pollution': 'air quality, pollution levels, or environmental monitoring',
                'Ecology': 'ecosystem dynamics, species populations, or ecological networks',
                'Public Health': 'disease spread, infection rates, or public health surveillance',
                'Hydrology': 'water flow, river networks, or hydrological cycles',
                'Oceanography': 'ocean currents, marine ecosystems, or oceanographic data',
                'Agriculture': 'crop yields, agricultural production, or farming networks',
                'Mobility': 'human mobility, migration patterns, or movement networks',
                'Climate': 'weather patterns, climate data, or atmospheric conditions'
            }
            domain_hint = f"\n\nPreferred domain/context: {domain_hints.get(self.domain, self.domain)}"
        
        prompt = SCENARIO_GENERATION_PROMPT.format(
            num_nodes=self.num_nodes, 
            max_seq_len=self.MAX_SEQ_LEN
        ) + domain_hint
        
        # Add previous scenario and feedback if this is a revision
        if previous_scenario and previous_feedback:
            revision_section = f"\n\n{'='*60}\nSCENARIO REVISION MODE\n{'='*60}\n\nYour task is to REVISE the following scenario based on judge feedback.\nDo NOT create a completely new scenario. Instead, FIX the specific issues identified.\n\n**PREVIOUS SCENARIO:**\n{previous_scenario}\n\n{'='*60}\nJUDGE FEEDBACK:\n{'='*60}\n{previous_feedback}\n{'='*60}\n\nPlease revise the above scenario to address the judge's feedback.\nKeep the overall structure and nodes, but fix the specific issues mentioned.\n"
            prompt = prompt + revision_section
        
        # 记录输入
        input_data = {
            "num_nodes": self.num_nodes,
            "max_seq_len": self.MAX_SEQ_LEN,
            "has_previous_scenario": previous_scenario is not None,
            "has_feedback": previous_feedback is not None,
            "mode": "revision" if (previous_scenario and previous_feedback) else "generation",
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt
        }
        
        response = self.model.generate_content(prompt)
        scenario = response.text
        
        # 记录输出
        if self.logger:
            self.logger.log_agent_interaction(
                agent_name="Agent 1: Scenario Generation",
                agent_type="generator",
                iteration=iteration,
                input_data=input_data,
                output_data={"scenario": scenario},
                metadata={
                    "scenario_length": len(scenario),
                    "is_revision": previous_scenario is not None,
                    "mode": "revision" if (previous_scenario and previous_feedback) else "generation"
                }
            )
        
        return scenario
    
    def generate_scenario_with_length_validation(self) -> Tuple[str, int]:
        """Generate scenario description with automatic length validation and retry"""
        max_retries = 3
        
        for attempt in range(max_retries):
            print(f"Generating scenario description - Attempt {attempt + 1}/{max_retries}")
            
            # Generate scenario
            scenario = self.generate_scenario_description()
            
            # Extract time information and calculate sequence length
            time_info = self._extract_time_info_from_scenario(scenario)
            
            if time_info["calculated_seq_len"]:
                calculated_len = time_info["calculated_seq_len"]
                
                if calculated_len <= self.MAX_SEQ_LEN:
                    print(f"✓ Generated time series length {calculated_len} meets maximum limit {self.MAX_SEQ_LEN}")
                    return scenario, calculated_len
                else:
                    print(f"⚠ Calculated sequence length {calculated_len} exceeds maximum limit {self.MAX_SEQ_LEN}, regenerating...")
                    print(f"  Time span: {time_info['time_span_raw']}")
                    print(f"  Sampling frequency: {time_info['sampling_frequency_raw']}")
                    
                    # Add more specific constraints for next attempt
                    if attempt < max_retries - 1:
                        print(f"Preparing attempt {attempt + 2} with stricter constraints...")
            else:
                print(f"⚠ Cannot extract time information from scenario, using default length {self.DEFAULT_SEQ_LEN}")
                return scenario, self.DEFAULT_SEQ_LEN
    
    def parse_scenario_to_structured_json(self, scenario: str, previous_feedback: str = None, iteration: int = 1) -> Dict[str, Any]:
        """Agent 2: Parse natural language scenario into structured JSON
        
        Args:
            scenario: Natural language scenario description
            previous_feedback: Feedback from Judge Agent 1 (if this is a retry)
            iteration: 当前迭代次数
        """
        
        # Use string replacement instead of format() to avoid issues with JSON braces
        prompt = SCENARIO_PARSING_PROMPT.replace("{scenario}", scenario)
        
        # Add feedback if this is a retry
        if previous_feedback:
            feedback_section = f"\n\n{'='*60}\nPREVIOUS ATTEMPT FEEDBACK FROM JUDGE:\n{'='*60}\n{previous_feedback}\n{'='*60}\n\nPlease address the issues mentioned above in your new parsing.\n"
            prompt = prompt + feedback_section
        
        # 记录输入
        input_data = {
            "scenario": scenario,
            "previous_feedback": previous_feedback,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt
        }
        
        response = self.model.generate_content(prompt)
        response_text = response.text
        print("\n=== Agent 2: Scenario Parsing Agent ===")
        if previous_feedback:
            print("(With judge feedback)")
        print("Raw response:")
        print(response_text[:300] + "..." if len(response_text) > 300 else response_text)
        print("-" * 50)
        
        # Extract JSON content
        json_text = self._extract_json_from_response(response_text)
        print("Extracted JSON:")
        print(json_text[:500] + "..." if len(json_text) > 500 else json_text)
        print("-" * 50)
        
        try:
            structured_json = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['nodes', 'edges', 'time_span', 'sampling_frequency', 
                             'variable', 'drift_patterns', 'adjacency_modulation']
            for field in required_fields:
                if field not in structured_json:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate node types
            for node in structured_json['nodes']:
                if node['type'] not in ['demand_source', 'propagation']:
                    raise ValueError(f"Invalid node type: {node['type']}")
            
            print("✓ Structured JSON validation passed")
            
            # 记录输出
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Agent 2: Scenario Parsing",
                    agent_type="parser",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"structured_json": structured_json},
                    metadata={
                        "has_feedback": previous_feedback is not None,
                        "num_nodes": len(structured_json['nodes']),
                        "num_edges": len(structured_json['edges']),
                        "validation_passed": True
                    }
                )
            
            return structured_json
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {e}"
            print(error_msg)
            print(f"Content attempted to parse: {json_text[:200]}...")
            
            # 记录错误
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Agent 2: Scenario Parsing",
                    agent_type="parser",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"error": error_msg, "raw_response": response_text},
                    metadata={"validation_passed": False, "error_type": "JSONDecodeError"}
                )
            
            raise ValueError("Invalid JSON format returned by Agent 2, please retry")
    
    def judge_scenario_parsing(self, scenario: str, parsed_json: Dict[str, Any], iteration: int = 1) -> Tuple[bool, Dict[str, Any]]:
        """Judge Agent 1: Validate if parsed JSON matches the original scenario"""
        
        print("\n=== Judge Agent 1: Scenario Parsing Validation ===")
        
        parsed_json_str = json.dumps(parsed_json, indent=2, ensure_ascii=False)
        
        prompt = JUDGE_SCENARIO_PARSING_PROMPT.format(
            expected_num_nodes=self.num_nodes,
            scenario=scenario,
            parsed_json=parsed_json_str
        )
        
        # 记录输入
        input_data = {
            "scenario": scenario[:500] + "..." if len(scenario) > 500 else scenario,
            "parsed_json": parsed_json,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt
        }
        
        response = self.model.generate_content(prompt)
        response_text = response.text
        print("Judge response:")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print("-" * 50)
        
        # Extract JSON from response
        json_text = self._extract_json_from_response(response_text)
        
        try:
            judgment = json.loads(json_text)
            approved = judgment.get("approved", False)
            
            if approved:
                print("✓ Judge Agent 1: APPROVED - Parsing is accurate")
            else:
                print("✗ Judge Agent 1: REJECTED - Issues found:")
                for issue in judgment.get("issues", []):
                    print(f"  - {issue['field']}: {issue['problem']}")
                    print(f"    Suggestion: {issue['suggestion']}")
                print(f"\nOverall: {judgment.get('overall_comment', 'N/A')}")
            
            # 记录输出
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Judge Agent 1: Scenario Parsing Validation",
                    agent_type="judge",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"judgment": judgment},
                    metadata={
                        "approved": approved,
                        "num_issues": len(judgment.get("issues", [])),
                        "overall_comment": judgment.get("overall_comment", "N/A")
                    }
                )
            
            return approved, judgment
            
        except json.JSONDecodeError as e:
            error_msg = f"Warning: Failed to parse judge response: {e}"
            print(error_msg)
            judgment = {"approved": False, "issues": [], "overall_comment": "Judge response parsing failed"}
            
            # 记录错误
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Judge Agent 1: Scenario Parsing Validation",
                    agent_type="judge",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"error": error_msg, "raw_response": response_text},
                    metadata={"approved": False, "error_type": "JSONDecodeError"}
                )
            
            # Default to rejection if we can't parse the judgment
            return False, judgment
    
    def parse_scenario_with_judge_loop(self, scenario: str, max_outer_iterations: int = 3, max_inner_iterations: int = 2) -> Dict[str, Any]:
        """Parse scenario with hierarchical Judge Agent 1 validation loop
        
        This implements a two-level feedback loop:
        - OUTER LOOP: Corrects Agent 1 (scenario generation logic)
        - INNER LOOP: Corrects Agent 2 (parsing fidelity)
        
        Args:
            scenario: Initial scenario text from Agent 1
            max_outer_iterations: Max iterations for scenario regeneration (Agent 1)
            max_inner_iterations: Max iterations for parsing correction (Agent 2)
        
        Returns:
            Validated parsed JSON
        """
        
        print("\n=== Agent 1 + Agent 2 + Judge Agent 1: Hierarchical Validation Loop ===")
        print(f"Maximum outer iterations (Agent 1): {max_outer_iterations}")
        print(f"Maximum inner iterations (Agent 2): {max_inner_iterations}")
        
        current_scenario = scenario
        agent1_feedback = None
        previous_scenario = None  # Track previous scenario for revision
        
        # OUTER LOOP: Scenario correction
        for outer_iter in range(max_outer_iterations):
            print(f"\n{'='*70}")
            print(f"OUTER LOOP - Iteration {outer_iter + 1}/{max_outer_iterations}")
            print(f"{'='*70}")
            
            # If this is a retry, revise scenario based on previous one and feedback
            if outer_iter > 0 and agent1_feedback and previous_scenario:
                print(f"\n🔄 Revising scenario based on Judge feedback...")
                print(f"   (Improving previous scenario, not generating from scratch)")
                current_scenario = self.generate_scenario_description(
                    previous_scenario=previous_scenario,
                    previous_feedback=agent1_feedback,
                    iteration=outer_iter + 1
                )
            
            agent2_feedback = None
            parsed_json = None
            
            # INNER LOOP: Parsing correction
            for inner_iter in range(max_inner_iterations):
                print(f"\n  --- Inner Loop - Iteration {inner_iter + 1}/{max_inner_iterations} ---")
                
                # Parse scenario with Agent 2 (with feedback if available)
                parsed_json = self.parse_scenario_to_structured_json(
                    current_scenario, 
                    agent2_feedback, 
                    iteration=inner_iter+1
                )
                
                # Judge validation (two-step diagnostic)
                approved, judgment = self.judge_scenario_parsing(
                    current_scenario, 
                    parsed_json, 
                    iteration=inner_iter+1
                )
                
                error_source = judgment.get('error_source', None)
                
                if approved:
                    print(f"\n✓ Scenario and parsing approved!")
                    print(f"   Total iterations: Outer={outer_iter + 1}, Inner={inner_iter + 1}")
                    return parsed_json
                
                elif error_source == 'agent2':
                    # AGENT 2 ERROR: Parsing fidelity issue
                    print(f"\n⚠ Judge identified PARSING ERROR (Agent 2's fault)")
                    print(f"   Issue type: Parsing Fidelity")
                    
                    if inner_iter < max_inner_iterations - 1:
                        # Continue inner loop with feedback to Agent 2
                        agent2_feedback = self._format_feedback_for_agent2(judgment)
                        print(f"\n  📋 Feedback for Agent 2:")
                        print(f"  {agent2_feedback[:200]}..." if len(agent2_feedback) > 200 else f"  {agent2_feedback}")
                    else:
                        print(f"\n⚠ Inner loop max iterations reached. Moving to outer loop.")
                        # Treat repeated parsing failures as a scenario problem
                        error_source = 'agent1'
                        break
                
                elif error_source == 'agent1':
                    # AGENT 1 ERROR: Scenario logic issue
                    print(f"\n⚠ Judge identified SCENARIO LOGIC ERROR (Agent 1's fault)")
                    print(f"   Issue type: Scenario Logic")
                    print(f"   Breaking inner loop to revise scenario...")
                    
                    # Save current scenario for revision and prepare feedback
                    previous_scenario = current_scenario
                    agent1_feedback = self._format_feedback_for_agent1(judgment)
                    break
                
                else:
                    # Fallback: treat as Agent 2 error
                    print(f"\n⚠ Warning: error_source unclear, treating as Agent 2 error")
                    error_source = 'agent2'
                    if inner_iter < max_inner_iterations - 1:
                        agent2_feedback = self._format_feedback_for_agent2(judgment)
            
            # After inner loop ends, check if we should continue outer loop
            if error_source == 'agent1':
                # Scenario logic error, continue outer loop
                if outer_iter < max_outer_iterations - 1:
                    print(f"\n  📋 Feedback for Agent 1:")
                    print(f"  {agent1_feedback[:300]}..." if len(agent1_feedback) > 300 else f"  {agent1_feedback}")
                    continue
                else:
                    print(f"\n⚠ Warning: Outer loop max iterations reached.")
                    break
            elif error_source == 'agent2':
                # Inner loop exhausted without approval
                print(f"\n⚠ Warning: Inner loop exhausted. Treating as scenario issue.")
                if outer_iter < max_outer_iterations - 1:
                    previous_scenario = current_scenario
                    agent1_feedback = "Parsing repeatedly failed. Please simplify the scenario description or provide clearer numerical values."
                    continue
                else:
                    break
        
        # If we've exhausted all iterations, return the last parsed result with a warning
        print(f"\n⚠ WARNING: Maximum iterations reached. Using last parsed result.")
        print(f"   This result may not be fully validated.")
        return parsed_json
    
    def _format_feedback_for_agent2(self, judgment: Dict[str, Any]) -> str:
        """Format feedback specifically for Agent 2 (parsing errors)"""
        feedback_text = f"PARSING FIDELITY ISSUES:\n\n"
        feedback_text += f"Overall: {judgment.get('feedback', 'Please improve parsing accuracy')}\n\n"
        
        parsing_issues = [issue for issue in judgment.get('issues', []) 
                         if issue.get('type') == 'Parsing Fidelity']
        
        if parsing_issues:
            feedback_text += f"Specific Issues ({len(parsing_issues)} found):\n"
            for idx, issue in enumerate(parsing_issues, 1):
                feedback_text += f"\n{idx}. Field: {issue.get('field', 'N/A')}\n"
                feedback_text += f"   Problem: {issue.get('problem', 'N/A')}\n"
                feedback_text += f"   Suggestion: {issue.get('suggestion', 'N/A')}\n"
        
        feedback_text += f"\n{judgment.get('overall_comment', '')}"
        return feedback_text
    
    def _format_feedback_for_agent1(self, judgment: Dict[str, Any]) -> str:
        """Format feedback specifically for Agent 1 (scenario logic errors)"""
        feedback_text = f"SCENARIO LOGIC ISSUES:\n\n"
        feedback_text += f"Overall: {judgment.get('feedback', 'Please improve scenario logic')}\n\n"
        
        logic_issues = [issue for issue in judgment.get('issues', []) 
                       if issue.get('type') == 'Scenario Logic']
        
        if logic_issues:
            feedback_text += f"Specific Issues ({len(logic_issues)} found):\n"
            for idx, issue in enumerate(logic_issues, 1):
                feedback_text += f"\n{idx}. Component: {issue.get('field', 'N/A')}\n"
                feedback_text += f"   Problem: {issue.get('problem', 'N/A')}\n"
                feedback_text += f"   Suggestion: {issue.get('suggestion', 'N/A')}\n"
        
        feedback_text += f"\n{judgment.get('overall_comment', '')}"
        return feedback_text
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON content from response text"""
        import re
        
        # Remove possible markdown code block markers
        text = response_text.strip()
        
        # Remove leading ```json or ```
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        # Remove trailing ```
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
        
        # Find the largest JSON object
        brace_count = 0
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    end_idx = i
                    break
        
        if start_idx != -1 and end_idx != -1:
            json_text = text[start_idx:end_idx+1]
            return json_text.strip()
        
        # If complete JSON object not found, try regex
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        
        # Finally try returning original text
        return text.strip()
    
    def generate_network_sde(self, structured_scenario: Dict[str, Any], seq_len: int = None, 
                             previous_feedback: str = None, iteration: int = 1) -> Dict[str, Any]:
        """Generate Network SDE parameters using Agents 3 & 4
        
        Args:
            structured_scenario: Structured scenario from Agent 2
            seq_len: Sequence length (optional)
            previous_feedback: Feedback from Judge Agent 2 (if this is a retry)
            iteration: 当前迭代次数
        """
        
        # Use instance seq_len if not provided
        if seq_len is None:
            seq_len = self.seq_len
        
        print(f"\n=== Network SDE Parameter Generation Pipeline ===")
        print(f"Number of nodes: {len(structured_scenario['nodes'])}")
        print(f"Sequence length: {seq_len}")
        
        # Agent 3: Generate SDE parameters from structured JSON
        sde_params = self._generate_sde_parameters(structured_scenario, previous_feedback, iteration)
        
        # Agent 4: Generate time-varying adjacency matrix from structured JSON
        time_varying_adj = self._generate_time_varying_adjacency(structured_scenario, previous_feedback, iteration)
        
        # Transfer drift_patterns to SDE parameters if not already present
        if "drift_patterns" in structured_scenario and "drift_patterns" not in sde_params:
            sde_params["drift_patterns"] = structured_scenario["drift_patterns"]
            
        # Calculate proper dt in hours based on sampling frequency
        true_dt_hours = getattr(self, "sampling_minutes", 60) / 60.0
        
        # Extract edge lags from structured scenario
        edge_lags = {f"{edge['source']}->{edge['target']}": edge['time_lag'] 
                     for edge in structured_scenario.get('edges', []) if 'time_lag' in edge}
        
        # Package everything together
        network_sde = {
            "structured_scenario": structured_scenario,
            "sequence_length": seq_len,
            "sde_parameters": sde_params,
            "time_varying_adjacency": time_varying_adj,
            "dt": true_dt_hours,
            "noise_correlation": self._generate_noise_correlation_matrix(),
            "edge_lags": edge_lags
        }
        
        return network_sde
    
    def _generate_sde_parameters(self, structured_scenario: Dict[str, Any], previous_feedback: str = None, iteration: int = 1) -> Dict[str, Any]:
        """Agent 3: Generate SDE parameters from structured JSON
        
        Args:
            structured_scenario: Structured scenario from Agent 2
            previous_feedback: Feedback from Judge Agent 2 (if this is a retry)
            iteration: 当前迭代次数
        """
        
        print("\n=== Agent 3: SDE Parameters Generation Agent ===")
        if previous_feedback:
            print("(With judge feedback)")
        
        # Convert structured scenario to JSON string for prompt
        structured_json_str = json.dumps(structured_scenario, indent=2, ensure_ascii=False)
        
        prompt = SDE_PARAMETERS_PROMPT.format(structured_scenario=structured_json_str)
        
        # Add feedback if this is a retry
        if previous_feedback:
            feedback_section = f"\n\n{'='*60}\nPREVIOUS ATTEMPT FEEDBACK FROM JUDGE:\n{'='*60}\n{previous_feedback}\n{'='*60}\n\nPlease address the issues mentioned above in your new parameter generation.\n"
            prompt = prompt + feedback_section
        
        # 记录输入
        input_data = {
            "structured_scenario": structured_scenario,
            "previous_feedback": previous_feedback,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt
        }
        
        response = self.model.generate_content(prompt)
        response_text = response.text
        print("Raw response:")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print("-" * 50)
        
        # Extract JSON content
        json_text = self._extract_json_from_response(response_text)
        
        try:
            sde_params = json.loads(json_text)
            
            # Validate constraints
            print("Validating SDE parameters...")
            
            # Check propagation nodes use mean_reverting only
            materialized = self._materialize_node_parameters(sde_params)
            for node_id, params in materialized.items():
                node_type = params.get('node_type', 'demand_source')
                drift_type = params.get('drift_type', 'mean_reverting')
                
                if node_type == 'propagation' and drift_type != 'mean_reverting':
                    raise ValueError(f"Constraint violation: Node {node_id} is propagation but uses drift_type={drift_type}")
                
                # Validate parameter ranges
                kappa = params.get('kappa', 0.25)
                lambda_val = params.get('lambda', 1.0)
                sigma = params.get('sigma', 0.3)
                
                if not (0.01 < kappa < 0.5):
                    print(f"Warning: Node {node_id} kappa={kappa} outside recommended range (0.01, 0.5)")
                if not (0.8 <= lambda_val <= 1.5):
                    print(f"Warning: Node {node_id} lambda={lambda_val} outside recommended range [0.8, 1.5]")
                if not (0 < sigma <= 0.5):
                    print(f"Warning: Node {node_id} sigma={sigma} outside recommended range (0, 0.5]")
            
            print("✓ SDE parameters validation passed")
            
            # 记录输出
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Agent 3: SDE Parameters Generation",
                    agent_type="generator",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"sde_parameters": sde_params},
                    metadata={
                        "has_feedback": previous_feedback is not None,
                        "validation_passed": True,
                        "num_nodes": len(structured_scenario['nodes'])
                    }
                )
            
            return sde_params
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {e}"
            print(error_msg)
            
            # 记录错误
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Agent 3: SDE Parameters Generation",
                    agent_type="generator",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"error": error_msg, "raw_response": response_text},
                    metadata={"validation_passed": False, "error_type": "JSONDecodeError"}
                )
            
            raise ValueError("Invalid JSON format returned by Agent 3")
    
    def _generate_time_varying_adjacency(self, structured_scenario: Dict[str, Any], previous_feedback: str = None, iteration: int = 1) -> Dict[str, Any]:
        """Agent 4: Generate time-varying adjacency matrix from structured JSON
        
        Args:
            structured_scenario: Structured scenario from Agent 2
            previous_feedback: Feedback from Judge Agent 2 (if this is a retry)
            iteration: 当前迭代次数
        """
        
        print("\n=== Agent 4: Time-Varying Adjacency Generation Agent ===")
        if previous_feedback:
            print("(With judge feedback)")
        
        # Convert structured scenario to JSON string for prompt
        structured_json_str = json.dumps(structured_scenario, indent=2, ensure_ascii=False)
        
        prompt = TIME_VARYING_ADJACENCY_PROMPT.format(structured_scenario=structured_json_str)
        
        # Add feedback if this is a retry
        if previous_feedback:
            feedback_section = f"\n\n{'='*60}\nPREVIOUS ATTEMPT FEEDBACK FROM JUDGE:\n{'='*60}\n{previous_feedback}\n{'='*60}\n\nPlease address the issues mentioned above in your new adjacency generation.\n"
            prompt = prompt + feedback_section
        
        # 记录输入
        input_data = {
            "structured_scenario": structured_scenario,
            "previous_feedback": previous_feedback,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt
        }
        
        response = self.model.generate_content(prompt)
        response_text = response.text
        print("Raw response:")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print("-" * 50)
        
        # Extract JSON content
        json_text = self._extract_json_from_response(response_text)
        
        try:
            time_varying_adj = json.loads(json_text)
            
            # Validate and enforce base_adjacency = 0.1 for all edges
            print("Enforcing base adjacency constraint (all edges = 0.1)...")
            
            # Use self.num_nodes instead of len(structured_scenario['nodes']) to match parameter generation
            num_nodes = self.num_nodes
            base_adj = np.zeros((num_nodes, num_nodes))
            
            # Set base_adjacency to 0.1 for directed edges in structured scenario
            for edge in structured_scenario['edges']:
                source = edge['source']
                target = edge['target']
                base_adj[source, target] = 0.1
                # No forced symmetry - edges are directional
            
            time_varying_adj["base_adjacency"] = base_adj.tolist()
            
            print(f"✓ Base adjacency enforced: {len(structured_scenario['edges'])} edges, all = 0.1")
            print(f"✓ Base adjacency matrix (directional):\n{base_adj}")
            
            # Validate time_modulation structure
            if 'time_modulation' not in time_varying_adj:
                raise ValueError("Missing time_modulation in adjacency output")
            
            print("✓ Time-varying adjacency validation passed")
            
            # 记录输出
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Agent 4: Time-Varying Adjacency Generation",
                    agent_type="generator",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"time_varying_adjacency": time_varying_adj},
                    metadata={
                        "has_feedback": previous_feedback is not None,
                        "validation_passed": True,
                        "num_edges": len(structured_scenario['edges']),
                        "has_time_patterns": len(time_varying_adj.get('time_modulation', {}).get('patterns', [])) > 0,
                        "has_weekly_patterns": time_varying_adj.get('time_modulation', {}).get('weekly_patterns', {}).get('enabled', False)
                    }
                )
            
            return time_varying_adj
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {e}"
            print(error_msg)
            
            # 记录错误
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Agent 4: Time-Varying Adjacency Generation",
                    agent_type="generator",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"error": error_msg, "raw_response": response_text},
                    metadata={"validation_passed": False, "error_type": "JSONDecodeError"}
                )
            
            raise ValueError("Invalid JSON format returned by Agent 4")
    
    def _generate_noise_correlation_matrix(self) -> np.ndarray:
        """Generate noise correlation matrix between nodes"""
        # For now, assume independent noise (identity matrix)
        # Could be extended to have correlated noise based on spatial proximity
        return np.eye(self.num_nodes)
    
    def judge_parameter_validation(self, structured_scenario: Dict[str, Any], 
                                   network_sde: Dict[str, Any], 
                                   ts_data: np.ndarray,
                                   temp_viz_path: str,
                                   iteration: int = 1,
                                   previous_judgment: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """Judge Agent 2: Validate parameters using multimodal LLM with visualization"""
        
        print("\n=== Judge Agent 2: Parameter Validation (Multimodal) ===")
        if previous_judgment:
            print("(With feedback from previous iteration)")
        
        # Prepare text content for the prompt
        structured_scenario_str = json.dumps(structured_scenario, indent=2, ensure_ascii=False)
        sde_params_str = json.dumps(network_sde['sde_parameters'], indent=2, ensure_ascii=False)
        adjacency_str = json.dumps(network_sde['time_varying_adjacency'], indent=2, ensure_ascii=False)
        
        # Format previous assessment section
        previous_assessment_section = ""
        if previous_judgment:
            previous_assessment_section = f"""
**Previous Assessment (Iteration {iteration - 1}):**
Please review your previous feedback below and check if the new parameters and visualization have addressed these issues.
```json
{json.dumps(previous_judgment, indent=2, ensure_ascii=False)}
```
"""
        
        prompt = JUDGE_PARAMETER_VALIDATION_PROMPT.format(
            structured_scenario=structured_scenario_str,
            sde_parameters=sde_params_str,
            time_varying_adjacency=adjacency_str,
            previous_assessment_section=previous_assessment_section
        )
        
        # 记录输入（不包括图像数据，只记录路径）
        input_data = {
            "structured_scenario": structured_scenario,
            "sde_parameters": network_sde['sde_parameters'],
            "time_varying_adjacency": network_sde['time_varying_adjacency'],
            "visualization_path": temp_viz_path,
            "ts_data_shape": ts_data.shape,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "previous_judgment": previous_judgment
        }
        
        # Upload image for multimodal analysis
        try:
            from PIL import Image
            image = Image.open(temp_viz_path)
            
            # Use Gemini's multimodal capability
            response = self.model.generate_content([prompt, image])
            response_text = response.text
            
            print("Judge response:")
            print(response_text[:600] + "..." if len(response_text) > 600 else response_text)
            print("-" * 50)
            
            # Extract JSON from response
            json_text = self._extract_json_from_response(response_text)
            
            judgment = json.loads(json_text)
            approved = judgment.get("approved", False)
            
            if approved:
                print("✓ Judge Agent 2: APPROVED - Parameters are reasonable")
            else:
                print("✗ Judge Agent 2: REJECTED - Issues found:")
                
                for issue in judgment.get("parameter_issues", []):
                    print(f"  - Node {issue['node_id']}, {issue['parameter']}: {issue['problem']}")
                    print(f"    Current: {issue['current_value']}, Suggested: {issue['suggested_value']}")
                
                for issue in judgment.get("adjacency_issues", []):
                    print(f"  - Edge {issue['edge']}: {issue['problem']}")
                    print(f"    Suggestion: {issue['suggestion']}")
                
                print(f"\nVisual Assessment: {judgment.get('visual_assessment', 'N/A')[:200]}...")
                print(f"Overall: {judgment.get('overall_comment', 'N/A')[:200]}...")
            
            # 记录输出
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Judge Agent 2: Parameter Validation",
                    agent_type="judge",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"judgment": judgment},
                    metadata={
                        "approved": approved,
                        "num_parameter_issues": len(judgment.get("parameter_issues", [])),
                        "num_adjacency_issues": len(judgment.get("adjacency_issues", [])),
                        "visual_assessment": judgment.get("visual_assessment", "N/A")[:100],
                        "overall_comment": judgment.get("overall_comment", "N/A")[:100]
                    }
                )
            
            return approved, judgment
            
        except Exception as e:
            error_msg = f"Warning: Multimodal judge validation failed: {e}"
            print(error_msg)
            judgment = {"approved": True, "parameter_issues": [], "adjacency_issues": [], 
                       "visual_assessment": "Judge validation failed", 
                       "overall_comment": "Auto-approved due to judge error"}
            
            # 记录错误
            if self.logger:
                self.logger.log_agent_interaction(
                    agent_name="Judge Agent 2: Parameter Validation",
                    agent_type="judge",
                    iteration=iteration,
                    input_data=input_data,
                    output_data={"error": error_msg, "judgment": judgment},
                    metadata={"approved": True, "error_type": str(type(e).__name__)}
                )
            
            # Default to approval if judge fails
            return True, judgment
    
    def generate_network_sde_with_judge_loop(self, structured_scenario: Dict[str, Any], 
                                             seq_len: int = None, 
                                             max_iterations: int = 3) -> Dict[str, Any]:
        """Generate Network SDE with Judge Agent 2 validation loop"""
        
        print("\n=== Agents 3 & 4 + Judge Agent 2: Parameter Generation with Validation Loop ===")
        print(f"Maximum iterations allowed: {max_iterations}")
        
        import tempfile
        import os
        import shutil
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for judge validation
        
        previous_feedback = None
        network_sde = None
        ts_data = None
        generation_info = None
        previous_judgment = None # Store judgment for next iteration's judge
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Generate SDE parameters and adjacency with feedback from previous iteration
            network_sde = self.generate_network_sde(structured_scenario, seq_len, previous_feedback, iteration=iteration+1)
            
            # Generate time series data for validation
            print("\nGenerating preview data for judge validation...")
            ts_data, generation_info = self.generate_spatiotemporal_data(network_sde)
            
            # Create visualization using the existing method
            with tempfile.NamedTemporaryFile(suffix='.png', prefix='judge_validation_', delete=False) as temp_f:
                temp_viz_path = temp_f.name
            
            print(f"Creating visualization for judge evaluation...")
            self.visualize_network_sde_results(ts_data, network_sde, generation_info, save_path=temp_viz_path)
            
            # Save a copy of the judge's input image to the session log directory
            if self.logger and self.logger.session_dir:
                log_image_path = self.logger.session_dir / f"judge_input_iter{iteration + 1}.png"
                try:
                    shutil.copy(temp_viz_path, log_image_path)
                    print(f"  🖼️  Judge input image saved to: {log_image_path}")
                except Exception as e:
                    print(f"  ⚠️ Could not save judge input image: {e}")

            # Judge validation (multimodal), now passing previous_judgment
            approved, judgment = self.judge_parameter_validation(
                structured_scenario, network_sde, ts_data, temp_viz_path,
                iteration=iteration+1, previous_judgment=previous_judgment
            )
            
            # Store the current judgment for the next iteration
            previous_judgment = judgment
            
            # Clean up temp file
            try:
                os.remove(temp_viz_path)
            except:
                pass
            
            if approved:
                print(f"\n✓ Parameters approved after {iteration + 1} iteration(s)")
                # Store the validated ts_data and generation_info in network_sde for later use
                network_sde['_validated_ts_data'] = ts_data
                network_sde['_validated_generation_info'] = generation_info
                return network_sde
            else:
                print(f"\n⚠ Iteration {iteration + 1} rejected, preparing feedback for next iteration...")
                
                # Prepare detailed feedback for next iteration
                if iteration < max_iterations - 1:
                    feedback_text = f"OVERALL COMMENT: {judgment.get('overall_comment', '')}\n"
                    feedback_text += f"\nVISUAL ASSESSMENT: {judgment.get('visual_assessment', '')}\n"
                    
                    # Parameter issues
                    param_issues = judgment.get('parameter_issues', [])
                    if param_issues:
                        feedback_text += f"\nPARAMETER ISSUES ({len(param_issues)} found):\n"
                        for idx, issue in enumerate(param_issues, 1):
                            feedback_text += f"\n{idx}. Node {issue.get('node_id', 'N/A')} - {issue.get('parameter', 'N/A')}\n"
                            feedback_text += f"   Current Value: {issue.get('current_value', 'N/A')}\n"
                            feedback_text += f"   Problem: {issue.get('problem', 'N/A')}\n"
                            feedback_text += f"   Suggested: {issue.get('suggested_value', 'N/A')}\n"
                    
                    # Adjacency issues
                    adj_issues = judgment.get('adjacency_issues', [])
                    if adj_issues:
                        feedback_text += f"\nADJACENCY ISSUES ({len(adj_issues)} found):\n"
                        for idx, issue in enumerate(adj_issues, 1):
                            feedback_text += f"\n{idx}. Edge {issue.get('edge', 'N/A')}\n"
                            feedback_text += f"   Problem: {issue.get('problem', 'N/A')}\n"
                            feedback_text += f"   Suggestion: {issue.get('suggestion', 'N/A')}\n"
                    
                    previous_feedback = feedback_text
                    print(f"\nFeedback prepared for iteration {iteration + 2}:")
                    print(feedback_text[:400] + "..." if len(feedback_text) > 400 else feedback_text)
        
        # If we've exhausted iterations, return the last result with a warning
        print(f"\n⚠ Warning: Maximum iterations ({max_iterations}) reached. Using last generated parameters.")
        if network_sde is not None:
            network_sde['_validated_ts_data'] = ts_data
            network_sde['_validated_generation_info'] = generation_info
        return network_sde
    
    def _materialize_node_parameters(self, hierarchical_params: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Convert hierarchical parameters to per-node materialized parameters"""
        
        global_defaults = hierarchical_params["global_defaults"]
        group_params = hierarchical_params.get("group_params", {})
        node_overrides = hierarchical_params.get("node_overrides", {})
        
        materialized = {}
        
        for node_id in range(self.num_nodes):
            node_id_str = str(node_id)
            
            # Start with global defaults
            params = global_defaults.copy()
            
            # Apply group parameters if specified
            if node_id_str in node_overrides and "group" in node_overrides[node_id_str]:
                group_name = node_overrides[node_id_str]["group"]
                if group_name in group_params:
                    params.update(group_params[group_name])
            
            # Apply node-specific overrides
            if node_id_str in node_overrides:
                node_override = node_overrides[node_id_str].copy()
                # Remove group key as it's not a parameter
                node_override.pop("group", None)
                params.update(node_override)
            
            # Ensure stability constraints are applied to global/group level params if no pattern overrides them
            if "drift_patterns" not in params:
                params["kappa"] = max(0.01, min(0.5, params.get("kappa", 0.25)))
                params["sigma"] = max(0.0, params.get("sigma", 0.8))
                params["lambda"] = max(0.8, min(1.5, params.get("lambda", 1.2)))
            
            # Enforce drift type constraint for propagation nodes
            node_type = params.get("node_type", "demand_source")
            if node_type == "propagation":
                if "drift_patterns" in params:
                    for pattern in params["drift_patterns"]:
                        if pattern.get("drift_type") != "mean_reverting":
                            print(f"Warning: Node {node_id} is propagation type but has a non-mean_reverting pattern. Forcing to 'mean_reverting'.")
                            pattern["drift_type"] = "mean_reverting"
                elif params.get("drift_type") != "mean_reverting":
                    print(f"Warning: Node {node_id} is propagation type but has drift_type='{params.get('drift_type')}'. Forcing to 'mean_reverting'")
                    params["drift_type"] = "mean_reverting"
            
            materialized[node_id] = params
        
        return materialized
    
    def generate_spatiotemporal_data(self, network_sde: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate spatial-temporal data using Network SDE"""
        
        self.sde_params = network_sde["sde_parameters"] # Set sde_params attribute
        seq_len = network_sde["sequence_length"]
        dt = network_sde["dt"]
        hierarchical_sde_params = network_sde["sde_parameters"]
        time_varying_adj = network_sde["time_varying_adjacency"]
        noise_corr = network_sde["noise_correlation"]
        
        print(f"\n=== Starting Spatiotemporal Data Generation ===")
        print(f"Sequence length: {seq_len}")
        print(f"Time step: {dt}")
        
        # Materialize node parameters from hierarchical structure
        materialized_params = self._materialize_node_parameters(hierarchical_sde_params)
        time_modulation = hierarchical_sde_params.get("time_modulation", {})
        
        print(f"✓ Parameter materialization completed, generated {len(materialized_params)}  node parameters")
        
        # Initialize data array
        data = np.zeros((self.num_nodes, seq_len))
        
        # Initialize independent SDE data (without network coupling) for comparison
        data_independent = np.zeros((self.num_nodes, seq_len))
        
        # Set initial conditions
        for i in range(self.num_nodes):
            node_params = materialized_params[i]
            # Use the node's main baseline as a fallback
            initial_baseline = node_params.get("baseline", 0)

            # If drift patterns exist, find the one for t=0 and use its baseline for initialization
            if "drift_patterns" in node_params:
                for pattern in node_params["drift_patterns"]:
                    start_time, end_time = pattern["time_range"]
                    if start_time <= 0 <= end_time:
                        # If the first pattern is sinusoidal, initialize at its t=0 value
                        if pattern.get("drift_type") == "sinusoidal":
                            baseline = pattern.get("baseline", initial_baseline)
                            amplitude = pattern.get("A", 0)
                            omega = pattern.get("omega", 0)
                            phi = pattern.get("phi", 0)
                            initial_baseline = baseline + amplitude * np.sin(omega * 0 + phi)
                            print(f"  Node {i}: Initializing with sinusoidal value at t=0: {initial_baseline:.2f}")
                        else:
                            initial_baseline = pattern.get("baseline", initial_baseline)
                            print(f"  Node {i}: Initializing with baseline from t=0 pattern: {initial_baseline:.2f}")
                        break  # Found the pattern for t=0
            
            # Initialize with the determined baseline plus small noise
            initial_val = initial_baseline + np.random.normal(0, 0.01 * abs(initial_baseline) + 1e-6) # Reduced noise
            data[i, 0] = initial_val
            data_independent[i, 0] = initial_val  # Same initial condition
        
        # Generate adjacency matrix for each time step
        adj_matrices = self._compute_time_varying_adjacency_matrices(time_varying_adj, seq_len)
        
        # Construct lag matrix from edge_lags
        lag_steps_mat = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        edge_lags = network_sde.get("edge_lags", {})
        if edge_lags:
            print(f"✓ Applying time lags: {edge_lags}")
                            # If edge_lags were specified in hours, convert to steps: int(round(hours * 60 / self.sampling_minutes))
            for k, v in edge_lags.items():
                if "->" in k:
                    s, t_ = map(int, k.split("->"))
                    lag_steps_mat[s, t_] = int(v)  # 已是"步"的话直接赋值
        
        # Preheating strategy: fill initial time steps with baseline values
        max_lag = int(lag_steps_mat.max()) if lag_steps_mat.max() > 0 else 0
        if max_lag > 0:
            print(f"✓ Preheating first {max_lag + 1} time steps with baseline values")
            for i in range(self.num_nodes):
                baseline = materialized_params[i]["baseline"]
                # 用 baseline + 微小噪声 预热前 max_lag+1 个点
                for t0 in range(1, min(seq_len, max_lag + 1)):
                    if data[i, t0] == 0.0:
                        data[i, t0] = baseline + np.random.normal(0, 0.02 * abs(baseline))
        
        # SDE integration using Euler-Maruyama method
        for t in range(1, seq_len):
            current_time = t * dt
            current_adj = adj_matrices[t]
            
            # Generate correlated noise
            dW = np.random.multivariate_normal(np.zeros(self.num_nodes), noise_corr * dt)
            
            for i in range(self.num_nodes):
                # Current state (network-coupled)
                X_i = data[i, t-1]
                
                # Current state (independent)
                X_i_indep = data_independent[i, t-1]
                
                # Drift term (same for both) - NOW TIME-VARYING
                drift = self._compute_drift_new(i, X_i, current_time, materialized_params[i], t)
                drift_indep = self._compute_drift_new(i, X_i_indep, current_time, materialized_params[i], t)
                
                # Diffusion term (same for both)
                diffusion = self._compute_diffusion_new(i, X_i, materialized_params[i])
                diffusion_indep = self._compute_diffusion_new(i, X_i_indep, materialized_params[i])
                
                # Coupling term (only for network-coupled version)
                if edge_lags:
                    coupling = self._compute_coupling_with_lag(
                        i, t, data, adj_matrices, materialized_params[i], lag_steps_mat
                    )
                else:
                    coupling = self._compute_coupling_new(i, data[:, t-1], current_adj, materialized_params[i])
                
                # External force (reserved for future use)
                external = 0.0
                
                # SDE update (network-coupled)
                data[i, t] = X_i + drift * dt + diffusion * dW[i] + coupling * dt + external
                
                # SDE update (independent, no coupling)
                data_independent[i, t] = X_i_indep + drift_indep * dt + diffusion_indep * dW[i]
                
                # Apply bounds to keep values realistic (allow negative for some scenarios)
                max_val = abs(materialized_params[i]["baseline"]) * 6 + 50
                min_val = -max_val if materialized_params[i]["baseline"] < 0 else 0
                data[i, t] = np.clip(data[i, t], min_val, max_val)
                data_independent[i, t] = np.clip(data_independent[i, t], min_val, max_val)
        
        # Package results
        generation_info = {
            "hierarchical_sde_parameters": hierarchical_sde_params,
            "materialized_parameters": materialized_params,
            "adjacency_matrices": adj_matrices,
            "dt": dt,
            "integration_method": "Euler-Maruyama",
            "independent_sde_data": data_independent  # Add independent SDE data for visualization
        }
        
        # Also add adjacency matrices to network_sde for visualization
        network_sde["adjacency_matrices"] = adj_matrices
        
        print(f"✓ Successfully generated spatiotemporal data for {self.num_nodes} nodes")
        print(f"✓ Data shape: {data.shape}")
        print(f"✓ Independent SDE data (no coupling) also generated for comparison")
        
        return data, generation_info
    
    def _compute_time_varying_adjacency_matrices(self, time_varying_adj: Dict[str, Any], seq_len: int) -> List[np.ndarray]:
        """Compute adjacency matrix for each time step"""
        
        base_adj = np.array(time_varying_adj["base_adjacency"], dtype=np.float64)
        modulation = time_varying_adj["time_modulation"]
        
        adj_matrices = []
        
        for t in range(seq_len):
            # Start with base adjacency
            current_adj = base_adj.copy().astype(np.float64)
            
            # Apply all patterns (unified approach)
            if "patterns" in modulation:
                current_adj = self._apply_time_modulation_patterns(current_adj, t, modulation["patterns"])
            
            adj_matrices.append(current_adj)
        
        return adj_matrices
    
    def _apply_time_modulation_patterns(self, adj_matrix, time_step, patterns):
        """Apply unified time modulation patterns to adjacency matrix
        
        Args:
            adj_matrix: Current adjacency matrix
            time_step: Current time step
            patterns: List of pattern dictionaries with time_range, edge_modulations
            
        Returns:
            Modified adjacency matrix
        """
        modified_adj = adj_matrix.copy()
        
        for pattern in patterns:
            time_range = pattern.get("time_range", [])
            if len(time_range) == 2:
                start = max(0, time_range[0])
                end = time_range[1] + 4
                if start <= time_step < end:
                    # Apply edge modulations for this pattern
                    for edge_key, mod in pattern.get("edge_modulations", {}).items():
                        if edge_key == "all_edges":
                            # Apply to all non-zero edges
                            modified_adj[modified_adj > 0] *= float(mod["multiplier"])
                        elif "->" in edge_key:
                            i, j = map(int, edge_key.split("->"))
                            if 0 <= i < modified_adj.shape[0] and 0 <= j < modified_adj.shape[1]:
                                modified_adj[i, j] *= float(mod["multiplier"])
        
        return modified_adj
    
    def _apply_daily_modulation(self, adj_matrix, time_step, daily_patterns):
        modified_adj = adj_matrix.copy()
        sampling_minutes = getattr(self, "sampling_minutes", 60)
        hour_of_day = ((time_step * sampling_minutes) // 60) % 24

        for pattern in daily_patterns.get("patterns", []):
            s, e = pattern["time_range"]
            if s > 24 or e > 24:
                s_mod, e_mod = s % 24, e % 24
                print(f"[warn] daily time_range {s}-{e} -> normalized to {s_mod}-{e_mod}")
                s, e = s_mod, e_mod
            if s <= hour_of_day < e:
                for edge_key, mod in pattern["edge_modulations"].items():
                    if "->" in edge_key:
                        i, j = map(int, edge_key.split("->"))
                        modified_adj[i, j] *= float(mod["multiplier"])
        return modified_adj
    
    
    def _apply_weekly_modulation(self, adj_matrix, time_step, weekly_patterns):
        # Weekly modulation now only affects baseline (in _compute_drift_new)
        # Return adjacency matrix unchanged to avoid double application
        return adj_matrix.copy()
    
    def _apply_seasonal_modulation(self, adj_matrix, time_step, seasonal_patterns):
        """Apply seasonal modulation to adjacency matrix
        
        Seasonal patterns have ~365-day cycles (e.g., tourism routes, agricultural flows, 
        holiday shopping, weather-dependent transport)
        
        Args:
            adj_matrix: Current adjacency matrix
            time_step: Current time step
            seasonal_patterns: Seasonal pattern configuration
            
        Returns:
            Modified adjacency matrix with seasonal modulation
        """
        modified_adj = adj_matrix.copy()
        sampling_minutes = getattr(self, "sampling_minutes", 60)
        
        # Calculate day of year (0-364)
        # CRITICAL FIX: For daily sampling, time_step IS the day number
        # For hourly sampling, we need to convert
        if sampling_minutes >= 24 * 60:  # Daily or longer sampling
            day_of_year = time_step % 365
        else:  # Hourly or sub-daily sampling
            total_minutes = time_step * sampling_minutes
            day_of_year = (total_minutes // (24 * 60)) % 365
        
        for pattern in seasonal_patterns.get("patterns", []):
            # Season defined by day range in year (e.g., summer: day 150-240)
            season_start = pattern.get("day_range", [0, 0])[0]
            season_end = pattern.get("day_range", [0, 0])[1]
            
            # Check if current day is in season (handle wrap-around for winter)
            in_season = False
            if season_start <= season_end:
                # Normal case: summer (day 150-240), spring (day 60-150)
                in_season = season_start <= day_of_year < season_end
            else:
                # Wrap-around case: winter (day 335 to day 60)
                in_season = day_of_year >= season_start or day_of_year < season_end
            
            if in_season:
                # Apply edge-specific modulations for this season
                for edge_key, mod in pattern.get("edge_modulations", {}).items():
                    if "->" in edge_key:
                        i, j = map(int, edge_key.split("->"))
                        if i < self.num_nodes and j < self.num_nodes:
                            multiplier = float(mod.get("multiplier", 1.0))
                            old_val = modified_adj[i, j]
                            modified_adj[i, j] *= multiplier
                            # Debug output (only print first few times)
                            if time_step < 3 or (time_step >= season_start and time_step <= season_start + 2):
                                print(f"  [t={time_step}] Seasonal: edge {i}->{j}: {old_val:.3f} × {multiplier:.1f} = {modified_adj[i, j]:.3f}")
        
        return modified_adj

    def _compute_drift_new(self, node_id: int, X_i: float, time: float, params: Dict[str, Any], 
                           time_step: int) -> float:
        """Compute drift term, now with time-varying patterns."""
        
        # 'params' contains the materialized parameters for the node.
        # We check if it has a 'drift_patterns' list.
        if "drift_patterns" in params:

            effective_time_step = time_step

            # Global repeat configuration is stored at the top level of sde_params
            drift_patterns_config = self.sde_params.get("drift_patterns", {})

            # Handle repeating patterns using repeat_period
            if drift_patterns_config.get("repeat", False) and "repeat_period" in drift_patterns_config:
                cycle_duration = drift_patterns_config["repeat_period"]
                if cycle_duration > 0:
                    effective_time_step = time_step % cycle_duration

            active_pattern = None
            # Iterate through the SDE-specific patterns defined for this node
            for pattern in params["drift_patterns"]:
                start_time, end_time = pattern["time_range"]
                # Time range is inclusive
                if start_time <= effective_time_step <= end_time:
                    active_pattern = pattern
                    break

            if active_pattern:
                time_modulation = active_pattern.get("time_modulation", {})

                # Combine base node params with active pattern-specific params
                combined_params = params.copy()
                combined_params.update(active_pattern)

                return self._compute_drift_logic(node_id, X_i, time, combined_params, time_modulation, effective_time_step)

        # Fallback if no patterns are defined for the node, or no pattern is active for the current time_step
        time_modulation = params.get("time_modulation", {}) # Legacy support
        return self._compute_drift_logic(node_id, X_i, time, params, time_modulation, time_step)

    def _compute_drift_logic(self, node_id: int, X_i: float, time: float, params: Dict[str, Any], 
                             time_modulation: Dict[str, Any], time_step: int) -> float:
        """The core logic for computing drift based on a set of parameters."""
        drift_type = params.get("drift_type", "mean_reverting")
        node_type = params.get("node_type", "demand_source")

        if drift_type == "mean_reverting":
            return self._compute_mean_reverting_drift(node_id, X_i, time, params, time_modulation, time_step, node_type)
        elif drift_type == "constant":
            return self._compute_constant_drift(params)
        elif drift_type == "sinusoidal":
            return self._compute_sinusoidal_drift(node_id, X_i, time, params, time_modulation, time_step, node_type)
        elif drift_type == "logistic":
            return self._compute_logistic_drift(X_i, params)
        else:
            print(f"Warning: Unknown drift_type '{drift_type}' for node {node_id}, using mean_reverting")
            return self._compute_mean_reverting_drift(node_id, X_i, time, params, time_modulation, time_step, node_type)
    
    def _compute_mean_reverting_drift(self, node_id: int, X_i: float, time: float, params: Dict[str, Any], 
                                    time_modulation: Dict[str, Any], time_step: int, node_type: str) -> float:
        """Compute mean-reverting drift: kappa * (mu_t - X_t)"""
        
        kappa = params.get("kappa", 0.25)
        baseline = params.get("baseline", 50.0)
        
        # Apply baseline modulations only for demand source nodes
        if node_type == "demand_source":
            effective_baseline = self._apply_baseline_modulations(baseline, node_id, time_modulation, time_step, params)
        else:
            # Propagation nodes use static baseline (minimal local background flow)
            effective_baseline = baseline
        
        # Basic mean-reverting drift
        base_drift = kappa * (effective_baseline - X_i)
        
        # Apply time modulation factor (does not change baseline itself)
        modulation_factor = self._compute_time_modulation_factor(time_modulation, time_step)
        
        return base_drift * modulation_factor
    
    def _compute_constant_drift(self, params: Dict[str, Any]) -> float:
        """Compute constant drift: alpha"""
        alpha = params.get("alpha", 0.0)
        return alpha
    
    def _compute_sinusoidal_drift(self, node_id: int, X_i: float, time: float, params: Dict[str, Any], 
                                time_modulation: Dict[str, Any], time_step: int, node_type: str) -> float:
        """Compute sinusoidal drift: kappa * (baseline + A * sin(omega * t + phi) - X_t)
        
        SINGLE HARMONIC ONLY - A, omega, phi must be scalars, not arrays.
        """
        
        kappa = params.get("kappa", 0.25)
        baseline = params.get("baseline", 50.0)
        amplitude = params.get("A", 10.0)  # Sinusoidal amplitude (scalar only)
        omega = params.get("omega", 2 * np.pi / 24)  # Default: daily cycle (scalar only)
        phi = params.get("phi", 0.0)  # Phase shift in radians (scalar only)
        
        # Validate that parameters are scalars, not arrays (single harmonic constraint)
        if not isinstance(amplitude, (int, float)) or not isinstance(omega, (int, float)) or not isinstance(phi, (int, float)):
            print(f"Warning: Node {node_id} sinusoidal parameters must be scalars. Converting arrays to first element.")
            amplitude = float(amplitude[0] if hasattr(amplitude, '__getitem__') else amplitude)
            omega = float(omega[0] if hasattr(omega, '__getitem__') else omega)
            phi = float(phi[0] if hasattr(phi, '__getitem__') else phi)
        
        # Calculate time-varying baseline with sinusoidal component and phase shift
        # CRITICAL FIX: Use time_step directly (already in correct units based on omega)
        # omega is defined in radians per time_step, not per hour
        sinusoidal_baseline = baseline + amplitude * np.sin(omega * time_step + phi)
        
        # Apply additional baseline modulations only for demand source nodes
        if node_type == "demand_source":
            effective_baseline = self._apply_baseline_modulations(sinusoidal_baseline, node_id, time_modulation, time_step, params)
        else:
            effective_baseline = sinusoidal_baseline
        
        # Mean-reverting drift with sinusoidal baseline
        base_drift = kappa * (effective_baseline - X_i)
        
        # Apply time modulation factor
        modulation_factor = self._compute_time_modulation_factor(time_modulation, time_step)
        
        return base_drift * modulation_factor
    
    def _compute_logistic_drift(self, X_i: float, params: Dict[str, Any]) -> float:
        """Compute logistic drift: r * X_t * (1 - X_t / K)"""
        
        growth_rate = params.get("r", 0.05)
        carrying_capacity = params.get("K", 100.0)
        
        # Prevent division by zero and negative values
        if carrying_capacity <= 0:
            carrying_capacity = 100.0
        
        # Logistic growth with saturation
        drift = growth_rate * X_i * (1 - X_i / carrying_capacity)
        
        return drift
    
    def _apply_baseline_modulations(self, baseline: float, node_id: int, time_modulation: Dict[str, Any], 
                                    time_step: int, node_params: Dict[str, Any] = None) -> float:
        """Apply time-dependent baseline modulations (weekend and seasonal cycles)
        
        Args:
            baseline: Base value to modulate
            node_id: Node identifier
            time_modulation: Global time modulation settings (for backward compatibility)
            time_step: Current time step
            node_params: Node-specific parameters (including per-node weekend_multiplier, seasonal_multiplier)
        """
        effective_baseline = baseline
        
        # Per-node weekend multiplier (new approach)
        if node_params and "weekend_multiplier" in node_params:
            weekend_multiplier = node_params.get("weekend_multiplier", 1.0)
            # Use consistent time calculation
            sampling_minutes = getattr(self, "sampling_minutes", 60)
            week_minutes = 7 * 24 * 60
            t_min = (time_step * sampling_minutes) % week_minutes
            weekend_start_min = 5 * 24 * 60 + 18 * 60  # Friday 18:00
            weekend_end_min = 7 * 24 * 60               # Sunday 24:00
            if weekend_start_min <= t_min < weekend_end_min:  # Weekend
                effective_baseline *= weekend_multiplier
        # NOTE: Global weekly_cycle removed in new schema
        # Time dynamics now encoded via node peaks and adjacency_modulation
        # elif time_modulation:
        #     weekly_cycle = time_modulation.get("weekly_cycle", {})
        #     if weekly_cycle.get("enabled", False):
        #         weekend_multiplier = weekly_cycle.get("weekend_multiplier", 1.0)
        #         sampling_minutes = getattr(self, "sampling_minutes", 60)
        #         week_minutes = 7 * 24 * 60
        #         t_min = (time_step * sampling_minutes) % week_minutes
        #         weekend_start_min = 5 * 24 * 60 + 18 * 60  # Friday 18:00
        #         weekend_end_min = 7 * 24 * 60               # Sunday 24:00
        #         if weekend_start_min <= t_min < weekend_end_min:  # Weekend
        #             effective_baseline *= weekend_multiplier
        
        # Seasonal modulation (per-node or global)
        seasonal_mult = None
        if node_params and "seasonal_multiplier" in node_params:
            seasonal_mult = node_params.get("seasonal_multiplier")
        elif time_modulation:
            seasonal_cycle = time_modulation.get("seasonal_cycle", {})
            if seasonal_cycle.get("enabled", False):
                seasonal_mult = seasonal_cycle.get("seasonal_multiplier")
        
        if seasonal_mult is not None:
            sampling_minutes = getattr(self, "sampling_minutes", 60)
            total_minutes = time_step * sampling_minutes
            day_of_year = (total_minutes // (24 * 60)) % 365
            
            # Check if current day is in any defined season
            for season in seasonal_mult:
                season_start = season.get("day_range", [0, 0])[0]
                season_end = season.get("day_range", [0, 0])[1]
                
                # Check if in season (handle wrap-around for winter)
                in_season = False
                if season_start <= season_end:
                    # Normal case: summer, spring, etc.
                    in_season = season_start <= day_of_year < season_end
                else:
                    # Wrap-around case: winter (day 335 to day 60)
                    in_season = day_of_year >= season_start or day_of_year < season_end
                
                if in_season:
                    effective_baseline *= season.get("multiplier", 1.0)
                    break  # Apply only first matching season
        
        return effective_baseline
    
    def _compute_time_modulation_factor(self, time_modulation: Dict[str, Any], time_step: int) -> float:
        """Compute time modulation factor for drift (does not affect baseline)
        
        NOTE: In new schema, time dynamics are encoded via:
        - Node-specific peaks (sinusoidal drift with amplitude/phase)
        - Adjacency modulation patterns (time-varying edge weights)
        Global daily_cycle removed from schema.
        """
        
        modulation_factor = 1.0
        
        # NOTE: Global daily_cycle removed in new schema
        # Time dynamics now encoded in node-specific sinusoidal drifts
        # daily_cycle = time_modulation.get("daily_cycle", {})
        # if daily_cycle.get("enabled", False):
        #     amplitude = daily_cycle.get("amplitude", 0.0)
        #     phase_hour = daily_cycle.get("phase_hour", 0)
        #     sampling_minutes = getattr(self, "sampling_minutes", 60)
        #     hour_of_day = ((time_step * sampling_minutes) // 60) % 24
        #     daily_factor = 1 + amplitude * np.sin(2 * np.pi * (hour_of_day - phase_hour) / 24)
        #     modulation_factor *= daily_factor
        
        return modulation_factor
    
    def _compute_diffusion_new(self, node_id: int, X_i: float, params: Dict[str, Any]) -> float:
        """Compute diffusion term using new hierarchical parameters"""
        
        sigma = params["sigma"]
        diffusion_shape = params.get("diffusion_shape", "constant")
        alpha = params.get("alpha", 0.0)
        
        if diffusion_shape == "constant":
            return sigma
        elif diffusion_shape == "sqrt":
            return sigma * np.sqrt(abs(X_i) + 1e-6)
        elif diffusion_shape == "linear":
            return sigma * (1 + alpha * abs(X_i))
        else:
            return sigma
    
    def _compute_coupling_new(self, node_id: int, X_all: np.ndarray, adj_matrix: np.ndarray, params: Dict[str, Any]) -> float:
        """Compute coupling term using new hierarchical parameters"""
        
        lambda_coupling = params["lambda"]
        
        coupling = 0.0
        
        for j in range(len(X_all)):
            if j != node_id and adj_matrix[j, node_id] > 0:
                # Linear coupling: lambda * A_ji * (X_j - X_i)
                coupling += lambda_coupling * adj_matrix[j, node_id] * (X_all[j] - X_all[node_id])
        
        return coupling
    
    def _compute_coupling_with_lag(
        self,
        node_id: int,
        t: int,                         # 当前时间步（用于取滞后）
        data: np.ndarray,               # 形状 (num_nodes, seq_len)
        adj_matrices: List[np.ndarray], # 所有时刻的邻接矩阵列表
        params: Dict[str, Any],         # 该节点参数（内含 lambda）
        lag_steps: np.ndarray           # 形状 (num_nodes, num_nodes)，lag(j->i) 的步数
    ) -> float:
        """耦合项：lambda * sum_j A_ji(t - lag_ji + 1) * (X_j(t - lag_ji) - X_i(t-1))
        
        Modified to make visual lag exactly equal to time_lag parameter.
        Now: visual_lag = time_lag (instead of time_lag + 1)
        
        IMPORTANT: time_lag must be >= 1 for this to work correctly.
        """
        lam = params["lambda"]
        if lam <= 0:
            return 0.0

        xi_prev = data[node_id, t-1]
        c = 0.0
        n = data.shape[0]

        for j in range(n):
            if j == node_id:
                continue
            
            lag = int(lag_steps[j, node_id])
            
            # 计算边权时刻：t - lag_ji + 1
            t_edge = max(0, min(len(adj_matrices) - 1, t - lag + 1))
            a_ji = float(adj_matrices[t_edge][j, node_id])
            
            if a_ji <= 0:
                continue

            # 修改：直接使用 t - lag，使得视觉延迟精确等于 time_lag 参数
            # 旧逻辑: t_src = max(0, (t-1) - lag)  -> visual_lag = time_lag + 1
            # 新逻辑: t_src = max(0, t - lag)      -> visual_lag = time_lag
            t_src = max(0, t - lag)
            xj_lag = float(data[j, t_src])

            c += lam * a_ji * (xj_lag - xi_prev)
        return c
    
    # Keep old functions for backward compatibility
    def _compute_drift(self, node_id: int, X_i: float, time: float, params: Dict[str, Any], time_step: int) -> float:
        """Compute drift term for SDE (legacy)"""
        
        drift_params = params["drift_params"]
        drift_type = drift_params["type"]
        
        base_drift = 0.0
        
        if drift_type == "mean_reverting":
            baseline = drift_params["baseline"]
            reversion_speed = drift_params["mean_reversion_speed"]
            base_drift = reversion_speed * (baseline - X_i)
        
        elif drift_type == "linear":
            trend_strength = drift_params["trend_strength"]
            base_drift = trend_strength
        
        elif drift_type == "logistic":
            baseline = drift_params["baseline"]
            carrying_capacity = drift_params["carrying_capacity"]
            growth_rate = drift_params.get("growth_rate", 0.1)
            base_drift = growth_rate * X_i * (1 - X_i / carrying_capacity)
        
        # Apply time-dependent modulation
        time_mod = params.get("time_dependent_modulation", {})
        
        # NOTE: Global daily_cycle removed in new schema
        # Time dynamics now in node-specific sinusoidal drift parameters
        # daily_cycle = time_mod.get("daily_cycle", {})
        # if daily_cycle.get("enabled", False):
        #     amplitude = daily_cycle["amplitude"]
        #     phase = daily_cycle["phase"]
        #     hour_of_day = (time_step % 24)
        #     daily_factor = 1 + amplitude * np.sin(2 * np.pi * (hour_of_day - phase) / 24)
        #     base_drift *= daily_factor
        
        return base_drift
    
    def _compute_diffusion(self, node_id: int, X_i: float, params: Dict[str, Any]) -> float:
        """Compute diffusion term for SDE (legacy)"""
        
        diffusion_params = params["diffusion_params"]
        diffusion_type = diffusion_params["type"]
        base_vol = diffusion_params["base_volatility"]
        
        if diffusion_type == "constant":
            return base_vol
        
        elif diffusion_type == "linear":
            scaling = diffusion_params.get("volatility_scaling", 0.1)
            return base_vol + scaling * abs(X_i)
        
        elif diffusion_type == "sqrt":
            scaling = diffusion_params.get("volatility_scaling", 0.1)
            return base_vol + scaling * np.sqrt(abs(X_i))
        
        return base_vol
    
    def _compute_coupling(self, node_id: int, X_all: np.ndarray, adj_matrix: np.ndarray, params: Dict[str, Any]) -> float:
        """Compute coupling term for SDE (legacy)"""
        
        coupling_params = params["coupling_params"]
        coupling_type = coupling_params["coupling_type"]
        coupling_strength = coupling_params["coupling_strength"]
        
        coupling = 0.0
        
        for j in range(len(X_all)):
            if j != node_id and adj_matrix[j, node_id] > 0:
                influence_strength = adj_matrix[j, node_id] * coupling_strength
                
                if coupling_type == "linear":
                    coupling += influence_strength * (X_all[j] - X_all[node_id])
                
                elif coupling_type == "threshold":
                    threshold = coupling_params.get("saturation_threshold", X_all[node_id])
                    if X_all[j] > threshold:
                        coupling += influence_strength * (X_all[j] - threshold)
                
                elif coupling_type == "sigmoid":
                    # Sigmoid coupling for smooth saturation effects
                    diff = X_all[j] - X_all[node_id]
                    coupling += influence_strength * np.tanh(diff / (X_all[node_id] + 1e-6))
        
        return coupling

    def visualize_network_sde_results(self, ts_data: np.ndarray, network_sde: Dict[str, Any], 
                                    generation_info: Dict[str, Any] = None,
                                    figsize: Tuple[int, int] = None, save_path: str = None):
        """Agent 6: Visualize Network SDE results with all nodes displayed
        
        Args:
            ts_data: Network-coupled SDE data
            network_sde: Network SDE configuration
            generation_info: Generation info containing independent SDE data (optional)
            figsize: Figure size
            save_path: Path to save the figure
        """
        
        # Set matplotlib to use English fonts
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        seq_len = network_sde["sequence_length"]
        graph_info = network_sde["structured_scenario"]
        
        # Extract independent SDE data if available
        independent_data = None
        if generation_info and "independent_sde_data" in generation_info:
            independent_data = generation_info["independent_sde_data"]
        
        # Calculate dynamic layout based on number of nodes
        # Top row: All nodes overview + individual nodes
        # Bottom row: Adjacency evolution, correlation matrix, network graph
        individual_plots = self.num_nodes
        cols = max(3, individual_plots + 1)  # At least 3 columns, more if needed
        
        # Auto-adjust figure size based on number of plots
        if figsize is None:
            figsize = (8 * cols, 12)  # increase width from 5*cols to 8*cols
        
        # Create subplot layout
        fig, axes = plt.subplots(2, cols, figsize=figsize)
        
        # Ensure axes is 2D array
        if cols == 1:
            axes = axes.reshape(2, 1)
        elif len(axes.shape) == 1:
            axes = axes.reshape(1, -1)
        
        # Extend node colors to support more nodes
        base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        node_colors = (base_colors * ((self.num_nodes // len(base_colors)) + 1))[:self.num_nodes]
        
        # Plot 1: Time series for all nodes (overview)
        ax_overview = axes[0, 0]
        for i in range(self.num_nodes):
            node_name = graph_info["nodes"][i]["name"]
            ax_overview.plot(ts_data[i], label=f'{node_name}', color=node_colors[i], linewidth=2)
        ax_overview.set_title('All Node Time Series Overview', fontsize=14, fontweight='bold')
        ax_overview.set_xlabel('Time Step')
        ax_overview.set_ylabel('Value')
        ax_overview.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_overview.grid(True, alpha=0.3)
        
        # Plot 2-N: Individual node plots
        # Get global variable from structured_scenario
        global_variable = graph_info.get("variable", "Value")
        
        for i in range(self.num_nodes):
            col_idx = i + 1
            if col_idx < cols:
                ax = axes[0, col_idx]
                node_info = graph_info["nodes"][i]
                
                # Plot network-coupled data (solid line)
                ax.plot(ts_data[i], color=node_colors[i], linewidth=2, label='Network-coupled', alpha=0.9)
                
                # Plot independent SDE data (dashed line) if available
                if independent_data is not None:
                    ax.plot(independent_data[i], color=node_colors[i], linewidth=2, 
                           linestyle='--', label='Independent (no coupling)', alpha=0.6)
                    ax.legend(loc='upper right', fontsize=8)
                
                ax.set_title(f'{node_info["name"]}\n({global_variable})', fontsize=12)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = np.mean(ts_data[i])
                std_val = np.std(ts_data[i])
                min_val = np.min(ts_data[i])
                max_val = np.max(ts_data[i])
                stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Hide unused subplots in top row
        for col_idx in range(self.num_nodes + 1, cols):
            axes[0, col_idx].set_visible(False)
        
        # Plot 3: Edge weight evolution over time for all edges
        ax3 = axes[1, 0]
        adj_matrices = network_sde.get("adjacency_matrices", [])
        
        if adj_matrices and len(graph_info["edges"]) > 0:
            # Define a diverse color palette with distinct colors for different edges
            edge_colors = [
                '#1f77b4',   # Blue
                '#ff7f0e',   # Orange
                '#2ca02c',   # Green
                '#d62728',   # Red
                '#9467bd',   # Purple
                '#8c564b',   # Brown
                '#e377c2',   # Pink
                '#7f7f7f',   # Gray
                '#bcbd22',   # Olive
                '#17becf',   # Cyan
                '#ff9896',   # Light Red
                '#98df8a',   # Light Green
                '#ffbb78',   # Light Orange
                '#aec7e8',   # Light Blue
                '#f7b6d3',   # Light Pink
                '#c5b0d5',   # Light Purple
                '#c49c94',   # Light Brown
                '#dbdb8d',   # Light Olive
                '#9edae5',   # Light Cyan
                '#c7c7c7'    # Light Gray
            ]
            
            # Plot evolution for each edge (including both directions for undirected graph)
            plotted_edges = set()  # To avoid duplicate plotting for undirected edges
            edge_stats = []
            
            for i, edge in enumerate(graph_info["edges"]):
                source, target = edge["source"], edge["target"]
                
                # Plot source->target
                adj_values = [adj_matrices[t][source, target] for t in range(seq_len)]
                source_name = graph_info["nodes"][source]["name"].split()[0]  # First word
                target_name = graph_info["nodes"][target]["name"].split()[0]  # First word
                edge_label = f"{source_name}→{target_name} ({source}→{target})"
                
                # Use a unique color for each edge
                color = edge_colors[i % len(edge_colors)]
                
                ax3.plot(adj_values, linewidth=2.5, color=color, label=edge_label, alpha=0.9)
                
                edge_stats.append({
                    'label': edge_label,
                    'values': adj_values,
                    'mean': np.mean(adj_values),
                    'std': np.std(adj_values),
                    'min': np.min(adj_values),
                    'max': np.max(adj_values)
                })
                
                # Also plot reverse direction if it's different (for undirected graph visualization)
                reverse_values = [adj_matrices[t][target, source] for t in range(seq_len)]
                if not np.array_equal(adj_values, reverse_values):
                    reverse_label = f"{target_name}→{source_name} ({target}→{source})"
                    # Use a different color for the reverse direction
                    reverse_color = edge_colors[(i + len(edge_colors)//2) % len(edge_colors)]
                    ax3.plot(reverse_values, linewidth=2.5, color=reverse_color, 
                            label=reverse_label, alpha=0.9)
                    
                    edge_stats.append({
                        'label': reverse_label,
                        'values': reverse_values,
                        'mean': np.mean(reverse_values),
                        'std': np.std(reverse_values),
                        'min': np.min(reverse_values),
                        'max': np.max(reverse_values)
                    })
            
            ax3.set_title("Edge Weight Evolution Over Time", fontsize=12, fontweight='bold')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Edge Weight Strength')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # Add comprehensive statistics
            if edge_stats:
                all_values = np.concatenate([stat['values'] for stat in edge_stats])
                global_mean = np.mean(all_values)
                global_std = np.std(all_values)
                global_min = np.min(all_values)
                global_max = np.max(all_values)
                
                # Print debug info to console
                print(f"\n[Visualization Debug] Edge Weight Statistics:")
                print(f"  Global: min={global_min:.3f}, max={global_max:.3f}, mean={global_mean:.3f}, std={global_std:.3f}")
                for stat in edge_stats:
                    print(f"  {stat['label']}: min={stat['min']:.3f}, max={stat['max']:.3f}, range={stat['max']-stat['min']:.3f}")
                
                stats_text = f'All Edges Statistics:\nMean: {global_mean:.3f}\nStd: {global_std:.3f}\nMin: {global_min:.3f}\nMax: {global_max:.3f}\nTotal Edges: {len(edge_stats)}'
                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=9, 
                        verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.9))
                
                # Add individual edge statistics with min/max
                detail_text = "Individual Edge Stats:\n"
                for stat in edge_stats[:3]:  # Show first 3 edges to avoid clutter
                    detail_text += f"{stat['label'][:15]}...: min={stat['min']:.2f}, max={stat['max']:.2f}\n"
                if len(edge_stats) > 3:
                    detail_text += f"... and {len(edge_stats)-3} more edges"
                
                ax3.text(0.02, 0.02, detail_text, transform=ax3.transAxes, fontsize=8, 
                        verticalalignment='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
            
        else:
            ax3.text(0.5, 0.5, 'No adjacency data available', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=12)
            ax3.set_title("Edge Weight Evolution", fontsize=12)
        
        # Plot 4: Cross-correlation matrix (place in second-to-last position)
        corr_col_idx = max(1, cols - 2)
        ax4 = axes[1, corr_col_idx]
        correlation_matrix = np.corrcoef(ts_data)
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title('Inter-Node Correlation Matrix', fontsize=12)
        ax4.set_xticks(range(self.num_nodes))
        ax4.set_yticks(range(self.num_nodes))
        ax4.set_xticklabels([f'Node {i}' for i in range(self.num_nodes)])
        ax4.set_yticklabels([f'Node {i}' for i in range(self.num_nodes)])
        plt.colorbar(im, ax=ax4)
        
        # Add correlation values as text
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                text = ax4.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        # Plot 5: Network graph visualization with actual edge weights (place in last position)
        ax5 = axes[1, cols - 1]
        G = nx.Graph()  # Use undirected graph since we treat it as undirected
        
        # Add nodes
        for node in graph_info["nodes"]:
            G.add_node(node["id"], name=node["name"])
        
        # Add edges with actual weights from adjacency matrix
        if adj_matrices:
            base_adj = np.array(adj_matrices[0])
            for edge in graph_info["edges"]:
                source, target = edge["source"], edge["target"]
                weight = base_adj[source, target]
                if weight > 0:
                    G.add_edge(source, target, weight=weight)
        else:
            # Fallback to default weights
            for edge in graph_info["edges"]:
                G.add_edge(edge["source"], edge["target"], weight=0.1)
        
        # Create layout
        if "spatial_layout" in graph_info and len(graph_info["spatial_layout"]) > 0:
            pos = {int(k): (v["x"], v["y"]) for k, v in graph_info["spatial_layout"].items()}
        else:
            pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax5, node_color=node_colors[:self.num_nodes], 
                              node_size=1200, alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax5, font_size=10, font_weight='bold')
        
        # Draw edges with thickness proportional to actual weight
        edges = G.edges(data=True)
        if edges:
            weights = [edge[2]['weight'] for edge in edges]
            max_weight = max(weights) if weights else 1
            # Scale edge widths: minimum 1, maximum 5
            scaled_widths = [1 + 4 * (w / max_weight) for w in weights]
            
            nx.draw_networkx_edges(G, pos, ax=ax5, width=scaled_widths, 
                                  alpha=0.7, edge_color='darkblue')
            
            # Add edge labels with weights
            edge_labels = {}
            for edge in edges:
                source, target, data = edge
                weight = data['weight']
                edge_labels[(source, target)] = f'{weight:.3f}'
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax5, font_size=8, 
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        ax5.set_title('Network Structure with Edge Weights', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # Add legend for edge weights
        if edges:
            legend_text = f'Edge Weight Range:\nMin: {min(weights):.3f}\nMax: {max(weights):.3f}\nEdge thickness ∝ weight'
            ax5.text(0.02, 0.98, legend_text, transform=ax5.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
        
        # Hide unused subplots in bottom row (between adjacency and correlation plots)
        for col_idx in range(1, max(1, cols - 2)):
            if col_idx < cols:
                axes[1, col_idx].set_visible(False)
        
        plt.tight_layout(pad=3.0)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Image saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def visualize_network_sde_html(self, ts_data: np.ndarray, network_sde: Dict[str, Any], 
                                    generation_info: Dict[str, Any] = None,
                                    save_path: str = None):
        """Generate interactive HTML visualization for Network SDE results
        
        Args:
            ts_data: Network-coupled SDE data
            network_sde: Network SDE configuration
            generation_info: Generation info containing independent SDE data (optional)
            save_path: Path to save the HTML file
        """
        
        import json
        
        seq_len = network_sde["sequence_length"]
        graph_info = network_sde["structured_scenario"]
        adj_matrices = network_sde.get("adjacency_matrices", [])
        
        # Extract node information
        nodes_data = []
        for i, node in enumerate(graph_info["nodes"]):
            nodes_data.append({
                "id": i,
                "name": node["name"],
                "type": node["type"],
                "description": node.get("description", ""),
                "timeseries": ts_data[i].tolist()
            })
        
        # Extract edge information
        edges_data = []
        for edge in graph_info["edges"]:
            source = edge["source"]
            target = edge["target"]
            # Get time-varying weights
            weights = [float(adj_matrices[t][source, target]) for t in range(seq_len)] if adj_matrices else [0.1] * seq_len
            edges_data.append({
                "source": source,
                "target": target,
                "relationship": edge.get("relationship", ""),
                "weights": weights
            })
        
        # Get spatial layout if available
        if "spatial_layout" in graph_info and graph_info["spatial_layout"]:
            layout = {int(k): {"x": v["x"], "y": v["y"]} for k, v in graph_info["spatial_layout"].items()}
            
            # Scale layout to fit SVG canvas (assuming SVG width ~900px, height 600px)
            # Find current bounds
            x_coords = [pos["x"] for pos in layout.values()]
            y_coords = [pos["y"] for pos in layout.values()]
            
            if x_coords and y_coords:
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # Add margins (80px on each side)
                margin = 80
                target_width = 900 - 2 * margin  # Target canvas width minus margins
                target_height = 600 - 2 * margin  # Target canvas height minus margins
                
                # Calculate scale factors
                current_width = max_x - min_x if max_x > min_x else 1
                current_height = max_y - min_y if max_y > min_y else 1
                
                scale_x = target_width / current_width
                scale_y = target_height / current_height
                
                # Use uniform scale (smaller of the two to ensure everything fits)
                scale = min(scale_x, scale_y)
                
                # Scale and center the layout
                for node_id in layout:
                    layout[node_id]["x"] = margin + (layout[node_id]["x"] - min_x) * scale
                    layout[node_id]["y"] = margin + (layout[node_id]["y"] - min_y) * scale
        else:
            # Generate circular layout centered in the SVG canvas
            import math
            layout = {}
            center_x, center_y = 450, 300  # Center of typical SVG canvas
            radius = 200
            
            for i in range(self.num_nodes):
                angle = 2 * math.pi * i / self.num_nodes
                layout[i] = {
                    "x": center_x + radius * math.cos(angle),
                    "y": center_y + radius * math.sin(angle)
                }
        
        # Calculate min/max for color scaling
        ts_min = float(np.min(ts_data))
        ts_max = float(np.max(ts_data))
        
        all_weights = []
        for edge in edges_data:
            all_weights.extend(edge["weights"])
        weight_min = float(min(all_weights)) if all_weights else 0
        weight_max = float(max(all_weights)) if all_weights else 1
        
        # Generate HTML
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network SDE Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        #container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }
        #info {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }
        #visualization {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        #network-container {
            flex: 2;
            min-height: 600px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: #fafafa;
            position: relative;
        }
        #timeseries-container {
            flex: 1;
            min-height: 600px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: white;
            padding: 10px;
            overflow-y: auto;
        }
        #controls {
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        #time-slider {
            width: 100%;
            margin: 10px 0;
        }
        #time-display {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }
        .control-buttons {
            text-align: center;
            margin-top: 10px;
        }
        button {
            padding: 10px 20px;
            margin: 0 5px;
            font-size: 14px;
            cursor: pointer;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            transition: background 0.3s;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .node {
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
            transition: r 0.2s;
        }
        .node:hover {
            stroke-width: 4px;
        }
        .node-label {
            font-size: 12px;
            font-weight: bold;
            pointer-events: none;
            text-anchor: middle;
        }
        .edge {
            fill: none;
            marker-end: url(#arrowhead);
            transition: stroke-width 0.3s;
        }
        .edge:hover {
            stroke-width: 6px !important;
        }
        .edge-label {
            font-size: 9px;
            fill: #333;
            font-weight: bold;
            pointer-events: none;
            text-shadow: 1px 1px 2px white, -1px -1px 2px white, 1px -1px 2px white, -1px 1px 2px white;
        }
        .legend {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-size: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .legend-title {
            font-weight: bold;
            margin-bottom: 10px;
        }
        .legend-gradient {
            width: 200px;
            height: 20px;
            margin: 5px 0;
        }
        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
        }
        .timeseries-plot {
            margin-bottom: 15px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
        }
        .timeseries-title {
            font-weight: bold;
            margin-bottom: 5px;
            color: #333;
        }
        .timeseries-chart {
            width: 100%;
            height: 80px;
        }
        .current-time-line {
            stroke: red;
            stroke-width: 2;
            stroke-dasharray: 5,5;
        }
        #stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }
        .stat-card {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>
    <div id="container">
        <h1>Network SDE Interactive Visualization</h1>
        <div id="info">
            <span id="variable-name"></span> | 
            <span id="num-nodes"></span> nodes | 
            <span id="seq-length"></span> time steps
        </div>
        
        <div id="controls">
            <div id="time-display">Time Step: <span id="current-time">0</span> / <span id="total-time">0</span></div>
            <input type="range" id="time-slider" min="0" max="100" value="0" step="1">
            <div class="control-buttons">
                <button id="play-btn">▶ Play</button>
                <button id="pause-btn" disabled>⏸ Pause</button>
                <button id="reset-btn">⏮ Reset</button>
                <label style="margin-left: 20px">
                    Speed: 
                    <select id="speed-select">
                        <option value="200">Slow</option>
                        <option value="100" selected>Normal</option>
                        <option value="50">Fast</option>
                        <option value="20">Very Fast</option>
                    </select>
                </label>
                <label style="margin-left: 20px">
                    <input type="checkbox" id="show-edge-labels" checked>
                    Show Edge Weights
                </label>
            </div>
        </div>
        
        <div id="visualization">
            <div id="network-container">
                <svg id="network-svg"></svg>
                <div class="legend">
                    <div class="legend-title">Node Value</div>
                    <svg class="legend-gradient" id="node-legend"></svg>
                    <div class="legend-labels">
                        <span id="node-min">0</span>
                        <span id="node-max">100</span>
                    </div>
                    <div class="legend-title" style="margin-top: 15px">Edge Weight</div>
                    <svg class="legend-gradient" id="edge-legend"></svg>
                    <div class="legend-labels">
                        <span id="edge-min">0</span>
                        <span id="edge-max">1</span>
                    </div>
                </div>
            </div>
            <div id="timeseries-container"></div>
        </div>
        
        <div id="stats"></div>
    </div>

    <script>
        // Data
        const nodesData = {{NODES_DATA}};
        const edgesData = {{EDGES_DATA}};
        const layout = {{LAYOUT}};
        const seqLen = {{SEQ_LEN}};
        const tsMin = {{TS_MIN}};
        const tsMax = {{TS_MAX}};
        const weightMin = {{WEIGHT_MIN}};
        const weightMax = {{WEIGHT_MAX}};
        const variableName = "{{VARIABLE}}";
        
        // Initialize display
        document.getElementById('variable-name').textContent = variableName;
        document.getElementById('num-nodes').textContent = nodesData.length;
        document.getElementById('seq-length').textContent = seqLen;
        document.getElementById('total-time').textContent = seqLen - 1;
        document.getElementById('time-slider').max = seqLen - 1;
        document.getElementById('node-min').textContent = tsMin.toFixed(2);
        document.getElementById('node-max').textContent = tsMax.toFixed(2);
        document.getElementById('edge-min').textContent = weightMin.toFixed(3);
        document.getElementById('edge-max').textContent = weightMax.toFixed(3);
        
        // Color scales
        const nodeColorScale = d3.scaleSequential(d3.interpolateYlOrRd)
            .domain([tsMin, tsMax]);
        // Edge color: from orange (low weight) to dark blue (high weight)
        const edgeColorScale = d3.scaleSequential()
            .domain([weightMin, weightMax])
            .interpolator(d3.interpolateRgb("#FF8C00", "#0000CD"));
        
        // Draw legends
        const nodeLegendSvg = d3.select('#node-legend');
        const edgeLegendSvg = d3.select('#edge-legend');
        
        const legendGradient = nodeLegendSvg.append('defs')
            .append('linearGradient')
            .attr('id', 'node-gradient')
            .attr('x1', '0%').attr('x2', '100%');
        legendGradient.append('stop').attr('offset', '0%').attr('stop-color', nodeColorScale(tsMin));
        legendGradient.append('stop').attr('offset', '100%').attr('stop-color', nodeColorScale(tsMax));
        nodeLegendSvg.append('rect').attr('width', 200).attr('height', 20).attr('fill', 'url(#node-gradient)');
        
        const edgeGradient = edgeLegendSvg.append('defs')
            .append('linearGradient')
            .attr('id', 'edge-gradient')
            .attr('x1', '0%').attr('x2', '100%');
        edgeGradient.append('stop').attr('offset', '0%').attr('stop-color', edgeColorScale(weightMin));
        edgeGradient.append('stop').attr('offset', '100%').attr('stop-color', edgeColorScale(weightMax));
        edgeLegendSvg.append('rect').attr('width', 200).attr('height', 20).attr('fill', 'url(#edge-gradient)');
        
        // Network visualization
        const networkContainer = document.getElementById('network-container');
        const width = networkContainer.clientWidth;
        const height = 600;
        
        const svg = d3.select('#network-svg')
            .attr('width', width)
            .attr('height', height);
        
        const g = svg.append("g");

        // Define arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 0 10 10')
            .attr('refX', 35)
            .attr('refY', 5)
            .attr('markerWidth', 8)
            .attr('markerHeight', 8)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M 0 0 L 10 5 L 0 10 z')
            .attr('fill', '#666');
        
        // Draw edges
        const edgesGroup = g.append('g').attr('class', 'edges');
        const edges = edgesGroup.selectAll('.edge')
            .data(edgesData)
            .enter().append('path')
            .attr('class', 'edge')
            .attr('d', d => {
                const sx = layout[d.source].x;
                const sy = layout[d.source].y;
                const tx = layout[d.target].x;
                const ty = layout[d.target].y;
                return `M ${sx} ${sy} L ${tx} ${ty}`;
            });
        
        // Draw edge labels
        const edgeLabels = edgesGroup.selectAll('.edge-label')
            .data(edgesData)
            .enter().append('text')
            .attr('class', 'edge-label')
            .attr('x', d => (layout[d.source].x + layout[d.target].x) / 2)
            .attr('y', d => (layout[d.source].y + layout[d.target].y) / 2);
        
        // Draw nodes
        const nodesGroup = g.append('g').attr('class', 'nodes');
        const nodes = nodesGroup.selectAll('.node')
            .data(nodesData)
            .enter().append('circle')
            .attr('class', 'node')
            .attr('cx', d => layout[d.id].x)
            .attr('cy', d => layout[d.id].y)
            .attr('r', 30)
            .on('click', (event, d) => {
                alert(`Node ${d.id}: ${d.name}\\n${d.description}`);
            });
        
        // Draw node labels
        const nodeLabels = nodesGroup.selectAll('.node-label')
            .data(nodesData)
            .enter().append('text')
            .attr('class', 'node-label')
            .attr('x', d => layout[d.id].x)
            .attr('y', d => layout[d.id].y + 45)
            .text(d => `Node ${d.id}`);
        
        const drag = d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);

        nodes.call(drag);

        function dragstarted(event, d) {
            d3.select(this).raise().attr("stroke", "black");
        }

        function dragged(event, d) {
            d3.select(this).attr("cx", d.x = event.x).attr("cy", d.y = event.y);
            nodeLabels.filter(n => n.id === d.id).attr("x", event.x).attr("y", event.y + 45);
            edges.filter(l => l.source === d.id || l.target === d.id)
                .attr('d', l => `M ${layout[l.source].x} ${layout[l.source].y} L ${layout[l.target].x} ${layout[l.target].y}`);
            edgeLabels.filter(l => l.source === d.id || l.target === d.id)
                .attr('x', l => (layout[l.source].x + layout[l.target].x) / 2)
                .attr('y', l => (layout[l.source].y + layout[l.target].y) / 2);
        }

        function dragended(event, d) {
            d3.select(this).attr("stroke", null);
        }

        svg.call(d3.zoom().on("zoom", function(event) {
            g.attr("transform", event.transform);
        }));
        
        // Draw timeseries plots
        const timeseriesContainer = d3.select('#timeseries-container');
        nodesData.forEach(node => {
            const plotDiv = timeseriesContainer.append('div')
                .attr('class', 'timeseries-plot');
            
            plotDiv.append('div')
                .attr('class', 'timeseries-title')
                .text(`Node ${node.id}: ${node.name.substring(0, 30)}...`);
            
            const plotSvg = plotDiv.append('svg')
                .attr('class', 'timeseries-chart')
                .attr('width', '100%')
                .attr('height', 80);
            
            const plotWidth = 280;
            const plotHeight = 70;
            const margin = {top: 5, right: 5, bottom: 5, left: 5};
            
            const xScale = d3.scaleLinear()
                .domain([0, seqLen - 1])
                .range([margin.left, plotWidth - margin.right]);
            
            const yScale = d3.scaleLinear()
                .domain([d3.min(node.timeseries), d3.max(node.timeseries)])
                .range([plotHeight - margin.bottom, margin.top]);
            
            const line = d3.line()
                .x((d, i) => xScale(i))
                .y(d => yScale(d));
            
            plotSvg.append('path')
                .datum(node.timeseries)
                .attr('fill', 'none')
                .attr('stroke', 'steelblue')
                .attr('stroke-width', 1.5)
                .attr('d', line);
            
            // Add current time line
            plotSvg.append('line')
                .attr('class', `current-time-line time-line-${node.id}`)
                .attr('x1', xScale(0))
                .attr('x2', xScale(0))
                .attr('y1', margin.top)
                .attr('y2', plotHeight - margin.bottom);
        });
        
        // Animation controls
        let currentTime = 0;
        let animationInterval = null;
        let animationSpeed = 100;
        
        function updateVisualization(timeStep) {
            currentTime = timeStep;
            document.getElementById('current-time').textContent = timeStep;
            document.getElementById('time-slider').value = timeStep;
            
            // Update nodes
            nodes.attr('fill', d => nodeColorScale(d.timeseries[timeStep]));
            
            // Update edges with dynamic width based on weight
            edges
                .attr('stroke', d => edgeColorScale(d.weights[timeStep]))
                .attr('stroke-width', d => {
                    // Map weight to stroke width (min 2px, max 8px)
                    const normalized = (d.weights[timeStep] - weightMin) / (weightMax - weightMin);
                    return 2 + normalized * 6;
                });
            
            // Update edge labels (make them semi-transparent and smaller)
            edgeLabels
                .text(d => d.weights[timeStep].toFixed(2))
                .attr('opacity', 0.7)
                .style('font-size', '9px')
                .style('font-weight', 'bold');
            
            // Update timeseries current time lines
            nodesData.forEach(node => {
                const xScale = d3.scaleLinear()
                    .domain([0, seqLen - 1])
                    .range([5, 275]);
                d3.select(`.time-line-${node.id}`)
                    .attr('x1', xScale(timeStep))
                    .attr('x2', xScale(timeStep));
            });
            
            // Update stats
            updateStats(timeStep);
        }
        
        function updateStats(timeStep) {
            const statsDiv = d3.select('#stats');
            statsDiv.html('');
            
            nodesData.forEach(node => {
                const card = statsDiv.append('div').attr('class', 'stat-card');
                card.append('div').attr('class', 'stat-label').text(`Node ${node.id}: ${node.name.substring(0, 25)}`);
                card.append('div').attr('class', 'stat-value').text(node.timeseries[timeStep].toFixed(2));
            });
        }
        
        function play() {
            if (animationInterval) return;
            document.getElementById('play-btn').disabled = true;
            document.getElementById('pause-btn').disabled = false;
            
            animationInterval = setInterval(() => {
                currentTime++;
                if (currentTime >= seqLen) {
                    currentTime = 0;
                }
                updateVisualization(currentTime);
            }, animationSpeed);
        }
        
        function pause() {
            if (animationInterval) {
                clearInterval(animationInterval);
                animationInterval = null;
                document.getElementById('play-btn').disabled = false;
                document.getElementById('pause-btn').disabled = true;
            }
        }
        
        function reset() {
            pause();
            updateVisualization(0);
        }
        
        // Event listeners
        document.getElementById('play-btn').addEventListener('click', play);
        document.getElementById('pause-btn').addEventListener('click', pause);
        document.getElementById('reset-btn').addEventListener('click', reset);
        document.getElementById('time-slider').addEventListener('input', (e) => {
            pause();
            updateVisualization(parseInt(e.target.value));
        });
        document.getElementById('speed-select').addEventListener('change', (e) => {
            animationSpeed = parseInt(e.target.value);
            if (animationInterval) {
                pause();
                play();
            }
        });
        document.getElementById('show-edge-labels').addEventListener('change', (e) => {
            edgeLabels.attr('opacity', e.target.checked ? 0.7 : 0);
        });
        
        // Initialize
        updateVisualization(0);
    </script>
</body>
</html>
        """
        
        # Replace placeholders
        html_content = html_template.replace('{{NODES_DATA}}', json.dumps(nodes_data))
        html_content = html_content.replace('{{EDGES_DATA}}', json.dumps(edges_data))
        html_content = html_content.replace('{{LAYOUT}}', json.dumps(layout))
        html_content = html_content.replace('{{SEQ_LEN}}', str(seq_len))
        html_content = html_content.replace('{{TS_MIN}}', str(ts_min))
        html_content = html_content.replace('{{TS_MAX}}', str(ts_max))
        html_content = html_content.replace('{{WEIGHT_MIN}}', str(weight_min))
        html_content = html_content.replace('{{WEIGHT_MAX}}', str(weight_max))
        html_content = html_content.replace('{{VARIABLE}}', graph_info.get('variable', 'Value'))
        
        # Save HTML file
        if save_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"output/network_sde_interactive_{timestamp}.html"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive HTML visualization saved to: {save_path}")
        print(f"Open this file in a web browser to view the interactive visualization.")
        
        return save_path

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information"""
        return {
            "num_nodes": self.num_nodes,
            "sequence_length": self.seq_len if self.seq_len is not None else "Not determined yet",
            "default_seq_len": self.DEFAULT_SEQ_LEN,
            "model": "gemini-2.5-flash",
            "seq_len_source": "Dynamic calculation (based on scenario)" if self.seq_len is not None else "Default value"
        }
    
    def print_config(self):
        """Print current configuration"""
        config = self.get_config_info()
        print("=== Network SDE Generator Configuration ===")
        print(f"Number of nodes: {config['num_nodes']}")
        print(f"Sequence length: {config['sequence_length']}")
        print(f"Length source: {config['seq_len_source']}")
        print(f"Default sequence length: {config['default_seq_len']}")
        print(f"LLM model: {config['model']}")
        print("=" * 35)
        print()

    def _extract_time_info_from_scenario(self, scenario: str) -> Dict[str, Any]:
        """Extract time span and sampling frequency from scenario description"""
        import re
        
        time_info = {
            "time_span": None,
            "sampling_frequency": None,
            "calculated_seq_len": None,
            "time_span_raw": None,
            "sampling_frequency_raw": None
        }
        
        # Try to extract TIME SPAN and SAMPLING FREQUENCY from formatted response
        time_span_match = re.search(r'TIME SPAN:\s*([^\n]+)', scenario, re.IGNORECASE)
        sampling_freq_match = re.search(r'SAMPLING FREQUENCY:\s*([^\n]+)', scenario, re.IGNORECASE)
        
        if time_span_match:
            time_info["time_span_raw"] = time_span_match.group(1).strip()
        if sampling_freq_match:
            time_info["sampling_frequency_raw"] = sampling_freq_match.group(1).strip()
        
        # Calculate sequence length based on extracted information
        if time_info["time_span_raw"] and time_info["sampling_frequency_raw"]:
            seq_len = self._calculate_sequence_length(
                time_info["time_span_raw"], 
                time_info["sampling_frequency_raw"]
            )
            time_info["calculated_seq_len"] = seq_len
        
        return time_info

    def _calculate_sequence_length(self, time_span_str: str, sampling_freq_str: str) -> int:
        """Calculate sequence length from time span and sampling frequency strings"""
        import re
        from datetime import datetime, timedelta
        
        # Parse time span
        total_minutes = 0
        
        # First try to parse date ranges (e.g., "January 1st, 2024 - January 31st, 2024")
        date_range_patterns = [
            r'(\w+ \d+(?:st|nd|rd|th)?,?\s+\d{4})\s*[-–]\s*(\w+ \d+(?:st|nd|rd|th)?,?\s+\d{4})',
            r'(\d{4}-\d{1,2}-\d{1,2})\s*[-–]\s*(\d{4}-\d{1,2}-\d{1,2})',
            r'(\d{1,2}/\d{1,2}/\d{4})\s*[-–]\s*(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in date_range_patterns:
            match = re.search(pattern, time_span_str, re.IGNORECASE)
            if match:
                start_date_str = match.group(1)
                end_date_str = match.group(2)
                
                try:
                    # Clean up date strings
                    start_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', start_date_str)
                    end_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', end_date_str)
                    
                    # Try different date formats
                    date_formats = [
                        '%B %d, %Y',  # January 1, 2024
                        '%B %d %Y',   # January 1 2024
                        '%Y-%m-%d',   # 2024-01-01
                        '%m/%d/%Y'    # 01/01/2024
                    ]
                    
                    start_date = None
                    end_date = None
                    
                    for fmt in date_formats:
                        try:
                            start_date = datetime.strptime(start_date_str, fmt)
                            end_date = datetime.strptime(end_date_str, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if start_date and end_date:
                        # Add one day to end_date to include the full end day
                        end_date = end_date + timedelta(days=1)
                        time_diff = end_date - start_date
                        total_minutes = int(time_diff.total_seconds() / 60)
                        print(f"Debug: Parsed date range - start: {start_date}, end: {end_date} (inclusive), duration: {total_minutes} minutes")
                        break
                        
                except Exception as e:
                    print(f"Debug: Failed to parse date range: {e}")
                    continue
        
        # If date range parsing failed, try simple duration patterns
        if total_minutes == 0:
            # Extract numbers and units from time span
            hours_match = re.search(r'(\d+)\s*hours?', time_span_str, re.IGNORECASE)
            days_match = re.search(r'(\d+)\s*days?', time_span_str, re.IGNORECASE) 
            weeks_match = re.search(r'(\d+)\s*weeks?', time_span_str, re.IGNORECASE)
            months_match = re.search(r'(\d+)\s*months?', time_span_str, re.IGNORECASE)
            years_match = re.search(r'(\d+)\s*years?', time_span_str, re.IGNORECASE)
            
            if hours_match:
                total_minutes = int(hours_match.group(1)) * 60
            elif days_match:
                total_minutes = int(days_match.group(1)) * 24 * 60
            elif weeks_match:
                total_minutes = int(weeks_match.group(1)) * 7 * 24 * 60
            elif months_match:
                total_minutes = int(months_match.group(1)) * 30 * 24 * 60  # Approximate
            elif years_match:
                total_minutes = int(years_match.group(1)) * 365 * 24 * 60  # Approximate
            elif '24' in time_span_str:
                total_minutes = 24 * 60
        
        # Parse sampling frequency
        sampling_minutes = 1  # Default to 1 minute
        
        # Try different frequency patterns
        freq_patterns = [
            (r'every\s+(\d+)\s*minutes?', 1),
            (r'every\s+(\d+)\s*hours?', 60),
            (r'(\d+)[-\s]*minute\s*intervals?', 1),
            (r'(\d+)[-\s]*hour\s*intervals?', 60),
            (r'(\d+)\s*minutes?', 1),
            (r'(\d+)\s*hours?', 60),
        ]
        
        for pattern, multiplier in freq_patterns:
            match = re.search(pattern, sampling_freq_str, re.IGNORECASE)
            if match:
                sampling_minutes = int(match.group(1)) * multiplier
                break
        
        # Check for common frequency keywords
        if sampling_minutes == 1:  # Still default, try keywords
            if 'hourly' in sampling_freq_str.lower() or '1 hour' in sampling_freq_str.lower():
                sampling_minutes = 60
            elif 'daily' in sampling_freq_str.lower():
                sampling_minutes = 24 * 60
            elif 'weekly' in sampling_freq_str.lower():
                sampling_minutes = 7 * 24 * 60
            elif 'monthly' in sampling_freq_str.lower():
                sampling_minutes = 30 * 24 * 60
        
        print(f"Debug: Total minutes: {total_minutes}, Sampling minutes: {sampling_minutes}")
        
        # Calculate sequence length
        if total_minutes > 0 and sampling_minutes > 0:
            seq_len = int(total_minutes / sampling_minutes)
            # Cap at maximum allowed length
            seq_len = min(seq_len, self.MAX_SEQ_LEN)
            print(f"Debug: Calculated sequence length: {seq_len} (limit: {self.MAX_SEQ_LEN})")
            
            # Save timing information for consistent dt calculation
            self.sampling_minutes = sampling_minutes
            self.time_span_minutes = total_minutes
            
            return seq_len
        
        print(f"Debug: Using default sequence length: {self.DEFAULT_SEQ_LEN}")
        # Fallback to default
        return self.DEFAULT_SEQ_LEN

def demo_network_sde_generation(enabled_judges: List[int] = None, enable_logging: bool = True,
                                num_nodes: int = 3, domain: str = 'traffic',
                                generate_viz: bool = True):
    """Demonstrate 6-Agent Network SDE Generation Pipeline with Judge Agents

    Args:
        enabled_judges: A list of integers specifying which judge agents to enable (e.g., [1, 2]).
        enable_logging: Enable interaction logging
        num_nodes: Number of nodes in the network
        domain: Domain type (traffic, epidemic, finance, etc.)
        generate_viz: Whether to render PNG/HTML visualizations. Disable for
            batch generation where the simulation is the only output that matters.
    """
    
    if enabled_judges is None:
        enabled_judges = [1, 2]

    num_enabled_judges = len(enabled_judges)
    if num_enabled_judges > 0:
        print(f"=== {6 + num_enabled_judges}-Agent Spatial-Temporal Data Generation Pipeline (with Judge Agents {enabled_judges}) ===\n")
    else:
        print("=== 6-Agent Spatial-Temporal Data Generation Pipeline ===\n")
    
    # Create output directory for saving figures
    import os
    os.makedirs("output", exist_ok=True)
    
    # 创建日志记录器（如果启用）
    logger = None
    if enable_logging:
        logger = AgentInteractionLogger()
        print()
    
    # Create generator with logger
    sde_gen = NetworkSDEGenerator(num_nodes=num_nodes, logger=logger)
    sde_gen.domain = domain  # Store domain for scenario generation
    
    # ===== AGENT 1: Scenario Generation Agent =====
    print("=" * 70)
    print("AGENT 1: Scenario Generation Agent")
    print("=" * 70)
    scenario, calculated_seq_len = sde_gen.generate_scenario_with_length_validation()
    # scenario = """{"scenario": "TIME SPAN: "1 day" SAMPLING FREQUENCY: "30 minutes" VARIABLE: "traffic flow (vehicles/hour)" NODES: - NODE 0: [type: DEMAND_SOURCE] [A large suburban residential zone that is a primary source of morning commuter traffic.] - NODE 1: [type: PROPAGATION] [A major highway interchange that connects the suburban zone to the urban core routes.] - NODE 2: [type: DEMAND_SOURCE] [A central business district (CBD) with high office density, generating evening commuter traffic.] - NODE 3: [type: PROPAGATION] [A primary arterial road that channels traffic from the highway interchange (NODE 1) into the CBD (NODE 2).] - NODE 4: [type: PROPAGATION] [A one-way bridge that serves as a main exit route from the CBD (NODE 2) back towards the highway interchange (NODE 1).] EDGES: - NODE 0 → NODE 1: [Morning commute route from the suburbs to the main highway, with a 30-minute (1 time step) travel lag.] - NODE 1 → NODE 3: [Connector route from the highway interchange to the arterial road leading into the city, with a 30-minute (1 time step) travel lag.] - NODE 3 → NODE 2: [Final approach from the arterial road into the CBD, with a 30-minute (1 time step) travel lag.] - NODE 2 → NODE 4: [Evening commute route from the CBD onto the exit bridge, with a 30-minute (1 time step) travel lag.] - NODE 4 → NODE 1: [Route from the exit bridge back to the main highway interchange, with a 30-minute (1 time step) travel lag.] - NODE 1 → NODE 0: [Return route from the highway interchange back to the suburban residential zone, with a 30-minute (1 time step) travel lag.] TEMPORAL PATTERNS: - NODE 0: - Behavior: Generates a single, sharp self-generated peak corresponding to the morning rush hour exodus. - baseline: 60 vehicles/hour - amplitude: 250 vehicles/hour - peak: 16 (corresponding to 8:00 AM) - propagated_variations: Experiences an influx of traffic from NODE 1 during the evening return commute, peaking around time step 38. - NODE 1: - Behavior: Acts as a central hub, experiencing two distinct peaks propagated from other nodes. - baseline: 70 vehicles/hour - amplitude: 0 - peak: null - propagated_variations: Receives the morning commute from NODE 0, peaking around time step 17. Receives the evening commute from NODE 4, peaking around time step 37. - NODE 2: - Behavior: Generates a single, strong self-generated peak corresponding to the evening rush hour as employees leave the CBD. - baseline: 50 vehicles/hour - amplitude: 200 vehicles/hour - peak: 35 (corresponding to 5:30 PM) - propagated_variations: Experiences an influx of morning commuter traffic propagated from NODE 3, peaking around time step 19. - NODE 3: - Behavior: A pure propagation node that channels the morning commute. - baseline: 55 vehicles/hour - amplitude: 0 - peak: null - propagated_variations: Receives a single major wave of traffic from NODE 1, peaking around time step 18. - NODE 4: - Behavior: A pure propagation node that channels the evening commute. - baseline: 65 vehicles/hour - amplitude: 0 - peak: null - propagated_variations: Receives a single major wave of traffic from NODE 2 as the evening commute begins, peaking around time step 36. - Edge Modulation: - Time 15-17: Edges affected: NODE 0 → NODE 1; Effect: strong; Description: Morning rush hour begins, facilitating outbound flow from the suburban residential zone toward the highway interchange. - Time 16-18: Edges affected: NODE 1 → NODE 3; Effect: strong; Description: Continuation of the morning commute wave moving from the interchange toward the urban arterial. - Time 17-19: Edges affected: NODE 3 → NODE 2; Effect: strong; Description: The morning traffic reaches the CBD, marking the tail end of the inbound commute. - Time 34-36: Edges affected: NODE 2 → NODE 4; Effect: strong; Description: Evening rush hour begins as outbound traffic leaves the CBD toward the exit bridge. - Time 35-37: Edges affected: NODE 4 → NODE 1; Effect: strong; Description: Evening flow continues from the bridge back to the highway interchange. - Time 36-38: Edges affected: NODE 1 → NODE 0; Effect: strong; Description: The return commute completes as traffic flows from the highway interchange back into the suburban residential zone."}"""
    # calculated_seq_len = 48
    sde_gen.seq_len = calculated_seq_len
    
    print("\nGenerated Scenario:")
    print(scenario)
    print(f"\n✓ Sequence length: {calculated_seq_len}")
    print("\n" + "="*70 + "\n")
    
    # ===== AGENT 2: Scenario Parsing Agent + JUDGE AGENT 1 =====
    print("=" * 70)
    if 1 in enabled_judges:
        print("AGENT 1 + AGENT 2 + JUDGE AGENT 1: Hierarchical Validation")
        print("(Outer loop validates scenario logic, inner loop validates parsing)")
    else:
        print("AGENT 2: Scenario Parsing Agent")
    print("=" * 70)
    
    if 1 in enabled_judges:
        structured_scenario = sde_gen.parse_scenario_with_judge_loop(
            scenario, 
            max_outer_iterations=3,  # Agent 1 scenario regeneration
            max_inner_iterations=2   # Agent 2 parsing correction
        )
    else:
        structured_scenario = sde_gen.parse_scenario_to_structured_json(scenario)
    
    print("\nParsed Structured JSON:")
    print(f"- Variable: {structured_scenario['variable']}")
    print(f"- Time span: {structured_scenario['time_span']}")
    print(f"- Sampling frequency: {structured_scenario['sampling_frequency']}")
    print(f"- Number of nodes: {len(structured_scenario['nodes'])}")
    print(f"- Number of edges: {len(structured_scenario['edges'])}")
    print(f"- Node types: {[n['type'] for n in structured_scenario['nodes']]}")
    print("\n" + "="*70 + "\n")
    
    # ===== AGENTS 3 & 4: SDE Parameters + Adjacency + JUDGE AGENT 2 =====
    print("=" * 70)
    if 2 in enabled_judges:
        print("AGENTS 3 & 4: SDE Parameters + Adjacency + JUDGE AGENT 2: Parameter Validation")
    else:
        print("AGENTS 3 & 4: SDE Parameters + Time-Varying Adjacency")
    print("=" * 70)
    
    ts_data = None
    generation_info = None
    if 2 in enabled_judges:
        network_sde = sde_gen.generate_network_sde_with_judge_loop(
            structured_scenario, 
            seq_len=calculated_seq_len,
            max_iterations=5
        )
        # Extract validated data if available
        ts_data = network_sde.pop('_validated_ts_data', None)
        generation_info = network_sde.pop('_validated_generation_info', None)
    else:
        network_sde = sde_gen.generate_network_sde(structured_scenario)
    
    
    # Configuration summary
    hierarchical_params = network_sde.get('sde_parameters', {})
    materialized_for_stats = sde_gen._materialize_node_parameters(hierarchical_params)
    
    drift_types = {}
    node_types = {}
    for node_id, params in materialized_for_stats.items():
        drift_type = params.get('drift_type', 'mean_reverting')
        node_type = params.get('node_type', 'demand_source')
        drift_types[drift_type] = drift_types.get(drift_type, 0) + 1
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"\n✓ Pipeline Summary:")
    print(f"  - Node types: {dict(node_types)}")
    print(f"  - Drift types: {dict(drift_types)}")
    print(f"  - Adjacency matrix: {len(structured_scenario['edges'])} edges")
    print(f"  - Time patterns: {len(network_sde['time_varying_adjacency']['time_modulation'].get('patterns', []))} pattern(s)")
    # Print pattern details
    patterns = network_sde['time_varying_adjacency']['time_modulation'].get('patterns', [])
    if patterns:
        print(f"  - Pattern time ranges: {[p.get('time_range', []) for p in patterns]}")
    print("\n" + "="*70 + "\n")
    
    # ===== AGENT 5: Simulation Agent =====
    print("=" * 70)
    print("AGENT 5: Simulation Agent")
    print("=" * 70)
    print("\nSDE Parameters Summary:")
    hierarchical_params = network_sde.get('sde_parameters', {})
    
    print("Global Default Parameters:")
    global_defaults = hierarchical_params.get('global_defaults', {})
    print(f"  • Drift type: {global_defaults.get('drift_type', 'unknown')}")
    print(f"  • Mean reversion speed (κ): {global_defaults.get('kappa', 'N/A')}")
    print(f"  • Baseline value: {global_defaults.get('baseline', 'N/A')}")
    print(f"  • Coupling strength (λ): {global_defaults.get('lambda', 'N/A')}")
    print(f"  • Diffusion coefficient (σ): {global_defaults.get('sigma', 'N/A')}")
    print(f"  • Diffusion shape: {global_defaults.get('diffusion_shape', 'N/A')}")
    
    group_params = hierarchical_params.get('group_params', {})
    if group_params:
        print("\nGroup Parameters:")
        for group_name, params in group_params.items():
            print(f"  • {group_name}: {params}")
    
    node_overrides = hierarchical_params.get('node_overrides', {})
    if node_overrides:
        print("\nNode Override Parameters:")
        for node_id, overrides in node_overrides.items():
            print(f"  • Node {node_id}: {overrides}")
    
    materialized = sde_gen._materialize_node_parameters(hierarchical_params)

    print("\nMaterialized node parameters:")
    for node_id, params in materialized.items():
        drift_type = params.get('drift_type', 'mean_reverting')
        node_type = params.get('node_type', 'demand_source')
        print(f"Node {node_id} ({node_type}): drift={drift_type}")
        print(f"  κ={params.get('kappa', 'N/A'):.3f}, baseline={params['baseline']:.1f}, λ={params['lambda']:.3f}, σ={params['sigma']:.3f}")
        
        # Show drift-specific parameters
        if drift_type == 'sinusoidal':
            A = params.get('A', 0)
            omega = params.get('omega', 0)
            phi = params.get('phi', 0)
            period = 2*np.pi/omega if omega > 0 else float('inf')
            # Convert phase to peak time (hours from midnight)
            peak_time = (-phi / omega) % (2*np.pi/omega) if omega > 0 else 0
            print(f"  Sinusoidal: A={A:.1f}, ω={omega:.3f} (period={period:.1f}h)")
            print(f"    Phase shift: φ={phi:.3f} rad (peak at {peak_time:.1f}h)")
        elif drift_type == 'constant':
            alpha = params.get('alpha', 0)
            print(f"  Constant: α={alpha:.3f}")
        elif drift_type == 'logistic':
            r = params.get('r', 0)
            K = params.get('K', 0)
            print(f"  Logistic: r={r:.3f}, K={K:.1f}")
        print()
    
    # NOTE: Preview baseline modulation removed (weekly_cycle no longer in schema)
    # Time dynamics now handled by node-specific peaks and adjacency_modulation
    # print("\nPreview of baseline modulation:")
    # weekend_mult = hierarchical_params['time_modulation'].get('weekly_cycle', {}).get('weekend_multiplier', 1.0)
    # if weekend_mult != 1.0:
    #     print(f"• Weekend baseline modulation: {weekend_mult:.1f}x")
    #     for node_id, params in materialized.items():
    #         original_baseline = params['baseline']
    #         weekend_baseline = original_baseline * weekend_mult
    #         print(f"  Node {node_id}: {original_baseline:.1f} → {weekend_baseline:.1f} (weekend)")
    
    
    print("\nTime-varying Adjacency Matrix Summary:")
    time_varying_adj = network_sde['time_varying_adjacency']
    base_adj = np.array(time_varying_adj['base_adjacency'])
    print(f"• Base adjacency matrix shape: {base_adj.shape}")
    print(f"• Non-zero connections: {np.count_nonzero(base_adj)}")
    
    modulation = time_varying_adj['time_modulation']
    patterns = modulation.get('patterns', [])
    print(f"• Time modulation patterns: {len(patterns)} active")
    if patterns:
        for i, p in enumerate(patterns, 1):
            print(f"  Pattern {i}: time_range {p.get('time_range', [])} affecting {len(p.get('edge_modulations', {}))} edge(s)")
    
    print("\n" + "="*70 + "\n")
    
    # Run simulation (use validated data if available from Judge Agent 2)
    if 2 in enabled_judges and ts_data is not None and generation_info is not None:
        print("Using validated time series data from Judge Agent 2...")
        print(f"✓ Data already generated and validated")
    else:
        print("Running SDE simulation...")
        ts_data, generation_info = sde_gen.generate_spatiotemporal_data(network_sde)
    
    print(f"\n✓ Generated {ts_data.shape[0]} time series")
    print(f"✓ Sequence length: {ts_data.shape[1]}")
    print(f"✓ Integration method: {generation_info['integration_method']}")
    print(f"✓ Time step (dt): {generation_info['dt']:.6f} hours")
    
    print("\n" + "="*70 + "\n")
    
    # ===== AGENT 6: Visualization Agent =====
    print("=" * 70)
    print("AGENT 6: Visualization Agent")
    print("=" * 70)
    
    # Save detailed results to files
    print("\nSaving results...")
    import datetime
    import pickle
    import json
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename prefix: domain_nodes_timestamp
    # Clean domain name for filename (remove special characters)
    domain_clean = domain.replace('&', 'And').replace(' ', '').replace('/', '_')
    file_prefix = f"{domain_clean}_node{num_nodes}_{timestamp}"
    
    # Save detailed description to text file
    desc_file_path = f"output/{file_prefix}_results.txt"
    with open(desc_file_path, 'w', encoding='utf-8') as f:
        if enabled_judges:
            f.write(f"=== {6 + len(enabled_judges)}-Agent Spatial-Temporal Data Generation Results (with Judge Agents {enabled_judges}) ===\n\n")
        else:
            f.write("=== 6-Agent Spatial-Temporal Data Generation Results ===\n\n")
        f.write(f"Generation time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Agent 1: Scenario Description\n")
        f.write("-" * 50 + "\n")
        f.write(scenario + "\n\n")
        
        f.write("Agent 2: Structured Scenario JSON\n")
        f.write("-" * 50 + "\n")
        f.write(f"Variable: {structured_scenario['variable']}\n")
        f.write(f"Time span: {structured_scenario['time_span']}\n")
        f.write(f"Sampling frequency: {structured_scenario['sampling_frequency']}\n")
        f.write(f"Sequence length: {sde_gen.seq_len}\n")
        f.write(f"Number of nodes: {len(structured_scenario['nodes'])}\n\n")
        
        f.write("Nodes:\n")
        for node in structured_scenario["nodes"]:
            f.write(f"  Node {node['id']}: [{node['type']}] {node['name']}\n")
            f.write(f"    {node['description']}\n")
        f.write("\nEdges:\n")
        for edge in structured_scenario["edges"]:
            f.write(f"  {edge['source']} -> {edge['target']}: {edge['relationship']}\n")
        f.write("\n")
        
        f.write("Agent 3: SDE Parameters (Hierarchical)\n")
        f.write("-" * 50 + "\n")
        hierarchical_params = network_sde['sde_parameters']
        
        f.write("\nGlobal Defaults:\n")
        global_defaults = hierarchical_params.get('global_defaults', {})
        for key, value in global_defaults.items():
            f.write(f"  {key}: {value}\n")
        
        group_params = hierarchical_params.get('group_params', {})
        if group_params:
            f.write("\nGroup Parameters:\n")
            for group_name, params in group_params.items():
                f.write(f"  {group_name}: {params}\n")
        
        node_overrides = hierarchical_params.get('node_overrides', {})
        if node_overrides:
            f.write("\nNode Overrides:\n")
            for node_id, overrides in node_overrides.items():
                f.write(f"  Node {node_id}: {overrides}\n")
        
        materialized_params = sde_gen._materialize_node_parameters(hierarchical_params)

        f.write("\nMaterialized Parameters (per node):\n")
        for node_id, params in materialized_params.items():
            # Check if node_id is within bounds
            if node_id < len(structured_scenario["nodes"]):
                node_info = structured_scenario["nodes"][node_id]
                f.write(f"\nNode {node_id}: [{node_info['type']}] {node_info['name']}\n")
            else:
                f.write(f"\nNode {node_id}: [type unknown] (node info not available)\n")
            
            f.write(f"  Drift type: {params.get('drift_type', 'mean_reverting')}\n")
            f.write(f"  κ (kappa): {params['kappa']:.4f}\n")
            f.write(f"  baseline: {params['baseline']:.2f}\n")
            f.write(f"  λ (lambda): {params['lambda']:.4f}\n")
            f.write(f"  σ (sigma): {params['sigma']:.4f}\n")
            f.write(f"  Diffusion shape: {params.get('diffusion_shape', 'constant')}\n")
            f.write("-" * 30 + "\n")
        
        f.write("\nAgent 4: Time-Varying Adjacency Matrix\n")
        f.write("-" * 50 + "\n")
        time_varying_adj = network_sde['time_varying_adjacency']
        f.write(f"Base adjacency (all edges = 0.1):\n{np.array(time_varying_adj['base_adjacency'])}\n\n")
        f.write(f"Time modulation:\n{json.dumps(time_varying_adj['time_modulation'], indent=2)}\n")
        
        f.write("\nAgent 5: Simulation Results\n")
        f.write("-" * 50 + "\n")
        f.write(f"Integration method: {generation_info['integration_method']}\n")
        f.write(f"Time step (dt): {generation_info['dt']:.6f} hours\n")
        f.write(f"Generated data shape: {ts_data.shape}\n")
        
    print(f"Detailed results saved to: {desc_file_path}")
    
    # Save complete dataset to pickle file
    complete_data_file = f"output/{file_prefix}_data.pkl"
    complete_data = {
        "timestamp": timestamp,
        "agent1_scenario": scenario,
        "agent2_structured_scenario": structured_scenario,
        "agent3_sde_parameters": network_sde['sde_parameters'],
        "agent4_time_varying_adjacency": network_sde['time_varying_adjacency'],
        "agent5_simulation_data": ts_data,
        "generation_info": generation_info,
        "seq_len": sde_gen.seq_len
    }
    
    with open(complete_data_file, 'wb') as f:
        pickle.dump(complete_data, f)
    print(f"Complete dataset saved to: {complete_data_file}")
    
    # Save as JSON (for cross-platform compatibility)
    json_file_path = f"output/{file_prefix}_data.json"
    json_data = {
        "timestamp": timestamp,
        "agent1_scenario": scenario,
        "agent2_structured_scenario": structured_scenario,
        "agent3_sde_parameters": network_sde["sde_parameters"],
        "agent4_time_varying_adjacency": network_sde["time_varying_adjacency"],
        "agent5_simulation_data": ts_data.tolist(),
        "seq_len": sde_gen.seq_len,
        "config": sde_gen.get_config_info()
    }
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON data saved: {json_file_path}")
    
    print("\n" + "="*70 + "\n")

    png_path = None
    html_path = None
    save_path = None

    if generate_viz:
        # Visualize results
        print("Generating visualization...")

        # Generate both PNG and HTML visualizations
        if num_nodes <= 3:
            # For small networks, generate both PNG (for quick preview) and HTML (for interactivity)
            print(f"Generating matplotlib visualization (PNG)...")
            png_path = f"output/{file_prefix}_viz.png"
            sde_gen.visualize_network_sde_results(ts_data, network_sde, generation_info, save_path=png_path)

            print(f"Generating HTML interactive visualization...")
            html_path = f"output/{file_prefix}_interactive.html"
            sde_gen.visualize_network_sde_html(ts_data, network_sde, generation_info, save_path=html_path)

            save_path = html_path  # Primary output is HTML
        else:
            # For larger networks, only generate HTML
            print(f"Generating HTML interactive visualization...")
            save_path = f"output/{file_prefix}_interactive.html"
            sde_gen.visualize_network_sde_html(ts_data, network_sde, generation_info, save_path=save_path)
    else:
        print("Skipping visualization (generate_viz=False).")
    
    print("\n" + "="*70)
    if enabled_judges:
        print(f"{6 + len(enabled_judges)}-AGENT PIPELINE (WITH JUDGE AGENTS {enabled_judges}) COMPLETED SUCCESSFULLY")
    else:
        print("6-AGENT PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Description: {desc_file_path}")
    print(f"  - Complete data (pickle): {complete_data_file}")
    print(f"  - JSON data: {json_file_path}")
    if generate_viz:
        if num_nodes <= 3 and png_path:
            print(f"  - Visualization (PNG): {png_path}")
            print(f"  - Visualization (HTML): {html_path}")
        elif save_path:
            print(f"  - Visualization (HTML): {save_path}")
    
    # 保存完整的Agent交互日志
    if logger:
        print("\n" + "="*70)
        print("保存Agent交互日志...")
        print("="*70)
        logger.save_complete_log()
    
    return {
        "agent1_scenario": scenario,
        "agent2_structured_scenario": structured_scenario,
        "agent3_sde_parameters": network_sde['sde_parameters'],
        "agent4_time_varying_adjacency": network_sde['time_varying_adjacency'],
        "agent5_simulation_data": ts_data,
        "generation_info": generation_info,
        "data_files": {
            "pickle": complete_data_file,
            "json": json_file_path,
            "description": desc_file_path,
            "visualization": save_path
        },
        "logger": logger
    }

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate spatial-temporal data with Network SDE')
    parser.add_argument('--num_nodes', type=int, default=3,
                       help='Number of nodes in the network (default: 3)')
    parser.add_argument('--domain', type=str, default='traffic', help='Domain/scenario type (default: traffic)')
    parser.add_argument('--judges', type=str, default='',
                        help='Specify which judge agents to enable, separated by commas (e.g., "1,2"). Use "none" to disable all judges.')

    args = parser.parse_args()

    enabled_judges = []
    if args.judges.lower() != 'none':
        try:
            enabled_judges = [int(j.strip()) for j in args.judges.split(',') if j.strip()]
        except ValueError:
            print("Error: Invalid input for --judges. Please provide a comma-separated list of integers (e.g., '1,2') or 'none'.")
            exit(1)

    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  - Number of nodes: {args.num_nodes}")
    print(f"  - Domain: {args.domain}")
    print(f"  - LLM model: {os.environ.get('LLM_MODEL', 'gpt-4o-mini')}")
    print(f"  - Judge agents: {'Enabled: ' + str(enabled_judges) if enabled_judges else 'Disabled'}")
    print(f"{'='*70}\n")

    result = demo_network_sde_generation(
        enabled_judges=enabled_judges,
        num_nodes=args.num_nodes,
        domain=args.domain,
    )
    
    # Output statistics
    print("\n=== Generation Results Statistics ===")
    print(f"Time series shape: {result['agent5_simulation_data'].shape}")
    print(f"Number of nodes: {len(result['agent2_structured_scenario']['nodes'])}")
    print(f"Integration method: {result['generation_info']['integration_method']}")
    
    # Calculate inter-series correlation
    correlation_matrix = np.corrcoef(result['agent5_simulation_data'])
    avg_correlation = np.mean(correlation_matrix[np.triu_indices(len(correlation_matrix), k=1)])
    print(f"Average inter-node correlation: {avg_correlation:.3f}")
