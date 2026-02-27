from __future__ import annotations
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from langchain_classic.agents import AgentExecutor
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import LLMResult
from langchain.tools import BaseTool
from lang_guardx.detection.core import Detector
from lang_guardx.langchain_layer.sql_policy import SQLPolicy, SQLPolicyEngine, PolicyVerdict


_BLOCKED_INPUT = (
    "I'm unable to process that request. "
    "It appears to contain content that violates the security policy."
)

_BLOCKED_TOOL = (
    "That action was blocked by the security policy. "
    "The requested tool or operation is not permitted in this context."
)

_BLOCKED_SEQUENCE = (
    "The agent's action sequence was flagged as a potential data exfiltration attempt. "
    "The operation has been aborted."
)



_DANGEROUS_TOOL_NAMES: Set[str] = {
    "terminal", "shell", "bash", "exec", "python_repl",
    "subprocess", "os_command", "system",
}

_DANGEROUS_TOOL_TYPES: Set[str] = {
    "ShellTool", "PythonREPLTool", "SubprocessTool",
}

# Tool names that suggest network/output (used in exfiltration sequence detection)
_NETWORK_TOOL_NAMES: Set[str] = {
    "requests_get", "requests_post", "http_request",
    "web_browser", "search", "send_email", "file_write",
}


@dataclass
class TraceStep:
    """
    One step in the agent's thought-action-observation loop.
    Matches LangChain's agent execution flow.
    """

    step_index: int
    thought: Optional[str] = None       
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None    
    tool_output: Optional[str] = None  
    tool_output_sanitized: Optional[str] = None  
    sql_generated: Optional[str] = None  
    sql_executed: Optional[str] = None  
    policy_verdict: Optional[PolicyVerdict] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class AgentTrace:
    user_input: str = ""
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    blocked_at_step: Optional[int] = None
    block_reason: Optional[str] = None
    all_tool_calls: List[str] = field(default_factory=list)
    all_sql_queries: List[str] = field(default_factory=list)
    indirect_flags_total: int = 0
    total_latency_ms: float = 0.0


class _AgentInterceptCallback(BaseCallbackHandler):
    def __init__(self, trace: AgentTrace) -> None:
        super().__init__()
        self._trace = trace
        self._current_step: Optional[TraceStep] = None
        self._step_index = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            text = response.generations[0][0].text
            if self._current_step is None:
                self._current_step = TraceStep(step_index=self._step_index)
            self._current_step.thought = text
        except (IndexError, AttributeError):
            pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        if self._current_step is None:
            self._current_step = TraceStep(step_index=self._step_index)

        self._current_step.tool_name = action.tool
        self._current_step.tool_input = str(action.tool_input)
        self._trace.all_tool_calls.append(action.tool)

        sql = self._extract_sql(str(action.tool_input))
        if sql:
            self._current_step.sql_generated = sql
            self._trace.all_sql_queries.append(sql)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self._current_step is None:
            self._current_step = TraceStep(step_index=self._step_index)

        self._current_step.tool_output = output
        self._trace.steps.append(self._current_step)
        self._current_step = None
        self._step_index += 1

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self._trace.final_answer = finish.return_values.get("output", "")

    def _extract_sql(self, tool_input: str) -> Optional[str]:
        stripped = tool_input.strip()
        upper = stripped.upper()
        if any(upper.startswith(kw) for kw in ("SELECT", "UPDATE", "DELETE", "DROP", "INSERT")):
            return stripped
        if '"query"' in stripped or "'query'" in stripped:
            import json
            try:
                data = json.loads(stripped)
                return data.get("query") or data.get("sql")
            except (json.JSONDecodeError, AttributeError):
                pass
        return None

def validate_agent_config(agent_executor: AgentExecutor) -> List[str]:
    warnings_found: List[str] = []

    tools: List[BaseTool] = getattr(agent_executor, "tools", [])

    for tool in tools:
        name = (tool.name or "").lower()
        type_name = type(tool).__name__

        if name in _DANGEROUS_TOOL_NAMES or type_name in _DANGEROUS_TOOL_TYPES:
            warnings_found.append(
                f"CRITICAL: Tool '{tool.name}' ({type_name}) allows "
                f"arbitrary execution. This is a high-risk capability "
                f"in a user-facing agent."
            )

    return warnings_found


def _check_tool_sequence(steps: List[TraceStep]) -> Optional[str]:
    tool_names = [
        s.tool_name.lower() for s in steps
        if s.tool_name
    ]

    # Pattern: any DB/SQL tool followed by any network/write tool
    for i, name in enumerate(tool_names[:-1]):
        next_name = tool_names[i + 1]
        is_db_tool = any(
            kw in name
            for kw in ("sql", "query", "database", "db", "search_db")
        )
        is_network_tool = next_name in _NETWORK_TOOL_NAMES or any(
            kw in next_name
            for kw in ("email", "post", "send", "write", "upload", "http")
        )
        if is_db_tool and is_network_tool:
            return (
                f"exfiltration_sequence: DB tool '{name}' followed by "
                f"network/write tool '{next_name}' — potential RI.2 pattern"
            )

    return None

class ProtectedAgent:
    def __init__(
        self,
        agent_executor: AgentExecutor,
        detector: Detector,
        policy_engine: SQLPolicyEngine,
        system_prompt: str = "",
        raise_on_dangerous_tools: bool = False,
    ) -> None:
        self._agent = agent_executor
        self._detector = detector
        self._policy_engine = policy_engine
        self._system_prompt = system_prompt

        config_warnings = validate_agent_config(agent_executor)
        for w in config_warnings:
            if raise_on_dangerous_tools:
                raise ValueError(w)
            else:
                warnings.warn(w, stacklevel=2)

    def run(self, user_input: str) -> tuple[str, AgentTrace]:
        trace = AgentTrace(user_input=user_input)
        start = time.monotonic()
        detection_result = self._detector.check(user_input)
        if detection_result.blocked:
            trace.block_reason = (
                f"{detection_result.reason}: {detection_result.detail}"
            )
            trace.total_latency_ms = (time.monotonic() - start) * 1000
            return _BLOCKED_INPUT, trace

        intercept_cb = _AgentInterceptCallback(trace=trace)

        try:
            result = self._agent.invoke(
                {"input": user_input},
                config={"callbacks": [intercept_cb]}
            )
            raw_response = result.get("output", "")
            
        except Exception as exc:
            trace.block_reason = f"agent_execution_error: {exc}"
            trace.total_latency_ms = (time.monotonic() - start) * 1000
            return _BLOCKED_INPUT, trace

        for step in trace.steps:
            step_start = time.monotonic()

            # SQL policy check on generated SQL
            if step.sql_generated:
                verdict = self._policy_engine.validate(step.sql_generated)
                step.policy_verdict = verdict
                step.sql_executed = verdict.safe_sql

                if verdict.blocked:
                    step.blocked = True
                    step.block_reason = "; ".join(verdict.violations)
                    trace.blocked_at_step = step.step_index
                    trace.block_reason = step.block_reason
                    trace.total_latency_ms = (time.monotonic() - start) * 1000
                    return _BLOCKED_TOOL, trace

            if step.tool_output:
                rows = self._parse_tool_output_to_rows(step.tool_output)
                if rows:
                    sanitized, flags = self._detector.scan_db_results(rows)
                    step.tool_output_sanitized = str(sanitized)
                    trace.indirect_flags_total += len(flags)

                    if flags and len(flags) == len(rows):
                        step.blocked = True
                        step.block_reason = "indirect_injection: all rows flagged"
                        trace.blocked_at_step = step.step_index
                        trace.block_reason = step.block_reason
                        trace.total_latency_ms = (time.monotonic() - start) * 1000
                        return _BLOCKED_TOOL, trace
                else:
                    step.tool_output_sanitized = step.tool_output

            step.latency_ms = (time.monotonic() - step_start) * 1000

        sequence_violation = _check_tool_sequence(trace.steps)
        if sequence_violation:
            trace.block_reason = sequence_violation
            trace.total_latency_ms = (time.monotonic() - start) * 1000
            return _BLOCKED_SEQUENCE, trace

        trace.final_answer = raw_response
        trace.total_latency_ms = (time.monotonic() - start) * 1000
        return raw_response, trace

    def _parse_tool_output_to_rows(
        self, tool_output: str
    ) -> List[Dict[str, Any]]:
        import ast
        try:
            parsed = ast.literal_eval(tool_output.strip())
            if isinstance(parsed, list):
                rows = []
                for i, item in enumerate(parsed):
                    if isinstance(item, dict):
                        rows.append(item)
                    elif isinstance(item, (list, tuple)):
                        rows.append({f"col_{j}": v for j, v in enumerate(item)})
                    else:
                        rows.append({"value": str(item)})
                return rows
        except (ValueError, SyntaxError):
            pass
        return []


def protect_agent(
    agent_executor: AgentExecutor,
    policy: Optional[SQLPolicy] = None,
    current_user_id: Optional[int] = None,
    system_prompt: str = "",
    detector: Optional[Detector] = None,
    raise_on_dangerous_tools: bool = False,
) -> ProtectedAgent:
    resolved_policy = policy or SQLPolicy.default()
    resolved_detector = detector or Detector()
    engine = SQLPolicyEngine(
        policy=resolved_policy,
        current_user_id=current_user_id,
    )

    return ProtectedAgent(
        agent_executor=agent_executor,
        detector=resolved_detector,
        policy_engine=engine,
        system_prompt=system_prompt,
        raise_on_dangerous_tools=raise_on_dangerous_tools,
    )