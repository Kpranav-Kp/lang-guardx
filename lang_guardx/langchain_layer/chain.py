from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_experimental.sql import SQLDatabaseChain
from lang_guardx.detection.core import Detector
from lang_guardx.langchain_layer.sql_policy import SQLPolicy, SQLPolicyEngine, PolicyVerdict

# Fallback Messages

_BLOCKED_INPUT = (
    "I'm unable to process that request. "
    "It appears to contain content that violates our usage policies. "
)

_BLOCKED_SQL = (
    "I'm unable to execute that SQL query. "
    "It was flagged by security policy as violating our SQL usage policies. "
)

_BLOCKED_RESULTS = (
    "The database returned content that could not be safely processed. "
    "Please contact support if you believe this is an error and try again. "
)

@dataclass
class ChainTrace:
    user_input: str = ""
    blocked_at: Optional[str] = None
    blocked_reason: Optional[str] = None
    generated_sql: Optional[str] = None
    policy_verdict: Optional[PolicyVerdict] = None
    executed_sql: Optional[str] = None
    db_rows_raw: List[Dict] = field(default_factory=list)
    db_rows_sanitized: List[Dict] = field(default_factory=list)
    indirect_flags: list[Any] = field(default_factory=list)
    llm_output: Optional[str] = None
    latency: float = 0.0

#SQL Intercept Callback

class _SQLInterceptCallBack(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.captured_sql: Optional[str] = None
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            text = response.generations[0][0].text
            if "SQLQuery:" in text:
                sql_part = text.split("SQLQuery:", 1)[1]
                sql_part = sql_part.split("SQLResult:")[0]
                self.captured_sql = sql_part.strip()
        except (IndexError, AttributeError):
            pass  
    
class ProtectedChain:
    def __init__(
        self,
        chain: SQLDatabaseChain,
        policy_engine: SQLPolicyEngine,
        detector: Detector,
        system_prompt: str = ""
    ) -> None:
        self._chain = chain
        self._policy_engine = policy_engine
        self._detector = detector
        self._system_prompt = system_prompt
    def run(self, user_input: str) -> tuple[str, ChainTrace]:
        trace = ChainTrace(user_input=user_input)
        start = time.monotonic()
        # Interception point 1: Layer 1 Input Gate
        detection_result = self._detector.check(user_input)
        if detection_result.blocked:
            trace.blocked_at = "input"
            trace.blocked_reason = (
                f"{detection_result.reason}: {detection_result.detail}"
            )
            trace.latency = (time.monotonic() - start) * 1000
            return _BLOCKED_INPUT, trace
        
        # Interception point 2: Layer 2 SQL Policy Enforcement SQL Gate
        sql_callback = _SQLInterceptCallBack()
        structured_input = self._build_structured_input(user_input)
        try:
            generated_sql_placeholder = "SELECT * from products LIMIT 10"
            trace.generated_sql = sql_callback.captured_sql or generated_sql_placeholder
        
        except Exception as exec:
            trace.blocked_at = "sql_generation"
            trace.blocked_reason = f"llm_error: {str(exec)}"
            trace.latency = (time.monotonic() - start) * 1000
            return _BLOCKED_INPUT, trace
        
        if trace.generated_sql:
            verdict = self._policy_engine.validate(trace.generated_sql)
            trace.policy_verdict = verdict

            if verdict.blocked:
                trace.blocked_at = "sql"
                trace.blocked_reason = "; ".join(verdict.violations)
                trace.latency = (time.monotonic() - start) * 1000
                return _BLOCKED_SQL, trace
            
            trace.executed_sql = verdict.safe_sql
        
        # Interception point 3: Layer 1's indirect output scanning Results Gate

        raw_rows: List[Dict] = []

        if raw_rows:
            sanitized_rows, flags = self._detector.scan_db_results(raw_rows)
            trace.db_rows_sanitized = sanitized_rows
            trace.indirect_flags = flags

            if flags:
                trace.blocked_at = "results"
                trace.blocked_reason = f"indirect_injection: {len(flags)} field(s) redacted"
                if len(flags) == len(raw_rows):
                    trace.latency = (time.monotonic() - start) * 1000
                    return _BLOCKED_RESULTS, trace
        else:
            trace.db_rows_sanitized = raw_rows

        response = "Placeholder response - wire up LLM final call here."
        trace.llm_output = response
        trace.latency = (time.monotonic() - start) * 1000
        return response, trace
    
    def _build_structured_input(self, user_input: str) -> str:
        import json
        structured = json.dumps(
            {
                "role_system": self._system_prompt,
                "role_user": user_input,
                "permittedoperations": self._policy_engine.policy.permitted_operations,
            },
            ensure_ascii=False,
        )
        return structured
    

# Public funtion
def protect_chain(
    chain: SQLDatabaseChain,
    policy: Optional[SQLPolicy] = None,
    current_user_id: Optional[int] = None,
    system_prompt: str = "",
    detector: Optional[Detector] = None,
) -> ProtectedChain:
    resolved_policy = policy or SQLPolicy.default()
    resolved_detector= detector or Detector()
    engine = SQLPolicyEngine(
        policy=resolved_policy,
        current_user_id=current_user_id,
    )
    
    return ProtectedChain(
        chain=chain,
        policy_engine=engine,
        detector=resolved_detector,
        system_prompt=system_prompt
    )