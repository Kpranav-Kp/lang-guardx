"""
lang_guardx/agent/adapter.py

Layer 2 — LangChain Adapter.
THIS IS THE ONLY FILE IN LAYER 2 THAT IMPORTS LANGCHAIN.

Latency improvement: sql_db_query_checker removed from tool list.
It adds one full LLM round trip per query and is redundant —
SQLPolicyEngine already validates all SQL before execution.

Trace: every run() returns (answer, AgentTrace).
AgentTrace records per-step verdicts, latency, block/rewrite counts,
and exfiltration flag for thesis metrics (RQ2 coverage, latency overhead).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from lang_guardx.detection.core import Detector

from .engine import SQLPolicyEngine
from .policy import PolicyVerdict, Verdict


@dataclass
class StepTrace:
    """One tool call in the agent reasoning loop."""

    step_index: int
    tool_name: str
    tool_input: str
    tool_output: str
    policy_verdict: PolicyVerdict | None = None
    blocked: bool = False
    latency_ms: float = 0.0
    layer3_flagged: bool = False


@dataclass
class AgentTrace:
    """
    Full execution trace for one agent.run() call.

    Thesis metric fields:
      steps            -> per-step verdicts (RQ2: P2SQL attack coverage)
      total_latency_ms -> end-to-end wall time (latency overhead vs PromptGuard 7.2%)
      policy_hit_count -> SQL queries intercepted by engine
      block_count      -> queries BLOCKED by policy
      rewrite_count    -> queries REWRITTEN (LIMIT added, user scope injected)
      exfiltration_flag -> RI.2 heuristic: DB tool followed by network tool
    """

    question: str
    steps: list[StepTrace] = field(default_factory=list)
    final_answer: str | None = None
    blocked_at_step: int | None = None
    block_reason: str | None = None
    exfiltration_flag: bool = False
    policy_hit_count: int = 0
    block_count: int = 0
    rewrite_count: int = 0
    total_latency_ms: float = 0.0
    layer3_hits: int = 0

    def summary(self) -> str:
        """One-line summary for smoke test output."""
        return (
            f"latency={self.total_latency_ms:.1f}ms | "
            f"steps={len(self.steps)} | "
            f"hits={self.policy_hit_count} | "
            f"blocked={self.block_count} | "
            f"rewritten={self.rewrite_count} | "
            f"exfil={self.exfiltration_flag} | "
            f"layer3={self.layer3_hits}"
        )


_DB_TOOL_KEYWORDS = {"sql", "query", "database", "db"}
_NETWORK_TOOL_KEYWORDS = {
    "requests_get",
    "requests_post",
    "http",
    "web",
    "send_email",
    "file_write",
    "upload",
    "search",
}


def _check_tool_sequence(steps: list[StepTrace]) -> str | None:
    """
    Detect RI.2 pattern: DB/SQL tool immediately followed by network/write tool.
    Heuristic — logs suspicious sequences, not confirmed attacks.
    Returns reason string if detected, None otherwise.
    """
    names = [s.tool_name.lower() for s in steps]
    for i in range(len(names) - 1):
        is_db = any(kw in names[i] for kw in _DB_TOOL_KEYWORDS)
        is_net = names[i + 1] in _NETWORK_TOOL_KEYWORDS or any(kw in names[i + 1] for kw in ("email", "post", "send", "write", "upload", "http"))
        if is_db and is_net:
            return f"Potential RI.2 exfiltration: '{names[i]}' -> '{names[i + 1]}'"
    return None


class _PolicyEnforcedDatabase(SQLDatabase):
    """
    Wraps SQLDatabase so every db.run() passes through SQLPolicyEngine.
    Inherits SQLDatabase only to satisfy toolkit type checks.
    All real SQLDatabase state is on the wrapped _db, delegated via __getattr__.
    """

    def __init__(self, db: SQLDatabase, engine: SQLPolicyEngine):
        object.__setattr__(self, "_db", db)
        object.__setattr__(self, "_guard", engine)
        object.__setattr__(self, "_last_verdict", None)
        object.__setattr__(self, "_trace", None)

    def _set_trace(self, trace: AgentTrace) -> None:
        object.__setattr__(self, "_trace", trace)

    def run(self, command: str, *args, **kwargs) -> str:
        guard: SQLPolicyEngine = object.__getattribute__(self, "_guard")
        trace: AgentTrace | None = object.__getattribute__(self, "_trace")

        verdict = guard.validate(command)
        object.__setattr__(self, "_last_verdict", verdict)

        if trace is not None:
            trace.policy_hit_count += 1
            if verdict.verdict == Verdict.BLOCKED:
                trace.block_count += 1
            elif verdict.verdict == Verdict.REWRITTEN:
                trace.rewrite_count += 1

        if verdict.verdict == Verdict.BLOCKED:
            return f"[LangGuardX BLOCKED] {'; '.join(verdict.violations)}"

        return str(self._db.run(str(verdict.safe_sql), *args, **kwargs))

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_db"), name)

    @property
    def last_verdict(self) -> PolicyVerdict | None:
        return object.__getattribute__(self, "_last_verdict")


_SYSTEM_PROMPT = """\
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results and return the answer.

Rules:
- Always look at the available tables first.
- Never query all columns — only ask for relevant columns.
- Limit results to at most {top_k} rows unless the user specifies otherwise.
- If a query is blocked by policy, tell the user clearly. Do NOT retry.
"""


class ProtectedSQLAgent:
    """
    LangChain SQL agent with Layer 2 policy enforcement at the DB boundary.

    Latency improvement over naive agent:
      - sql_db_query_checker removed (saves 1 LLM call per query, ~500ms-2s)
      - SQLPolicyEngine validates SQL in <2ms, replaces checker entirely

    Usage:
        policy = SQLPolicy(permitted_tables=["products"], max_rows=50)
        engine = SQLPolicyEngine(policy)
        agent  = ProtectedSQLAgent(llm=llm, db=db, engine=engine)
        answer, trace = agent.run("How many products?")
        print(trace.summary())
    """

    def __init__(
        self,
        llm: BaseChatModel,
        db: SQLDatabase,
        engine: SQLPolicyEngine,
        top_k: int = 10,
    ) -> None:
        self._protected_db = _PolicyEnforcedDatabase(db, engine)

        toolkit = SQLDatabaseToolkit(db=self._protected_db, llm=llm)
        all_tools = toolkit.get_tools()

        self._tools = [t for t in all_tools if t.name != "sql_db_query_checker"]
        self._detector = Detector()

        system_prompt = _SYSTEM_PROMPT.format(
            dialect=db.dialect,
            top_k=top_k,
        )

        self._agent = create_agent(
            model=llm,
            tools=self._tools,
            system_prompt=system_prompt,
        )

    def run(self, question: str) -> tuple[str, AgentTrace]:
        """
        Run the agent. Returns (answer, AgentTrace).
        AgentTrace.summary() gives a one-line metric string for smoke tests.
        """
        trace = AgentTrace(question=question)
        self._protected_db._set_trace(trace)
        cb = _TraceCallback(trace, self._protected_db, self._detector)

        start = time.monotonic()
        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=RunnableConfig(callbacks=[cb]),
            )
            messages = result.get("messages", [])
            answer = messages[-1].content if messages else str(result)
        except Exception as exc:
            answer = f"[LangGuardX] Agent error: {exc}"
            trace.block_reason = str(exc)

        trace.total_latency_ms = (time.monotonic() - start) * 1000
        trace.final_answer = answer

        exfil = _check_tool_sequence(trace.steps)
        if exfil:
            trace.exfiltration_flag = True
            trace.block_reason = exfil

        return answer, trace

    @property
    def last_blocked(self) -> bool:
        v = self._protected_db.last_verdict
        return v is not None and v.verdict == Verdict.BLOCKED


class _TraceCallback(BaseCallbackHandler):
    """
    Records tool invocations into AgentTrace.
    Inherits from BaseCallbackHandler to satisfy RunnableConfig type requirements.
    """

    def __init__(self, trace: AgentTrace, db: _PolicyEnforcedDatabase, detector: Detector) -> None:
        super().__init__()
        self._trace = trace
        self._db = db
        self._detector = detector
        self._step_start = 0.0
        self._step_index = 0
        self._pending: StepTrace | None = None

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self._step_start = time.monotonic()
        self._pending = StepTrace(
            step_index=self._step_index,
            tool_name=serialized.get("name", "unknown"),
            tool_input=input_str,
            tool_output="",
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self._pending is None:
            return
        step = self._pending
        step.tool_output = output
        step.latency_ms = (time.monotonic() - self._step_start) * 1000

        if step.tool_name == "sql_db_query":
            verdict = self._db.last_verdict
            step.policy_verdict = verdict
            if verdict and verdict.verdict == Verdict.BLOCKED:
                step.blocked = True
                self._trace.blocked_at_step = step.step_index
                self._trace.block_reason = "; ".join(verdict.violations)

            if not step.blocked:
                try:
                    sanitized_list, flags = self._detector.scan_db_strings([output])
                    if flags:
                        step.tool_output = sanitized_list[0]
                        step.layer3_flagged = True
                        self._trace.layer3_hits += 1
                except Exception as e:
                    print(f"[Layer 3 warning] Scanner failed: {e}")

        self._trace.steps.append(step)
        self._step_index += 1
        self._pending = None

    def on_llm_start(self, *a: Any, **kw: Any) -> None:
        pass

    def on_llm_end(self, *a: Any, **kw: Any) -> None:
        pass

    def on_llm_error(self, *a: Any, **kw: Any) -> None:
        pass

    def on_tool_error(self, *a: Any, **kw: Any) -> None:
        pass

    def on_chain_start(self, *a: Any, **kw: Any) -> None:
        pass

    def on_chain_end(self, *a: Any, **kw: Any) -> None:
        pass

    def on_chain_error(self, *a: Any, **kw: Any) -> None:
        pass

    def on_agent_action(self, *a: Any, **kw: Any) -> None:
        pass

    def on_agent_finish(self, *a: Any, **kw: Any) -> None:
        pass
