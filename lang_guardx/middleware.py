from __future__ import annotations

import time
from typing import Any, Protocol

from lang_guardx.agent.adapter import (
    _SYSTEM_PROMPT,
    AgentTrace,
    StepTrace,
    _check_tool_sequence,
    _PolicyEnforcedDatabase,
    _TraceCallback,
)
from lang_guardx.detection.core import Detector
from lang_guardx.events import EventBus


class AgentMiddleware(Protocol):
    """Pluggable agent backend for LangGuardX.

    Implement this protocol to integrate LangGuardX with any agent
    framework.  The ``run()`` method is called **after** Layer 1
    (input detection) has passed.  It is responsible for:

    * Generating SQL from the user's question
    * Executing SQL through LangGuardX's policy-enforced database
    * Collecting execution traces
    * Detecting exfiltration patterns

    The default implementation is :class:`LangChainSQLMiddleware`.
    """

    def run(self, question: str, detector: Detector, event_bus: EventBus) -> tuple[str, AgentTrace]:
        """Execute the agent for *question*.

        Returns ``(final_answer, trace)``.
        """
        ...


class LangChainSQLMiddleware:
    """LangChain agent backend with full LangGuardX policy enforcement.

    Wraps a LangChain ``create_agent()`` SQL agent with:

    * Layer 2 — every SQL query is validated through ``SQLPolicyEngine``
    * Layer 3 — every DB result is scanned for indirect injection
    * Exfiltration detection — DB-tool → network-tool sequences are flagged

    Usage::

        from lang_guardx import LangGuardX
        from lang_guardx.middleware import LangChainSQLMiddleware
        from langchain_community.utilities import SQLDatabase

        guard = LangGuardX("config.yaml")
        guard.use_middleware(LangChainSQLMiddleware(llm=llm, db=SQLDatabase.from_uri(...)))
        answer, trace = guard.run_agent("Show me top products")
    """

    def __init__(
        self,
        llm: Any,
        db: Any,
        engine: Any,
        top_k: int = 10,
    ) -> None:
        try:
            from langchain.agents import create_agent
            from langchain_community.agent_toolkits import SQLDatabaseToolkit
        except ImportError as exc:
            raise ImportError("LangChainSQLMiddleware requires the langchain extras. Install with: pip install langguardx[langchain]") from exc

        self._protected_db = _PolicyEnforcedDatabase(db, engine)
        toolkit = SQLDatabaseToolkit(db=self._protected_db, llm=llm)
        all_tools = toolkit.get_tools()
        self._tools = [t for t in all_tools if t.name != "sql_db_query_checker"]
        system_prompt = _SYSTEM_PROMPT.format(dialect=db.dialect, top_k=top_k)
        self._agent = create_agent(
            model=llm,
            tools=self._tools,
            system_prompt=system_prompt,
        )

    def run(self, question: str, detector: Detector, event_bus: EventBus) -> tuple[str, AgentTrace]:
        self._protected_db._set_trace(None)
        trace = AgentTrace(question=question)
        self._protected_db._set_trace(trace)
        cb = _TraceCallback(trace, self._protected_db, detector)

        try:
            from langchain_core.runnables import RunnableConfig
        except ImportError as exc:
            raise ImportError("LangChainSQLMiddleware requires the langchain extras. Install with: pip install langguardx[langchain]") from exc

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
        from .agent.policy import Verdict

        v = self._protected_db.last_verdict
        return v is not None and v.verdict == Verdict.BLOCKED


class DirectSQLMiddleware:
    """Minimal middleware that executes a fixed SQL query for each question.

    Useful for testing or for use cases where the SQL is not generated
    by an LLM but provided directly by the caller.

    Usage::

        guard.use_middleware(
            DirectSQLMiddleware(
                db=db,
                engine=engine,
                sql_map={
                    "top products": "SELECT name FROM products ORDER BY sales DESC LIMIT 10",
                },
            )
        )
    """

    def __init__(
        self,
        db: Any,
        engine: Any,
        sql_map: dict[str, str] | None = None,
        default_sql: str | None = None,
    ) -> None:
        self._protected_db = _PolicyEnforcedDatabase(db, engine)
        self._sql_map = sql_map or {}
        self._default_sql = default_sql

    def run(self, question: str, detector: Detector, event_bus: EventBus) -> tuple[str, AgentTrace]:
        trace = AgentTrace(question=question)

        sql = self._sql_map.get(question.strip().lower(), self._default_sql)
        if sql is None:
            return f"[LangGuardX] No mapped SQL for: {question}", trace

        self._protected_db._set_trace(trace)
        start = time.monotonic()
        try:
            output = self._protected_db.run(sql)
            trace.policy_hit_count += 1
        except Exception as exc:
            output = f"[LangGuardX] Error: {exc}"

        trace.total_latency_ms = (time.monotonic() - start) * 1000
        trace.final_answer = output

        step = StepTrace(
            step_index=0,
            tool_name="sql_db_query",
            tool_input=sql,
            tool_output=output,
        )
        step.latency_ms = trace.total_latency_ms
        trace.steps.append(step)

        return output, trace
