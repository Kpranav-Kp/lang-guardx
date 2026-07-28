"""LangChain adapter — backward-compatible wrapper for :class:`ProtectedSQLAgent`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .policy import PolicyVerdict, Verdict

logger = logging.getLogger(__name__)


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
    """Full execution trace for one agent.run() call."""

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
        return (
            f"latency={self.total_latency_ms:.1f}ms | "
            f"steps={len(self.steps)} | "
            f"hits={self.policy_hit_count} | "
            f"blocked={self.block_count} | "
            f"rewritten={self.rewrite_count} | "
            f"exfil={self.exfiltration_flag} | "
            f"layer3={self.layer3_hits}"
        )


# ── Internal helpers (shared with middleware) ─────────────────────────────────

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
    names = [s.tool_name.lower() for s in steps]
    for i in range(len(names) - 1):
        is_db = any(kw in names[i] for kw in _DB_TOOL_KEYWORDS)
        is_net = names[i + 1] in _NETWORK_TOOL_KEYWORDS or any(kw in names[i + 1] for kw in ("email", "post", "send", "write", "upload", "http"))
        if is_db and is_net:
            return f"Potential RI.2 exfiltration: '{names[i]}' -> '{names[i + 1]}'"
    return None


try:
    from langchain_community.utilities import SQLDatabase as _SQLDatabase
except ImportError:
    _SQLDatabase: type = object  # type: ignore


class _PolicyEnforcedDatabase(_SQLDatabase):  # type: ignore
    """Wraps a LangChain SQLDatabase to enforce SQL policy on every query."""

    def __init__(self, db, engine):
        object.__setattr__(self, "_db", db)
        object.__setattr__(self, "_guard", engine)
        object.__setattr__(self, "_last_verdict", None)
        object.__setattr__(self, "_trace", None)

    def _set_trace(self, trace: AgentTrace | None) -> None:
        object.__setattr__(self, "_trace", trace)

    def run(self, command: str, *args, **kwargs) -> str:
        guard = object.__getattribute__(self, "_guard")
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

        db = object.__getattribute__(self, "_db")
        return str(db.run(str(verdict.safe_sql), *args, **kwargs))

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_db"), name)

    @property
    def last_verdict(self) -> PolicyVerdict | None:
        return object.__getattribute__(self, "_last_verdict")


_SYSTEM_PROMPT = """\
You are a SQL agent for {dialect} database.
You may ONLY answer questions related to the database schema (products, orders, customers, etc.).
If the user asks anything not about the database, reply: "I can only answer questions about the store database."
- Use sql_db_list_tables first.
- Never use SELECT *.
- Limit to {top_k} rows.
- If a query is blocked, say so and stop.
"""


try:
    from langchain_core.callbacks import BaseCallbackHandler as _CallbackBase
except ImportError:
    _CallbackBase: type = object  # type: ignore


class _TraceCallback(_CallbackBase):  # type: ignore
    """LangChain callback that records StepTrace and runs Layer 3 scanning."""

    def __init__(self, trace: AgentTrace, db: _PolicyEnforcedDatabase, detector) -> None:
        import time

        self._trace = trace
        self._db = db
        self._detector = detector
        self._time = time
        self._step_start = 0.0
        self._step_index = 0
        self._pending: StepTrace | None = None

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        self._step_start = self._time.monotonic()
        self._pending = StepTrace(
            step_index=self._step_index,
            tool_name=serialized.get("name", "unknown"),
            tool_input=input_str,
            tool_output="",
        )

    def on_tool_end(self, output, **kwargs) -> None:
        if self._pending is None:
            return
        step = self._pending

        if hasattr(output, "content"):
            output_str = output.content
        elif not isinstance(output, str):
            output_str = str(output)
        else:
            output_str = output

        step.tool_output = output_str
        step.latency_ms = (self._time.monotonic() - self._step_start) * 1000

        if step.tool_name == "sql_db_query":
            verdict = self._db.last_verdict
            step.policy_verdict = verdict
            if verdict and verdict.verdict == Verdict.BLOCKED:
                step.blocked = True
                self._trace.blocked_at_step = step.step_index
                self._trace.block_reason = "; ".join(verdict.violations)

            if not step.blocked:
                try:
                    sanitized_list, flags = self._detector.scan_db_strings([output_str])
                    if flags:
                        step.tool_output = sanitized_list[0]
                        step.layer3_flagged = True
                        self._trace.layer3_hits += 1
                except Exception as e:
                    logger.warning("Layer 3 scanner failed: %s", e)

        self._trace.steps.append(step)
        self._step_index += 1
        self._pending = None


# ── Backward-compatible wrapper ───────────────────────────────────────────────


class ProtectedSQLAgent:
    """Legacy LangChain agent wrapper.

    .. deprecated::
        Use ``LangGuardX.use_middleware(LangChainSQLMiddleware(...))`` instead.
    """

    def __init__(self, llm, db, engine, top_k: int = 10) -> None:
        from lang_guardx.detection.core import Detector

        from ..middleware import LangChainSQLMiddleware

        self._middleware = LangChainSQLMiddleware(llm, db, engine, top_k=top_k)
        self._detector = Detector()

    def run(self, question: str) -> tuple[str, AgentTrace]:
        from lang_guardx.events import EventBus

        result = self._detector.check(question)
        if result.blocked:
            trace = AgentTrace(question=question)
            trace.block_reason = f"Layer 1 blocked: {result.reason} - {result.detail}"
            trace.block_count = 1
            return f"[LangGuardX BLOCKED] {trace.block_reason}", trace

        return self._middleware.run(question, self._detector, EventBus())

    @property
    def last_blocked(self) -> bool:
        return self._middleware.last_blocked
