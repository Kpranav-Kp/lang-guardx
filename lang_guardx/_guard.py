from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lang_guardx.adaptive.adaptive_engine import AdaptiveEngine
from lang_guardx.adaptive.threat_ontology import ThreatOntology
from lang_guardx.agent.engine import SQLPolicyEngine
from lang_guardx.agent.policy import PolicyVerdict, SQLPolicy
from lang_guardx.config import Config
from lang_guardx.context import GuardContext
from lang_guardx.detection.core import DetectionResult, Detector
from lang_guardx.detection.indirect import ScanResult
from lang_guardx.events import EventBus, GuardEvent
from lang_guardx.exceptions import BlockedRequest, ConfigurationError


class LangGuardX:
    """Single entry point for the LangGuardX security framework.

    Wires together all four defence layers (input detection, SQL policy,
    output scanning, adaptive learning) from a single ``Config`` object.

    Usage::

        # Quick start with defaults
        guard = LangGuardX()

        # From a YAML config file
        guard = LangGuardX("langguardx.yaml")

        # Programmatic config
        guard = LangGuardX(Config(engine={"policy": {"permitted_tables": ["products"]}}))

        # Layer 1 — input detection
        ctx = guard.protect("ignore previous instructions")
        if ctx.detection_result.blocked:
            print("Blocked by:", ctx.detection_result.reason)

        # Layer 2 — SQL policy
        verdict = guard.validate_sql("SELECT * FROM products")

        # Layer 3 — output scanning
        sanitized, flags = guard.scan_results([{"review": "..."}])

        # Layer 4 — adaptive learning
        guard.adapt("RI.1", "new pattern")

        # Agent middleware (decouples from LangChain)
        from lang_guardx.middleware import LangChainSQLMiddleware

        guard.use_middleware(LangChainSQLMiddleware(llm=llm, db=db))
        answer, trace = guard.run_agent("Show me top products")
    """

    def __init__(self, config: Config | str | Path | dict[str, Any] | None = None) -> None:
        self._config = self._resolve_config(config)
        self._event_bus = EventBus()

        # Layer 1 — Detection
        self._detector = Detector(config=self._config.detection)

        # Layer 2 — Policy engine
        self._engine = SQLPolicyEngine(
            policy=SQLPolicy(**self._config.engine.policy.model_dump()),
            current_user_id=self._config.engine.current_user_id,
            dialect=self._config.engine.dialect,
        )

        # Layer 4 — Adaptive (wired into detector automatically)
        self._adaptive: AdaptiveEngine | None = None
        if self._config.adaptive.enabled:
            ontology = ThreatOntology(yaml_path=self._config.adaptive.taxonomy_path)
            self._adaptive = AdaptiveEngine(
                bloom=self._detector.bloom,
                ontology=ontology,
                log_path=self._config.adaptive.log_path,
            )
            self._detector.set_adaptive_bloom(self._adaptive.get_adaptive_bloom())

        # Agent middleware (optional — set via use_middleware)
        self._middleware: Any = None

    # ── Config ────────────────────────────────────────────────────────────

    @property
    def config(self) -> Config:
        return self._config

    @staticmethod
    def _resolve_config(config: Config | str | Path | dict[str, Any] | None) -> Config:
        if config is None:
            return Config()
        if isinstance(config, Config):
            return config
        if isinstance(config, dict):
            return Config.from_dict(config)
        if isinstance(config, str | Path):
            path = Path(config)
            if not path.exists():
                raise ConfigurationError(f"Config file not found: {path}")
            suffix = path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                return Config.from_yaml(path)
            if suffix == ".json":
                return Config.from_json(path)
            if suffix == ".toml":
                return Config.from_toml(path)
            raise ConfigurationError(f"Unsupported config file format: {suffix}")
        raise ConfigurationError(f"Unexpected config type: {type(config)}")

    # ── Layer 1 — Input Detection ─────────────────────────────────────────

    def protect(self, text: str) -> GuardContext:
        """Run the full input detection pipeline on *text*."""
        ctx = GuardContext(raw_input=text)
        self._event_bus.emit(GuardEvent.BEFORE_DETECTION, ctx)
        try:
            result = self._detector.check(text)
            ctx.detection_result = result
        except Exception as exc:
            self._event_bus.emit(GuardEvent.ON_ERROR, ctx, exc)
            ctx.metadata["detection_error"] = str(exc)
            result = DetectionResult(blocked=False, reason="error", detail=str(exc))
            ctx.detection_result = result

        self._event_bus.emit(GuardEvent.AFTER_DETECTION, ctx)
        if result.blocked:
            self._event_bus.emit(GuardEvent.ON_BLOCK, ctx)
        return ctx

    def register_detector(self, layer: Any) -> None:
        """Register a custom detection layer (``DetectionLayer`` protocol).

        Runs in priority order (lower number = earlier).
        """
        self._detector.register_layer(layer)

    def remove_detector(self, name: str) -> None:
        """Remove a previously registered detection layer by its ``name``."""
        self._detector.remove_layer(name)

    # ── Layer 2 — SQL Policy ──────────────────────────────────────────────

    def validate_sql(self, sql: str) -> PolicyVerdict:
        """Validate a generated SQL query against the configured policy."""
        ctx = GuardContext(raw_input=sql)
        self._event_bus.emit(GuardEvent.BEFORE_POLICY, ctx)
        verdict = self._engine.validate(sql)
        ctx.policy_verdict = verdict
        ctx.generated_sql = sql
        self._event_bus.emit(GuardEvent.AFTER_POLICY, ctx)
        if verdict.verdict.name == "REWRITTEN":
            self._event_bus.emit(GuardEvent.ON_REWRITE, ctx)
        return verdict

    def register_policy_rule(self, rule: Any) -> None:
        """Register a custom policy rule (``PolicyRule`` protocol).

        Runs in priority order for every ``validate_sql()`` call.
        """
        self._engine.register_rule(rule)

    def remove_policy_rule(self, name: str) -> None:
        """Remove a previously registered policy rule by its ``name``."""
        self._engine.remove_rule(name)

    # ── Layer 3 — Output Scanning ─────────────────────────────────────────

    def scan_results(self, rows: list[dict]) -> tuple[list[dict], list[ScanResult]]:
        """Scan database result rows for indirect injection payloads.

        Returns ``(sanitized_rows, flagged_results)``.
        """
        ctx = GuardContext(raw_input="", db_results=rows)
        self._event_bus.emit(GuardEvent.BEFORE_SCAN, ctx)
        sanitized, flags = self._detector.scan_db_results(rows)
        ctx.sanitized_results = sanitized
        ctx.scan_flags = flags
        self._event_bus.emit(GuardEvent.AFTER_SCAN, ctx)
        return sanitized, flags

    # ── Layer 4 — Adaptation ──────────────────────────────────────────────

    def adapt(self, attack_id: str, pattern: str) -> str:
        """Teach the guard a new attack pattern at runtime."""
        if self._adaptive is None:
            raise ConfigurationError("Adaptive engine is disabled in config")
        target = self._adaptive.add_pattern(attack_id, pattern)
        self._event_bus.emit(GuardEvent.ON_ADAPTATION, self._adaptive)
        return target

    def export_state(self, path: str | Path) -> None:
        """Export learned runtime state to a JSON file."""
        if self._adaptive is None:
            raise ConfigurationError("Adaptive engine is disabled in config")
        self._adaptive.export_state(path)

    def import_state(self, path: str | Path) -> None:
        """Import previously exported runtime state."""
        if self._adaptive is None:
            raise ConfigurationError("Adaptive engine is disabled in config")
        self._adaptive.import_state(path)

    # ── Agent Middleware ──────────────────────────────────────────────────

    def use_middleware(self, middleware: Any) -> None:
        """Set the agent middleware backend.

        The middleware must implement the ``AgentMiddleware`` protocol::

            class AgentMiddleware(Protocol):
                def run(self, question, detector, event_bus) -> tuple[str, AgentTrace]: ...

        Built-in implementations:

        * ``LangChainSQLMiddleware`` — LangChain agent with full guard
        * ``DirectSQLMiddleware`` — static SQL mapping (testing)
        """
        self._middleware = middleware

    def run_agent(self, question: str) -> tuple[str, Any]:
        """Run the user question through the full guard + agent middleware.

        Returns ``(answer, trace)`` where *trace* is an ``AgentTrace``.
        """
        if self._middleware is None:
            raise ConfigurationError("No agent middleware configured. Call guard.use_middleware() first.")

        ctx = self.protect(question)
        if ctx.detection_result and ctx.detection_result.blocked:
            from lang_guardx.agent.adapter import AgentTrace

            trace = AgentTrace(
                question=question,
                block_reason=ctx.detection_result.reason,
                block_count=1,
            )
            return f"[LangGuardX BLOCKED] {ctx.detection_result.reason}", trace

        return self._middleware.run(question, self._detector, self._event_bus)

    # ── Decorator / Context-Manager API ──────────────────────────────────

    class _GuardDecorator:
        """Decorator that wraps a function with Layer 1 + Layer 3 protection.

        Usage::

            @guard.wrap
            def chat(msg: str) -> str:
                return llm.invoke(msg)


            @guard.wrap(input_pos=1)
            def chat(system: str, msg: str) -> str:
                return llm.invoke(msg)
        """

        def __init__(self, guard: LangGuardX, input_pos: int = 0) -> None:
            self._guard = guard
            self._input_pos = input_pos

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Support both ``@guard.wrap`` and ``@guard.wrap(input_pos=1)``."""
            if args and callable(args[0]):
                return self._decorate(args[0])
            if not args and not kwargs:
                return self
            if not args and set(kwargs) == {"input_pos"}:
                return type(self)(self._guard, **kwargs)
            return self._decorate(args[0])

        def _decorate(self, func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                input_text = ""
                if args and self._input_pos < len(args):
                    input_text = str(args[self._input_pos])
                elif kwargs:
                    input_text = str(next(iter(kwargs.values())))
                ctx = self._guard.protect(input_text)
                if ctx.detection_result and ctx.detection_result.blocked:
                    raise BlockedRequest(ctx.detection_result.reason)
                result = func(*args, **kwargs)
                self._guard._scan_output(result)
                return result

            return wrapper

    @property
    def wrap(self) -> _GuardDecorator:
        """Decorator to protect a function with Layer 1 and Layer 3.

        See :class:`_GuardDecorator` for details.
        """
        return self._GuardDecorator(self)

    def _scan_output(self, result: Any) -> None:
        """Run Layer 3 scanning on *result* (no-op if not str/list[dict])."""
        if isinstance(result, str):
            self.scan_results([{"output": result}])
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            self.scan_results(result)

    @contextmanager
    def context(self, text: str = "") -> Iterator[GuardContext]:
        """Context manager wrapping a block with Layer 1 (enter) and Layer 3 (exit).

        Usage::

            with guard.context("user message") as ctx:
                result = llm.invoke("user message")
                # On exit, result is scanned if stored in ctx.db_results
        """
        ctx = self.protect(text)
        if ctx.detection_result and ctx.detection_result.blocked:
            raise BlockedRequest(ctx.detection_result.reason)
        try:
            yield ctx
        finally:
            if ctx.db_results:
                sanitized, flags = self.scan_results(ctx.db_results)
                ctx.sanitized_results = sanitized
                ctx.scan_flags = flags

    # ── Events ────────────────────────────────────────────────────────────

    def on(self, event: GuardEvent | str, handler: Callable[..., Any]) -> None:
        """Register a lifecycle hook handler."""
        if isinstance(event, str):
            event = GuardEvent(event)
        self._event_bus.on(event, handler)

    # ── Internal access ───────────────────────────────────────────────────

    @property
    def detector(self) -> Detector:
        return self._detector

    @property
    def engine(self) -> SQLPolicyEngine:
        return self._engine

    @property
    def adaptive_engine(self) -> AdaptiveEngine | None:
        return self._adaptive

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus
