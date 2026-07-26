from __future__ import annotations

from collections.abc import Callable
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
from lang_guardx.exceptions import ConfigurationError


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
        from lang_guardx import Config

        guard = LangGuardX(Config(engine={"policy": {"permitted_tables": ["products"]}}))

        # Run detection on user input
        ctx = guard.protect("ignore previous instructions")
        if ctx.detection_result.blocked:
            print("Blocked by:", ctx.detection_result.reason)

        # Validate generated SQL
        verdict = guard.validate_sql("SELECT * FROM products")
        if verdict.verdict.name == "BLOCKED":
            print("SQL blocked:", verdict.violations)

        # Scan DB results for indirect injection
        sanitized, flags = guard.scan_results([{"review": "..."}])

        # Register custom lifecycle hooks
        guard.on(GuardEvent.ON_BLOCK, lambda ctx: log_alert(ctx))
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
            ontology = ThreatOntology(
                yaml_path=self._config.adaptive.taxonomy_path,
            )
            self._adaptive = AdaptiveEngine(
                bloom=self._detector.bloom,
                ontology=ontology,
                log_path=self._config.adaptive.log_path,
            )
            self._detector.set_adaptive_bloom(self._adaptive.get_adaptive_bloom())

    # ── Config ────────────────────────────────────────────────────────────

    @property
    def config(self) -> Config:
        """The resolved configuration."""
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
        """Run the full input detection pipeline on *text*.

        Returns a :class:`GuardContext` with ``detection_result`` populated.
        Does **not** raise — check ``ctx.detection_result.blocked`` instead.
        """
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
        """Register a custom detection layer (must implement ``DetectionLayer`` protocol).

        The layer will be called for every ``protect()`` invocation in
        priority order (lower number = runs first).
        """
        self._detector.register_layer(layer)

    def remove_detector(self, name: str) -> None:
        """Remove a previously registered detection layer by its ``name``."""
        self._detector.remove_layer(name)

    # ── Layer 2 — SQL Policy ──────────────────────────────────────────────

    def validate_sql(self, sql: str) -> PolicyVerdict:
        """Validate a generated SQL query against the configured policy.

        Returns a :class:`PolicyVerdict` with verdict PASSED, REWRITTEN, or BLOCKED.
        """
        ctx = GuardContext(raw_input=sql)
        self._event_bus.emit(GuardEvent.BEFORE_POLICY, ctx)
        verdict = self._engine.validate(sql)
        ctx.policy_verdict = verdict
        ctx.generated_sql = sql
        self._event_bus.emit(GuardEvent.AFTER_POLICY, ctx)
        if verdict.verdict.name == "REWRITTEN":
            self._event_bus.emit(GuardEvent.ON_REWRITE, ctx)
        return verdict

    # ── Layer 3 — Output Scanning ─────────────────────────────────────────

    def scan_results(self, rows: list[dict]) -> tuple[list[dict], list[ScanResult]]:
        """Scan database result rows for indirect injection payloads.

        Returns ``(sanitized_rows, flagged_results)``.
        Flagged content is replaced with a redaction placeholder.
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
        """Teach the guard a new attack pattern at runtime.

        The pattern is classified via the P2SQL threat ontology and
        routed to the appropriate detector (e.g. Bloom filter, regex).

        Returns the update target that received the pattern (e.g. ``"bloom_corpus"``).
        """
        if self._adaptive is None:
            raise ConfigurationError("Adaptive engine is disabled in config")
        target = self._adaptive.add_pattern(attack_id, pattern)
        self._event_bus.emit(GuardEvent.ON_ADAPTATION, self._adaptive)
        return target

    def export_state(self, path: str | Path) -> None:
        """Export learned runtime state (adaptive patterns, etc.) to a JSON file."""
        if self._adaptive is None:
            raise ConfigurationError("Adaptive engine is disabled in config")
        self._adaptive.export_state(path)

    def import_state(self, path: str | Path) -> None:
        """Import previously exported runtime state."""
        if self._adaptive is None:
            raise ConfigurationError("Adaptive engine is disabled in config")
        self._adaptive.import_state(path)

    # ── Events ────────────────────────────────────────────────────────────

    def on(self, event: GuardEvent | str, handler: Callable[..., Any]) -> None:
        """Register a lifecycle hook handler.

        Example::

            guard.on("on_block", lambda ctx: send_alert(ctx))
            guard.on(GuardEvent.ON_UNCERTAIN, log_for_review)
        """
        if isinstance(event, str):
            event = GuardEvent(event)
        self._event_bus.on(event, handler)

    # ── Internal access (for advanced use / testing) ──────────────────────

    @property
    def detector(self) -> Detector:
        """The internal Layer 1 detector (advanced use)."""
        return self._detector

    @property
    def engine(self) -> SQLPolicyEngine:
        """The internal SQL policy engine (advanced use)."""
        return self._engine

    @property
    def adaptive_engine(self) -> AdaptiveEngine | None:
        """The internal adaptive engine (advanced use)."""
        return self._adaptive

    @property
    def event_bus(self) -> EventBus:
        """The internal event bus (advanced use)."""
        return self._event_bus
