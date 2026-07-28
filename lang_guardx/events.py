from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GuardEvent(Enum):
    """Lifecycle events emitted by the LangGuardX pipeline."""

    BEFORE_DETECTION = "before_detection"
    AFTER_DETECTION = "after_detection"
    BEFORE_POLICY = "before_policy"
    AFTER_POLICY = "after_policy"
    BEFORE_SCAN = "before_scan"
    AFTER_SCAN = "after_scan"
    ON_BLOCK = "on_block"
    ON_REWRITE = "on_rewrite"
    ON_UNCERTAIN = "on_uncertain"
    ON_ADAPTATION = "on_adaptation"
    ON_EXFILTRATION = "on_exfiltration"
    ON_ERROR = "on_error"


_handler = Callable[..., Any]


class EventBus:
    """Simple pub-sub event bus for lifecycle hooks.

    Usage::

        bus = EventBus()
        bus.on(GuardEvent.ON_BLOCK, lambda ctx: print(f"Blocked: {ctx.raw_input}"))
        bus.emit(GuardEvent.ON_BLOCK, ctx)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: dict[GuardEvent, list[_handler]] = {}

    def on(self, event: GuardEvent, handler: _handler) -> None:
        """Register a handler for an event."""
        with self._lock:
            self._handlers.setdefault(event, []).append(handler)

    def once(self, event: GuardEvent, handler: _handler) -> None:
        """Register a one-shot handler for an event."""

        def wrapper(*args: Any, **kwargs: Any) -> None:
            self.off(event, wrapper)
            handler(*args, **kwargs)

        self.on(event, wrapper)

    def off(self, event: GuardEvent, handler: _handler) -> None:
        """Remove a specific handler for an event."""
        with self._lock:
            handlers = self._handlers.get(event, [])
            if handler in handlers:
                handlers.remove(handler)

    def emit(self, event: GuardEvent, *args: Any, **kwargs: Any) -> None:
        """Emit an event, calling all registered handlers in order."""
        with self._lock:
            handlers = list(self._handlers.get(event, []))
        for handler in handlers:
            try:
                handler(*args, **kwargs)
            except Exception as exc:
                logger.warning("Handler %r failed for event %s: %s", handler, event.value, exc)

    def clear(self) -> None:
        """Remove all handlers."""
        with self._lock:
            self._handlers.clear()
