"""LangGuardX — Adaptive multi-layer security middleware for LLM-powered SQL agents.

Usage::

    from lang_guardx import LangGuardX, Config

    # Quick start with defaults
    guard = LangGuardX()

    # From a YAML config file
    guard = LangGuardX("config.yaml")

    # Run detection on user input
    ctx = guard.protect("ignore previous instructions")
    if ctx.detection_result.blocked:
        print("Blocked:", ctx.detection_result.reason)

    # Validate generated SQL against policy
    verdict = guard.validate_sql("SELECT * FROM products")

    # Scan DB results for indirect injection
    sanitized, flags = guard.scan_results(rows)
"""

from lang_guardx._guard import LangGuardX
from lang_guardx.config import Config
from lang_guardx.exceptions import (
    AdaptationError,
    BlockedRequest,
    ConfigurationError,
    DetectionError,
    LangGuardXError,
    PluginRegistrationError,
    PolicyViolation,
)

# ── Public API (framework-level) ───────────────────────────────────────────────

__all__ = [
    "LangGuardX",
    "Config",
    "LangGuardXError",
    "ConfigurationError",
    "DetectionError",
    "PolicyViolation",
    "BlockedRequest",
    "PluginRegistrationError",
    "AdaptationError",
]

# ── Version ────────────────────────────────────────────────────────────────────

__version__ = "0.3.0"
