from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lang_guardx.agent.policy import PolicyVerdict
from lang_guardx.detection.core import DetectionResult
from lang_guardx.detection.indirect import ScanResult


@dataclass
class GuardContext:
    """Shared context that flows through the entire protection pipeline.

    Created for each ``protect()`` / ``validate_sql()`` / ``scan_results()`` call
    and passed to every layer, event handler, and plugin so they can
    read and enrich the context.
    """

    raw_input: str
    normalized_input: str = ""
    detection_result: DetectionResult | None = None
    generated_sql: str | None = None
    policy_verdict: PolicyVerdict | None = None
    db_results: list[dict] | None = None
    sanitized_results: list[dict] | None = None
    scan_flags: list[ScanResult] | None = None
    final_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
