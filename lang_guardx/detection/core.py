from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Protocol

from lang_guardx.config import DetectionConfig

from .bloom import BloomDetector
from .indirect import IndirectScanner, ScanResult
from .regex import RegexDetector
from .sql_intent import SQLIntentClassifier

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    blocked: bool
    reason: str = ""
    detail: str = ""
    confidence: float = 0.0


def _normalize(text: str) -> str:
    text = re.sub(r"[\u200B-\u200D\u00AD\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Detection Layer Protocol ───────────────────────────────────────────────────


class DetectionLayer(Protocol):
    """Pluggable detection layer.

    Implement this protocol to add custom detection logic to the
    pipeline.  Layers are run in ``priority`` order (lowest first).
    If ``detect()`` returns a ``DetectionResult`` with ``blocked=True``
    the pipeline short-circuits immediately.
    """

    name: str
    priority: int

    def detect(self, text: str) -> DetectionResult | None:
        """Run detection on *text*.

        Return ``None`` to let the next layer decide.
        Return a ``DetectionResult`` with ``blocked=True`` to short-circuit.
        Return a ``DetectionResult`` with ``blocked=False`` to signal
        an informational result (e.g. UNCERTAIN) — the pipeline
        stores it and continues to the next layer.
        """


# ── Built-in Layers ───────────────────────────────────────────────────────────


class _BloomLayer:
    """Layer 1a — fast Bloom-filter pre-filter."""

    name = "bloom"
    priority = 10

    def __init__(self, bloom: BloomDetector) -> None:
        self._bloom = bloom

    def detect(self, text: str) -> DetectionResult | None:
        if self._bloom.might_be_attack(text):
            return DetectionResult(
                blocked=True,
                reason="bloom_filter",
                detail="known injection signature matched",
            )
        return None


class _AdaptiveBloomLayer:
    """Layer 1b — adaptive Bloom filter learned at runtime.

    Reads the detector's ``adaptive_bloom`` attribute so it stays in
    sync when ``Detector.set_adaptive_bloom()`` is called.
    """

    name = "adaptive_bloom"
    priority = 20

    def __init__(self, detector: Detector) -> None:
        self._detector = detector

    def detect(self, text: str) -> DetectionResult | None:
        bloom = self._detector.adaptive_bloom
        if bloom is not None and bloom.might_be_attack(text):
            return DetectionResult(
                blocked=True,
                reason="adaptive_bloom",
                detail="runtime-learned pattern matched",
            )
        return None


class _RegexLayer:
    """Layer 1c — regex / Aho-Corasick pattern matching."""

    name = "regex"
    priority = 30

    def __init__(self, regex: RegexDetector) -> None:
        self._regex = regex

    def detect(self, text: str) -> DetectionResult | None:
        match = self._regex.check(text)
        if match.matched:
            return DetectionResult(
                blocked=True,
                reason="regex_detector",
                detail=f"{match.pattern_name} [{match.category}]",
            )
        return None


class _BertLayer:
    """Layer 1d — DistilBERT neural classifier with BRM."""

    name = "distilbert"
    priority = 40

    def __init__(
        self,
        bert: SQLIntentClassifier | None,
        brm_cost_fp: float = 1.0,
        brm_cost_fn: float = 2.0,
        brm_uncertain_ratio: float = 0.2,
    ) -> None:
        self._bert = bert
        self._brm_cost_fp = brm_cost_fp
        self._brm_cost_fn = brm_cost_fn
        self._brm_uncertain_ratio = brm_uncertain_ratio
        self._consecutive_errors = 0
        self._disabled = False

    def detect(self, text: str) -> DetectionResult | None:
        if self._disabled or self._bert is None:
            return None
        try:
            label, conf = self._bert.predict(text)
            decision, _ = self._bert.decide(
                text,
                cost_fp=self._brm_cost_fp,
                cost_fn=self._brm_cost_fn,
                uncertain_ratio=self._brm_uncertain_ratio,
            )
            self._consecutive_errors = 0
            if decision == "BLOCK":
                return DetectionResult(blocked=True, reason="distilbert_brm", detail=label, confidence=conf)
            if decision == "UNCERTAIN":
                logger.info("UNCERTAIN: %s", text[:100])
                return DetectionResult(blocked=False, reason="uncertain", detail=label, confidence=conf)
            return None
        except Exception:
            self._consecutive_errors += 1
            if self._consecutive_errors >= 5:
                logger.error("Disabling BERT layer after 5 consecutive failures")
                self._disabled = True
            return None


# ── Detector Orchestrator ──────────────────────────────────────────────────────


class Detector:
    """Main Layer 1 detection interface.

    Instantiate once and reuse across requests.

    Two construction paths:

    **Legacy** (positional args, backward-compatible)::

        detector = Detector(model_path="...", distilbert_threshold=0.75)

    **Config-based** (recommended for new code)::

        detector = Detector(config=DetectionConfig(...))
    """

    def __init__(
        self,
        model_path: str | None = None,
        distilbert_threshold: float = 0.75,
        bloom_corpus_path: str | None = None,
        config: DetectionConfig | None = None,
    ) -> None:
        # ── Resolve configuration ────────────────────────────────────────
        if config is not None:
            cfg = config
        else:
            cfg = DetectionConfig()

        # ── Layer 1a — Bloom filter ──────────────────────────────────────
        self.bloom = BloomDetector.with_defaults()
        if cfg.bloom.corpus_path:
            self.bloom.load_corpus_from_file(cfg.bloom.corpus_path)
        if bloom_corpus_path:
            self.bloom.load_corpus_from_file(bloom_corpus_path)

        # ── Layer 1c — Regex ─────────────────────────────────────────────
        self.regex = RegexDetector()

        # ── Layer 1d — DistilBERT ────────────────────────────────────────
        resolved_model_path = model_path or cfg.distilbert.model_path
        resolved_threshold = distilbert_threshold if model_path is not None else cfg.distilbert.threshold
        self.bert: SQLIntentClassifier | None = None
        if cfg.distilbert.enabled:
            try:
                self.bert = SQLIntentClassifier(
                    model_path=resolved_model_path,
                    threshold=resolved_threshold,
                )
            except Exception:
                logger.warning("DistilBERT layer disabled due to init error. Install ml extras or check model path.")

        # ── Layer 3 — Indirect Scanner ───────────────────────────────────
        self.scanner = IndirectScanner(
            bloom_detector=self.bloom,
            regex_detector=self.regex,
            bert_classifier=self.bert,
            normalizer_fn=_normalize,
        )

        # ── Adaptive bloom (initially None; injected by AdaptiveEngine) ──
        self._adaptive_bloom: BloomDetector | None = None

        # ── Master switch ──────────────────────────────────────────────────
        self._enabled = cfg.enabled

        # ── Layer registry (thread-safe: copy-on-read) ──────────────────
        self._lock = threading.Lock()
        self._layers: list[DetectionLayer] = []
        self._register_default_layers(cfg)

    def _register_default_layers(self, cfg: DetectionConfig) -> None:
        """Register the built-in detection layers in priority order."""
        self._layers = [
            _BloomLayer(self.bloom),
            _AdaptiveBloomLayer(self),
            _RegexLayer(self.regex),
            _BertLayer(
                self.bert,
                brm_cost_fp=cfg.distilbert.brm_cost_fp,
                brm_cost_fn=cfg.distilbert.brm_cost_fn,
                brm_uncertain_ratio=cfg.distilbert.brm_uncertain_ratio,
            ),
        ]

    # ── Adaptive bloom ───────────────────────────────────────────────────

    @property
    def adaptive_bloom(self) -> BloomDetector | None:
        return self._adaptive_bloom

    def set_adaptive_bloom(self, adaptive_bloom: BloomDetector | None) -> None:
        """Inject the adaptive Bloom filter (from AdaptiveEngine).

        The ``_AdaptiveBloomLayer`` reads this attribute directly, so
        it takes effect on the next ``check()`` call.
        """
        self._adaptive_bloom = adaptive_bloom

    # ── Layer registration (plugin API) ──────────────────────────────────

    def register_layer(self, layer: DetectionLayer) -> None:
        """Register a custom detection layer.

        The layer will be called for every ``check()`` invocation in
        priority order.  See :class:`DetectionLayer` for the contract.
        """
        with self._lock:
            self._layers.append(layer)

    def remove_layer(self, name: str) -> None:
        """Remove a previously registered layer by its ``name``."""
        with self._lock:
            self._layers = [layer for layer in self._layers if layer.name != name]

    def list_layers(self) -> list[DetectionLayer]:
        """Return the list of registered detection layers (sorted by priority)."""
        return sorted(self._layers, key=lambda x: x.priority)

    # ── Main detection pipeline ──────────────────────────────────────────

    def check(self, text: str) -> DetectionResult:
        """Run the full detection pipeline on *text*.

        Pipeline order (by layer priority):
          normalize → bloom → adaptive_bloom → regex → distilbert → custom

        Returns ``DetectionResult`` immediately on the first blocked hit.
        If no layer blocks, returns the last non-blocking result
        (e.g. UNCERTAIN) or a clean pass.
        """
        if not self._enabled:
            return DetectionResult(blocked=False, reason="detection_disabled")

        normalized = _normalize(text)
        last_non_blocking: DetectionResult | None = None

        with self._lock:
            layers = list(self._layers)
        for layer in sorted(layers, key=lambda x: x.priority):
            result = layer.detect(normalized)
            if result is None:
                continue
            if result.blocked:
                return result
            last_non_blocking = result

        if last_non_blocking is not None:
            return last_non_blocking
        return DetectionResult(blocked=False)

    # ── Layer 3 passthrough ──────────────────────────────────────────────

    def scan_db_results(self, db_rows: list[dict]) -> tuple[list[dict], list[ScanResult]]:
        return self.scanner.scan(db_rows)

    def scan_db_strings(self, texts: list[str]) -> tuple[list[str], list[ScanResult]]:
        return self.scanner.scan_string_list(texts)
