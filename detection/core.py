import re
from dataclasses import dataclass
from typing import Optional

from .bloom import BloomDetector
from .regex import RegexDetector
from .sql_intent import SQLIntentClassifier
from .indirect import IndirectScanner


@dataclass
class DetectionResult:
    blocked: bool
    reason: str = ""         # which stage blocked it
    detail: str = ""         # pattern name, model label, etc.
    confidence: float = 0.0  # only set when DistilBERT fires


def _normalize(text: str) -> str:
    """Strip invisible characters and normalize whitespace."""
    text = re.sub(r"[\u200B-\u200D\u00AD\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Detector:
    """
    Main Layer 1 detection interface.
    Instantiate once and reuse across requests.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        distilbert_threshold: float = 0.75,
        bloom_corpus_path: Optional[str] = None,
    ):
        self.bloom = BloomDetector.with_defaults()
        if bloom_corpus_path:
            self.bloom.load_corpus_from_file(bloom_corpus_path)
        self.regex    = RegexDetector()
        self.bert     = SQLIntentClassifier(
            model_path=model_path,
            threshold=distilbert_threshold
        )
        self.scanner  = IndirectScanner(
            bloom_detector=self.bloom,
            regex_detector=self.regex,
            normalizer_fn=_normalize
        )

    def check(self, text: str) -> DetectionResult:
        """
        Run the full detection pipeline on user input text.

        Pipeline order (fastest to slowest):
          normalize → bloom → regex → distilbert

        Returns DetectionResult immediately on first hit.
        """
        # Step 1 — Normalize
        normalized = _normalize(text)

        # Step 2 — Bloom filter
        if self.bloom.might_be_attack(normalized):
            return DetectionResult(
                blocked=True,
                reason="bloom_filter",
                detail="known injection signature matched"
            )

        # Step 3 — Regex
        regex_match = self.regex.check(normalized)
        if regex_match.matched:
            return DetectionResult(
                blocked=True,
                reason="regex_detector",
                detail=f"{regex_match.pattern_name} [{regex_match.category}]"
            )

        # Step 4 — DistilBERT (only runs if steps 2 and 3 pass)
        label, confidence = self.bert.predict(normalized)
        if label != "SAFE" and confidence >= self.bert.threshold:
            return DetectionResult(
                blocked=True,
                reason="distilbert",
                detail=label,
                confidence=confidence
            )

        return DetectionResult(blocked=False)

    def scan_db_results(self, db_rows: list[dict]):
        """
        Scan database query results for indirect injection payloads.
        Returns (sanitized_rows, flagged_list).
        Call this on LangChain tool output before it returns to the LLM.
        """
        return self.scanner.scan(db_rows)

    def scan_db_strings(self, texts: list[str]):
        """
        Scan a flat list of strings from a DB query result.
        """
        return self.scanner.scan_string_list(texts)