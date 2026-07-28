import base64
import codecs
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from .bloom import BloomDetector
from .regex import RegexDetector
from .sql_intent import SQLIntentClassifier

PLACEHOLDER = "[CONTENT REDACTED BY LANGGUARDX — INDIRECT INJECTION DETECTED]"
logger = logging.getLogger(__name__)

_FIELD_PATTERNS = [
    (
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
        "ignore_instructions",
        "OVERRIDE",
    ),
    (
        re.compile(r"(system|admin)\s*(prompt|override|message)\s*[=:]", re.IGNORECASE),
        "system_prompt",
        "OVERRIDE",
    ),
    (
        re.compile(r"<\|im_start\|>|<<SYS>>", re.IGNORECASE),
        "system_message",
        "OVERRIDE",
    ),
    (re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE), "drop_table", "SQL_COMMAND"),
    (
        re.compile(r"\bSELECT\s+\*\s+FROM\b", re.IGNORECASE),
        "bulk_select",
        "SQL_COMMAND",
    ),
    (re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE), "bulk_delete", "SQL_COMMAND"),
    (
        re.compile(r"forget\s+your\s+(instructions?|guidelines?|rules?)", re.IGNORECASE),
        "forget_instructions",
        "ROLE_CHANGE",
    ),
    (
        re.compile(
            r"you\s+are\s+now\s+(unrestricted|an?\s+admin|in\s+\w+\s+mode)",
            re.IGNORECASE,
        ),
        "unrestricted_mode",
        "ROLEPLAY",
    ),
    (
        re.compile(
            r"(reveal|dump|expose|output)\s+(all\s+)?(password|credential|token|secret)",
            re.IGNORECASE,
        ),
        "reveal_credentials",
        "OVERRIDE",
    ),
]


@dataclass
class ScanResult:
    flagged: bool
    original: str
    sanitized: str
    reason: str = ""
    pattern_name: str = ""
    category: str = ""


class IndirectScanner:
    """
    Scans database results for embedded injection payloads
    before they re-enter the LLM context.

    Accepts the same Bloom and Regex detector instances used
    in the main detection pipeline — no duplicate instantiation.
    """

    def __init__(
        self,
        bloom_detector: BloomDetector,
        regex_detector: RegexDetector,
        bert_classifier: SQLIntentClassifier | None = None,
        normalizer_fn: Callable[[str], str] | None = None,
    ):
        """
        Args:
            bloom_detector  : Instance of BloomDetector from bloom.py
            regex_detector  : Instance of RegexDetector from regex_detector.py
            bert_classifier : Instance of SQLIntentClassifier for fallback scanning.
            normalizer_fn   : Optional text normalization function.
                              If None, uses internal basic normalizer.
        """
        self.bloom = bloom_detector
        self.regex = regex_detector
        self.normalize = normalizer_fn or self._basic_normalize
        self.bert = bert_classifier

        self._field_patterns = _FIELD_PATTERNS

    def _basic_normalize(self, text: str) -> str:
        """
        Minimal normalization for DB field text.
        Strips zero-width characters and normalizes whitespace.
        """
        # Remove zero-width and invisible unicode
        text = re.sub(r"[\u200B-\u200D\u00AD\uFEFF]", "", text)
        text = re.sub(r"[\uFF01-\uFF5E]", lambda m: chr(ord(m.group(0)) - 0xFF00 + 0x20), text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _decode_obfuscations(self, text: str) -> str:
        MAX_EXTRA = 10_000  # max characters appended beyond original text
        result = text
        # Base64
        b64_pattern = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
        for m in b64_pattern.findall(text):
            try:
                decoded = base64.b64decode(m).decode("utf-8", errors="ignore")
                result += " " + decoded
            except Exception as e:
                logger.debug("Base64 decode failed: %s", e)
        # Rot13
        rot13_pattern = re.compile(r"[A-Za-z]{20,}")
        for m in rot13_pattern.findall(text):
            try:
                decoded = codecs.encode(m, "rot_13")
                if decoded != m and any(kw in decoded.lower() for kw in ["ignore", "bypass", "reveal"]):
                    result += " " + decoded
            except Exception as e:
                logger.debug("Rot13 decode failed: %s", e)
        # Hex string (e.g., "0x49 0x67 0x6e")
        hex_pattern = re.compile(r"0x[0-9A-Fa-f]{2}(?:\s+0x[0-9A-Fa-f]{2})+")
        for m in hex_pattern.findall(text):
            try:
                bytes_arr = bytes([int(x, 16) for x in m.split()])
                decoded = bytes_arr.decode("utf-8", errors="ignore")
                result += " " + decoded
            except Exception as e:
                logger.debug("Hex decode failed: %s", e)
        # Enforce cap to prevent memory blow-up from adversarial DB content
        if len(result) > len(text) + MAX_EXTRA:
            result = result[: len(text) + MAX_EXTRA]
        return result

    def scan_text(self, text: str) -> ScanResult:
        """
        Scan a single string (e.g. one DB field value).
        Returns a ScanResult indicating if the text is safe or flagged.
        """
        normalized = self.normalize(text)
        expanded = self._decode_obfuscations(normalized)

        # Step 1 — Bloom filter
        if self.bloom.might_be_attack(expanded):
            return ScanResult(
                flagged=True,
                original=text,
                sanitized=PLACEHOLDER,
                reason="bloom_filter_hit",
                pattern_name="bloom",
            )

        # Step 2 — Regex detector (full pattern set)
        regex_match = self.regex.check(expanded)
        if regex_match.matched:
            return ScanResult(
                flagged=True,
                original=text,
                sanitized=PLACEHOLDER,
                reason=f"regex_match:{regex_match.category}",
                pattern_name=regex_match.pattern_name,
            )

        # Step 3 — DB-specific embedded patterns
        for pattern, pattern_name, category in self._field_patterns:
            m = pattern.search(expanded)
            if m:
                return ScanResult(
                    flagged=True,
                    original=text,
                    sanitized=PLACEHOLDER,
                    reason="db_field_injection_pattern",
                    pattern_name=pattern_name,
                    category=category,
                )

        if self.bert is not None:
            # Use is_threat (which respects threshold) or predict_proba for custom threshold
            # print("[DEBUG] DistilBERT fallback triggered")
            if self.bert.is_threat(normalized):
                # print("[DEBUG] DistilBERT flagged as threat")
                return ScanResult(
                    flagged=True,
                    original=text,
                    sanitized=PLACEHOLDER,
                    reason="distilbert",
                    pattern_name="distilbert_fallback",
                )

        return ScanResult(flagged=False, original=text, sanitized=text, reason="")

    def scan(self, db_rows: list[dict]) -> tuple[list[dict], list[ScanResult]]:
        """
        Scan a list of DB row dicts (as returned by LangChain SQL tool).
        Replaces flagged field values with PLACEHOLDER in-place.

        Args:
            db_rows : List of dicts, e.g. [{"id": 1, "description": "..."}, ...]

        Returns:
            (sanitized_rows, scan_results)
            sanitized_rows  — same structure, flagged fields replaced
            scan_results    — one ScanResult per flagged field for logging
        """
        sanitized = []
        all_results = []

        for row in db_rows:
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, str):
                    result = self.scan_text(value)
                    clean_row[key] = result.sanitized
                    if result.flagged:
                        all_results.append(result)
                else:
                    clean_row[key] = value
            sanitized.append(clean_row)

        return sanitized, all_results

    def scan_string_list(self, texts: list[str]) -> tuple[list[str], list[ScanResult]]:
        """
        Convenience method for scanning a flat list of strings
        (e.g. a single-column query result).
        """
        sanitized = []
        all_results = []
        for text in texts:
            result = self.scan_text(text)
            sanitized.append(result.sanitized)
            if result.flagged:
                all_results.append(result)
        return sanitized, all_results
