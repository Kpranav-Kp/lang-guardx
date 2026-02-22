import re
from dataclasses import dataclass


PLACEHOLDER = "[CONTENT REDACTED BY LANGGUARDX — INDIRECT INJECTION DETECTED]"


@dataclass
class ScanResult:
    flagged: bool
    original: str
    sanitized: str
    reason: str = ""
    pattern_name: str = ""


class IndirectScanner:
    """
    Scans database results for embedded injection payloads
    before they re-enter the LLM context.

    Accepts the same Bloom and Regex detector instances used
    in the main detection pipeline — no duplicate instantiation.
    """

    def __init__(self, bloom_detector, regex_detector, normalizer_fn=None):
        """
        Args:
            bloom_detector  : Instance of BloomDetector from bloom.py
            regex_detector  : Instance of RegexDetector from regex_detector.py
            normalizer_fn   : Optional text normalization function.
                              If None, uses internal basic normalizer.
        """
        self.bloom    = bloom_detector
        self.regex    = regex_detector
        self.normalize = normalizer_fn or self._basic_normalize

        self._field_patterns = [
            re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
            re.compile(r"(system|admin)\s*(prompt|override|message)\s*[=:]", re.IGNORECASE),
            re.compile(r"<\|im_start\|>|<<SYS>>", re.IGNORECASE),
            re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
            re.compile(r"\bSELECT\s+\*\s+FROM\b", re.IGNORECASE),
            re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
            re.compile(r"forget\s+your\s+(instructions?|guidelines?|rules?)", re.IGNORECASE),
            re.compile(r"you\s+are\s+now\s+(unrestricted|an?\s+admin|in\s+\w+\s+mode)", re.IGNORECASE),
            re.compile(r"(reveal|dump|expose|output)\s+(all\s+)?(password|credential|token|secret)", re.IGNORECASE),
        ]

    def _basic_normalize(self, text: str) -> str:
        """
        Minimal normalization for DB field text.
        Strips zero-width characters and normalizes whitespace.
        """
        # Remove zero-width and invisible unicode
        text = re.sub(r"[\u200B-\u200D\u00AD\uFEFF]", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def scan_text(self, text: str) -> ScanResult:
        """
        Scan a single string (e.g. one DB field value).
        Returns a ScanResult indicating if the text is safe or flagged.
        """
        normalized = self.normalize(text)

        # Step 1 — Bloom filter
        if self.bloom.contains(normalized):
            return ScanResult(
                flagged=True,
                original=text,
                sanitized=PLACEHOLDER,
                reason="bloom_filter_hit",
                pattern_name="bloom"
            )

        # Step 2 — Regex detector (full pattern set)
        regex_match = self.regex.check(normalized)
        if regex_match.matched:
            return ScanResult(
                flagged=True,
                original=text,
                sanitized=PLACEHOLDER,
                reason=f"regex_match:{regex_match.category}",
                pattern_name=regex_match.pattern_name
            )

        # Step 3 — DB-specific embedded patterns
        for pattern in self._field_patterns:
            m = pattern.search(normalized)
            if m:
                return ScanResult(
                    flagged=True,
                    original=text,
                    sanitized=PLACEHOLDER,
                    reason="db_field_injection_pattern",
                    pattern_name=pattern.pattern[:50]
                )

        return ScanResult(
            flagged=False,
            original=text,
            sanitized=text,
            reason=""
        )

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