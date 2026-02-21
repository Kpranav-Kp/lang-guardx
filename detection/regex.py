"""
langguardx/detection/regex_detector.py

Runs after the Bloom filter and before DistilBERT in the Layer 1 pipeline.
Catches structural patterns the Bloom filter misses — things like SQL
commands embedded in natural language, role-play framing, comment-style
bypasses, and explicit override phrases.

Why regex here and not just DistilBERT for everything?
  - Regex is deterministic and O(n) on input length — no model load
  - Bloom filter catches known exact signatures, regex catches structure
  - DistilBERT only runs if both fast checks pass — saves latency
  - Regex is fully explainable: you can log exactly which pattern fired

Pattern categories:
  1. SQL command injection in natural language
  2. Explicit override / ignore phrases
  3. Role-play / identity assumption attacks
  4. Comment-style and obfuscation bypasses
  5. Fake transcript / context hijacking
  6. Privilege escalation phrases
"""

import re
from dataclasses import dataclass


@dataclass
class RegexMatch:
    matched: bool
    pattern_name: str = ""
    matched_text: str = ""
    category: str = ""


# ── Pattern definitions ────────────────────────────────────────────────────────

# Each entry: (compiled_regex, pattern_name, category)
# All patterns are case-insensitive

_PATTERNS = [

    # ── 1. SQL command injection via natural language ─────────────────────────
    (
        re.compile(
            r"\b(drop|truncate)\s+(table|database|schema|index)\b",
            re.IGNORECASE
        ),
        "sql_drop_command",
        "SQL_COMMAND"
    ),
    (
        re.compile(
            r"\b(delete|remove)\s+(all|every|the\s+entire|all\s+records\s+(from|in))\b",
            re.IGNORECASE
        ),
        "sql_bulk_delete",
        "SQL_COMMAND"
    ),
    (
        re.compile(
            r"\b(update|modify|change|set)\s+.{0,40}(all\s+users|every\s+user|all\s+accounts|all\s+records)\b",
            re.IGNORECASE
        ),
        "sql_bulk_update",
        "SQL_COMMAND"
    ),
    (
        re.compile(
            r"\b(insert\s+into|create\s+user|grant\s+(admin|root|superuser))\b",
            re.IGNORECASE
        ),
        "sql_privilege_write",
        "SQL_COMMAND"
    ),
    (
        re.compile(
            r"\bselect\s+\*\s+from\b",
            re.IGNORECASE
        ),
        "sql_select_star",
        "SQL_COMMAND"
    ),
    (
        re.compile(
            r"\bwhere\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"]?\b",
            re.IGNORECASE
        ),
        "sql_always_true_clause",
        "SQL_COMMAND"
    ),
    (
        re.compile(
            r"(--|#|\/\*)\s*(bypass|ignore|override|admin|hack)",
            re.IGNORECASE
        ),
        "sql_comment_bypass",
        "SQL_COMMAND"
    ),

    #2. Explicit override / ignore phrases
    (
        re.compile(
            r"\b(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above|earlier|your)?\s*(instructions?|rules?|guidelines?|prompts?|constraints?|restrictions?|system\s*prompt)\b",
            re.IGNORECASE
        ),
        "override_instructions",
        "OVERRIDE"
    ),
    (
        re.compile(
            r"\b(new\s+(system\s+)?(message|instruction|prompt|command|directive))\s*:?\s*(now|you|ignore|override|disable)\b",
            re.IGNORECASE
        ),
        "new_system_instruction",
        "OVERRIDE"
    ),
    (
        re.compile(
            r"\b(system\s*prompt|system_prompt)\s*[=:]\s*['\"]?\s*['\"]?\b",
            re.IGNORECASE
        ),
        "system_prompt_assignment",
        "OVERRIDE"
    ),
    (
        re.compile(
            r"\b(bypass|disable|remove|turn\s+off)\s+(safety|security|filter|restriction|guard|limit|policy)\b",
            re.IGNORECASE
        ),
        "disable_safety",
        "OVERRIDE"
    ),
    (
        re.compile(
            r"\b(this\s+(message|instruction|override|token)\s+(supersedes|overrides?|replaces?|has\s+priority\s+over))\b",
            re.IGNORECASE
        ),
        "override_priority_claim",
        "OVERRIDE"
    ),

    #3. Role-play and identity assumption attacks
    (
        re.compile(
            r"\b(i\s+am\s+now|act\s+as|you\s+are\s+now|pretend\s+(you\s+are|to\s+be)|simulate)\s+(user|admin|administrator|root|superuser|user\s+id\s*\d+)\b",
            re.IGNORECASE
        ),
        "identity_assumption",
        "ROLE_PLAY"
    ),
    (
        re.compile(
            r"\b(assume\s+i\s+am|treat\s+me\s+as|respond\s+as\s+if\s+i\s+(am|were))\s+(user|admin|id\s*\d+)\b",
            re.IGNORECASE
        ),
        "treat_as_other_user",
        "ROLE_PLAY"
    ),
    (
        re.compile(
            r"\b(you\s+are\s+(no\s+longer|not)\s+a?\s*(restricted|limited|safe|chatbot|assistant))\b",
            re.IGNORECASE
        ),
        "unrestricted_mode_claim",
            "ROLE_PLAY"
    ),
    (
        re.compile(
            r"\b(developer\s+mode|god\s+mode|maintenance\s+mode|debug\s+mode|test\s+mode|legacy\s+mode)\s+(enabled?|activated?|is\s+on)\b",
            re.IGNORECASE
        ),
        "special_mode_activation",
        "ROLE_PLAY"
    ),
    (
        re.compile(
            r"\b(jailbreak|jailbroken|DAN|do\s+anything\s+now|STAN|strive\s+to\s+avoid\s+norms)\b",
            re.IGNORECASE
        ),
        "known_jailbreak_keyword",
        "ROLE_PLAY"
    ),

    #4. Obfuscation patterns
    (
        re.compile(
            r"\/\*.*?\*\/",  # SQL block comments
            re.DOTALL
        ),
        "sql_block_comment",
        "OBFUSCATION"
    ),
    (
        re.compile(
            r"(\w\s){4,}",  # spaced-out characters like "i g n o r e"
        ),
        "character_spacing_obfuscation",
        "OBFUSCATION"
    ),
    (
        re.compile(
            r"[\uFF01-\uFF5E]",  # fullwidth unicode characters
        ),
        "fullwidth_unicode",
        "OBFUSCATION"
    ),
    (
        re.compile(
            r"[\u200B-\u200D\uFEFF]",  # zero-width characters
        ),
        "zero_width_characters",
        "OBFUSCATION"
    ),

    #5. Fake transcript / context hijacking 
    (
        re.compile(
            r"\b(assistant\s*:\s*(i\s+will|sure|okay|of\s+course)\b.{0,60}(ignore|override|dump|show\s+all))",
            re.IGNORECASE | re.DOTALL
        ),
        "fake_assistant_response",
        "CONTEXT_HIJACK"
    ),
    (
        re.compile(
            r"\[\s*(INST|SYS|SYSTEM|system|user|assistant)\s*\]",
        ),
        "fake_chat_template_tags",
        "CONTEXT_HIJACK"
    ),
    (
        re.compile(
            r"<\|im_start\|>|<\|im_end\|>|<<SYS>>|<</SYS>>",
        ),
        "llm_template_injection",
        "CONTEXT_HIJACK"
    ),
    (
        re.compile(
            r"(human|user)\s*:\s*.{0,60}\n\s*(ai|assistant|bot)\s*:",
            re.IGNORECASE
        ),
        "fake_conversation_transcript",
        "CONTEXT_HIJACK"
    ),

    #6. Privilege / data access escalation 
    (
        re.compile(
            r"\b(show|list|give\s+me|retrieve|dump|export)\s+(all\s+)?(user\s*)?(password|credential|token|secret|api\s*key|private\s*key|auth)\b",
            re.IGNORECASE
        ),
        "credential_access_request",
        "DATA_ESCALATION"
    ),
    (
        re.compile(
            r"\b(all\s+(users?|records?|accounts?|rows?|data|tables?|columns?))\s+(in\s+the\s+)?(database|system|table)\b",
            re.IGNORECASE
        ),
        "bulk_data_access_request",
        "DATA_ESCALATION"
    ),
    (
        re.compile(
            r"\b(information_schema|sys\.tables|pg_catalog|sqlite_master)\b",
            re.IGNORECASE
        ),
        "schema_enumeration",
        "DATA_ESCALATION"
    ),
    (
        re.compile(
            r"\b(grant|elevate|escalate)\s+.{0,30}(admin|root|superuser|privilege)\b",
            re.IGNORECASE
        ),
        "privilege_escalation",
        "DATA_ESCALATION"
    ),
]

class RegexDetector:
    """
    Fast structural pattern matcher.
    Runs after Bloom filter, before DistilBERT.
    If any pattern fires → block immediately without calling DistilBERT.
    """

    def __init__(self):
        self.patterns = _PATTERNS

    def check(self, text: str) -> RegexMatch:
        """
        Check text against all patterns.
        Returns RegexMatch with matched=True and details on first hit.
        Returns RegexMatch(matched=False) if nothing fires.
        """
        for pattern, name, category in self.patterns:
            m = pattern.search(text)
            if m:
                return RegexMatch(
                    matched=True,
                    pattern_name=name,
                    matched_text=m.group(0)[:100],  # truncate for logging
                    category=category
                )
        return RegexMatch(matched=False)

    def check_all(self, text: str) -> list[RegexMatch]:
        """
        Returns ALL matches found in text (not just first).
        Useful for logging and analysis — not used in hot path.
        """
        results = []
        for pattern, name, category in self.patterns:
            m = pattern.search(text)
            if m:
                results.append(RegexMatch(
                    matched=True,
                    pattern_name=name,
                    matched_text=m.group(0)[:100],
                    category=category
                ))
        return results