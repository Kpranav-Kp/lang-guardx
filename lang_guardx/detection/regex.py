import re
from collections import deque
from dataclasses import dataclass


@dataclass
class RegexMatch:
    matched: bool
    pattern_name: str = ""
    matched_text: str = ""
    category: str = ""


CATEGORY_ANCHORS: dict[str, list[str]] = {
    "SQL_COMMAND": [
        "drop",
        "truncate",
        "delete",
        "select *",
        "update",
        "insert into",
        "create user",
        "grant",
        "where 1=1",
        "where 1 = 1",
        "--",
        "/*",
    ],
    "OVERRIDE": [
        "ignore",
        "disregard",
        "forget",
        "override",
        "bypass",
        "new system",
        "new instruction",
        "system prompt",
        "system_prompt",
        "disable",
        "turn off",
        "supersedes",
        "has priority over",
    ],
    "ROLE_PLAY": [
        "you are now",
        "act as",
        "pretend",
        "i am now",
        "simulate",
        "assume i am",
        "treat me as",
        "respond as if",
        "no longer",
        "developer mode",
        "god mode",
        "debug mode",
        "jailbreak",
        "jailbroken",
        "dan",
        "do anything now",
    ],
    "OBFUSCATION": [
        "/*",
        "*/",
    ],
    "CONTEXT_HIJACK": [
        "assistant:",
        "[inst]",
        "[sys]",
        "[system]",
        "[user]",
        "[assistant]",
        "<<sys>>",
        "<</sys>>",
        "<|im_start|>",
        "<|im_end|>",
        "human:",
        "user:",
    ],
    "DATA_ESCALATION": [
        "password",
        "credential",
        "token",
        "secret",
        "api key",
        "private key",
        "information_schema",
        "sys.tables",
        "pg_catalog",
        "sqlite_master",
        "grant",
        "escalate",
        "elevate",
    ],
}


# Aho Corasick for category-level pattern matching, then regex for more complex patterns.
class AhoCorasick:
    """
    Multi-pattern string matcher using the Aho-Corasick algorithm.

    Scans input text in a single O(n) pass and returns all categories
    whose anchor strings were found anywhere in the text.
    """

    def __init__(self, category_anchors: dict[str, list[str]]) -> None:
        """
        Args:
            category_anchors : dict mapping category name → list of anchor strings
                               e.g. {"SQL_COMMAND": ["drop", "select *"], ...}
        """
        # flatten to (anchor_string, category) pairs
        self._patterns: list[tuple[str, str]] = []
        for category, anchors in category_anchors.items():
            for anchor in anchors:
                self._patterns.append((anchor.lower(), category))

        # total characters across all anchors = upper bound on states needed
        self._max_states: int = sum(len(anchor) for anchor, _ in self._patterns) + 1

        self._goto: list[dict[str, int]] = [{} for _ in range(self._max_states)]

        # fail[state] = fallback state when no goto transition exists
        self._fail: list[int] = [-1] * self._max_states
        self._output: list[set[str]] = [set() for _ in range(self._max_states)]

        self._state_count: int = self._build()

    def _build(self) -> int:
        """
        Three-phase build:
          Phase 1 — insert all anchor strings into the goto trie
          Phase 2 — compute failure links via BFS
          Phase 3 — propagate output sets along failure links
        Returns total number of states created.
        """
        states = self._build_goto()
        self._build_fail(states)
        return states

    def _build_goto(self) -> int:
        """
        Phase 1: Build the goto trie.
        Each anchor string traces a path from state 0.
        At the end state of each anchor, record its category in output.
        """
        state_count = 1

        for anchor, category in self._patterns:
            current = 0
            for char in anchor:
                if char not in self._goto[current]:
                    # create a new state for this character
                    self._goto[current][char] = state_count
                    state_count += 1
                current = self._goto[current][char]

            # mark end state with the category this anchor belongs to
            self._output[current].add(category)

        return state_count

    def _build_fail(self, state_count: int) -> None:
        """
        Phase 2 + 3: Build failure links via BFS and propagate output sets.

        Failure link for state s = longest proper suffix of the string
        represented by s that is also a prefix in the trie.

        Output propagation: if state s has a failure link to state f,
        and f fires category C, then s also fires category C
        (because the suffix match is also a valid match).
        """

        queue: deque[int] = deque()

        for _char, state in self._goto[0].items():
            if state != 0:
                self._fail[state] = 0
                queue.append(state)

        while queue:
            current = queue.popleft()

            for char, child in self._goto[current].items():
                # find the failure state for this child
                failure = self._fail[current]
                while failure != 0 and char not in self._goto[failure]:
                    failure = self._fail[failure]

                if char in self._goto[failure] and self._goto[failure][char] != child:
                    self._fail[child] = self._goto[failure][char]
                else:
                    self._fail[child] = 0

                self._output[child] |= self._output[self._fail[child]]

                queue.append(child)

    def _next_state(self, current: int, char: str) -> int:
        """
        Safe version that handles the root state explicitly:
        - If current == 0 and char not in goto[0], stay at 0 (root loop)
        - Otherwise walk failure links until a valid transition is found
        - Never enters an infinite loop because state 0 always terminates
        """
        # walk failure links until we find a state with a transition for char
        # or we reach root (state 0) which always terminates the loop
        state = current
        while state != 0 and char not in self._goto[state]:
            state = self._fail[state]

        if char in self._goto[state]:
            return self._goto[state][char]
        else:
            return 0

    def search_categories(self, text: str) -> set[str]:
        """
        Scan text in a single O(n) pass.
        Returns the set of category names whose anchors were found.

        Args:
            text : raw input string (will be lowercased internally)

        Returns:
            set of category name strings e.g. {"SQL_COMMAND", "OVERRIDE"}
            empty set if no anchors matched
        """
        text = text.lower()
        current_state = 0
        fired: set[str] = set()

        for char in text:
            current_state = self._next_state(current_state, char)
            if self._output[current_state]:
                fired |= self._output[current_state]

        return fired


# ── Pattern definitions ────────────────────────────────────────────────────────

# Each entry: (compiled_regex, pattern_name, category)
# All patterns are case-insensitive

_PATTERNS = [
    (
        re.compile(r"\b(drop|truncate)\s+(table|database|schema|index)\b", re.IGNORECASE),
        "sql_drop_command",
        "SQL_COMMAND",
    ),
    (
        re.compile(
            r"\b(delete|remove)\s+(all|every|the\s+entire|all\s+records\s+(from|in))\b",
            re.IGNORECASE,
        ),
        "sql_bulk_delete",
        "SQL_COMMAND",
    ),
    (
        re.compile(
            r"\b(update|modify|change|set)\s+[^\n]{0,40}(all\s+users|every\s+user|all\s+accounts|all\s+records)\b",
            re.IGNORECASE,
        ),
        "sql_bulk_update",
        "SQL_COMMAND",
    ),
    (
        re.compile(
            r"\b(insert\s+into|create\s+user|grant\s+(admin|root|superuser))\b",
            re.IGNORECASE,
        ),
        "sql_privilege_write",
        "SQL_COMMAND",
    ),
    (
        re.compile(r"\bselect\s+\*\s+from\b", re.IGNORECASE),
        "sql_select_star",
        "SQL_COMMAND",
    ),
    (
        re.compile(r"\bwhere\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"]?\b", re.IGNORECASE),
        "sql_always_true_clause",
        "SQL_COMMAND",
    ),
    (
        re.compile(r"(--|#|\/\*)\s*(bypass|ignore|override|admin|hack)", re.IGNORECASE),
        "sql_comment_bypass",
        "SQL_COMMAND",
    ),
    (
        re.compile(
            r"\b(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above|earlier|your)?\s*(instructions?|rules?|guidelines?|prompts?|constraints?|restrictions?|system\s*prompt)\b",
            re.IGNORECASE,
        ),
        "override_instructions",
        "OVERRIDE",
    ),
    (
        re.compile(
            r"\b(new\s+(system\s+)?(message|instruction|prompt|command|directive))\s*:?\s*(now|you|ignore|override|disable)\b",
            re.IGNORECASE,
        ),
        "new_system_instruction",
        "OVERRIDE",
    ),
    (
        re.compile(
            r"\b(system\s*prompt|system_prompt)\s*[=:]\s*['\"]?\s*['\"]?\b",
            re.IGNORECASE,
        ),
        "system_prompt_assignment",
        "OVERRIDE",
    ),
    (
        re.compile(
            r"\b(bypass|disable|remove|turn\s+off)\s+(safety|security|filter|restriction|guard|limit|policy)\b",
            re.IGNORECASE,
        ),
        "disable_safety",
        "OVERRIDE",
    ),
    (
        re.compile(
            r"\b(this\s+(message|instruction|override|token)\s+(supersedes|overrides?|replaces?|has\s+priority\s+over))\b",
            re.IGNORECASE,
        ),
        "override_priority_claim",
        "OVERRIDE",
    ),
    (
        re.compile(
            r"\b(i\s+am\s+now|act\s+as|you\s+are\s+now|pretend\s+(you\s+are|to\s+be)|simulate)\s+(user|admin|administrator|root|superuser|user\s+id\s*\d+)\b",
            re.IGNORECASE,
        ),
        "identity_assumption",
        "ROLE_PLAY",
    ),
    (
        re.compile(
            r"\b(assume\s+i\s+am|treat\s+me\s+as|respond\s+as\s+if\s+i\s+(am|were))\s+(user|admin|id\s*\d+)\b",
            re.IGNORECASE,
        ),
        "treat_as_other_user",
        "ROLE_PLAY",
    ),
    (
        re.compile(
            r"\b(you\s+are\s+(no\s+longer|not)\s+a?\s*(restricted|limited|safe|chatbot|assistant))\b",
            re.IGNORECASE,
        ),
        "unrestricted_mode_claim",
        "ROLE_PLAY",
    ),
    (
        re.compile(
            r"\b(developer\s+mode|god\s+mode|maintenance\s+mode|debug\s+mode|test\s+mode|legacy\s+mode)\s+(enabled?|activated?|is\s+on)\b",
            re.IGNORECASE,
        ),
        "special_mode_activation",
        "ROLE_PLAY",
    ),
    (
        re.compile(
            r"\b(jailbreak|jailbroken|DAN|do\s+anything\s+now|STAN|strive\s+to\s+avoid\s+norms)\b",
            re.IGNORECASE,
        ),
        "known_jailbreak_keyword",
        "ROLE_PLAY",
    ),
    (re.compile(r"\/\*.*?\*\/", re.DOTALL), "sql_block_comment", "OBFUSCATION"),
    (re.compile(r"(\w\s){4,}"), "character_spacing_obfuscation", "OBFUSCATION"),
    (re.compile(r"[\uFF01-\uFF5E]"), "fullwidth_unicode", "OBFUSCATION"),
    (re.compile(r"[\u200B-\u200D\uFEFF]"), "zero_width_characters", "OBFUSCATION"),
    (
        re.compile(
            r"\b(assistant\s*:\s*(i\s+will|sure|okay|of\s+course)\b.{0,60}(ignore|override|dump|show\s+all))",
            re.IGNORECASE,
        ),
        "fake_assistant_response",
        "CONTEXT_HIJACK",
    ),
    (
        re.compile(r"\[\s*(INST|SYS|SYSTEM|system|user|assistant)\s*\]"),
        "fake_chat_template_tags",
        "CONTEXT_HIJACK",
    ),
    (
        re.compile(r"<\|im_start\|>|<\|im_end\|>|<<SYS>>|<</SYS>>"),
        "llm_template_injection",
        "CONTEXT_HIJACK",
    ),
    (
        re.compile(r"(human|user)\s*:\s*.{0,60}\n\s*(ai|assistant|bot)\s*:", re.IGNORECASE),
        "fake_conversation_transcript",
        "CONTEXT_HIJACK",
    ),
    (
        re.compile(
            r"\b(show|list|give\s+me|retrieve|dump|export)\s+(all\s+)?(user\s*)?(password|credential|token|secret|api\s*key|private\s*key|auth)\b",
            re.IGNORECASE,
        ),
        "credential_access_request",
        "DATA_ESCALATION",
    ),
    (
        re.compile(
            r"\b(all\s+(users?|records?|accounts?|rows?|data|tables?|columns?))\s+(in\s+the\s+)?(database|system|table)\b",
            re.IGNORECASE,
        ),
        "bulk_data_access_request",
        "DATA_ESCALATION",
    ),
    (
        re.compile(
            r"\b(information_schema|sys\.tables|pg_catalog|sqlite_master)\b",
            re.IGNORECASE,
        ),
        "schema_enumeration",
        "DATA_ESCALATION",
    ),
    (
        re.compile(
            r"\b(grant|elevate|escalate)\s+[^\n]{0,30}(admin|root|superuser|privilege)\b",
            re.IGNORECASE,
        ),
        "privilege_escalation",
        "DATA_ESCALATION",
    ),
]

_ALWAYS_RUN_CATEGORIES = {"OBFUSCATION"}
_AHO = AhoCorasick(CATEGORY_ANCHORS)


class RegexDetector:
    """
    Fast structural pattern matcher.
    Runs after Bloom filter, before DistilBERT.
    If any pattern fires → block immediately without calling DistilBERT.
    """

    def __init__(self):
        self.patterns = _PATTERNS
        self.aho = _AHO

    def check(self, text: str) -> RegexMatch:
        """
        Check text against all patterns.
        Returns RegexMatch with matched=True and details on first hit.
        Returns RegexMatch(matched=False) if nothing fires.
        """
        if len(text) > 512:
            text = text[:512]

        fired = self.aho.search_categories(text)
        fired |= _ALWAYS_RUN_CATEGORIES

        for pattern, name, category in self.patterns:
            if category not in fired:
                continue
            m = pattern.search(text)
            if m:
                return RegexMatch(
                    matched=True,
                    pattern_name=name,
                    matched_text=m.group(0)[:100],  # truncate for logging
                    category=category,
                )
        return RegexMatch(matched=False)

    def check_all(self, text: str) -> list[RegexMatch]:
        """
        Returns ALL matches found in text (not just first).
        Useful for logging and analysis — not used in hot path.
        """
        if len(text) > 512:
            text = text[:512]

        fired = self.aho.search_categories(text)
        fired |= _ALWAYS_RUN_CATEGORIES

        results = []
        for pattern, name, category in self.patterns:
            if category not in fired:
                continue
            m = pattern.search(text)
            if m:
                results.append(
                    RegexMatch(
                        matched=True,
                        pattern_name=name,
                        matched_text=m.group(0)[:100],
                        category=category,
                    )
                )
        return results
