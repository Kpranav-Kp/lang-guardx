from __future__ import annotations

import threading

from lang_guardx.config import EngineConfig

from .policy import PolicyVerdict, SQLPolicy
from .rules import DEFAULT_RULES, PolicyRule, RuleState


class SQLPolicyEngine:
    """SQL policy validation engine.

    Runs a configurable pipeline of :class:`PolicyRule` instances in
    priority order.  Built-in rules cover the full P2SQL attack taxonomy.

    Usage::

        engine = SQLPolicyEngine(policy)
        verdict = engine.validate("SELECT name FROM products")

        # Add a custom rule
        engine.register_rule(MyCustomRule())
    """

    def __init__(
        self,
        policy: SQLPolicy,
        current_user_id: int | None = None,
        dialect: str = "sqlite",
    ) -> None:
        self._lock = threading.Lock()
        self._policy = policy
        self._user_id = current_user_id
        self._dialect = dialect
        self._rules: list[PolicyRule] = list(DEFAULT_RULES)

    @classmethod
    def from_config(cls, config: EngineConfig) -> SQLPolicyEngine:
        """Create an engine from an :class:`EngineConfig`."""
        return cls(
            policy=SQLPolicy(
                permitted_operations=config.policy.permitted_operations,
                permitted_tables=config.policy.permitted_tables,
                restricted_columns=config.policy.restricted_columns,
                scoped_tables=config.policy.scoped_tables,
                require_user_scope=config.policy.require_user_scope,
                max_rows=config.policy.max_rows,
            ),
            current_user_id=config.current_user_id,
            dialect=config.dialect,
        )

    # ── Rule registry ────────────────────────────────────────────────────

    def register_rule(self, rule: PolicyRule) -> None:
        """Register a custom policy rule.

        The rule will run for every ``validate()`` call in priority order.
        """
        with self._lock:
            self._rules.append(rule)

    def remove_rule(self, name: str) -> None:
        """Remove a previously registered rule by its ``name``."""
        with self._lock:
            self._rules = [r for r in self._rules if r.name != name]

    def list_rules(self) -> list[PolicyRule]:
        """Return the list of registered policy rules (sorted by priority)."""
        with self._lock:
            return sorted(self._rules, key=lambda r: r.priority)

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self, sql: str) -> PolicyVerdict:
        """Run the full SQL policy validation pipeline on *sql*.

        Returns a :class:`PolicyVerdict` with verdict PASSED, REWRITTEN, or BLOCKED.
        """
        state = RuleState(
            original_sql=sql.strip(),
            user_id=self._user_id,
        )

        with self._lock:
            rules = list(self._rules)
        for rule in sorted(rules, key=lambda r: r.priority):
            rule.apply(state, self._policy, self._dialect)
            if state.blocked:
                return PolicyVerdict.blocked(
                    original_sql=state.original_sql,
                    reason="; ".join(state.violations),
                )

        if state.rewritten and state.tree is not None:
            safe_sql = state.tree.sql(dialect=self._dialect)
            return PolicyVerdict.rewritten(state.original_sql, safe_sql)

        return PolicyVerdict.passed(state.original_sql)
