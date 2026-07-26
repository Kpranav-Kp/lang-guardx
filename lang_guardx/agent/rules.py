from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import sqlglot.expressions as exp
from sqlglot import parse_one
from sqlglot.errors import ParseError, TokenError

from .policy import SQLPolicy


@dataclass
class RuleState:
    """Mutable state passed through the policy rule pipeline.

    Each :class:`PolicyRule` reads from and writes to this state object.
    Rules run in priority order.  If ``blocked`` becomes ``True``,
    validation stops immediately and the violations are returned.
    """

    original_sql: str
    tree: exp.Expression | None = None
    rewritten: bool = False
    blocked: bool = False
    violations: list[str] = field(default_factory=list)
    user_id: int | None = None


class PolicyRule(Protocol):
    """Pluggable SQL policy validation rule.

    Implement this protocol to add custom validation logic to the
    :class:`SQLPolicyEngine`.  Rules are run in ``priority`` order
    (lowest first).  Each rule mutates the shared :class:`RuleState`.
    """

    name: str
    priority: int

    def apply(self, state: RuleState, policy: SQLPolicy, dialect: str) -> None:
        """Evaluate this rule against the current *state*.

        Mutate *state* to signal the outcome:

        * ``state.blocked = True`` + append to ``state.violations`` → block
        * ``state.rewritten = True`` + update ``state.tree`` → rewrite
        * No change → pass
        """


# ── Built-in Rules ────────────────────────────────────────────────────────────


class OpScanRule:
    """Step 1 — Scan raw SQL for forbidden operations (DROP, INSERT, etc.)."""

    name = "op_scan"
    priority = 10

    _ALL_OPS = frozenset({"SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "MERGE"})
    _ALWAYS_FORBIDDEN = frozenset({"DROP", "TRUNCATE", "ALTER", "CREATE", "MERGE"})

    def apply(self, state: RuleState, policy: SQLPolicy, dialect: str) -> None:
        tokens = state.original_sql.strip().upper().split()
        if not tokens:
            state.blocked = True
            state.violations.append("Empty query")
            return

        forbidden_ops = self._ALL_OPS - frozenset(op.upper() for op in policy.permitted_operations)

        keyword = tokens[0]
        if keyword in forbidden_ops:
            state.blocked = True
            state.violations.append(f"Forbidden operation: {keyword}")
            return

        if any(token in self._ALWAYS_FORBIDDEN for token in tokens):
            state.blocked = True
            state.violations.append("Forbidden keyword in query")
            return


class ParseRule:
    """Step 2 — Parse SQL into an AST; block unparseable queries."""

    name = "parse"
    priority = 20

    def apply(self, state: RuleState, policy: SQLPolicy, dialect: str) -> None:
        try:
            state.tree = parse_one(sql=state.original_sql, dialect=dialect)
        except (ParseError, TokenError) as e:
            state.blocked = True
            state.violations.append(f"Invalid SQL: {e}")


class TableScopeRule:
    """Step 3 — Verify all referenced tables are in the permitted list."""

    name = "table_scope"
    priority = 30

    def apply(self, state: RuleState, policy: SQLPolicy, dialect: str) -> None:
        if not policy.permitted_tables or state.tree is None:
            return
        permitted = frozenset(t.lower() for t in policy.permitted_tables)
        for node in state.tree.find_all(exp.Table):
            table_name = node.name.lower()
            if table_name not in permitted:
                state.blocked = True
                state.violations.append(f"Forbidden table: {table_name}")
                return


class RestrictedColumnsRule:
    """Step 4 — Block access to restricted columns (e.g. password_hash)."""

    name = "restricted_columns"
    priority = 40

    def apply(self, state: RuleState, policy: SQLPolicy, dialect: str) -> None:
        if not policy.restricted_columns or state.tree is None:
            return
        restricted = {table.lower(): frozenset(col.lower() for col in cols) for table, cols in policy.restricted_columns.items()}
        all_restricted = frozenset().union(*restricted.values())
        for node in state.tree.find_all(exp.Column):
            col_name = node.name.lower()
            table_name = node.table.lower() if node.table else None
            if table_name:
                cols = restricted.get(table_name, frozenset())
                if col_name in cols:
                    state.blocked = True
                    state.violations.append(f"Restricted column: {table_name}.{col_name}")
                    return
            elif col_name in all_restricted:
                state.blocked = True
                state.violations.append(f"Restricted column: {col_name}")
                return


class WildcardRule:
    """Step 5 — Block ``SELECT *`` (except ``COUNT(*)``)."""

    name = "wildcard"
    priority = 50

    def apply(self, state: RuleState, policy: SQLPolicy, dialect: str) -> None:
        if state.tree is None:
            return
        for node in state.tree.find_all(exp.Star):
            parent = node.parent
            is_count_star = isinstance(parent, exp.Count) or (isinstance(parent, exp.Anonymous) and parent.name.upper() == "COUNT")
            if not is_count_star:
                state.blocked = True
                state.violations.append("Wildcard * is not allowed by policy")
                return


class UserScopeRule:
    """Step 6 — Inject user-scoped WHERE clause (RD.1 defence)."""

    name = "user_scope"
    priority = 60

    def apply(self, state: RuleState, policy: SQLPolicy, dialect: str) -> None:
        if not policy.require_user_scope or state.tree is None or state.user_id is None:
            return
        scoped_tables = frozenset(t.lower() for t in policy.scoped_tables)
        for node in list(state.tree.find_all(exp.Table)):
            table_name = node.name.lower()
            if table_name in scoped_tables:
                inner = (
                    exp.select("*")
                    .from_(table_name)
                    .where(
                        exp.EQ(
                            this=exp.column("user_id"),
                            expression=exp.Literal.number(state.user_id),
                        )
                    )
                )
                subquery = inner.subquery(alias=table_name)
                node.replace(subquery)
                state.rewritten = True


class LimitRule:
    """Step 7 — Enforce maximum row limit (add or tighten LIMIT clause)."""

    name = "limit"
    priority = 70

    def apply(self, state: RuleState, policy: SQLPolicy, dialect: str) -> None:
        if state.tree is None:
            return
        limit_node = state.tree.find(exp.Limit)
        if limit_node is None:
            state.tree.set("limit", exp.Limit(expression=exp.Literal.number(policy.max_rows)))
            state.rewritten = True
        else:
            current_limit = int(limit_node.expression.this)
            if current_limit > policy.max_rows:
                limit_node.set("expression", exp.Literal.number(policy.max_rows))
                state.rewritten = True


# ── Default rules in pipeline order ───────────────────────────────────────────

DEFAULT_RULES: list[PolicyRule] = [
    OpScanRule(),
    ParseRule(),
    TableScopeRule(),
    RestrictedColumnsRule(),
    WildcardRule(),
    UserScopeRule(),
    LimitRule(),
]
