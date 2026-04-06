from __future__ import annotations

import sqlglot.expressions as exp
from sqlglot import parse_one
from sqlglot.errors import ParseError, TokenError

from .policy import PolicyVerdict, SQLPolicy


class SQLPolicyEngine:
    _ALL_OPS: frozenset[str] = frozenset(
        {
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "TRUNCATE",
            "ALTER",
            "CREATE",
            "MERGE",
        }
    )

    def __init__(
        self,
        policy: SQLPolicy,
        current_user_id: int | None = None,
        dialect: str = "sqlite",
    ) -> None:
        self._policy = policy
        self._user_id = current_user_id
        self._dialect = dialect
        self._permitted_tables_set = frozenset(t.lower() for t in policy.permitted_tables)
        self._restricted_cols_set = {table.lower(): frozenset(col.lower() for col in cols) for table, cols in policy.restricted_columns.items()}
        self._all_restricted = frozenset().union(*self._restricted_cols_set.values())
        self._forbidden_ops_set = self._ALL_OPS - frozenset(op.upper() for op in policy.permitted_operations)
        self._scoped_tables_set = frozenset(t.lower() for t in policy.scoped_tables)

    def validate(self, sql: str) -> PolicyVerdict:
        sql = sql.strip()

        verdict = self._step1_op_scan(sql)
        if verdict is not None:
            return verdict

        tree, verdict = self._step2_parse(sql)
        if verdict is not None:
            return verdict

        if tree is None:
            raise ValueError("Parsed SQL tree is None")
        verdict = self._step3_table_scope(tree, sql)
        if verdict is not None:
            return verdict

        verdict = self._step4_restricted_cols(tree, sql)
        if verdict is not None:
            return verdict

        verdict = self._step5_wildcard(tree, sql)
        if verdict is not None:
            return verdict

        rewritten = False
        tree, rewritten = self._step6_user_scope(tree, rewritten)
        tree, rewritten = self._step7_limit(tree, rewritten)

        return self._step8_verdict(sql, tree, rewritten)

    def _step1_op_scan(self, sql: str) -> PolicyVerdict | None:
        sql_tokens = sql.strip().upper().split()
        if not sql_tokens:
            return PolicyVerdict.blocked(sql, "Empty query")
        keyword = sql_tokens[0]
        if keyword in self._forbidden_ops_set:
            return PolicyVerdict.blocked(sql, f"Forbidden operation: {keyword}")
        always_forbidden = {"DROP", "TRUNCATE", "ALTER", "CREATE", "MERGE"}
        if any(token in always_forbidden for token in sql_tokens):
            return PolicyVerdict.blocked(sql, "Forbidden keyword in query")
        return None

    def _step2_parse(self, sql: str) -> tuple[exp.Expression | None, PolicyVerdict | None]:
        try:
            tree = parse_one(sql=sql, dialect=self._dialect)
            return (tree, None)
        except (ParseError, TokenError) as e:
            return (None, PolicyVerdict.blocked(sql, f"Invalid SQL: {e}"))

    def _step3_table_scope(self, tree: exp.Expression, sql: str) -> PolicyVerdict | None:
        if not self._permitted_tables_set:
            return None
        for node in tree.find_all(exp.Table):
            table_name = node.name.lower()
            if table_name not in self._permitted_tables_set:
                return PolicyVerdict.blocked(original_sql=sql, reason=f"Forbidden table: {table_name}")

        return None

    def _step4_restricted_cols(self, tree: exp.Expression, sql: str) -> PolicyVerdict | None:
        if not self._restricted_cols_set:
            return None
        for node in tree.find_all(exp.Column):
            col_name = node.name.lower()
            table_name = node.table.lower() if node.table else None
            if table_name:
                restricted_cols = self._restricted_cols_set.get(table_name, frozenset())
                if col_name in restricted_cols:
                    return PolicyVerdict.blocked(
                        original_sql=sql,
                        reason=f"Restricted column: {table_name}.{col_name}",
                    )
            else:
                if col_name in self._all_restricted:
                    return PolicyVerdict.blocked(original_sql=sql, reason=f"Restricted column: {col_name}")
        return None

    def _step5_wildcard(self, tree: exp.Expression, sql: str) -> PolicyVerdict | None:
        for node in tree.find_all(exp.Star):
            parent = node.parent
            if isinstance(parent, exp.Count) or isinstance(parent, exp.Anonymous) and parent.name.upper() == "COUNT":
                return PolicyVerdict.blocked(
                    original_sql=sql,
                    reason="Wildcard SELECT (COUNT(*)) is not allowed by policy",
                )
            else:
                return PolicyVerdict.blocked(original_sql=sql, reason="Wildcard * is not allowed by policy")
        return None

    def _step6_user_scope(self, tree: exp.Expression, rewritten: bool):
        if not self._policy.require_user_scope or self._user_id is None:
            return (tree, rewritten)
        for node in tree.find_all(exp.Table):
            table_name = node.name.lower()
            if table_name in self._scoped_tables_set:
                inner = (
                    exp.select("*")
                    .from_(table_name)
                    .where(
                        exp.EQ(
                            this=exp.column("user_id"),
                            expression=exp.Literal.number(self._user_id),
                        )
                    )
                )
                subquery = inner.subquery(alias=table_name)
                node.replace(subquery)
                rewritten = True
        return (tree, rewritten)

    def _step7_limit(self, tree: exp.Expression, rewritten: bool):
        limit_node = tree.find(exp.Limit)
        if limit_node is None:
            tree.set("limit", exp.Limit(expression=exp.Literal.number(self._policy.max_rows)))
            rewritten = True
        else:
            current_limit = int(limit_node.expression.this)
            if current_limit > self._policy.max_rows:
                limit_node.set(
                    "expression",
                    exp.Literal.number(self._policy.max_rows),
                )
                rewritten = True
        return (tree, rewritten)

    def _step8_verdict(
        self,
        original_sql: str,
        tree: exp.Expression,
        rewritten: bool,
    ) -> PolicyVerdict:
        if rewritten:
            safe_sql = tree.sql(dialect=self._dialect)
            return PolicyVerdict.rewritten(original_sql, safe_sql)

        return PolicyVerdict.passed(original_sql)
