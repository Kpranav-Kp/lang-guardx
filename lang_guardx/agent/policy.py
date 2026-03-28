"""
lang_guardx/agent/policy.py

This file has ZERO LangChain imports. It operates entirely on plain Python
strings and sqlglot ASTs. The LangChain adapter (adapter.py) is the only
file that touches LangChain. Keep it that way.

Build order (complete each section before moving to the next):
  [1] SQLPolicy dataclass
  [2] Verdict enum + PolicyVerdict dataclass
  [3] SQLPolicyEngine.__init__
  [4] validate() — Step 1: Operation Keyword Scan
  [5] validate() — Step 2: AST Parse
  [6] validate() — Steps 3-5: Table / Column / Wildcard checks
  [7] validate() — Steps 6-7: Rewrite steps (user scope + LIMIT)
  [8] validate() — Step 8: Return final verdict

Run your tests after each section. Do NOT move to the next section
until the current one has passing tests.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import sqlglot
import sqlglot.expressions as exp
from sqlglot import parse_one
from sqlglot.errors import ParseError, TokenError


# Dataclass representing the SQL policy configuration.

@dataclass
class SQLPolicy:
    permitted_operations : list[str] 
    permitted_tables : list[str]
    restricted_columns : dict[str, list[str]]
    scoped_tables : list[str]
    require_user_scope : bool = False
    max_rows : int = 1000
    

class Verdict(Enum):
    PASSED = "PASSED"
    REWRITTEN = "REWRITTEN"
    BLOCKED = "BLOCKED"


@dataclass
class PolicyVerdict:
    verdict : Verdict
    original_sql : str
    safe_sql : Optional[str]
    violations : list[str]

    @classmethod
    def passed(cls, original_sql : str) -> PolicyVerdict:
        return cls(
            verdict=Verdict.PASSED,
            original_sql=original_sql,
            safe_sql=original_sql,
            violations=[],
        )

    @classmethod
    def rewritten(cls, original_sql : str, safe_sql : str) -> PolicyVerdict:
        return cls(
            verdict=Verdict.REWRITTEN,
            original_sql=original_sql,
            safe_sql=safe_sql,
            violations=[],
        )

    @classmethod
    def blocked(cls, original_sql : str, reason : str) -> PolicyVerdict:
        return cls(
            verdict=Verdict.BLOCKED,
            original_sql=original_sql,
            safe_sql=None,
            violations=[reason],
        )



class SQLPolicyEngine:
    _ALL_OPS: frozenset[str] = frozenset({
        "SELECT", "INSERT", "UPDATE", "DELETE",
        "DROP", "TRUNCATE", "ALTER", "CREATE", "MERGE"
    })

    def __init__(
        self,
        policy: SQLPolicy,
        current_user_id: Optional[int] = None,
    ) -> None:
        self._policy = policy
        self._user_id = current_user_id
        self._permitted_tables_set = frozenset(t.lower() for t in policy.permitted_tables)
        self._restricted_cols_set = {
            table.lower(): frozenset(cols.lower() for cols in cols)
            for table, cols in policy.restricted_columns.items()
        }
        self._all_restricted = frozenset().union(*self._restricted_cols_set.values())
        self._forbidden_ops_set = self._ALL_OPS - frozenset(op.upper() for op in policy.permitted_operations)



    def validate(self, sql: str) -> PolicyVerdict:
        sql = sql.strip()
 
        verdict = self._step1_op_scan(sql)
        if verdict is not None:
            return verdict
 
        tree, verdict = self._step2_parse(sql)
        if verdict is not None:
            return verdict
 
        assert tree is not None  # tree is guaranteed non-None if verdict is None
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


    def _step1_op_scan(self, sql: str) -> Optional[PolicyVerdict]:
        sql_tokens = sql.strip().upper().split()
        if not sql_tokens:
            return PolicyVerdict.blocked(sql, "Empty query")
        keyword = sql_tokens[0]
        if keyword in self._forbidden_ops_set:
            return PolicyVerdict.blocked(sql, f"Forbidden operation: {keyword}")
        always_forbidden = {"DROP", "TRUNCATE", "ALTER", "CREATE", "MERGE"}
        if any(token in always_forbidden for token in sql_tokens):
            return PolicyVerdict.blocked(sql, f"Forbidden keyword in query")
        return None


    def _step2_parse(self, sql: str, dialect: str = "sqlite") -> tuple[Optional[exp.Expression], Optional[PolicyVerdict]]:
        try:
            tree = parse_one(sql=sql, dialect=dialect)
            return (tree, None)
        except (ParseError, TokenError) as e:
            return (None, PolicyVerdict.blocked(sql, f"Invalid SQL: {e}"))

    def _step3_table_scope(self, tree: exp.Expression, sql: str) -> Optional[PolicyVerdict]:
        if not self._permitted_tables_set:
            return None
        for node in tree.find_all(exp.Table):
            table_name = node.name.lower()
            if table_name not in self._permitted_tables_set:
                return PolicyVerdict.blocked(
                    original_sql=sql,
                    reason=f"Forbidden table: {table_name}"
                )
            
        return None


    def _step4_restricted_cols(self, tree: exp.Expression, sql: str) -> Optional[PolicyVerdict]:
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
                        reason=f"Restricted column: {table_name}.{col_name}"
                    )
            else:
                if col_name in self._all_restricted:
                    return PolicyVerdict.blocked(
                        original_sql=sql,
                        reason=f"Restricted column: {col_name}"
                    )
        return None


    def _step5_wildcard(self, tree: exp.Expression, sql: str) -> Optional[PolicyVerdict]:
        for node in tree.find_all(exp.Star):
            parent = node.parent
            if isinstance(parent, exp.Count) or isinstance(parent, exp.Anonymous) and parent.name.upper() == "COUNT":
                return PolicyVerdict.blocked(
                    original_sql=sql,
                    reason="Wildcard SELECT (COUNT(*)) is not allowed by policy"
                )
            else:
                return PolicyVerdict.blocked(
                    original_sql=sql,
                    reason="Wildcard * is not allowed by policy"
                )
        return None

    def _step6_user_scope(self, tree: exp.Expression, rewritten: bool):
        if not self._policy.require_user_scope or self._user_id is None:
            return (tree, rewritten)
        for node in tree.find_all(exp.Table):
            table_name = node.name.lower()
            if table_name in (t.lower() for t in self._policy.scoped_tables):
                inner = exp.select("*").from_(table_name).where(
                    exp.EQ(
                        this=exp.column("user_id"),
                        expression=exp.Literal.number(self._user_id)
                    )
                )
                subquery = inner.subquery(alias=table_name)
                node.replace(subquery)
                rewritten = True
        return (tree, rewritten)


    def _step7_limit(self, tree: exp.Expression, rewritten: bool):
        limit_node = tree.find(exp.Limit)
        if limit_node is None:
            tree.set(
                "limit",
                exp.Limit(expression=exp.Literal.number(self._policy.max_rows))
            )
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
            safe_sql = tree.sql(dialect="sqlite")
            return PolicyVerdict.rewritten(original_sql, safe_sql)
    
        return PolicyVerdict.passed(original_sql)