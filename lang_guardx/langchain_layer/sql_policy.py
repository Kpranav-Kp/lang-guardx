from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sqlglot
import sqlglot.expressions as exp
import sqlglot.errors as errors



@dataclass
class SQLPolicy:
    # Only these SQL operations are allowed. Any other operation is a violation.
    permitted_operations: List[str] = field(
        default_factory=lambda: ["SELECT"]
    )

    # If non-empty, only these tables may be queried. Empty = all tables allowed.
    permitted_tables: List[str] = field(default_factory=list)

    # Columns that must never appear in SELECT results, keyed by table name.
    restricted_columns: Dict[str, List[str]] = field(default_factory=dict)

    # If True, every query to a scoped table must include WHERE user_id = current_user.
    require_user_scope: bool = False

    # Column name used for user scoping (typically "user_id").
    user_scope_column: str = "user_id"

    # Maximum rows any query may return. Prevents bulk data dumps.
    max_rows: int = 100

    # Tables that require user-scope enforcement.
    scoped_tables: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SQLPolicy":
        return cls(
            permitted_operations=data.get("permitted_operations", ["SELECT"]),
            permitted_tables=data.get("permitted_tables", []),
            restricted_columns=data.get("restricted_columns", {}),
            require_user_scope=data.get("require_user_scope", False),
            user_scope_column=data.get("user_scope_column", "user_id"),
            max_rows=data.get("max_rows", 100),
            scoped_tables=data.get("scoped_tables", []),
        )

    @classmethod
    def default(cls) -> "SQLPolicy":
        return cls(
            permitted_operations=["SELECT"],
            permitted_tables=[],
            restricted_columns={},
            require_user_scope=False,
            max_rows=50,
        )


@dataclass
class PolicyVerdict:

    blocked: bool
    rewritten_sql: Optional[str]
    violations: List[str]
    original_sql: str

    @property
    def safe_sql(self) -> Optional[str]:
        if self.blocked:
            return None
        return self.rewritten_sql if self.rewritten_sql else self.original_sql


class SQLPolicyEngine:
    # SQL operation keywords that indicate write/destroy operations.
    _WRITE_OPERATIONS = {
        "DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE",
        "ALTER", "CREATE", "REPLACE", "MERGE",
    }

    def __init__(
        self,
        policy: Optional[SQLPolicy] = None,
        current_user_id: Optional[int] = None,
        dialect: str = "postgres",
    ) -> None:
        self.policy = policy or SQLPolicy.default()
        self.current_user_id = current_user_id
        self.dialect = dialect


    def validate(self, sql: str) -> PolicyVerdict:
        violations: List[str] = []

        #Step 1: Operation type check (hard block on writes) 
        op_violation = self._check_operation(sql)
        if op_violation:
            violations.append(op_violation)
            return PolicyVerdict(
                blocked=True,
                rewritten_sql=None,
                violations=violations,
                original_sql=sql,
            )

        #Step 2: Parse to AST (sqlglot) 
        try:
            tree = sqlglot.parse_one(sql, dialect=self.dialect)
        except errors.ParseError as exc:
            violations.append(f"sql_parse_error: {exc}")
            return PolicyVerdict(
                blocked=True,
                rewritten_sql=None,
                violations=violations,
                original_sql=sql,
            )

        #Step 3: Table scope check 
        table_violation = self._check_tables(tree)
        if table_violation:
            violations.append(table_violation)
            return PolicyVerdict(
                blocked=True,
                rewritten_sql=None,
                violations=violations,
                original_sql=sql,
            )

        #Step 4: Restricted column check 
        col_violation = self._check_restricted_columns(tree)
        if col_violation:
            violations.append(col_violation)
            # Column violations are auto-fixable by rewriting
            # (rewriter will strip restricted columns from SELECT list).

        #Step 5: Wildcard SELECT check
        if self._has_wildcard(tree):
            violations.append("wildcard_select: SELECT * is not permitted")
            # Auto-fixable: rewriter will expand to permitted columns if known,
            # or block if no column whitelist exists.

        #Step 6: Rewrite
        rewritten: Optional[str] = None
        if violations:
            rewrite_result = self._rewrite(tree, violations)
            if rewrite_result is None:
                # Rewriting failed or was not possible — block.
                return PolicyVerdict(
                    blocked=True,
                    rewritten_sql=None,
                    violations=violations,
                    original_sql=sql,
                )
            rewritten = rewrite_result
            # Clear violations that were fixed by rewriting.
            violations = [v for v in violations if not self._was_fixed(v)]

        #Step 7: Apply user scope injection if policy requires it 
        target_tree = sqlglot.parse_one(rewritten, dialect=self.dialect) if rewritten else tree
        scoped_sql = self._inject_user_scope(target_tree)
        if scoped_sql:
            rewritten = scoped_sql

        #Step 8: Enforce LIMIT 
        final_tree = sqlglot.parse_one(rewritten or sql, dialect=self.dialect)
        limit_sql = self._enforce_limit(final_tree)
        if limit_sql:
            rewritten = limit_sql

        return PolicyVerdict(
            blocked=False,
            rewritten_sql=rewritten,
            violations=violations,
            original_sql=sql,
        )


    def _check_operation(self, sql: str) -> Optional[str]:
        upper = sql.upper()
        for op in self._WRITE_OPERATIONS:
            if re.search(rf"\b{op}\b", upper):
                if op not in [o.upper() for o in self.policy.permitted_operations]:
                    return f"forbidden_operation: {op} is not in permitted_operations"
        return None

    def _check_tables(self, tree: exp.Expression) -> Optional[str]:
        if not self.policy.permitted_tables:
            return None  

        permitted_lower = {t.lower() for t in self.policy.permitted_tables}
        for table in tree.find_all(exp.Table):
            name = table.name.lower() if table.name else ""
            if name and name not in permitted_lower:
                return f"forbidden_table: '{name}' is not in permitted_tables"
        return None

    def _check_restricted_columns(self, tree: exp.Expression) -> Optional[str]:
        if not self.policy.restricted_columns:
            return None

        for col in tree.find_all(exp.Column):
            col_name = col.name.lower() if col.name else ""
            table_name = col.table.lower() if col.table else ""

            # Check table-specific restriction
            if table_name and table_name in {
                k.lower() for k in self.policy.restricted_columns
            }:
                restricted = [
                    c.lower()
                    for c in self.policy.restricted_columns.get(table_name, [])
                ]
                if col_name in restricted:
                    return (
                        f"restricted_column: '{col_name}' on table '{table_name}' "
                        f"is not permitted"
                    )

            # Check column-only restriction (applies across all tables)
            for tbl, cols in self.policy.restricted_columns.items():
                if col_name in [c.lower() for c in cols] and not table_name:
                    return f"restricted_column: '{col_name}' is restricted"

        return None

    def _has_wildcard(self, tree: exp.Expression) -> bool:
        """Returns True if the query contains SELECT *."""
        for col in tree.find_all(exp.Star):
            return True
        return False


    def _rewrite(
        self, tree: exp.Expression, violations: List[str]
    ) -> Optional[str]:
        """
        Attempt to auto-fix violations in the AST.
        Returns rewritten SQL string, or None if the violation cannot be fixed.
        """
        # Wildcard: we cannot safely expand SELECT * without a column whitelist.
        # For now, block it. In a full implementation you would substitute
        # the permitted column list from the policy config.
        has_wildcard = any("wildcard_select" in v for v in violations)
        if has_wildcard:
            # TODO: expand to explicit column list from policy config
            return None  # Block for now — safe default

        # Restricted column: remove the column from SELECT list
        # This is a simplified approach; a production version would
        # walk the AST and prune exp.Column nodes that are restricted.
        # For the starter scaffold, return None to block.
        has_restricted_col = any("restricted_column" in v for v in violations)
        if has_restricted_col:
            return None  # Block — safer than partial rewrite

        return None

    def _was_fixed(self, violation: str) -> bool:
        """
        Returns True if this violation type is one the rewriter handles.
        Used to clean the violations list after a successful rewrite.
        """
        fixable_prefixes = ("limit_exceeded",)
        return any(violation.startswith(p) for p in fixable_prefixes)

    def _inject_user_scope(self, tree: exp.Expression) -> Optional[str]:
        if not self.policy.require_user_scope:
            return None
        if self.current_user_id is None:
            return None

        scoped_tables = (
            {t.lower() for t in self.policy.scoped_tables}
            if self.policy.scoped_tables
            else None  # None means: scope all tables
        )

        modified = False
        for table in tree.find_all(exp.Table):
            name = table.name.lower() if table.name else ""
            if not name:
                continue
            if scoped_tables is not None and name not in scoped_tables:
                continue

            # Build nested subquery: SELECT * FROM <table> WHERE user_id = <id>
            # sqlglot expression building
            subquery = exp.select("*").from_(name).where(
                exp.EQ(
                    this=exp.Column(
                        this=exp.Identifier(
                            this=self.policy.user_scope_column, quoted=False
                        )
                    ),
                    expression=exp.Literal.number(self.current_user_id),
                )
            ).subquery(alias=name)

            table.replace(subquery)
            modified = True

        if not modified:
            return None

        return tree.sql(dialect=self.dialect)

    def _enforce_limit(self, tree: exp.Expression) -> Optional[str]:
        """
        Ensures a LIMIT clause exists and does not exceed policy max_rows.
        Adds or tightens the limit to policy.max_rows.
        """
        existing_limit = tree.find(exp.Limit)

        if existing_limit is None:
            # No LIMIT clause — add one
            tree.set(
                "limit",
                exp.Limit(this=exp.Literal.number(self.policy.max_rows)),
            )
            return tree.sql(dialect=self.dialect)

        # LIMIT exists — check if it exceeds max_rows
        try:
            current = int(existing_limit.this.this)
            if current > self.policy.max_rows:
                existing_limit.set(
                    "this", exp.Literal.number(self.policy.max_rows)
                )
                return tree.sql(dialect=self.dialect)
        except (AttributeError, ValueError):
            pass

        return None