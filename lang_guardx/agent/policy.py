from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Dataclass representing the SQL policy configuration.


@dataclass
class SQLPolicy:
    permitted_operations: list[str]
    permitted_tables: list[str]
    restricted_columns: dict[str, list[str]] = field(default_factory=dict)
    scoped_tables: list[str] = field(default_factory=list)
    require_user_scope: bool = False
    max_rows: int = 1000

    @classmethod
    def from_dict(cls, d: dict) -> SQLPolicy:
        """Load policy from a plain dict (JSON/YAML deserialised)."""
        if "permitted_operations" not in d:
            raise ValueError("Policy must include 'permitted_operations' field.")
        ops = d["permitted_operations"]
        if not isinstance(ops, list) or not all(isinstance(op, str) for op in ops):
            raise ValueError("'permitted_operations' must be a list of strings.")
        if "permitted_tables" not in d:
            raise ValueError("Policy must include 'permitted_tables' field with at least one table.")
        tables = d["permitted_tables"]
        if not isinstance(tables, list) or not all(isinstance(t, str) for t in tables):
            raise ValueError("'permitted_tables' must be a list of strings.")
        if "restricted_columns" not in d:
            raise ValueError("Policy must include 'restricted_columns' field (can't be empty dict).")

        return cls(
            permitted_operations=ops,
            permitted_tables=tables,
            restricted_columns=d.get("restricted_columns", {}),
            scoped_tables=d.get("scoped_tables", []),
            require_user_scope=d.get("require_user_scope", False),
            max_rows=d.get("max_rows", 1000),
        )


class Verdict(Enum):
    PASSED = "PASSED"
    REWRITTEN = "REWRITTEN"
    BLOCKED = "BLOCKED"


@dataclass
class PolicyVerdict:
    verdict: Verdict
    original_sql: str
    safe_sql: str | None
    violations: list[str]

    @classmethod
    def passed(cls, original_sql: str) -> PolicyVerdict:
        return cls(
            verdict=Verdict.PASSED,
            original_sql=original_sql,
            safe_sql=original_sql,
            violations=[],
        )

    @classmethod
    def rewritten(cls, original_sql: str, safe_sql: str) -> PolicyVerdict:
        return cls(
            verdict=Verdict.REWRITTEN,
            original_sql=original_sql,
            safe_sql=safe_sql,
            violations=[],
        )

    @classmethod
    def blocked(cls, original_sql: str, reason: str) -> PolicyVerdict:
        return cls(
            verdict=Verdict.BLOCKED,
            original_sql=original_sql,
            safe_sql=None,
            violations=[reason],
        )
