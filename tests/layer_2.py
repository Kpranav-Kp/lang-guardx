"""
tests/layer_2_engine.py
=======================
Layer 2 engine and policy tests — no LLM, no DB, no API key.

Usage:
    python -m pytest tests/layer_2_engine.py -v
    python tests/layer_2_engine.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lang_guardx.agent.engine import SQLPolicyEngine
from lang_guardx.agent.policy import PolicyVerdict, SQLPolicy, Verdict

REPS = 200  # engine is fast, more reps = better numbers


# ── Shared policy fixture ─────────────────────────────────────────────────────


def make_policy(**kwargs):
    permitted_operations = kwargs.get("permitted_operations", ["SELECT"])
    permitted_tables = kwargs.get("permitted_tables", ["products", "orders", "reviews", "customers"])
    restricted_columns = kwargs.get(
        "restricted_columns",
        {
            "employees": ["salary", "password_hash"],
            "customers": ["email", "phone"],
        },
    )
    scoped_tables = kwargs.get("scoped_tables", ["orders"])
    require_user_scope = kwargs.get("require_user_scope", False)
    max_rows = kwargs.get("max_rows", 50)
    return SQLPolicy(
        permitted_operations=permitted_operations,
        permitted_tables=permitted_tables,
        restricted_columns=restricted_columns,
        scoped_tables=scoped_tables,
        require_user_scope=require_user_scope,
        max_rows=max_rows,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SQLPolicy
# ═══════════════════════════════════════════════════════════════════════════════


class TestSQLPolicy:
    def test_basic_instantiation(self):
        p = SQLPolicy(permitted_operations=["SELECT"], permitted_tables=["products"])
        assert p.max_rows == 1000
        assert p.require_user_scope is False

    def test_from_dict_valid(self):
        d = {
            "permitted_operations": ["SELECT"],
            "permitted_tables": ["products"],
            "restricted_columns": {},
            "max_rows": 100,
        }
        p = SQLPolicy.from_dict(d)
        assert p.max_rows == 100

    def test_from_dict_missing_operations_raises(self):
        with pytest.raises(ValueError):
            SQLPolicy.from_dict({"permitted_tables": ["products"]})

    def test_from_dict_missing_tables_raises(self):
        with pytest.raises(ValueError):
            SQLPolicy.from_dict({"permitted_operations": ["SELECT"]})

    def test_from_dict_missing_restricted_columns_defaults_to_empty(self):
        """restricted_columns is optional; defaults to empty dict."""
        policy = SQLPolicy.from_dict(
            {
                "permitted_operations": ["SELECT"],
                "permitted_tables": ["products"],
                # restricted_columns absent
            }
        )
        assert policy.restricted_columns == {}

    def test_from_dict_defaults_applied(self):
        d = {
            "permitted_operations": ["SELECT"],
            "permitted_tables": ["products"],
            "restricted_columns": {},
        }
        p = SQLPolicy.from_dict(d)
        assert p.max_rows == 1000
        assert p.require_user_scope is False
        assert p.scoped_tables == []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SQLPolicyEngine: BLOCKED verdicts
# ═══════════════════════════════════════════════════════════════════════════════


class TestSQLPolicyEngineBlocked:
    @pytest.fixture(scope="class")
    def engine(self):
        return SQLPolicyEngine(make_policy())

    # ── U.1 — Forbidden verb ─────────────────────────────────────────────────

    def test_u1_drop_table(self, engine):
        v = engine.validate("DROP TABLE users")
        assert v.verdict == Verdict.BLOCKED

    def test_u1_delete_from(self, engine):
        v = engine.validate("DELETE FROM orders")
        assert v.verdict == Verdict.BLOCKED

    def test_u1_truncate(self, engine):
        v = engine.validate("TRUNCATE TABLE products")
        assert v.verdict == Verdict.BLOCKED

    def test_u1_alter(self, engine):
        v = engine.validate("ALTER TABLE users ADD COLUMN x TEXT")
        assert v.verdict == Verdict.BLOCKED

    def test_u1_insert(self, engine):
        v = engine.validate("INSERT INTO products VALUES (1, 'test')")
        assert v.verdict == Verdict.BLOCKED

    def test_u1_update(self, engine):
        v = engine.validate("UPDATE products SET price = 0")
        assert v.verdict == Verdict.BLOCKED

    def test_u1_create(self, engine):
        v = engine.validate("CREATE TABLE evil (id INT)")
        assert v.verdict == Verdict.BLOCKED

    # ── U.2 — Wildcard ───────────────────────────────────────────────────────

    def test_u2_select_star(self, engine):
        v = engine.validate("SELECT * FROM products")
        assert v.verdict == Verdict.BLOCKED

    def test_u2_count_star(self, engine):
        v = engine.validate("SELECT COUNT(*) FROM products")
        assert v.verdict == Verdict.BLOCKED

    def test_u2_star_with_where(self, engine):
        v = engine.validate("SELECT * FROM products WHERE id = 1")
        assert v.verdict == Verdict.BLOCKED

    # ── U.3 — Restricted column ──────────────────────────────────────────────

    def test_u3_salary_qualified(self, engine):
        v = engine.validate("SELECT employees.salary FROM employees")
        assert v.verdict == Verdict.BLOCKED

    def test_u3_password_hash_unqualified(self, engine):
        v = engine.validate("SELECT password_hash FROM employees")
        assert v.verdict == Verdict.BLOCKED

    def test_u3_email_qualified(self, engine):
        v = engine.validate("SELECT customers.email FROM customers")
        assert v.verdict == Verdict.BLOCKED

    # ── RI.2 — Schema probing ────────────────────────────────────────────────

    def test_ri2_information_schema(self, engine):
        v = engine.validate("SELECT * FROM information_schema.tables")
        assert v.verdict == Verdict.BLOCKED

    def test_ri2_sqlite_master(self, engine):
        v = engine.validate("SELECT * FROM sqlite_master")
        assert v.verdict == Verdict.BLOCKED

    # ── Forbidden table ──────────────────────────────────────────────────────

    def test_forbidden_table_blocked(self, engine):
        v = engine.validate("SELECT id FROM employees")
        assert v.verdict == Verdict.BLOCKED

    def test_forbidden_table_in_join_blocked(self, engine):
        v = engine.validate("SELECT p.name FROM products p JOIN employees e ON p.id = e.id")
        assert v.verdict == Verdict.BLOCKED

    # ── Invalid SQL ──────────────────────────────────────────────────────────

    def test_invalid_sql_blocked(self, engine):
        v = engine.validate("NOT VALID SQL !!!!")
        assert v.verdict == Verdict.BLOCKED

    def test_empty_query_blocked(self, engine):
        v = engine.validate("")
        assert v.verdict == Verdict.BLOCKED


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SQLPolicyEngine: REWRITTEN verdicts
# ═══════════════════════════════════════════════════════════════════════════════


class TestSQLPolicyEngineRewritten:
    @pytest.fixture(scope="class")
    def engine(self):
        return SQLPolicyEngine(make_policy())

    def test_limit_added_when_absent(self, engine):
        v = engine.validate("SELECT product_name FROM products")
        assert v.verdict == Verdict.REWRITTEN
        assert "LIMIT" in v.safe_sql.upper()

    def test_limit_tightened_when_too_large(self, engine):
        v = engine.validate("SELECT product_name FROM products LIMIT 9999")
        assert v.verdict == Verdict.REWRITTEN
        assert "50" in v.safe_sql

    def test_rewritten_sql_is_not_none(self, engine):
        v = engine.validate("SELECT product_name FROM products")
        assert v.safe_sql is not None

    def test_rewritten_sql_contains_table(self, engine):
        v = engine.validate("SELECT product_name FROM products")
        assert "products" in v.safe_sql.lower()

    def test_user_scope_injected_when_required(self):
        policy = make_policy(
            require_user_scope=True,
            scoped_tables=["orders"],
        )
        engine = SQLPolicyEngine(policy, current_user_id=42)
        v = engine.validate("SELECT id FROM orders")
        assert v.verdict == Verdict.REWRITTEN
        assert v.safe_sql is not None
        assert "42" in v.safe_sql

    def test_rd2_limit_bypass_rewritten(self, engine):
        v = engine.validate("SELECT id FROM products LIMIT 99999")
        assert v.verdict == Verdict.REWRITTEN
        assert "99999" not in v.safe_sql


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — P2SQL Full Coverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestP2SQLCoverage:
    """All 7 P2SQL attack patterns in one place for thesis Table 2."""

    @pytest.fixture(scope="class")
    def engine(self):
        return SQLPolicyEngine(make_policy())

    def test_u1_forbidden_verb(self, engine):
        assert engine.validate("DROP TABLE users").verdict == Verdict.BLOCKED

    def test_u2_wildcard(self, engine):
        assert engine.validate("SELECT * FROM products").verdict == Verdict.BLOCKED

    def test_u3_restricted_column(self, engine):
        assert engine.validate("SELECT salary FROM employees").verdict == Verdict.BLOCKED

    def test_rd1_cross_user(self):
        policy = make_policy(require_user_scope=True, scoped_tables=["orders"])
        engine = SQLPolicyEngine(policy, current_user_id=1)
        assert engine.validate("SELECT id FROM orders").verdict == Verdict.REWRITTEN

    def test_rd2_limit_bypass(self, engine):
        assert engine.validate("SELECT id FROM products LIMIT 9999").verdict == Verdict.REWRITTEN

    def test_ri2_schema_probe(self, engine):
        assert engine.validate("SELECT * FROM sqlite_master").verdict == Verdict.BLOCKED

    def test_safe_legitimate_query(self, engine):
        v = engine.validate("SELECT product_name FROM products")
        assert v.verdict in (Verdict.REWRITTEN, Verdict.PASSED)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PolicyVerdict structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyVerdict:
    def test_blocked_has_violations(self):
        v = PolicyVerdict.blocked("DROP TABLE x", "Forbidden operation: DROP")
        assert v.verdict == Verdict.BLOCKED
        assert len(v.violations) == 1
        assert v.safe_sql is None

    def test_rewritten_has_safe_sql(self):
        v = PolicyVerdict.rewritten("SELECT * FROM x", "SELECT id FROM x LIMIT 50")
        assert v.verdict == Verdict.REWRITTEN
        assert v.safe_sql == "SELECT id FROM x LIMIT 50"
        assert v.violations == []

    def test_passed_safe_sql_equals_original(self):
        v = PolicyVerdict.passed("SELECT id FROM products")
        assert v.verdict == Verdict.PASSED
        assert v.safe_sql == v.original_sql


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LATENCY BASELINE
# ═══════════════════════════════════════════════════════════════════════════════


def _measure(fn, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": round(statistics.mean(times), 4),
        "p95_ms": round(sorted(times)[int(0.95 * len(times))], 4),
    }


def run_latency_report():
    W = 72
    print("\n" + "═" * W)
    print("  LANGGUARDX LAYER 2 — ENGINE LATENCY BASELINE")
    print(f"  {REPS} reps per case")
    print("═" * W)

    engine = SQLPolicyEngine(make_policy())

    cases = [
        ("BLOCKED — forbidden verb", "DROP TABLE users"),
        ("BLOCKED — wildcard", "SELECT * FROM products"),
        ("BLOCKED — restricted column", "SELECT salary FROM employees"),
        ("BLOCKED — forbidden table", "SELECT id FROM employees"),
        ("BLOCKED — invalid SQL", "NOT VALID SQL"),
        ("REWRITTEN — limit added", "SELECT product_name FROM products"),
        ("REWRITTEN — limit tightened", "SELECT product_name FROM products LIMIT 9999"),
        ("REWRITTEN — complex with join", "SELECT p.name, o.total FROM products p JOIN orders o ON p.id = o.product_id"),
    ]

    print(f"\n  {'Case':<40} {'Verdict':<12} {'Mean':>9} {'p95':>9}")
    print(f"  {'':─<40} {'':─<12} {'(ms)':>9} {'(ms)':>9}")

    for label, sql in cases:
        s = _measure(lambda q=sql: engine.validate(q))
        verdict = engine.validate(sql).verdict.value
        print(f"  {label:<40} {verdict:<12} {s['mean_ms']:>9} {s['p95_ms']:>9}")

    print("\n  ➜  Target: <2ms mean for all cases (vs PromptGuard LLM checker)")
    print("═" * W + "\n")


if __name__ == "__main__":
    run_latency_report()
