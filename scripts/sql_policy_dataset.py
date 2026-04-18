#!/usr/bin/env python
"""
build_layer2_dataset_enhanced.py

Creates an enhanced dataset for RQ2 with:
- Complex benign and malicious SQL (joins, subqueries, CTEs)
- Queries that test LIMIT rewrite inside subqueries
- ~5% noise (extra spaces, mixed case, trailing semicolons, etc.)

Output: data/rq2_enhanced_dataset.csv
"""

import random
import string
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "rq2_enhanced_dataset.csv"

# Schema
ALLOWED_TABLES = ["products", "orders", "order_items", "categories", "reviews"]
ALLOWED_COLUMNS = {
    "products": ["product_id", "product_name", "price", "stock_quantity"],
    "orders": ["order_id", "customer_id", "total_amount", "status"],
    "order_items": ["item_id", "order_id", "product_id", "quantity", "unit_price"],
    "categories": ["category_id", "category_name", "description"],
    "reviews": ["review_id", "product_id", "review_text", "rating"],
}
RESTRICTED_COLS = {"employees": ["salary"], "customers": ["email", "phone"]}
MAX_ROWS = 50

# Dataset size
TOTAL_SAMPLES = 2600
BENIGN_COUNT = 600
MALICIOUS_COUNT = TOTAL_SAMPLES - BENIGN_COUNT
NOISE_PROB = 0.05   # 5% of rows get cosmetic noise

# ----------------------------------------------------------------------
# Noise helpers
# ----------------------------------------------------------------------
def add_noise(sql: str) -> str:
    """Add cosmetic noise: random case, extra spaces, trailing semicolon."""
    if random.random() > NOISE_PROB:
        return sql
    # Random case mixing
    if random.random() > 0.5:
        sql = ''.join(c.upper() if random.random() > 0.7 else c.lower() for c in sql)
    # Extra spaces
    if random.random() > 0.5:
        sql = sql.replace(" ", "  ")
    # Trailing semicolon
    if random.random() > 0.5 and not sql.strip().endswith(';'):
        sql += ";"
    return sql

# ----------------------------------------------------------------------
# Complex benign SQL (policy‑compliant)
# ----------------------------------------------------------------------
def generate_benign_sql(n: int) -> list:
    benign = []
    # Predefined complex patterns
    templates = [
        # Simple SELECT (already covered, but keep)
        lambda: f"SELECT {random.choice(ALLOWED_COLUMNS['products'])} FROM products",
        # Join with two tables
        lambda: f"SELECT p.product_name, o.total_amount FROM products p JOIN orders o ON p.product_id = o.product_id",
        # Join with three tables
        lambda: f"SELECT p.product_name, c.category_name, r.rating FROM products p JOIN categories c ON p.category_id = c.category_id JOIN reviews r ON p.product_id = r.product_id",
        # Subquery in WHERE
        lambda: f"SELECT product_name FROM products WHERE price > (SELECT AVG(price) FROM products)",
        # Subquery in FROM
        lambda: f"SELECT sub.product_name FROM (SELECT product_name, price FROM products WHERE price > 100) AS sub",
        # Subquery with LIMIT (to test rewrite inside subquery)
        lambda: f"SELECT product_name FROM (SELECT product_name FROM products LIMIT 1000) AS sub WHERE product_name LIKE 'A%'",
        # CTE (common table expression)
        lambda: f"WITH high_price AS (SELECT product_id, product_name FROM products WHERE price > 500) SELECT product_name FROM high_price",
        # Nested subquery with LIMIT
        lambda: f"SELECT product_name FROM (SELECT product_name, price FROM (SELECT product_name, price FROM products LIMIT 200) AS inner) AS outer",
        # Aggregation with GROUP BY
        lambda: f"SELECT category_id, AVG(price) FROM products GROUP BY category_id",
        # Complex WHERE with multiple conditions
        lambda: f"SELECT product_name FROM products WHERE price BETWEEN 50 AND 200 AND stock_quantity > 0",
    ]
    for _ in range(n):
        template = random.choice(templates)
        query = template()
        # Randomly add a LIMIT (may exceed MAX_ROWS to trigger rewrite)
        if random.random() > 0.6:
            limit = random.randint(1, MAX_ROWS * 2)
            query += f" LIMIT {limit}"
        query = add_noise(query)
        benign.append({"prompt": query, "label": 0, "source": "synthetic_benign"})
    return benign

# ----------------------------------------------------------------------
# Complex malicious SQL (policy violations)
# ----------------------------------------------------------------------
def generate_malicious_sql(n: int) -> list:
    malicious = []
    all_tables = ALLOWED_TABLES + list(RESTRICTED_COLS.keys())

    # Templates for each attack category, some with subqueries/complexity
    # U.1 - Forbidden verbs (can't be in subquery easily, but we keep simple)
    verbs = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE"]
    # U.2 - Wildcard SELECT (including in subquery)
    # U.3 - Restricted columns (including in subquery)
    # RI.2 - Schema probing
    # RD.2 - LIMIT bypass (large LIMIT)

    # Predefined complex malicious patterns
    templates = [
        # U.1 (simple)
        lambda: f"{random.choice(verbs)} TABLE {random.choice(all_tables)}",
        # U.2 - wildcard in main query
        lambda: f"SELECT * FROM {random.choice(ALLOWED_TABLES)}",
        # U.2 - wildcard in subquery
        lambda: f"SELECT sub.* FROM (SELECT * FROM {random.choice(ALLOWED_TABLES)}) AS sub",
        # U.2 - wildcard in CTE
        lambda: f"WITH w AS (SELECT * FROM {random.choice(ALLOWED_TABLES)}) SELECT * FROM w",
        # U.3 - restricted column in main
        lambda: f"SELECT {random.choice(RESTRICTED_COLS['employees'])} FROM employees",
        # U.3 - restricted column in subquery
        lambda: f"SELECT (SELECT {random.choice(RESTRICTED_COLS['customers'])} FROM customers LIMIT 1) FROM products",
        # RI.2 - schema probe
        lambda: f"SELECT * FROM {random.choice(['information_schema.tables', 'sqlite_master'])}",
        # RI.2 - schema probe in subquery
        lambda: f"SELECT (SELECT name FROM sqlite_master) FROM products",
        # RD.2 - LIMIT bypass
        lambda: f"SELECT {random.choice(ALLOWED_COLUMNS['products'])} FROM products LIMIT {random.randint(MAX_ROWS+1, 10000)}",
        # RD.2 - LIMIT bypass inside subquery
        lambda: f"SELECT product_name FROM (SELECT product_name FROM products LIMIT 10000) AS sub",
    ]

    # Distribute evenly
    per_template = n // len(templates)
    remainder = n % len(templates)
    for _ in range(per_template):
        for t in templates:
            sql = t()
            sql = add_noise(sql)
            malicious.append({"prompt": sql, "label": 1, "source": "synthetic_malicious"})
    for _ in range(remainder):
        t = random.choice(templates)
        sql = t()
        sql = add_noise(sql)
        malicious.append({"prompt": sql, "label": 1, "source": "synthetic_malicious"})

    return malicious[:n]

# ----------------------------------------------------------------------
# P2SQL extraction (unchanged, but we add noise to extracted attacks)
# ----------------------------------------------------------------------
def extract_p2sql_sql_attacks() -> list:
    p2sql_dir = DATA_DIR / "P2SQL"
    attacks = []
    if p2sql_dir.exists():
        for file in p2sql_dir.rglob("*.sql"):
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for line in content.splitlines():
                    line = line.strip()
                    if line and any(kw in line.upper() for kw in ["SELECT", "DROP", "INSERT", "UPDATE", "DELETE", "CREATE"]):
                        attacks.append({"prompt": add_noise(line), "label": 1, "source": "P2SQL"})
        for rq_dir in ["RQ1/prompts", "RQ2/prompts"]:
            prompt_dir = p2sql_dir / rq_dir
            if prompt_dir.exists():
                for file in prompt_dir.glob("*.txt"):
                    with open(file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and any(kw in line.upper() for kw in ["SELECT", "DROP", "INSERT", "UPDATE", "DELETE", "CREATE"]):
                                attacks.append({"prompt": add_noise(line), "label": 1, "source": "P2SQL"})
    if not attacks:
        print("No SQL attacks found in P2SQL repo; using fallback patterns.")
        attacks = [
            {"prompt": "DROP TABLE users", "label": 1, "source": "P2SQL_fallback"},
            {"prompt": "SELECT * FROM products", "label": 1, "source": "P2SQL_fallback"},
            {"prompt": "SELECT salary FROM employees", "label": 1, "source": "P2SQL_fallback"},
            {"prompt": "SELECT id FROM orders", "label": 1, "source": "P2SQL_fallback"},
            {"prompt": "SELECT product_name FROM products LIMIT 9999", "label": 1, "source": "P2SQL_fallback"},
            {"prompt": "SELECT * FROM sqlite_master", "label": 1, "source": "P2SQL_fallback"},
        ]
    return attacks

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Building enhanced RQ2 dataset with complex queries and noise...")
    benign = generate_benign_sql(BENIGN_COUNT)
    p2sql_attacks = extract_p2sql_sql_attacks()
    print(f"Extracted {len(p2sql_attacks)} SQL attacks from P2SQL repo.")
    synthetic_mal_needed = MALICIOUS_COUNT - len(p2sql_attacks)
    if synthetic_mal_needed < 0:
        p2sql_attacks = random.sample(p2sql_attacks, MALICIOUS_COUNT)
        synthetic_mal_needed = 0
    malicious_synthetic = generate_malicious_sql(synthetic_mal_needed)
    all_rows = benign + p2sql_attacks + malicious_synthetic
    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Enhanced dataset saved to {OUTPUT_PATH}")
    print(f"   Total rows: {len(df)}")
    print(f"   Malicious (label=1): {len(df[df['label'] == 1])}")
    print(f"   Benign (label=0): {len(df[df['label'] == 0])}")
    print("\nSample rows:")
    print(df.head(10))

if __name__ == "__main__":
    main()
