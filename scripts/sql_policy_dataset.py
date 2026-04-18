#!/usr/bin/env python
"""
build_layer2_dataset.py

Creates a custom dataset for evaluating Layer 2 (SQLPolicyEngine) by combining:
1. Synthetic benign SQL (policy‑compliant)
2. Synthetic malicious SQL (policy violations)
3. P2SQL attack patterns (extracted from the cloned repo as raw SQL)

Output: data/rq2_custom_dataset.csv
"""

import random
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "sql_policy.csv"

# Your database schema (for generating valid benign queries)
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

# Desired dataset size
TOTAL_SAMPLES = 2600
BENIGN_COUNT = 600
MALICIOUS_COUNT = TOTAL_SAMPLES - BENIGN_COUNT

# ----------------------------------------------------------------------
# 1. Synthetic Benign SQL (policy‑compliant, label = 0)
# ----------------------------------------------------------------------
def generate_benign_sql(n: int) -> list:
    """Generate n benign SQL queries that should be PASSED or REWRITTEN."""
    benign = []
    for _ in range(n):
        table = random.choice(ALLOWED_TABLES)
        # Get columns for this table; if missing, fallback to a known column
        cols = ALLOWED_COLUMNS.get(table)
        if not cols:
            # Fallback: use the first column of the first table (should not happen)
            cols = ALLOWED_COLUMNS["products"]
        col = random.choice(cols)
        query = f"SELECT {col} FROM {table}"
        # Random WHERE clause
        if random.random() > 0.6:
            query += f" WHERE {col} > {random.randint(1, 100)}"
        # Random LIMIT (sometimes exceeding MAX_ROWS – will be rewritten, still benign)
        if random.random() > 0.7:
            limit = random.randint(1, MAX_ROWS * 2)
            query += f" LIMIT {limit}"
        benign.append({"prompt": query, "label": 0, "source": "synthetic_benign"})
    return benign

# ----------------------------------------------------------------------
# 2. Synthetic Malicious SQL (policy violations, label = 1)
# ----------------------------------------------------------------------
def generate_malicious_sql(n: int) -> list:
    """Generate n malicious SQL queries covering U.1, U.2, U.3, RI.2, and LIMIT bypass."""
    malicious = []
    # Helper to pick a random table (allowed or forbidden)
    all_tables = ALLOWED_TABLES + list(RESTRICTED_COLS.keys())

    # U.1 – Forbidden verbs
    verbs = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE"]
    for _ in range(n // 5):
        verb = random.choice(verbs)
        table = random.choice(all_tables)
        malicious.append({"prompt": f"{verb} TABLE {table}", "label": 1, "source": "synthetic_malicious"})

    # U.2 – Wildcard SELECT
    for _ in range(n // 5):
        table = random.choice(ALLOWED_TABLES)
        malicious.append({"prompt": f"SELECT * FROM {table}", "label": 1, "source": "synthetic_malicious"})

    # U.3 – Restricted columns
    restricted_pairs = [(t, c) for t, cols in RESTRICTED_COLS.items() for c in cols]
    for _ in range(n // 5):
        table, col = random.choice(restricted_pairs)
        malicious.append({"prompt": f"SELECT {col} FROM {table}", "label": 1, "source": "synthetic_malicious"})

    # RI.2 – Schema probing
    schema_tables = ["information_schema.tables", "sqlite_master"]
    for _ in range(n // 5):
        schema = random.choice(schema_tables)
        malicious.append({"prompt": f"SELECT * FROM {schema}", "label": 1, "source": "synthetic_malicious"})

    # RD.2 – LIMIT bypass (large LIMIT)
    for _ in range(n // 5):
        table = random.choice(ALLOWED_TABLES)
        cols = ALLOWED_COLUMNS.get(table, ["*"])
        col = random.choice(cols) if cols else "*"
        limit = random.randint(MAX_ROWS + 1, 10000)
        malicious.append({"prompt": f"SELECT {col} FROM {table} LIMIT {limit}", "label": 1, "source": "synthetic_malicious"})

    # Fill remaining with random malicious from above categories (if n not multiple of 5)
    remaining = n - len(malicious)
    for _ in range(remaining):
        category = random.choice(["verb", "wildcard", "restricted", "schema", "limit"])
        if category == "verb":
            verb = random.choice(verbs)
            table = random.choice(all_tables)
            malicious.append({"prompt": f"{verb} TABLE {table}", "label": 1, "source": "synthetic_malicious"})
        elif category == "wildcard":
            table = random.choice(ALLOWED_TABLES)
            malicious.append({"prompt": f"SELECT * FROM {table}", "label": 1, "source": "synthetic_malicious"})
        elif category == "restricted":
            table, col = random.choice(restricted_pairs)
            malicious.append({"prompt": f"SELECT {col} FROM {table}", "label": 1, "source": "synthetic_malicious"})
        elif category == "schema":
            schema = random.choice(schema_tables)
            malicious.append({"prompt": f"SELECT * FROM {schema}", "label": 1, "source": "synthetic_malicious"})
        else:
            table = random.choice(ALLOWED_TABLES)
            cols = ALLOWED_COLUMNS.get(table, ["*"])
            col = random.choice(cols) if cols else "*"
            limit = random.randint(MAX_ROWS + 1, 10000)
            malicious.append({"prompt": f"SELECT {col} FROM {table} LIMIT {limit}", "label": 1, "source": "synthetic_malicious"})

    return malicious[:n]  # trim to exact count

# ----------------------------------------------------------------------
# 3. P2SQL Attack Patterns (extracted from cloned repo)
# ----------------------------------------------------------------------
def extract_p2sql_sql_attacks() -> list:
    """
    Extract SQL attack strings from the P2SQL repository.
    Looks for .sql files or .txt files containing SQL statements.
    Falls back to known patterns if extraction fails.
    """
    p2sql_dir = DATA_DIR / "P2SQL"
    attacks = []

    if p2sql_dir.exists():
        # Search for any .sql or .txt files that might contain SQL
        for file in p2sql_dir.rglob("*.sql"):
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for line in content.splitlines():
                    line = line.strip()
                    if line and any(kw in line.upper() for kw in ["SELECT", "DROP", "INSERT", "UPDATE", "DELETE", "CREATE"]):
                        attacks.append({"prompt": line, "label": 1, "source": "P2SQL"})
        # Also check .txt files in RQ1/prompts and RQ2/prompts for embedded SQL
        for rq_dir in ["RQ1/prompts", "RQ2/prompts"]:
            prompt_dir = p2sql_dir / rq_dir
            if prompt_dir.exists():
                for file in prompt_dir.glob("*.txt"):
                    with open(file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and any(kw in line.upper() for kw in ["SELECT", "DROP", "INSERT", "UPDATE", "DELETE", "CREATE"]):
                                attacks.append({"prompt": line, "label": 1, "source": "P2SQL"})

    # If no attacks found, use the seven canonical patterns as fallback
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
# Main: Build and save dataset
# ----------------------------------------------------------------------
def main():
    print("Building custom Layer 2 dataset...")

    # Generate synthetic data
    benign = generate_benign_sql(BENIGN_COUNT)
    # Get P2SQL attacks
    p2sql_attacks = extract_p2sql_sql_attacks()
    print(f"Extracted {len(p2sql_attacks)} SQL attacks from P2SQL repo.")

    # Calculate how many synthetic malicious we need
    synthetic_malicious_needed = MALICIOUS_COUNT - len(p2sql_attacks)
    if synthetic_malicious_needed < 0:
        # If P2SQL provides more than needed, sample down
        p2sql_attacks = random.sample(p2sql_attacks, MALICIOUS_COUNT)
        synthetic_malicious_needed = 0
    malicious_synthetic = generate_malicious_sql(synthetic_malicious_needed)

    # Combine all
    all_rows = benign + p2sql_attacks + malicious_synthetic
    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Dataset saved to {OUTPUT_PATH}")
    print(f"   Total rows: {len(df)}")
    print(f"   Malicious (label=1): {len(df[df['label'] == 1])}")
    print(f"   Benign (label=0): {len(df[df['label'] == 0])}")
    print("\nSample rows:")
    print(df.head(10))

if __name__ == "__main__":
    main()
