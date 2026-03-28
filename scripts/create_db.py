import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

def create_toy_database(db_path: str = "toy_store.db"):
    """Create a toy SQLite database with sample e-commerce data"""
    
    # Remove existing database if it exists
    Path(db_path).unlink(missing_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # 1. Categories Table
    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL,
            description TEXT
        )
    """)
    
    categories = [
        ("Electronics", "Gadgets and electronic devices"),
        ("Clothing", "Apparel and fashion items"),
        ("Books", "Physical and digital books"),
        ("Home & Garden", "Home improvement and gardening"),
        ("Sports", "Sports equipment and accessories"),
        ("Toys", "Children's toys and games"),
    ]
    cursor.executemany("INSERT INTO categories (category_name, description) VALUES (?, ?)", categories)
    
    # 2. Products Table
    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            category_id INTEGER,
            price REAL NOT NULL,
            stock_quantity INTEGER DEFAULT 0,
            supplier TEXT,
            created_date DATE,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)
    
    products = [
        ("Wireless Headphones", 1, 89.99, 150, "TechCorp", "2023-01-15"),
        ("Smart Watch", 1, 199.50, 80, "GadgetInc", "2023-02-20"),
        ("Running Shoes", 5, 120.00, 200, "SportyBrand", "2023-03-10"),
        ("Coffee Maker", 4, 79.99, 60, "HomeEssentials", "2023-01-05"),
        ("Fantasy Novel", 3, 15.99, 300, "BookHouse", "2023-04-12"),
        ("Building Blocks Set", 6, 45.50, 100, "ToyWorld", "2023-05-01"),
        ("Winter Jacket", 2, 150.00, 75, "FashionCo", "2023-06-15"),
        ("Yoga Mat", 5, 35.00, 120, "SportyBrand", "2023-07-20"),
        ("Bluetooth Speaker", 1, 55.00, 90, "TechCorp", "2023-08-10"),
        ("Cookbook", 3, 25.00, 50, "BookHouse", "2023-09-01"),
    ]
    cursor.executemany("""
        INSERT INTO products (product_name, category_id, price, stock_quantity, supplier, created_date) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, products)
    
    # 3. Customers Table
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE,
            phone TEXT,
            city TEXT,
            country TEXT DEFAULT 'USA',
            registration_date DATE,
            loyalty_points INTEGER DEFAULT 0
        )
    """)
    
    customers = [
        ("John", "Smith", "john.smith@email.com", "555-0101", "New York", "USA", "2023-01-10", 150),
        ("Sarah", "Johnson", "sarah.j@email.com", "555-0102", "Los Angeles", "USA", "2023-02-15", 320),
        ("Michael", "Chen", "m.chen@email.com", "555-0103", "San Francisco", "USA", "2023-03-20", 75),
        ("Emma", "Davis", "emma.d@email.com", "555-0104", "Chicago", "USA", "2023-04-05", 200),
        ("Robert", "Wilson", "rob.w@email.com", "555-0105", "Houston", "USA", "2023-05-12", 50),
        ("Lisa", "Brown", "lisa.brown@email.com", "555-0106", "Phoenix", "USA", "2023-06-18", 410),
        ("David", "Lee", "david.lee@email.com", "555-0107", "Seattle", "USA", "2023-07-22", 180),
        ("Jennifer", "Taylor", "jen.t@email.com", "555-0108", "Miami", "USA", "2023-08-30", 95),
    ]
    cursor.executemany("""
        INSERT INTO customers (first_name, last_name, email, phone, city, country, registration_date, loyalty_points) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, customers)
    
    # 4. Employees Table
    cursor.execute("""
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            position TEXT,
            department TEXT,
            hire_date DATE,
            salary REAL,
            manager_id INTEGER,
            FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
        )
    """)
    
    employees = [
        ("Alice", "Manager", "Store Manager", "Management", "2022-01-15", 75000, None),
        ("Bob", "Williams", "Sales Supervisor", "Sales", "2022-03-20", 55000, 1),
        ("Carol", "Martinez", "Sales Associate", "Sales", "2022-06-10", 35000, 2),
        ("Dan", "Anderson", "Sales Associate", "Sales", "2023-01-05", 35000, 2),
        ("Eve", "Thompson", "Inventory Manager", "Operations", "2022-04-12", 48000, 1),
        ("Frank", "Garcia", "Cashier", "Sales", "2023-02-28", 32000, 2),
    ]
    cursor.executemany("""
        INSERT INTO employees (first_name, last_name, position, department, hire_date, salary, manager_id) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, employees)
    
    # 5. Orders Table
    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            employee_id INTEGER,
            order_date DATE,
            total_amount REAL,
            status TEXT DEFAULT 'pending',
            shipping_city TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        )
    """)
    
    # Generate orders for last 6 months
    base_date = datetime(2024, 9, 1)
    orders = []
    for i in range(50):  # 50 orders
        customer_id = random.randint(1, 8)
        employee_id = random.randint(2, 6)  # Exclude manager (id 1) from direct sales
        days_ago = random.randint(0, 180)
        order_date = (base_date - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        total = round(random.uniform(50, 500), 2)
        status = random.choice(["completed", "completed", "completed", "pending", "shipped", "cancelled"])
        city = random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])
        orders.append((customer_id, employee_id, order_date, total, status, city))
    
    cursor.executemany("""
        INSERT INTO orders (customer_id, employee_id, order_date, total_amount, status, shipping_city) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, orders)
    
    # 6. Order Items Table (junction table)
    cursor.execute("""
        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            unit_price REAL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)
    
    # Generate order items
    order_items = []
    for order_id in range(1, 51):
        num_items = random.randint(1, 4)
        for _ in range(num_items):
            product_id = random.randint(1, 10)
            quantity = random.randint(1, 3)
            # Get actual product price
            cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
            price = cursor.fetchone()[0]
            order_items.append((order_id, product_id, quantity, price))
    
    cursor.executemany("""
        INSERT INTO order_items (order_id, product_id, quantity, unit_price) 
        VALUES (?, ?, ?, ?)
    """, order_items)
    
    # 7. Reviews Table
    cursor.execute("""
        CREATE TABLE reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            customer_id INTEGER,
            rating INTEGER CHECK(rating >= 1 AND rating <= 5),
            review_text TEXT,
            review_date DATE,
            FOREIGN KEY (product_id) REFERENCES products(product_id),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)
    
    reviews = [
        (1, 1, 5, "Excellent sound quality!", "2024-01-15"),
        (1, 2, 4, "Good but battery could be better", "2024-02-20"),
        (3, 3, 5, "Most comfortable shoes ever", "2024-03-10"),
        (2, 4, 3, "Average watch for the price", "2024-01-25"),
        (5, 1, 5, "Couldnt put it down!", "2024-04-05"),
        (6, 5, 4, "Kids love it", "2024-05-12"),
        (7, 6, 2, "Too expensive for quality", "2024-06-18"),
        (4, 7, 5, "Makes great coffee every morning", "2024-07-22"),
    ]
    cursor.executemany("""
        INSERT INTO reviews (product_id, customer_id, rating, review_text, review_date) 
        VALUES (?, ?, ?, ?, ?)
    """, reviews)
    
    conn.commit()
    conn.close()

    
    return db_path

def verify_database(db_path: str = "toy_store.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("-" * 50)
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables: {[t[0] for t in tables]}")
    
    # Count records in each table
    for table in [t[0] for t in tables]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  - {table}: {count} records")
    
    # Sample query
    print("\nSample Data (First 3 products):")
    cursor.execute("""
        SELECT p.product_name, c.category_name, p.price, p.stock_quantity 
        FROM products p 
        JOIN categories c ON p.category_id = c.category_id 
        LIMIT 3
    """)
    for row in cursor.fetchall():
        print(f"  {row}")
    
    conn.close()


if __name__ == "__main__":
    db_file = create_toy_database()
    verify_database(db_file)