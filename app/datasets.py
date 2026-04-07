"""
Seeded dataset generators for all 3 tasks.
Each function returns a reproducible pandas DataFrame with intentional data quality issues.
"""
import numpy as np
import pandas as pd


def generate_task1_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Task 1 — Easy: 200 rows, 5 columns.
    Problems: ~15% missing age, ~10% missing total_price. Nothing else.
    """
    rng = np.random.RandomState(seed)
    n = 200

    ages        = rng.randint(18, 70, n).astype(float)
    prices      = rng.uniform(10.0, 5000.0, n)
    quantities  = rng.randint(1, 20, n)
    countries   = rng.choice(["India", "USA", "UK", "Germany", "France"], n)
    cust_ids    = [f"C{str(i).zfill(4)}" for i in range(1, n + 1)]

    # Inject missing
    ages[rng.choice(n, size=int(n * 0.15), replace=False)] = np.nan
    prices = prices.astype(float)
    prices[rng.choice(n, size=int(n * 0.10), replace=False)] = np.nan

    return pd.DataFrame({
        "customer_id": cust_ids,
        "age":         ages,
        "total_price": prices,
        "quantity":    quantities,
        "country":     countries,
    })


def generate_task2_dataset(seed: int = 123) -> pd.DataFrame:
    """
    Task 2 — Medium: 500 rows, 8 columns.
    Problems: missing values, outliers in age/quantity, inconsistent gender strings, bad categories.
    """
    rng = np.random.RandomState(seed)
    n = 500

    cust_ids   = [f"C{str(i).zfill(4)}" for i in range(1, n + 1)]
    ages       = rng.randint(18, 70, n).astype(float)
    genders    = list(rng.choice(["Male", "Female"], n))
    countries  = rng.choice(["India", "USA", "UK", "Germany", "France"], n)
    categories = list(rng.choice(["Electronics", "Clothing", "Food", "Books", "Sports"], n))
    quantities = rng.randint(1, 20, n).astype(float)
    unit_prices = rng.uniform(10.0, 2000.0, n)
    total_prices = (quantities * unit_prices).astype(float)

    # Missing
    ages[rng.choice(n, size=int(n * 0.15), replace=False)] = np.nan
    unit_prices[rng.choice(n, size=int(n * 0.10), replace=False)] = np.nan
    total_prices[rng.choice(n, size=int(n * 0.10), replace=False)] = np.nan

    # Outliers — impossible ages
    for i in rng.choice(n, size=15, replace=False):
        ages[i] = rng.choice([150, 180, 200, -5])

    # Outliers — negative quantities
    for i in rng.choice(n, size=10, replace=False):
        quantities[i] = rng.choice([-3, -5, -10])

    # Inconsistent gender labels
    bad_gender = ["M", "male", "F", "female", "MALE", "FEMALE"]
    for i in rng.choice(n, size=30, replace=False):
        genders[i] = rng.choice(bad_gender)

    # Bad category values
    for i in rng.choice(n, size=15, replace=False):
        categories[i] = "???"

    return pd.DataFrame({
        "customer_id":     cust_ids,
        "age":             ages,
        "gender":          genders,
        "country":         countries,
        "product_category": categories,
        "quantity":        quantities,
        "unit_price":      unit_prices,
        "total_price":     total_prices,
    })


def generate_task3_dataset(seed: int = 456) -> pd.DataFrame:
    """
    Task 3 — Hard: 1000 rows, 12 columns.
    Problems: missing, outliers, noise + hidden correlations baked in via seed:
      - High discount  → higher return_flag
      - Electronics    → higher unit_price AND higher return rate
      - CLV correlates with quantity × unit_price
    """
    rng = np.random.RandomState(seed)
    n = 1000

    cust_ids    = [f"C{str(i).zfill(4)}" for i in range(1, n + 1)]
    ages        = rng.randint(18, 70, n).astype(float)
    genders     = list(rng.choice(["Male", "Female"], n))
    countries   = rng.choice(["India", "USA", "UK", "Germany", "France"], n)
    categories  = rng.choice(["Electronics", "Clothing", "Food", "Books", "Sports"], n)
    quantities  = rng.randint(1, 20, n).astype(float)
    discounts   = rng.uniform(0.0, 0.5, n)
    pay_methods = rng.choice(["UPI", "Credit Card", "PayPal", "Debit Card", "NetBanking"], n)
    order_months = rng.randint(1, 13, n)

    # Electronics → higher price
    unit_prices = np.where(
        categories == "Electronics",
        rng.uniform(500.0, 3000.0, n),
        rng.uniform(10.0, 500.0, n),
    )

    # Hidden correlation 1: high discount → higher return probability
    return_prob = 0.10 + 0.60 * discounts
    return_flags = (rng.uniform(0, 1, n) < return_prob).astype(int)

    # Hidden correlation 2: Electronics → 45 % return rate regardless
    elec_mask = categories == "Electronics"
    return_flags[elec_mask] = (rng.uniform(0, 1, int(elec_mask.sum())) < 0.45).astype(int)

    # Hidden correlation 3: CLV ~ quantity × price
    clv = quantities * unit_prices * rng.uniform(1.5, 3.5, n) + rng.normal(0, 100, n)

    total_prices = quantities * unit_prices * (1.0 - discounts) + rng.normal(0, 50, n)

    # Missing values
    ages[rng.choice(n, size=int(n * 0.12), replace=False)] = np.nan
    clv_arr = clv.copy()
    clv_arr[rng.choice(n, size=int(n * 0.08), replace=False)] = np.nan

    # Outliers in age
    for i in rng.choice(n, size=20, replace=False):
        ages[i] = rng.choice([150, 180, 200])

    # Inconsistent gender
    bad_gender = ["M", "male", "F", "female"]
    for i in rng.choice(n, size=50, replace=False):
        genders[i] = rng.choice(bad_gender)

    return pd.DataFrame({
        "customer_id":            cust_ids,
        "age":                    ages,
        "gender":                 genders,
        "country":                countries,
        "product_category":       categories,
        "quantity":               quantities,
        "unit_price":             unit_prices,
        "discount":               discounts,
        "total_price":            total_prices,
        "return_flag":            return_flags,
        "payment_method":         pay_methods,
        "customer_lifetime_value": clv_arr,
        "order_month":            order_months,
    })
