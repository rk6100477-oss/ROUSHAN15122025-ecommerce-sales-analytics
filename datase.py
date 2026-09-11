# ============================================================
# TASK 3: DATASET SELECTION & DATA PREPARATION
# Domain: E-Commerce Sales
# ============================================================

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_PATH = Path(__file__).resolve().with_name("ecommerce_sales.csv")
OUTPUT_PATH = Path(__file__).resolve().with_name("ecommerce_sales_cleaned.csv")


def show_head(df: pd.DataFrame, title: str, rows: int = 5) -> None:
    """Print a DataFrame preview without requiring IPython display."""
    print(f"\n{title}")
    print(df.head(rows).to_string(index=False))


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform the e-commerce sales dataset."""
    cleaned = df.copy()

    if cleaned.empty:
        return cleaned

    if "OrderDate" in cleaned.columns:
        cleaned["OrderDate"] = pd.to_datetime(cleaned["OrderDate"], errors="coerce")

    for column in ["Age", "ShippingCost", "Rating"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in ["City", "PaymentMethod"]:
        if column in cleaned.columns:
            mode_value = cleaned[column].mode(dropna=True)
            if not mode_value.empty:
                cleaned[column] = cleaned[column].fillna(mode_value.iloc[0])

    if "OrderID" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset="OrderID", keep="first")
    else:
        cleaned = cleaned.drop_duplicates()

    if "Quantity" in cleaned.columns:
        cleaned = cleaned[cleaned["Quantity"].notna() & (cleaned["Quantity"] > 0)]

    if "Price" in cleaned.columns:
        cleaned["Price"] = cleaned["Price"].abs()

    if "Rating" in cleaned.columns:
        cleaned = cleaned[
            cleaned["Rating"].notna()
            & (cleaned["Rating"] >= 1)
            & (cleaned["Rating"] <= 5)
        ]

    if "DiscountPercent" in cleaned.columns:
        cleaned = cleaned[
            cleaned["DiscountPercent"].notna()
            & (cleaned["DiscountPercent"] >= 0)
            & (cleaned["DiscountPercent"] <= 100)
        ]

    text_columns = [
        "Gender",
        "City",
        "ProductCategory",
        "ProductName",
        "PaymentMethod",
    ]
    for column in text_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip().str.title()

    if {"Price", "Quantity", "DiscountPercent"}.issubset(cleaned.columns):
        cleaned["TotalAmount"] = (
            cleaned["Price"] * cleaned["Quantity"] * (1 - cleaned["DiscountPercent"] / 100)
        )

    if "OrderDate" in cleaned.columns:
        cleaned["OrderMonth"] = cleaned["OrderDate"].dt.month
        cleaned["OrderYear"] = cleaned["OrderDate"].dt.year

    if "Age" in cleaned.columns:
        cleaned["AgeGroup"] = pd.cut(
            cleaned["Age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["Below 25", "26-35", "36-45", "46-55", "56+"],
            right=True,
        )

    for column in ["Price", "Quantity"]:
        if column not in cleaned.columns:
            continue

        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned[f"{column}_Capped"] = cleaned[column].clip(lower=lower, upper=upper)

    return cleaned


def create_sample_dataset(path: Path) -> None:
    """Create a minimal demo dataset if the source CSV is missing."""
    sample_data = [
        {
            "OrderID": "ORD-1001",
            "CustomerID": "CUST-101",
            "Gender": "Male",
            "Age": 28,
            "City": "New York",
            "ProductCategory": "Electronics",
            "ProductName": "Laptop",
            "Quantity": 2,
            "Price": 800,
            "OrderDate": "2024-01-15",
            "PaymentMethod": "Credit Card",
            "ShippingCost": 25,
            "Rating": 4.5,
            "DiscountPercent": 10,
        },
        {
            "OrderID": "ORD-1002",
            "CustomerID": "CUST-102",
            "Gender": "Female",
            "Age": 35,
            "City": "Chicago",
            "ProductCategory": "Home",
            "ProductName": "Lamp",
            "Quantity": 3,
            "Price": -200,
            "OrderDate": "2024-02-20",
            "PaymentMethod": "Debit Card",
            "ShippingCost": 15,
            "Rating": 3,
            "DiscountPercent": 20,
        },
        {
            "OrderID": "ORD-1003",
            "CustomerID": "CUST-103",
            "Gender": "Male",
            "Age": 42,
            "City": "Boston",
            "ProductCategory": "Fashion",
            "ProductName": "Sneakers",
            "Quantity": 1,
            "Price": 120,
            "OrderDate": "2024-03-18",
            "PaymentMethod": "UPI",
            "ShippingCost": 18,
            "Rating": 5,
            "DiscountPercent": 5,
        },
        {
            "OrderID": "ORD-1004",
            "CustomerID": "CUST-104",
            "Gender": "Female",
            "Age": 50,
            "City": "Seattle",
            "ProductCategory": "Grocery",
            "ProductName": "Organic Tea",
            "Quantity": 5,
            "Price": 30,
            "OrderDate": "2024-04-07",
            "PaymentMethod": "Credit Card",
            "ShippingCost": 12,
            "Rating": 4,
            "DiscountPercent": 15,
        },
    ]
    pd.DataFrame(sample_data).to_csv(path, index=False)


def main() -> None:
    """Load the dataset, clean it, and save the processed output."""
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}. Creating a sample dataset for initialization.")
        create_sample_dataset(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)
    print("Number of Rows and Columns:")
    print(df.shape)
    show_head(df, "First 5 Records:")

    cleaned = prepare_dataset(df)
    print("\nFinal Dataset Shape:")
    print(cleaned.shape)
    print("\nFinal Missing Values:")
    print(cleaned.isnull().sum())
    print("\nFinal Duplicate Rows:")
    print(cleaned.duplicated().sum())
    show_head(cleaned, "Prepared Dataset Preview:")

    cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCleaned dataset saved as: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
