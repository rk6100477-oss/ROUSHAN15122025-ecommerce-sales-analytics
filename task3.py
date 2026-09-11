# ============================================================
# TASK 3: DATASET SELECTION & DATA PREPARATION
# Domain: E-Commerce Sales
# ============================================================

from pathlib import Path

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import pandas as pd  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

try:
    from IPython.display import display  # pyright: ignore[reportMissingModuleSource]
except ImportError:
    def display(obj=None, *args, **kwargs):
        if obj is not None:
            print(obj)

DATASET_PATH = Path(__file__).resolve().with_name("ecommerce_sales.csv")
OUTPUT_PATH = Path(__file__).resolve().with_name("ecommerce_sales_cleaned.csv")


def create_sample_dataset(path: Path) -> None:
    """Create a minimal demo dataset if the CSV file is missing."""
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
    ]
    pd.DataFrame(sample_data).to_csv(path, index=False)


# ------------------------------------------------------------
# 3.1 Dataset Selection & Loading
# ------------------------------------------------------------

if not DATASET_PATH.exists():
    print(f"Dataset not found at {DATASET_PATH}. Creating a sample dataset.")
    create_sample_dataset(DATASET_PATH)

# Load the E-Commerce Sales dataset
df = pd.read_csv(DATASET_PATH)

# Display first 5 records
print("First 5 Records:")
display(df.head())

# ------------------------------------------------------------
# 3.2 Data Inspection
# ------------------------------------------------------------

print("Number of Rows and Columns:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
display(df.head())

# ------------------------------------------------------------
# 3.3 Identification of Columns and Data Types
# ------------------------------------------------------------

print("Columns and Data Types:")
print(df.dtypes)

print("\nDetailed Column Information:")
column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Non-Null Values": df.notnull().sum(),
    "Missing Values": df.isnull().sum()
})

display(column_info)

# ------------------------------------------------------------
# 3.4 Identification of Missing Values
# ------------------------------------------------------------

print("Missing Values in Each Column:")
missing_values = df.isnull().sum()
print(missing_values)

print("\nMissing Value Percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.round(2))

# ------------------------------------------------------------
# 3.5 Handling Missing Values
# ------------------------------------------------------------

# Convert OrderDate into datetime format
df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

# Numerical columns: fill missing values with median
numerical_columns = ["Age", "ShippingCost", "Rating"]

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

# Categorical columns: fill missing values with mode
categorical_columns = ["City", "PaymentMethod"]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("Missing Values After Treatment:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 3.6 Removal of Duplicate Records
# ------------------------------------------------------------

print("Duplicate Rows Before Removal:", df.duplicated().sum())

# Remove exact duplicate rows
df = df.drop_duplicates()

print("Duplicate Rows After Removal:", df.duplicated().sum())

# Remove duplicate OrderID records
print("Duplicate OrderIDs Before Removal:",
      df["OrderID"].duplicated().sum())

df = df.drop_duplicates(subset="OrderID", keep="first")

print("Duplicate OrderIDs After Removal:",
      df["OrderID"].duplicated().sum())

# ------------------------------------------------------------
# 3.7 Removal of Incorrect Records
# ------------------------------------------------------------

# Remove records with Quantity <= 0
df = df[df["Quantity"] > 0]

# Correct negative Price values
df["Price"] = df["Price"].abs()

# Keep only valid ratings between 1 and 5
df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]

# Keep only valid DiscountPercent between 0 and 100
df = df[
    (df["DiscountPercent"] >= 0) &
    (df["DiscountPercent"] <= 100)
]

# ------------------------------------------------------------
# 3.8 Standardization of Text Data
# ------------------------------------------------------------

text_columns = [
    "Gender",
    "City",
    "ProductCategory",
    "ProductName",
    "PaymentMethod"
]

for column in text_columns:
    df[column] = df[column].astype(str).str.strip().str.title()

# ------------------------------------------------------------
# 3.9 Preparation of Dataset for Analysis
# ------------------------------------------------------------

# Calculate Total Amount after discount
df["TotalAmount"] = (
    df["Price"] *
    df["Quantity"] *
    (1 - df["DiscountPercent"] / 100)
)

# Create Order Month and Order Year
df["OrderMonth"] = df["OrderDate"].dt.month
df["OrderYear"] = df["OrderDate"].dt.year

# Create Age Groups
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=["Below 25", "26-35", "36-45", "46-55", "56+"]
)

# IQR Capping for Price
Q1_price = df["Price"].quantile(0.25)
Q3_price = df["Price"].quantile(0.75)
IQR_price = Q3_price - Q1_price

lower_price = Q1_price - 1.5 * IQR_price
upper_price = Q3_price + 1.5 * IQR_price

df["Price_Capped"] = df["Price"].clip(
    lower=lower_price,
    upper=upper_price
)

# IQR Capping for Quantity
Q1_quantity = df["Quantity"].quantile(0.25)
Q3_quantity = df["Quantity"].quantile(0.75)
IQR_quantity = Q3_quantity - Q1_quantity

lower_quantity = Q1_quantity - 1.5 * IQR_quantity
upper_quantity = Q3_quantity + 1.5 * IQR_quantity

df["Quantity_Capped"] = df["Quantity"].clip(
    lower=lower_quantity,
    upper=upper_quantity
)

# ------------------------------------------------------------
# 3.10 Final Dataset Inspection
# ------------------------------------------------------------

print("Final Dataset Shape:")
print(df.shape)

print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nFinal Duplicate Rows:")
print(df.duplicated().sum())

print("\nFinal Dataset:")
display(df.head())

# ------------------------------------------------------------
# 3.11 Save Prepared Dataset
# ------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned dataset saved as: {OUTPUT_PATH.name}")