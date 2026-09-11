# ============================================================
# TASK 4: EXPLORATORY DATA ANALYSIS (EDA)
# Dataset: E-Commerce Sales
# ============================================================

import importlib
from typing import Any

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def display(obj: Any) -> None:
    try:
        ipython_display = importlib.import_module("IPython.display")
        ipython_display.display(obj)
    except Exception:
        print(obj)

# ------------------------------------------------------------
# 4.1 Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("ecommerce_sales.csv")

print("Initial Dataset Shape:", df.shape)

# ------------------------------------------------------------
# 4.2 Data Inspection
# ------------------------------------------------------------


def available_columns(columns):
    return [column for column in columns if column in df.columns]


print("\nFirst 5 Rows:")
display(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nDescriptive Statistics:")
display(df.describe(include="all"))

# ------------------------------------------------------------
# 4.3 Data Cleaning
# ------------------------------------------------------------

# Convert OrderDate to datetime
df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

# Remove exact duplicate rows
print("\nExact Duplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Remove duplicate OrderIDs
print("Duplicate OrderIDs:", df["OrderID"].duplicated().sum())
df = df.drop_duplicates(subset="OrderID", keep="first")

# Remove incorrect Quantity values
df = df[df["Quantity"] > 0]

# Correct negative Price values
df["Price"] = df["Price"].abs()

# Keep valid Rating values
df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]

# Keep valid DiscountPercent values
df = df[
    (df["DiscountPercent"] >= 0) &
    (df["DiscountPercent"] <= 100)
]

# Standardize categorical/text columns
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
# 4.4 Missing-Value Handling
# ------------------------------------------------------------

print("\nMissing Values Before Treatment:")
display(df.isnull().sum())

# Numerical missing values → median
for column in ["Age", "ShippingCost", "Rating"]:
    df[column] = df[column].fillna(df[column].median())

# Categorical missing values → mode
for column in ["City", "PaymentMethod"]:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Treatment:")
display(df.isnull().sum())

# ------------------------------------------------------------
# 4.5 Feature Engineering
# ------------------------------------------------------------

# Total Amount after discount
df["TotalAmount"] = (
    df["Price"] *
    df["Quantity"] *
    (1 - df["DiscountPercent"] / 100)
)

# Date-based features
df["OrderMonth"] = df["OrderDate"].dt.month
df["OrderYear"] = df["OrderDate"].dt.year

# Age groups
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=["Below 25", "26-35", "36-45", "46-55", "56+"]
)

# ------------------------------------------------------------
# 4.6 Descriptive Statistics
# ------------------------------------------------------------

print("\nNumerical Descriptive Statistics:")
numerical_columns = available_columns([
    "Age",
    "Price",
    "Quantity",
    "DiscountPercent",
    "ShippingCost",
    "DeliveryDays",
    "Rating",
    "TotalAmount"
])

if not numerical_columns:
    raise ValueError("No numeric columns are available for analysis.")

display(df[numerical_columns].describe().round(2))

print("\nMean Values:")
display(df[numerical_columns].mean().round(2))

print("\nMedian Values:")
display(df[numerical_columns].median().round(2))

# ------------------------------------------------------------
# 4.7 Correlation Analysis
# ------------------------------------------------------------

correlation_columns = available_columns([
    "Age",
    "Price",
    "Quantity",
    "DiscountPercent",
    "ShippingCost",
    "DeliveryDays",
    "Rating",
    "TotalAmount"
])

if not correlation_columns:
    raise ValueError("No correlation columns are available for analysis.")

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation Matrix:")
display(correlation_matrix.round(2))

# Price and TotalAmount correlation
print(
    "\nCorrelation between Price and TotalAmount:",
    round(df["Price"].corr(df["TotalAmount"]), 2)
)

# Quantity and TotalAmount correlation
print(
    "Correlation between Quantity and TotalAmount:",
    round(df["Quantity"].corr(df["TotalAmount"]), 2)
)

# ------------------------------------------------------------
# 4.8 Important Patterns and Trends
# ------------------------------------------------------------

# Revenue by Product Category
category_revenue = (
    df.groupby("ProductCategory")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nRevenue by Product Category:")
display(category_revenue.round(2))

# Average Order Value by Product Category
category_aov = (
    df.groupby("ProductCategory")["TotalAmount"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Order Value by Product Category:")
display(category_aov.round(2))

# Monthly Revenue
monthly_revenue = (
    df.groupby(["OrderYear", "OrderMonth"])["TotalAmount"]
    .sum()
    .reset_index()
)

monthly_revenue["YearMonth"] = (
    monthly_revenue["OrderYear"].astype(str)
    + "-"
    + monthly_revenue["OrderMonth"].astype(str).str.zfill(2)
)

print("\nMonthly Revenue:")
display(monthly_revenue)

# Revenue by City
city_revenue = (
    df.groupby("City")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Cities by Revenue:")
display(city_revenue.head(10).round(2))

# Average Order Value by Age Group
agegroup_aov = (
    df.groupby("AgeGroup", observed=False)["TotalAmount"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Order Value by Age Group:")
display(agegroup_aov.round(2))

# Revenue by Gender
gender_revenue = (
    df.groupby("Gender")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nRevenue by Gender:")
display(gender_revenue.round(2))

# Average Rating by Product Category
category_rating = (
    df.groupby("ProductCategory")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Rating by Product Category:")
display(category_rating.round(2))

# Payment Method Usage
payment_usage = (
    df["PaymentMethod"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPayment Method Share (%):")
display(payment_usage)

# ------------------------------------------------------------
# 4.9 Outlier Identification Using IQR
# ------------------------------------------------------------

def find_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]

    return Q1, Q3, IQR, lower_limit, upper_limit, outliers


outlier_columns = available_columns([
    "Price",
    "Quantity",
    "TotalAmount",
    "ShippingCost",
    "DeliveryDays"
])

print("\nOutlier Analysis:")

for column in outlier_columns:
    Q1, Q3, IQR, lower, upper, outliers = find_outliers(df, column)

    print(f"\n{column}")
    print("Q1:", round(Q1, 2))
    print("Q3:", round(Q3, 2))
    print("IQR:", round(IQR, 2))
    print("Lower Limit:", round(lower, 2))
    print("Upper Limit:", round(upper, 2))
    print("Number of Outliers:", len(outliers))
    print("Outlier Percentage:",
          round((len(outliers) / len(df)) * 100, 2), "%")

# ------------------------------------------------------------
# 4.10 IQR Outlier Treatment
# ------------------------------------------------------------

# Price outlier treatment
Q1_price = df["Price"].quantile(0.25)
Q3_price = df["Price"].quantile(0.75)
IQR_price = Q3_price - Q1_price

lower_price = Q1_price - 1.5 * IQR_price
upper_price = Q3_price + 1.5 * IQR_price

df["Price_Capped"] = df["Price"].clip(
    lower=lower_price,
    upper=upper_price
)

# Quantity outlier treatment
Q1_quantity = df["Quantity"].quantile(0.25)
Q3_quantity = df["Quantity"].quantile(0.75)
IQR_quantity = Q3_quantity - Q1_quantity

lower_quantity = Q1_quantity - 1.5 * IQR_quantity
upper_quantity = Q3_quantity + 1.5 * IQR_quantity

df["Quantity_Capped"] = df["Quantity"].clip(
    lower=lower_quantity,
    upper=upper_quantity
)

print("\nOutlier Treatment Completed.")

# ------------------------------------------------------------
# 4.11 Distribution / Skewness Analysis
# ------------------------------------------------------------

print("\nSkewness of Numerical Variables:")
skewness = df[numerical_columns].skew().sort_values(ascending=False)
display(skewness.round(2))

# ------------------------------------------------------------
# 4.12 Key Findings Automatically Generated
# ------------------------------------------------------------

top_category = category_revenue.idxmax()
top_category_revenue = category_revenue.max()

top_city = city_revenue.idxmax()
peak_month = monthly_revenue.loc[
    monthly_revenue["TotalAmount"].idxmax(),
    "YearMonth"
]

highest_age_group = agegroup_aov.idxmax()

price_total_corr = df["Price"].corr(df["TotalAmount"])
quantity_total_corr = df["Quantity"].corr(df["TotalAmount"])

print("\n================ KEY FINDINGS ================")

print(
    f"1. {top_category} generated the highest total revenue "
    f"of approximately ₹{top_category_revenue:,.2f}."
)

print(
    f"2. The highest monthly revenue was recorded in {peak_month}."
)

print(
    f"3. {top_city} was the top revenue-generating city."
)

print(
    f"4. The {highest_age_group} age group had the highest average order value."
)

print(
    f"5. Price and TotalAmount showed a correlation of "
    f"{price_total_corr:.2f}."
)

print(
    f"6. Quantity and TotalAmount showed a correlation of "
    f"{quantity_total_corr:.2f}."
)

print(
    f"7. The dataset contains {len(df)} records after cleaning."
)

print(
    "8. Missing values were handled using median imputation "
    "for numerical variables and mode imputation for categorical variables."
)

print(
    "9. Outliers were identified using the IQR method and "
    "Price/Quantity values were capped for analysis."
)

# ------------------------------------------------------------
# 4.13 Final Dataset Check
# ------------------------------------------------------------

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

print("\nRemaining Duplicate Rows:")
print(df.duplicated().sum())

# Save EDA-ready dataset
df.to_csv("ecommerce_sales_eda_ready.csv", index=False)

print("\nEDA-ready dataset saved as: ecommerce_sales_eda_ready.csv")