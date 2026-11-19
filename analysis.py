"""
Data Science Tool Mastery Project
Sample retail sales analysis using Pandas, NumPy, Matplotlib and Seaborn.
Run from this folder: python analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retail_sales_sample.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_and_clean_data(path):
    """Load the CSV, parse dates, remove duplicate rows and handle missing values."""
    data = pd.read_csv(path, parse_dates=["Order_Date"])
    data = data.drop_duplicates()
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Unit_Price"] = pd.to_numeric(data["Unit_Price"], errors="coerce")
    data = data.dropna(subset=["Order_Date", "Product", "Quantity", "Unit_Price"])
    data["Revenue"] = data["Quantity"] * data["Unit_Price"]
    return data

def summarize_data(data):
    """Print useful descriptive statistics and grouped summaries."""
    print("\\n--- Dataset overview ---")
    print(data.info())
    print("\\n--- Descriptive statistics ---")
    print(data[["Quantity", "Unit_Price", "Revenue"]].describe())
    print("\\nTotal revenue:", round(data["Revenue"].sum(), 2))
    print("\\nRevenue by product:")
    print(data.groupby("Product")["Revenue"].sum().sort_values(ascending=False))
    print("\\nRevenue by city:")
    print(data.groupby("City")["Revenue"].sum().sort_values(ascending=False))
    print("\\nCorrelation matrix:")
    print(data[["Quantity", "Unit_Price", "Revenue"]].corr(numeric_only=True))

def create_visualizations(data):
    """Save charts to the outputs folder."""
    sns.set_theme(style="whitegrid")

    product_revenue = data.groupby("Product", as_index=False)["Revenue"].sum()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=product_revenue, x="Product", y="Revenue")
    plt.title("Total Revenue by Product")
    plt.xlabel("Product")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revenue_by_product.png", dpi=150)
    plt.close()

    city_revenue = data.groupby("City", as_index=False)["Revenue"].sum()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=city_revenue, x="City", y="Revenue")
    plt.title("Total Revenue by City")
    plt.xlabel("City")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "revenue_by_city.png", dpi=150)
    plt.close()

    daily = data.groupby("Order_Date", as_index=False)["Revenue"].sum()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=daily, x="Order_Date", y="Revenue", marker="o")
    plt.title("Daily Revenue Trend")
    plt.xlabel("Order date")
    plt.ylabel("Revenue")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "daily_revenue_trend.png", dpi=150)
    plt.close()

def main():
    data = load_and_clean_data(DATA_FILE)
    summarize_data(data)
    create_visualizations(data)
    data.to_csv(OUTPUT_DIR / "cleaned_sales_data.csv", index=False)
    print(f"\\nAnalysis complete. Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
