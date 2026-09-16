"""
Data Warehouse Module
=====================
Star schema builder and SQL-style analytical queries using pandas.
Demonstrates joins, aggregations, CTEs (method chaining),
window functions (.rank(), .shift()), and time-based analysis.
"""

import pandas as pd
import numpy as np


# ─── Star Schema Builder ────────────────────────────────────────────


def create_dim_product(df):
    """Create the Product dimension table."""
    dim = df[["Product", "Category"]].drop_duplicates().reset_index(drop=True)
    dim["Product_ID"] = range(1, len(dim) + 1)
    return dim


def create_dim_platform(df):
    """Create the Platform dimension table."""
    dim = df[["Platform"]].drop_duplicates().reset_index(drop=True)
    dim["Platform_ID"] = range(1, len(dim) + 1)
    return dim


def create_dim_issue(df):
    """Create the Issue dimension table."""
    dim = df[["Issue"]].drop_duplicates().reset_index(drop=True)
    dim["Issue_ID"] = range(1, len(dim) + 1)
    return dim


def create_dim_date(df, date_col="Date"):
    """Create the Date dimension table with Year, Month, Day, Quarter."""
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    dim = df_copy[[date_col]].drop_duplicates().reset_index(drop=True)
    dim["Date_ID"] = range(1, len(dim) + 1)
    dim["Year"] = dim[date_col].dt.year
    dim["Month"] = dim[date_col].dt.month
    dim["Day"] = dim[date_col].dt.day
    dim["Quarter"] = dim[date_col].dt.quarter
    return dim


def create_fact_review(df, dim_product, dim_platform, dim_issue, dim_date):
    """
    Create the Fact table by merging dimension IDs.

    Returns
    -------
    pd.DataFrame
        Fact table with foreign keys and measures.
    """
    fact = df.copy()
    fact["Date"] = pd.to_datetime(fact["Date"])
    fact["Review_ID"] = range(1, len(fact) + 1)

    # Merge dimension keys
    fact = fact.merge(dim_product, on=["Product", "Category"], how="left")
    fact = fact.merge(dim_platform, on="Platform", how="left")
    fact = fact.merge(dim_issue, on="Issue", how="left")
    fact = fact.merge(dim_date, on="Date", how="left")

    # Select final columns (keys + measures)
    fact = fact[[
        "Review_ID",
        "Product_ID",
        "Platform_ID",
        "Issue_ID",
        "Date_ID",
        "Rating",
        "Sentiment_score",
        "RSI",
    ]]

    return fact


def build_star_schema(df):
    """
    Build the complete star schema from the processed dataset.

    Returns
    -------
    dict
        Keys: 'fact_review', 'dim_product', 'dim_platform', 'dim_issue', 'dim_date'.
    """
    dim_product = create_dim_product(df)
    dim_platform = create_dim_platform(df)
    dim_issue = create_dim_issue(df)
    dim_date = create_dim_date(df)
    fact_review = create_fact_review(df, dim_product, dim_platform, dim_issue, dim_date)

    schema = {
        "fact_review": fact_review,
        "dim_product": dim_product,
        "dim_platform": dim_platform,
        "dim_issue": dim_issue,
        "dim_date": dim_date,
    }

    print(f"Star Schema built:")
    for name, table in schema.items():
        print(f"  {name}: {table.shape[0]:,} rows × {table.shape[1]} columns")

    return schema


def save_warehouse(schema, base_path):
    """Save all warehouse tables to CSV."""
    for name, table in schema.items():
        path = f"{base_path}/{name}.csv"
        table.to_csv(path, index=False)
    print(f"✅ Warehouse tables saved to {base_path}")


# ─── SQL-Style Analytical Queries ────────────────────────────────────


def query_product_risk_ranking(schema):
    """
    SQL equivalent:
        SELECT p.Product, p.Category, AVG(f.RSI) AS Avg_RSI,
               RANK() OVER (ORDER BY AVG(f.RSI) DESC) AS Risk_Rank
        FROM fact_review f
        JOIN dim_product p ON f.Product_ID = p.Product_ID
        GROUP BY p.Product, p.Category
        ORDER BY Risk_Rank
    """
    fact = schema["fact_review"]
    dim_p = schema["dim_product"]

    result = (
        fact.merge(dim_p, on="Product_ID")
        .groupby(["Product", "Category"])
        .agg(Avg_RSI=("RSI", "mean"), Review_Count=("RSI", "count"))
        .reset_index()
        .sort_values("Avg_RSI", ascending=False)
    )
    result["Risk_Rank"] = result["Avg_RSI"].rank(ascending=False, method="dense").astype(int)
    return result


def query_quarterly_rsi_change(schema):
    """
    SQL equivalent:
        WITH quarterly AS (
            SELECT d.Year, d.Quarter, AVG(f.RSI) AS Avg_RSI
            FROM fact_review f JOIN dim_date d ON f.Date_ID = d.Date_ID
            GROUP BY d.Year, d.Quarter
        )
        SELECT *, Avg_RSI - LAG(Avg_RSI) OVER (ORDER BY Year, Quarter) AS QoQ_Change
        FROM quarterly
    """
    fact = schema["fact_review"]
    dim_d = schema["dim_date"]

    quarterly = (
        fact.merge(dim_d, on="Date_ID")
        .groupby(["Year", "Quarter"])
        .agg(Avg_RSI=("RSI", "mean"))
        .reset_index()
        .sort_values(["Year", "Quarter"])
    )
    quarterly["QoQ_Change"] = quarterly["Avg_RSI"].diff()
    return quarterly


def query_platform_category_crosstab(schema):
    """
    SQL equivalent:
        SELECT pl.Platform, p.Category,
               AVG(f.RSI) AS Avg_RSI, COUNT(*) AS N
        FROM fact_review f
        JOIN dim_product p ON f.Product_ID = p.Product_ID
        JOIN dim_platform pl ON f.Platform_ID = pl.Platform_ID
        GROUP BY pl.Platform, p.Category
    """
    fact = schema["fact_review"]
    dim_p = schema["dim_product"]
    dim_pl = schema["dim_platform"]

    result = (
        fact.merge(dim_p, on="Product_ID")
        .merge(dim_pl, on="Platform_ID")
        .groupby(["Platform", "Category"])
        .agg(Avg_RSI=("RSI", "mean"), N=("RSI", "count"))
        .reset_index()
    )
    return result


def query_products_above_avg_risk(schema):
    """
    SQL equivalent (subquery pattern):
        SELECT p.Product, AVG(f.RSI) AS Avg_RSI
        FROM fact_review f JOIN dim_product p ON f.Product_ID = p.Product_ID
        GROUP BY p.Product
        HAVING AVG(f.RSI) > (SELECT AVG(RSI) FROM fact_review)
        ORDER BY Avg_RSI DESC
    """
    fact = schema["fact_review"]
    dim_p = schema["dim_product"]

    global_avg = fact["RSI"].mean()

    product_avg = (
        fact.merge(dim_p, on="Product_ID")
        .groupby("Product")
        .agg(Avg_RSI=("RSI", "mean"))
        .reset_index()
    )
    above_avg = product_avg[product_avg["Avg_RSI"] > global_avg].sort_values(
        "Avg_RSI", ascending=False
    )
    above_avg["Pct_Above_Avg"] = ((above_avg["Avg_RSI"] - global_avg) / global_avg * 100).round(2)

    return above_avg, global_avg


def query_issue_impact_by_platform(schema):
    """
    SQL equivalent:
        SELECT pl.Platform, i.Issue,
               AVG(f.RSI) AS Avg_RSI,
               COUNT(*) AS N,
               RANK() OVER (PARTITION BY pl.Platform ORDER BY AVG(f.RSI) DESC) AS Issue_Rank
        FROM fact_review f
        JOIN dim_platform pl ON f.Platform_ID = pl.Platform_ID
        JOIN dim_issue i ON f.Issue_ID = i.Issue_ID
        GROUP BY pl.Platform, i.Issue
    """
    fact = schema["fact_review"]
    dim_pl = schema["dim_platform"]
    dim_i = schema["dim_issue"]

    result = (
        fact.merge(dim_pl, on="Platform_ID")
        .merge(dim_i, on="Issue_ID")
        .groupby(["Platform", "Issue"])
        .agg(Avg_RSI=("RSI", "mean"), N=("RSI", "count"))
        .reset_index()
    )
    result["Issue_Rank"] = (
        result.groupby("Platform")["Avg_RSI"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    return result.sort_values(["Platform", "Issue_Rank"])


def query_monthly_cumulative_reviews(schema):
    """
    SQL equivalent (window function):
        SELECT d.Year, d.Month,
               COUNT(*) AS Monthly_Reviews,
               SUM(COUNT(*)) OVER (ORDER BY d.Year, d.Month) AS Cumulative_Reviews
        FROM fact_review f JOIN dim_date d ON f.Date_ID = d.Date_ID
        GROUP BY d.Year, d.Month
    """
    fact = schema["fact_review"]
    dim_d = schema["dim_date"]

    monthly = (
        fact.merge(dim_d, on="Date_ID")
        .groupby(["Year", "Month"])
        .agg(Monthly_Reviews=("Review_ID", "count"))
        .reset_index()
        .sort_values(["Year", "Month"])
    )
    monthly["Cumulative_Reviews"] = monthly["Monthly_Reviews"].cumsum()
    return monthly
