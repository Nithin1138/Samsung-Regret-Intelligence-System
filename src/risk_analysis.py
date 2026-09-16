"""
Risk Analysis Module
====================
Risk scoring engine with trend-based growth,
dynamic threshold classification, and temporal intelligence
(acceleration detection, consistent-increase identification).
"""

import pandas as pd
import numpy as np
from src.rsi import (
    calculate_monthly_rsi,
    calculate_rolling_rsi,
    calculate_rsi_trend_slope,
    calculate_review_volume_trend,
    calculate_risk_acceleration,
    calculate_negative_sentiment_trend,
)


def build_risk_scorecard(df, verbose=True):
    """
    Build a comprehensive product risk scorecard using trend-based metrics.

    Risk_Score = 0.35 × Avg_RSI (normalized)
              + 0.25 × RSI_Slope (normalized, trend-based growth)
              + 0.20 × Issue_Density
              + 0.10 × Neg_Sentiment_Ratio
              + 0.10 × Risk_Acceleration (normalized)

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset with RSI, Issue, Sentiment_score, Date, Product.

    Returns
    -------
    pd.DataFrame
        Risk scorecard with one row per product.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Component 1: Average RSI ---
    product_rsi = df.groupby("Product")["RSI"].mean().rename("Avg_RSI")

    # --- Component 2: RSI Trend Slope (replaces endpoint growth) ---
    monthly = calculate_monthly_rsi(df)
    trends = calculate_rsi_trend_slope(monthly)
    trends = trends.set_index("Product")["RSI_Slope"]

    # --- Component 3: Issue Density ---
    issue_density = (
        df[df["Issue"] != "none"].groupby("Product").size()
        / df.groupby("Product").size()
    ).fillna(0).rename("Issue_Density")

    # --- Component 4: Negative Sentiment Ratio ---
    df["Is_Negative"] = (df["Sentiment_score"] < 0).astype(int)
    neg_ratio = df.groupby("Product")["Is_Negative"].mean().rename("Neg_Ratio")

    # --- Component 5: Risk Acceleration ---
    monthly_rolling = calculate_rolling_rsi(monthly)
    accel = calculate_risk_acceleration(monthly_rolling)
    accel = accel.set_index("Product")["Risk_Acceleration"]

    # --- Combine ---
    risk_df = pd.DataFrame({
        "Avg_RSI": product_rsi,
        "RSI_Slope": trends,
        "Issue_Density": issue_density,
        "Neg_Ratio": neg_ratio,
        "Risk_Acceleration": accel,
    }).fillna(0)

    # --- Normalize each component to [0, 1] ---
    def _minmax(series):
        rng = series.max() - series.min()
        if rng == 0:
            return pd.Series(0.0, index=series.index)
        return (series - series.min()) / rng

    risk_df["Avg_RSI_norm"] = _minmax(risk_df["Avg_RSI"])
    risk_df["RSI_Slope_norm"] = _minmax(risk_df["RSI_Slope"])
    risk_df["Issue_Density_norm"] = _minmax(risk_df["Issue_Density"])
    risk_df["Neg_Ratio_norm"] = _minmax(risk_df["Neg_Ratio"])
    risk_df["Accel_norm"] = _minmax(risk_df["Risk_Acceleration"])

    # --- Weighted Risk Score ---
    risk_df["Risk_Score"] = (
        0.35 * risk_df["Avg_RSI_norm"]
        + 0.25 * risk_df["RSI_Slope_norm"]
        + 0.20 * risk_df["Issue_Density_norm"]
        + 0.10 * risk_df["Neg_Ratio_norm"]
        + 0.10 * risk_df["Accel_norm"]
    )

    # --- Dynamic Threshold Classification ---
    q1 = risk_df["Risk_Score"].quantile(0.25)
    q2 = risk_df["Risk_Score"].quantile(0.50)
    q3 = risk_df["Risk_Score"].quantile(0.75)

    def _classify(score):
        if score <= q1:
            return "Low"
        elif score <= q2:
            return "Moderate"
        elif score <= q3:
            return "High"
        else:
            return "Critical"

    risk_df["Risk_Level"] = risk_df["Risk_Score"].apply(_classify)
    risk_df = risk_df.sort_values("Risk_Score", ascending=False)

    if verbose:
        print(f"\n{'='*60}")
        print("  PRODUCT RISK SCORECARD")
        print(f"{'='*60}")
        print(f"  Total products: {len(risk_df)}")
        print(f"  Risk thresholds: Low ≤ {q1:.4f}, Moderate ≤ {q2:.4f}, High ≤ {q3:.4f}, Critical > {q3:.4f}")
        print(f"\n  Risk Level Distribution:")
        print(f"  {risk_df['Risk_Level'].value_counts().to_dict()}")
        print(f"\n  Top 5 Critical Products:")
        top5 = risk_df.head(5)[["Avg_RSI", "RSI_Slope", "Issue_Density", "Risk_Score", "Risk_Level"]]
        print(top5.to_string())

    return risk_df


def identify_consistently_worsening_products(monthly_df, min_consecutive=3, product_col="Product"):
    """
    Identify products whose RSI has been increasing for N+ consecutive months.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Output from calculate_monthly_rsi().
    min_consecutive : int
        Minimum number of consecutive months of RSI increase.

    Returns
    -------
    pd.DataFrame
        Products with their longest streak of consecutive RSI increases.
    """
    df = monthly_df.sort_values([product_col, "Month"]).copy()
    df["RSI_Change"] = df.groupby(product_col)["Monthly_RSI"].diff()
    df["Increasing"] = (df["RSI_Change"] > 0).astype(int)

    def _max_streak(group):
        streak = 0
        max_streak = 0
        for val in group["Increasing"]:
            if val == 1:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    streaks = (
        df.groupby(product_col)
        .apply(_max_streak, include_groups=False)
        .reset_index(name="Max_Consecutive_Increase")
    )
    streaks = streaks[streaks["Max_Consecutive_Increase"] >= min_consecutive]
    return streaks.sort_values("Max_Consecutive_Increase", ascending=False)
