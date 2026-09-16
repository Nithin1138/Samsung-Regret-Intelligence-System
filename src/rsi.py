"""
Regret Severity Index (RSI) Module
===================================
Core RSI computation, monthly aggregation,
rolling window smoothing, and trend slope calculation.
"""

import pandas as pd
import numpy as np
from scipy import stats


def calculate_rsi(row):
    """
    Calculate per-review Regret Severity Index.

    RSI = 0.4 × neg_sentiment + 0.3 × rating_deviation + 0.2 × issue_flag + 0.1 × specificity

    Parameters
    ----------
    row : pd.Series
        Must contain: Sentiment_score, Rating, Issue.

    Returns
    -------
    float
        RSI value clamped to [0, 1].
    """
    sentiment = row["Sentiment_score"]
    rating = row["Rating"]

    neg_sent = max(0, -sentiment)
    rating_dev = abs(5 - rating) / 4
    keyword_intensity = 1 if row["Issue"] != "none" else 0
    specificity = 0.6  # fixed for consistency

    rsi = (
        0.4 * neg_sent
        + 0.3 * rating_dev
        + 0.2 * keyword_intensity
        + 0.1 * specificity
    )

    return min(max(rsi, 0), 1)


def add_rsi_column(df, rsi_col="RSI"):
    """Add per-review RSI column to the DataFrame."""
    df[rsi_col] = df.apply(calculate_rsi, axis=1)
    return df


# ─── Temporal RSI Metrics ───────────────────────────────────────────


def calculate_monthly_rsi(df, product_col="Product", date_col="Date", rsi_col="RSI"):
    """
    Calculate monthly average RSI per product.

    Returns
    -------
    pd.DataFrame
        Columns: Product, Month (Period), Monthly_RSI, Review_Count.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["Month"] = df[date_col].dt.to_period("M")

    monthly = (
        df.groupby([product_col, "Month"])
        .agg(
            Monthly_RSI=(rsi_col, "mean"),
            Review_Count=(rsi_col, "count"),
        )
        .reset_index()
    )

    return monthly


def calculate_rolling_rsi(monthly_df, window=3, product_col="Product"):
    """
    Calculate rolling average RSI over a specified window (default 3 months).

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Output from calculate_monthly_rsi().
    window : int
        Rolling window size in months.

    Returns
    -------
    pd.DataFrame
        With added Rolling_RSI column.
    """
    monthly_df = monthly_df.sort_values([product_col, "Month"])
    monthly_df["Rolling_RSI"] = (
        monthly_df.groupby(product_col)["Monthly_RSI"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    return monthly_df


def calculate_rsi_trend_slope(monthly_df, product_col="Product"):
    """
    Calculate the linear regression slope of monthly RSI per product.
    Positive slope = worsening regret trend.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Output from calculate_monthly_rsi().

    Returns
    -------
    pd.DataFrame
        Columns: Product, RSI_Slope, RSI_Intercept, R_Squared, N_Months.
    """

    def _slope(group):
        if len(group) < 2:
            return pd.Series({
                "RSI_Slope": 0.0,
                "RSI_Intercept": group["Monthly_RSI"].mean(),
                "R_Squared": 0.0,
                "N_Months": len(group),
            })
        x = np.arange(len(group))
        y = group["Monthly_RSI"].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return pd.Series({
            "RSI_Slope": slope,
            "RSI_Intercept": intercept,
            "R_Squared": r_value ** 2,
            "N_Months": len(group),
        })

    trends = (
        monthly_df.sort_values([product_col, "Month"])
        .groupby(product_col)
        .apply(_slope, include_groups=False)
        .reset_index()
    )

    return trends


def calculate_negative_sentiment_trend(df, product_col="Product", date_col="Date"):
    """
    Calculate the rolling 3-month negative sentiment ratio trend per product.

    Returns
    -------
    pd.DataFrame
        Columns: Product, Month, Neg_Ratio, Rolling_Neg_Ratio.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["Month"] = df[date_col].dt.to_period("M")
    df["Is_Negative"] = (df["Sentiment_score"] < 0).astype(int)

    monthly_neg = (
        df.groupby([product_col, "Month"])
        .agg(Neg_Ratio=("Is_Negative", "mean"))
        .reset_index()
        .sort_values([product_col, "Month"])
    )

    monthly_neg["Rolling_Neg_Ratio"] = (
        monthly_neg.groupby(product_col)["Neg_Ratio"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    return monthly_neg


def calculate_review_volume_trend(monthly_df, product_col="Product"):
    """
    Calculate the slope of review volume over time per product.

    Returns
    -------
    pd.DataFrame
        Columns: Product, Volume_Slope.
    """

    def _vol_slope(group):
        if len(group) < 2:
            return 0.0
        x = np.arange(len(group))
        y = group["Review_Count"].values.astype(float)
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope

    vol_trends = (
        monthly_df.sort_values([product_col, "Month"])
        .groupby(product_col)
        .apply(_vol_slope, include_groups=False)
        .reset_index(name="Volume_Slope")
    )

    return vol_trends


def calculate_risk_acceleration(monthly_df, product_col="Product"):
    """
    Calculate risk acceleration (second derivative of RSI trend).
    Positive acceleration = risk is increasing faster.

    Uses the slope of the rolling RSI's first differences.

    Returns
    -------
    pd.DataFrame
        Columns: Product, Risk_Acceleration.
    """
    df = monthly_df.copy().sort_values([product_col, "Month"])
    df["RSI_Diff"] = df.groupby(product_col)["Monthly_RSI"].diff()

    def _accel(group):
        diffs = group["RSI_Diff"].dropna()
        if len(diffs) < 2:
            return 0.0
        x = np.arange(len(diffs))
        slope, _, _, _, _ = stats.linregress(x, diffs.values)
        return slope

    accel = (
        df.groupby(product_col)
        .apply(_accel, include_groups=False)
        .reset_index(name="Risk_Acceleration")
    )

    return accel
