"""
ETL Pipeline Module
===================
Orchestrates the full ETL flow:
  Load → Validate → Clean → Transform (Sentiment + Issues + RSI) → Save
"""

import pandas as pd
from src.data_quality import run_quality_report
from src.preprocessing import clean_dataset
from src.sentiment import add_sentiment_scores
from src.issue_extraction import add_issue_column
from src.rsi import add_rsi_column


def load_raw_data(filepath):
    """Load raw CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} raw records from {filepath}")
    return df


def run_etl_pipeline(
    input_path,
    output_path=None,
    add_dates=False,
    verbose=True,
):
    """
    Full ETL pipeline.

    Parameters
    ----------
    input_path : str
        Path to raw CSV file.
    output_path : str, optional
        Path to save processed CSV. If None, does not save.
    add_dates : bool
        If True, generates random dates (for datasets missing a Date column).
    verbose : bool
        Print progress and reports.

    Returns
    -------
    pd.DataFrame
        Fully processed dataset with Sentiment_score, Issue, and RSI columns.
    """
    # 1. Load
    df = load_raw_data(input_path)

    # 2. Data Quality Report
    if verbose:
        print("\n--- Data Quality Validation ---")
    run_quality_report(df, verbose=verbose)

    # 3. Clean
    if verbose:
        print("\n--- Preprocessing ---")
    df = clean_dataset(df, verbose=verbose)

    # 4. Add dates if missing
    if add_dates and "Date" not in df.columns:
        from datetime import datetime, timedelta
        import random

        def random_date():
            start = datetime(2022, 1, 1)
            end = datetime(2025, 1, 1)
            return start + timedelta(days=random.randint(0, (end - start).days))

        df["Date"] = [random_date() for _ in range(len(df))]
        if verbose:
            print("  Added random dates (2022-01-01 to 2025-01-01)")

    # 5. Sentiment Analysis
    if verbose:
        print("\n--- Sentiment Analysis ---")
    df = add_sentiment_scores(df)
    if verbose:
        print(f"  Sentiment scores computed for {len(df):,} reviews")

    # 6. Issue Extraction
    if verbose:
        print("\n--- Issue Extraction ---")
    df = add_issue_column(df)
    if verbose:
        issue_dist = df["Issue"].value_counts()
        issues_found = len(df[df["Issue"] != "none"])
        print(f"  Issues identified in {issues_found:,} reviews ({issues_found/len(df)*100:.1f}%)")

    # 7. RSI Calculation
    if verbose:
        print("\n--- RSI Calculation ---")
    df = add_rsi_column(df)
    if verbose:
        print(f"  RSI range: [{df['RSI'].min():.4f}, {df['RSI'].max():.4f}]")
        print(f"  RSI mean:  {df['RSI'].mean():.4f}")

    # 8. Save
    if output_path:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\n✅ ETL COMPLETE — Saved {len(df):,} records to {output_path}")

    return df
