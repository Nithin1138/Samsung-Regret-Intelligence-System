"""
Data Quality Validation Module
==============================
Validates raw data before processing.
Checks for missing values, duplicates, invalid ratings,
empty text, and invalid platform/product values.
Produces a structured quality report with exact counts.
"""

import pandas as pd
import numpy as np


def validate_ratings(df, rating_col="Rating"):
    """Identify rows with ratings outside the valid 1–5 range."""
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    invalid_mask = (df[rating_col] < 1) | (df[rating_col] > 5) | df[rating_col].isna()
    return invalid_mask


def validate_text(df, text_col="Review"):
    """Identify rows with empty or whitespace-only review text."""
    empty_mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
    return empty_mask


def validate_platforms(df, platform_col="Platform", valid_platforms=None):
    """Identify rows with invalid platform values (case-insensitive)."""
    if valid_platforms is None:
        valid_platforms = {"amazon", "flipkart"}
    normalized = df[platform_col].astype(str).str.strip().str.lower()
    invalid_mask = ~normalized.isin(valid_platforms)
    return invalid_mask


def validate_dates(df, date_col="Date"):
    """Identify rows with invalid or missing dates."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    invalid_mask = dates.isna()
    return invalid_mask


def find_duplicates(df, subset=None):
    """Identify duplicate rows based on subset columns."""
    if subset is None:
        subset = ["Product", "Review"]
    dup_mask = df.duplicated(subset=subset, keep="first")
    return dup_mask


def run_quality_report(df, verbose=True):
    """
    Run a full data quality validation and return a structured report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset to validate.
    verbose : bool
        If True, print the report to stdout.

    Returns
    -------
    dict
        Quality report with counts for each check.
    """
    total_raw = len(df)

    # --- Checks ---
    dup_mask = find_duplicates(df)
    invalid_rating_mask = validate_ratings(df.copy())
    empty_text_mask = validate_text(df)
    invalid_platform_mask = validate_platforms(df)
    invalid_date_mask = validate_dates(df)
    missing_product = df["Product"].isna() | (df["Product"].astype(str).str.strip() == "")

    report = {
        "total_raw_records": total_raw,
        "duplicate_reviews": int(dup_mask.sum()),
        "invalid_ratings": int(invalid_rating_mask.sum()),
        "empty_review_text": int(empty_text_mask.sum()),
        "invalid_platforms": int(invalid_platform_mask.sum()),
        "invalid_dates": int(invalid_date_mask.sum()),
        "missing_products": int(missing_product.sum()),
    }

    # Combined invalid (union of all issues)
    any_issue = dup_mask | invalid_rating_mask | empty_text_mask | invalid_platform_mask | invalid_date_mask | missing_product
    report["total_invalid_records"] = int(any_issue.sum())
    report["estimated_clean_records"] = total_raw - report["total_invalid_records"]

    if verbose:
        print("=" * 50)
        print("     DATA QUALITY REPORT")
        print("=" * 50)
        print(f"  Raw records:           {report['total_raw_records']:,}")
        print(f"  Duplicate reviews:     {report['duplicate_reviews']:,}")
        print(f"  Invalid ratings:       {report['invalid_ratings']:,}")
        print(f"  Empty review text:     {report['empty_review_text']:,}")
        print(f"  Invalid platforms:     {report['invalid_platforms']:,}")
        print(f"  Invalid dates:         {report['invalid_dates']:,}")
        print(f"  Missing products:      {report['missing_products']:,}")
        print("-" * 50)
        print(f"  Total invalid:         {report['total_invalid_records']:,}")
        print(f"  Estimated clean:       {report['estimated_clean_records']:,}")
        print("=" * 50)

    return report
