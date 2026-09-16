"""
Preprocessing Module
====================
Text cleaning, deduplication, rating validation,
missing value handling, and data normalization.
"""

import pandas as pd
import numpy as np
import re


def clean_text(text):
    """
    Clean review text: lowercase, strip whitespace,
    remove special characters (keep alphanumeric and basic punctuation).
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s.,!?'-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_rating(rating_val):
    """
    Coerce rating to numeric float.
    Handles string formats like '4 out of 5', '3.5/5', etc.
    Returns NaN for truly unparseable values.
    """
    if pd.isna(rating_val):
        return np.nan
    s = str(rating_val).strip()
    match = re.search(r"(\d+\.?\d*)", s)
    if match:
        val = float(match.group(1))
        if 1 <= val <= 5:
            return val
    return np.nan


def remove_duplicates(df, subset=None, verbose=True):
    """
    Remove duplicate rows. Reports count of removed duplicates.

    Parameters
    ----------
    df : pd.DataFrame
    subset : list, optional
        Columns to check for duplicates. Default: ['Product', 'Review'].

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    if subset is None:
        subset = ["Product", "Review"]

    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=subset)
    after = len(df)

    if verbose:
        print(f"  Duplicates removed: {before - after:,}")

    return df


def handle_missing_values(df, verbose=True):
    """
    Drop rows missing critical fields (Product, Review, Rating).
    Reports the count of dropped rows.
    """
    before = len(df)
    df = df.dropna(subset=["Product", "Review", "Rating"])
    after = len(df)

    if verbose:
        print(f"  Missing-value rows removed: {before - after:,}")

    return df


def fix_ratings_column(df, rating_col="Rating", verbose=True):
    """
    Parse and validate the Rating column.
    Coerces to float, drops rows where rating is invalid.
    """
    df[rating_col] = df[rating_col].apply(fix_rating)
    invalid_count = int(df[rating_col].isna().sum())

    if verbose and invalid_count > 0:
        print(f"  Invalid ratings dropped: {invalid_count:,}")

    df = df.dropna(subset=[rating_col])
    df[rating_col] = df[rating_col].astype(float)
    return df


def normalize_text_columns(df):
    """Normalize Product and Platform columns: strip + title case."""
    if "Product" in df.columns:
        df["Product"] = df["Product"].astype(str).str.strip()
    if "Platform" in df.columns:
        df["Platform"] = df["Platform"].astype(str).str.strip()
    return df


def clean_dataset(df, verbose=True):
    """
    Full preprocessing pipeline:
    1. Normalize text columns
    2. Remove duplicates
    3. Handle missing values
    4. Fix ratings
    5. Clean review text

    Returns the cleaned DataFrame.
    """
    if verbose:
        print(f"\n{'='*50}")
        print("     PREPROCESSING PIPELINE")
        print(f"{'='*50}")
        print(f"  Starting records: {len(df):,}")

    df = normalize_text_columns(df)
    df = remove_duplicates(df, verbose=verbose)
    df = handle_missing_values(df, verbose=verbose)
    df = fix_ratings_column(df, verbose=verbose)
    df["Review"] = df["Review"].apply(clean_text)

    # Remove rows with empty reviews after cleaning
    df = df[df["Review"].str.len() > 0]

    if verbose:
        print(f"  Final clean records: {len(df):,}")
        print(f"{'='*50}\n")

    return df.reset_index(drop=True)
