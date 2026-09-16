"""
Data Mining Module
==================
Clustering (KMeans), Association Rule Mining (Apriori),
Outlier Detection (IsolationForest), and
Temporal Risk Prediction (replaces circular LogReg).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# ─── Clustering ──────────────────────────────────────────────────────


def perform_clustering(df, features=None, n_clusters=3, random_state=42):
    """
    KMeans clustering on RSI (or custom features).

    Parameters
    ----------
    df : pd.DataFrame
    features : list, optional
        Columns to cluster on. Default: ['RSI'].
    n_clusters : int
    random_state : int

    Returns
    -------
    tuple (pd.DataFrame, KMeans)
        DataFrame with Cluster and Cluster_Label columns, and fitted model.
    """
    if features is None:
        features = ["RSI"]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[features])

    # Auto-label clusters by mean RSI
    cluster_means = df.groupby("Cluster")["RSI"].mean()
    sorted_clusters = cluster_means.sort_values().index.tolist()

    labels = ["Low Risk", "Medium Risk", "High Risk"]
    if n_clusters > 3:
        labels = [f"Risk Level {i}" for i in range(n_clusters)]
    label_map = {c: labels[i] for i, c in enumerate(sorted_clusters[:len(labels)])}
    df["Cluster_Label"] = df["Cluster"].map(label_map).fillna("Unknown")

    return df, kmeans


# ─── Association Rule Mining ─────────────────────────────────────────


def perform_association_mining(df, min_support=0.01, min_confidence=0.3):
    """
    Apriori association rule mining on Issue × High_Risk × Platform.

    Returns
    -------
    pd.DataFrame
        Association rules sorted by lift.
    """
    from mlxtend.frequent_patterns import apriori, association_rules

    df_apriori = df[["Issue", "High_Risk", "Platform"]].copy()
    df_apriori["High_Risk"] = df_apriori["High_Risk"].astype(str)
    df_encoded = pd.get_dummies(df_apriori)

    frequent_items = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by="lift", ascending=False)

    return rules


# ─── Outlier Detection ───────────────────────────────────────────────


def detect_outliers(df, features=None, contamination=0.05, random_state=42):
    """
    IsolationForest outlier detection.

    Returns
    -------
    pd.DataFrame
        With Outlier column (-1 = outlier, 1 = normal).
    """
    if features is None:
        features = ["RSI"]

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    df["Outlier"] = iso.fit_predict(df[features])

    n_outliers = (df["Outlier"] == -1).sum()
    print(f"  Outliers detected: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

    return df, iso


# ─── Temporal Risk Prediction (Replaces Circular LogReg) ─────────────


def build_temporal_features(df, product_col="Product", date_col="Date"):
    """
    Build per-product, per-quarter features for temporal risk prediction.

    Features (computed from current quarter):
        - avg_rsi: Average RSI
        - review_count: Number of reviews
        - neg_ratio: Fraction of reviews with negative sentiment
        - issue_density: Fraction of reviews with issues != 'none'
        - avg_rating: Average rating
        - avg_sentiment: Average sentiment score

    Target (from next quarter):
        - future_high_risk: 1 if next quarter's avg RSI > 0.5, else 0

    Uses a time-based split: earlier quarters for training, later for testing.

    Returns
    -------
    pd.DataFrame
        Feature table with one row per (Product, Quarter).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["Quarter"] = df[date_col].dt.to_period("Q")
    df["Is_Negative"] = (df["Sentiment_score"] < 0).astype(int)
    df["Has_Issue"] = (df["Issue"] != "none").astype(int)

    # Aggregate per product per quarter
    quarterly = (
        df.groupby([product_col, "Quarter"])
        .agg(
            avg_rsi=("RSI", "mean"),
            review_count=("RSI", "count"),
            neg_ratio=("Is_Negative", "mean"),
            issue_density=("Has_Issue", "mean"),
            avg_rating=("Rating", "mean"),
            avg_sentiment=("Sentiment_score", "mean"),
        )
        .reset_index()
        .sort_values([product_col, "Quarter"])
    )

    # Create target: next quarter's high risk flag
    quarterly["future_avg_rsi"] = quarterly.groupby(product_col)["avg_rsi"].shift(-1)
    quarterly["future_high_risk"] = (quarterly["future_avg_rsi"] > 0.5).astype(int)

    # Drop rows without a future quarter (can't have target)
    quarterly = quarterly.dropna(subset=["future_avg_rsi"])

    return quarterly


def train_temporal_risk_model(quarterly_features, test_ratio=0.25, verbose=True):
    """
    Train a Logistic Regression model using time-based split.

    Parameters
    ----------
    quarterly_features : pd.DataFrame
        Output from build_temporal_features().
    test_ratio : float
        Fraction of time periods to hold out for testing.
    verbose : bool

    Returns
    -------
    dict
        Keys: 'model', 'X_train', 'X_test', 'y_train', 'y_test',
              'y_pred', 'feature_names', 'report', 'confusion_matrix'.
    """
    feature_cols = [
        "avg_rsi", "review_count", "neg_ratio",
        "issue_density", "avg_rating", "avg_sentiment",
    ]

    # Time-based split: sort by quarter, split chronologically
    sorted_quarters = sorted(quarterly_features["Quarter"].unique())
    n_quarters = len(sorted_quarters)
    split_idx = int(n_quarters * (1 - test_ratio))

    train_quarters = sorted_quarters[:split_idx]
    test_quarters = sorted_quarters[split_idx:]

    train_mask = quarterly_features["Quarter"].isin(train_quarters)
    test_mask = quarterly_features["Quarter"].isin(test_quarters)

    X_train = quarterly_features.loc[train_mask, feature_cols]
    y_train = quarterly_features.loc[train_mask, "future_high_risk"]
    X_test = quarterly_features.loc[test_mask, feature_cols]
    y_test = quarterly_features.loc[test_mask, "future_high_risk"]

    if verbose:
        print(f"\n{'='*50}")
        print("  TEMPORAL RISK PREDICTION")
        print(f"{'='*50}")
        print(f"  Training quarters: {train_quarters[0]} to {train_quarters[-1]} ({len(train_quarters)} quarters)")
        print(f"  Testing quarters:  {test_quarters[0]} to {test_quarters[-1]} ({len(test_quarters)} quarters)")
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Train target distribution: {dict(y_train.value_counts())}")
        print(f"  Test target distribution:  {dict(y_test.value_counts())}")

    # Train
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Results
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    if verbose:
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"  Confusion Matrix:")
        print(cm)

    # Feature importance
    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_[0],
        "Abs_Coefficient": np.abs(model.coef_[0]),
    }).sort_values("Abs_Coefficient", ascending=False)

    if verbose:
        print(f"\n  Feature Importance:")
        print(importance.to_string(index=False))

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": feature_cols,
        "feature_importance": importance,
        "report": report,
        "confusion_matrix": cm,
        "train_quarters": train_quarters,
        "test_quarters": test_quarters,
    }
