"""
Sentiment Analysis Module
=========================
VADER-based sentiment analysis for customer reviews.
"""

import pandas as pd
import nltk


def _ensure_vader():
    """Download VADER lexicon if not already present."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def compute_sentiment(text):
    """
    Compute VADER compound sentiment score for a single text.

    Returns
    -------
    float
        Compound score in [-1, 1].
    """
    _ensure_vader()
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(str(text))["compound"]


def add_sentiment_scores(df, text_col="Review", score_col="Sentiment_score"):
    """
    Add VADER compound sentiment scores to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
        Column containing review text.
    score_col : str
        Name for the new sentiment score column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the sentiment score column added.
    """
    _ensure_vader()
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    df[score_col] = df[text_col].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
    return df


def classify_sentiment(score, pos_threshold=0.05, neg_threshold=-0.05):
    """
    Classify a compound sentiment score into Positive / Negative / Neutral.
    """
    if score >= pos_threshold:
        return "Positive"
    elif score <= neg_threshold:
        return "Negative"
    else:
        return "Neutral"


def add_sentiment_labels(df, score_col="Sentiment_score", label_col="Sentiment_label"):
    """Add categorical sentiment labels based on compound scores."""
    df[label_col] = df[score_col].apply(classify_sentiment)
    return df
