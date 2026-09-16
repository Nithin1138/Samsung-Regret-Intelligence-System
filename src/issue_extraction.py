"""
Issue Extraction Module
=======================
Rule-based issue categorization from customer review text.
Maps keywords to issue categories for defect identification.
"""

import pandas as pd

# Keyword map: issue category → list of trigger keywords
ISSUE_KEYWORDS = {
    "battery": [
        "battery", "drain", "backup", "charging", "charge",
        "power", "life", "dead",
    ],
    "heating": [
        "heat", "overheat", "hot", "warm", "temperature",
    ],
    "display": [
        "display", "screen", "flicker", "touch", "pixel",
        "crack", "brightness",
    ],
    "performance": [
        "slow", "lag", "hang", "freeze", "crash",
        "unresponsive", "delay",
    ],
    "camera": [
        "camera", "blur", "photo", "focus", "video", "picture",
    ],
    "sound": [
        "sound", "audio", "speaker", "volume", "mic",
    ],
    "connectivity": [
        "wifi", "bluetooth", "network", "signal", "disconnect",
    ],
    "build_quality": [
        "build", "material", "fragile", "plastic", "cheap",
        "broken", "scratch",
    ],
    "software": [
        "update", "bug", "glitch", "app", "os", "ui",
        "interface", "software",
    ],
    "general_issue": [
        "bad", "worst", "terrible", "horrible", "defective",
        "faulty", "poor", "pathetic", "useless", "waste",
        "disappointed", "regret",
    ],
}


def extract_issue(text, keyword_map=None):
    """
    Extract the primary issue category from review text
    using keyword matching.

    Parameters
    ----------
    text : str
        Review text.
    keyword_map : dict, optional
        Custom keyword map. Defaults to ISSUE_KEYWORDS.

    Returns
    -------
    str
        Issue category name, or 'none' if no match.
    """
    if keyword_map is None:
        keyword_map = ISSUE_KEYWORDS

    text = str(text).lower()

    for issue, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text:
                return issue

    return "none"


def add_issue_column(df, text_col="Review", issue_col="Issue"):
    """
    Add issue category column to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
        Column containing review text.
    issue_col : str
        Name for the new issue column.

    Returns
    -------
    pd.DataFrame
    """
    df[issue_col] = df[text_col].apply(extract_issue)
    return df


def get_issue_distribution(df, issue_col="Issue"):
    """Return value counts of issue categories."""
    return df[issue_col].value_counts()


def extract_sentence_aspects(text, sia=None, keyword_map=None):
    """
    Sentence-level rule-based aspect extraction.
    Splits review into clauses/sentences and analyzes sentiment per aspect.
    Example: 'Battery drains quickly but the camera is excellent.'
    Produces:
        [{'aspect': 'battery', 'sentiment': 'Negative', 'compound': -0.42},
         {'aspect': 'camera', 'sentiment': 'Positive', 'compound': 0.58}]

    Parameters
    ----------
    text : str
        Review text.
    sia : SentimentIntensityAnalyzer, optional
        VADER sentiment analyzer.
    keyword_map : dict, optional
        Aspect keyword dictionary.

    Returns
    -------
    list of dict
        Extracted aspects with their localized sentiment and compound score.
    """
    import re
    if keyword_map is None:
        keyword_map = ISSUE_KEYWORDS

    if sia is None:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

    # Split by punctuation and contrastive conjunctions
    clauses = re.split(r'[\.\?!;]|\b(?:but|however|although|though|whereas|yet)\b', str(text), flags=re.IGNORECASE)
    results = []

    for clause in clauses:
        clause_str = clause.strip().lower()
        if not clause_str or len(clause_str) < 3:
            continue

        for aspect, keywords in keyword_map.items():
            if aspect == "general_issue":
                continue  # General issue is an overall category, not a specific component aspect
            # Check whole-word matches
            matched = False
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', clause_str):
                    matched = True
                    break
            if matched:
                compound = sia.polarity_scores(clause)['compound']
                # Use VADER compound polarity, with fallback to negative defect cues
                negative_cues = {"drain", "worst", "bad", "slow", "lag", "heat", "hot", "flicker", "defect", "poor", "broken", "dead", "issue", "problem", "hang"}
                positive_cues = {"good", "great", "excellent", "best", "fast", "smooth", "clear", "sharp", "long", "amazing", "love"}

                if compound >= 0.05:
                    sentiment = "Positive"
                elif compound <= -0.05:
                    sentiment = "Negative"
                elif any(w in clause_str for w in negative_cues):
                    sentiment = "Negative"
                    compound = -0.35
                elif any(w in clause_str for w in positive_cues):
                    sentiment = "Positive"
                    compound = 0.35
                else:
                    sentiment = "Neutral"

                results.append({
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "compound": compound,
                    "clause": clause.strip()
                })
                break  # One match per aspect per clause

    return results


def build_aspect_sentiment_table(df, text_col="Review", sia=None):
    """
    Extracts sentence-level aspects across the entire dataset into a flattened tabular format
    ideal for Aspect-Based Sentiment Analysis and Power BI reporting.
    """
    rows = []
    for idx, row in df.iterrows():
        review_id = row.get("Review_ID", idx + 1)
        product = row.get("Product", "Unknown")
        category = row.get("Category", "Unknown")
        platform = row.get("Platform", "Unknown")
        aspects = extract_sentence_aspects(row[text_col], sia=sia)

        for asp in aspects:
            rows.append({
                "Review_ID": review_id,
                "Product": product,
                "Category": category,
                "Platform": platform,
                "Aspect": asp["aspect"],
                "Aspect_Sentiment": asp["sentiment"],
                "Aspect_Compound": asp["compound"],
                "Clause": asp["clause"]
            })

    return pd.DataFrame(rows)
