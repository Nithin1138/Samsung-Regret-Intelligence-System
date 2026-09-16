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
