"""
Power BI Data Preparation & Export Script
=========================================
Generates clean, normalized dimensional tables and aggregated datasets
tailored specifically for Power BI ingestion.

Output files in outputs/powerbi/:
1. powerbi_fact_reviews.csv
2. powerbi_dim_product.csv
3. powerbi_dim_date.csv
4. powerbi_dim_platform.csv
5. powerbi_dim_issue.csv
6. powerbi_product_risk_summary.csv
7. powerbi_temporal_trends.csv
8. powerbi_aspect_sentiments.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.issue_extraction import build_aspect_sentiment_table
from src.warehouse import build_star_schema

def export_powerbi_datasets():
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "powerbi")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating Power BI assets in: {output_dir}")

    # 1. Load Processed Dataset
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "samsung_processed_dataset.csv")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')
    print(f"Loaded {len(df):,} processed review records.")

    # 2. Build and export Star Schema tables
    schema = build_star_schema(df)
    schema['fact_review'].to_csv(os.path.join(output_dir, "powerbi_fact_reviews.csv"), index=False)
    schema['dim_product'].to_csv(os.path.join(output_dir, "powerbi_dim_product.csv"), index=False)
    schema['dim_date'].to_csv(os.path.join(output_dir, "powerbi_dim_date.csv"), index=False)
    schema['dim_platform'].to_csv(os.path.join(output_dir, "powerbi_dim_platform.csv"), index=False)
    schema['dim_issue'].to_csv(os.path.join(output_dir, "powerbi_dim_issue.csv"), index=False)
    print("✅ Exported 5 Star Schema dimensional tables.")

    # 3. Build Product Risk Summary with Trend Slopes
    slope_records = {}
    for product, group in df.groupby('Product'):
        monthly_p = group.groupby('YearMonth')['RSI'].mean().reset_index()
        if len(monthly_p) >= 3:
            x = np.arange(len(monthly_p))
            y = monthly_p['RSI'].values
            res = stats.linregress(x, y)
            slope_records[product] = res.slope
        else:
            slope_records[product] = 0.0

    prod_summary = df.groupby(['Product', 'Category']).agg(
        Total_Reviews=('RSI', 'count'),
        Avg_Rating=('Rating', 'mean'),
        Avg_Sentiment=('Sentiment_score', 'mean'),
        Avg_RSI=('RSI', 'mean'),
        High_Risk_Reviews=('RSI', lambda s: (s > 0.5).sum()),
        Negative_Reviews=('Sentiment_score', lambda s: (s < -0.05).sum()),
        Issue_Count=('Issue', lambda i: (i != 'none').sum())
    ).reset_index()

    prod_summary['Trend_Slope'] = prod_summary['Product'].map(slope_records).fillna(0)
    prod_summary['High_Risk_Pct'] = (prod_summary['High_Risk_Reviews'] / prod_summary['Total_Reviews'] * 100).round(2)
    prod_summary['Negative_Pct'] = (prod_summary['Negative_Reviews'] / prod_summary['Total_Reviews'] * 100).round(2)
    prod_summary['Issue_Density_Pct'] = (prod_summary['Issue_Count'] / prod_summary['Total_Reviews'] * 100).round(2)

    # Normalize slope
    min_s, max_s = prod_summary['Trend_Slope'].min(), prod_summary['Trend_Slope'].max()
    norm_slope = (prod_summary['Trend_Slope'] - min_s) / (max_s - min_s) if max_s > min_s else 0.5

    prod_summary['Composite_Risk_Score'] = (
        0.30 * prod_summary['Avg_RSI'] +
        0.25 * norm_slope +
        0.20 * (prod_summary['Issue_Density_Pct'] / 100.0) +
        0.15 * (prod_summary['Negative_Pct'] / 100.0) +
        0.10 * (prod_summary['Total_Reviews'] / prod_summary['Total_Reviews'].max())
    ).round(4)

    prod_summary['Risk_Tier'] = pd.qcut(prod_summary['Composite_Risk_Score'], q=3, labels=['Low Risk', 'Moderate Risk', 'Critical Risk'])
    prod_summary = prod_summary.sort_values('Composite_Risk_Score', ascending=False)
    prod_summary.to_csv(os.path.join(output_dir, "powerbi_product_risk_summary.csv"), index=False)
    print("✅ Exported product risk summary.")

    # 4. Build Temporal Trends Table (Monthly & 3-Month Rolling RSI)
    temporal = df.groupby(['YearMonth', 'Category', 'Platform']).agg(
        Review_Volume=('RSI', 'count'),
        Avg_Rating=('Rating', 'mean'),
        Avg_Sentiment=('Sentiment_score', 'mean'),
        Avg_RSI=('RSI', 'mean')
    ).reset_index()
    temporal['YearMonth'] = temporal['YearMonth'].astype(str)
    temporal['Rolling_3M_RSI'] = temporal.groupby(['Category', 'Platform'])['Avg_RSI'].transform(lambda s: s.rolling(3, min_periods=1).mean()).round(4)
    temporal.to_csv(os.path.join(output_dir, "powerbi_temporal_trends.csv"), index=False)
    print("✅ Exported temporal trends dataset.")

    # 5. Extract Aspect Sentiments (Priority 7 sentence-level extraction for Power BI)
    print("Extracting sentence-level aspect sentiments across dataset...")
    aspect_df = build_aspect_sentiment_table(df)
    aspect_df.to_csv(os.path.join(output_dir, "powerbi_aspect_sentiments.csv"), index=False)
    print(f"✅ Exported {len(aspect_df):,} aspect sentiment occurrences to powerbi_aspect_sentiments.csv")

    print("\n🎉 All Power BI datasets generated successfully!")

if __name__ == "__main__":
    export_powerbi_datasets()
