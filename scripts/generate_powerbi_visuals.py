"""
Generate Professional Power BI Dashboard Visual Renders
======================================================
Produces high-resolution visual layouts representing the 5 Power BI dashboard pages:
1. page1_overview.png
2. page2_product_risk.png
3. page3_temporal_intelligence.png
4. page4_issue_analysis.png
5. page5_platform_analysis.png
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POWERBI_DIR = os.path.join(PROJECT_ROOT, "outputs", "powerbi")
os.makedirs(POWERBI_DIR, exist_ok=True)

# Set clean aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#bdc3c7'
plt.rcParams['axes.linewidth'] = 0.8

# Load datasets
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "samsung_processed_dataset.csv"))
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
aspect_df = pd.read_csv(os.path.join(POWERBI_DIR, "powerbi_aspect_sentiments.csv"))
prod_summary = pd.read_csv(os.path.join(POWERBI_DIR, "powerbi_product_risk_summary.csv"))

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
def generate_page1_overview():
    fig = plt.figure(figsize=(15, 9), facecolor='#f8f9fa')
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Top Banner Title
    fig.text(0.08, 0.94, "SAMSUNG REGRET INTELLIGENCE — EXECUTIVE OVERVIEW", fontsize=18, fontweight='bold', color='#1a252f')
    fig.text(0.08, 0.915, "Multi-Platform E-Commerce Product Health & Quality Monitoring Dashboard", fontsize=11, color='#7f8c8d')

    # KPI 1: Total Reviews
    ax_kpi1 = fig.add_subplot(gs[0, 0])
    ax_kpi1.set_facecolor('#ffffff')
    ax_kpi1.text(0.5, 0.65, f"{len(df):,}", ha='center', va='center', fontsize=26, fontweight='bold', color='#2c3e50')
    ax_kpi1.text(0.5, 0.25, "TOTAL REVIEWS", ha='center', va='center', fontsize=11, fontweight='bold', color='#7f8c8d')
    ax_kpi1.axis('off')

    # KPI 2: Avg Star Rating
    ax_kpi2 = fig.add_subplot(gs[0, 1])
    ax_kpi2.set_facecolor('#ffffff')
    ax_kpi2.text(0.5, 0.65, f"{df['Rating'].mean():.2f} ★", ha='center', va='center', fontsize=26, fontweight='bold', color='#f39c12')
    ax_kpi2.text(0.5, 0.25, "AVERAGE RATING", ha='center', va='center', fontsize=11, fontweight='bold', color='#7f8c8d')
    ax_kpi2.axis('off')

    # KPI 3: Avg Regret Index
    ax_kpi3 = fig.add_subplot(gs[0, 2])
    ax_kpi3.set_facecolor('#ffffff')
    ax_kpi3.text(0.5, 0.65, f"{df['RSI'].mean():.3f}", ha='center', va='center', fontsize=26, fontweight='bold', color='#e74c3c')
    ax_kpi3.text(0.5, 0.25, "AVERAGE RSI", ha='center', va='center', fontsize=11, fontweight='bold', color='#7f8c8d')
    ax_kpi3.axis('off')

    # KPI 4: High Risk Flagged Reviews
    ax_kpi4 = fig.add_subplot(gs[0, 3])
    ax_kpi4.set_facecolor('#ffffff')
    high_risk_count = (df['RSI'] > 0.5).sum()
    ax_kpi4.text(0.5, 0.65, f"{high_risk_count:,} ({high_risk_count/len(df)*100:.1f}%)", ha='center', va='center', fontsize=22, fontweight='bold', color='#c0392b')
    ax_kpi4.text(0.5, 0.25, "HIGH RISK REVIEWS (RSI > 0.5)", ha='center', va='center', fontsize=10, fontweight='bold', color='#7f8c8d')
    ax_kpi4.axis('off')

    # Chart 1: Category Distribution Donut
    ax_cat = fig.add_subplot(gs[1:3, 0:2])
    cat_counts = df['Category'].value_counts()
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
    wedges, texts, autotexts = ax_cat.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
                                          startangle=140, colors=colors[:len(cat_counts)],
                                          wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2))
    plt.setp(autotexts, size=10, weight="bold")
    ax_cat.set_title("Review Distribution by Product Category", fontsize=13, fontweight='bold', pad=15)

    # Chart 2: Monthly Ingestion & Average RSI
    ax_trend = fig.add_subplot(gs[1:3, 2:4])
    monthly = df.groupby('Month').agg(Volume=('RSI', 'count'), Avg_RSI=('RSI', 'mean')).reset_index()
    monthly['Month_Str'] = monthly['Month'].astype(str)

    ax_trend.bar(monthly['Month_Str'], monthly['Volume'], color='#ecf0f1', edgecolor='#bdc3c7', label='Review Count')
    ax_rsi = ax_trend.twinx()
    ax_rsi.plot(monthly['Month_Str'], monthly['Avg_RSI'], color='#e74c3c', marker='o', linewidth=2.2, label='Avg RSI')
    ax_trend.set_xticklabels(monthly['Month_Str'], rotation=45, ha='right', fontsize=8)
    ax_trend.set_ylabel("Monthly Review Count", color='#7f8c8d')
    ax_rsi.set_ylabel("Average RSI", color='#e74c3c')
    ax_rsi.grid(False)
    ax_trend.set_title("Longitudinal Volume & Remorse Severity Trend", fontsize=13, fontweight='bold')

    out_path = os.path.join(POWERBI_DIR, "page1_overview.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated {out_path}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: PRODUCT RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
def generate_page2_product_risk():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#f8f9fa')
    fig.suptitle("PRODUCT RISK ANALYSIS & CRITICAL PRIORITIZATION", fontsize=16, fontweight='bold', y=0.98, color='#1a252f')

    # Top 10 Critical Products by Composite Risk
    top10 = prod_summary.head(10).sort_values('Composite_Risk_Score', ascending=True)
    palette = ['#e74c3c' if r == 'Critical Risk' else '#f39c12' for r in top10['Risk_Tier']]
    bars = ax1.barh(top10['Product'], top10['Composite_Risk_Score'], color=palette, edgecolor='white', height=0.65)
    ax1.set_title("Top 10 High-Risk Products (Composite Score)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Composite Risk Score (RSI 30% + Slope 25% + Issues 20% + Neg 15% + Vol 10%)")
    for bar in bars:
        ax1.text(bar.get_width() + 0.01, bar.get_y() + 0.15, f"{bar.get_width():.3f}", va='center', fontsize=9, fontweight='bold')

    # Scatter: Avg RSI vs Trend Slope (Matrix)
    scatter = ax2.scatter(prod_summary['Avg_RSI'], prod_summary['Trend_Slope'],
                          s=prod_summary['Total_Reviews'] * 1.5,
                          c=prod_summary['Composite_Risk_Score'], cmap='YlOrRd', alpha=0.8, edgecolors='#2c3e50')
    ax2.axvline(df['RSI'].mean(), color='#7f8c8d', linestyle='--', alpha=0.7, label='Average RSI Threshold')
    ax2.axhline(0, color='#7f8c8d', linestyle='-', alpha=0.5)
    ax2.set_title("Product Risk Matrix: Severity vs Temporal Deterioration", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Average RSI (Current Severity)")
    ax2.set_ylabel("RSI Trend Slope (Velocity of Deterioration)")
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Composite Risk Score")
    ax2.legend(loc='upper left')

    out_path = os.path.join(POWERBI_DIR, "page2_product_risk.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated {out_path}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: TEMPORAL INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════
def generate_page3_temporal():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), facecolor='#f8f9fa')
    fig.suptitle("TEMPORAL INTELLIGENCE & ACCELERATION MONITORING", fontsize=16, fontweight='bold', y=0.98, color='#1a252f')

    # Category Rolling 3-Month RSI
    cat_monthly = df.groupby(['Category', 'Month'])['RSI'].mean().reset_index()
    cat_monthly['Month_Str'] = cat_monthly['Month'].astype(str)
    cat_monthly['Rolling_3M'] = cat_monthly.groupby('Category')['RSI'].transform(lambda s: s.rolling(3, min_periods=1).mean())

    for cat in df['Category'].unique():
        sub = cat_monthly[cat_monthly['Category'] == cat]
        ax1.plot(sub['Month_Str'], sub['Rolling_3M'], marker='.', label=cat, linewidth=2)

    ax1.set_title("3-Month Rolling RSI Trends Across Categories", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Rolling Average RSI")
    ax1.legend(loc='upper right')
    ax1.tick_params(axis='x', rotation=45)

    # Risk Acceleration (Deterioration slope of slopes)
    top_worsening = prod_summary.sort_values('Trend_Slope', ascending=False).head(5)['Product'].tolist()
    for prod in top_worsening:
        p_data = df[df['Product'] == prod].groupby('Month')['RSI'].mean().reset_index()
        p_data['Month_Str'] = p_data['Month'].astype(str)
        p_data['Slope_Diff'] = p_data['RSI'].diff().rolling(2, min_periods=1).mean()
        ax2.plot(p_data['Month_Str'], p_data['Slope_Diff'], marker='o', label=prod[:22], linewidth=1.8)

    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_title("Risk Acceleration Curve (2nd Derivative Δ Slope) — Top Deteriorating Products", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Year-Month")
    ax2.set_ylabel("Risk Acceleration Rate")
    ax2.legend(loc='upper left')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    out_path = os.path.join(POWERBI_DIR, "page3_temporal_intelligence.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated {out_path}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: ISSUE & ASPECT SENTIMENT ANALYSIS (Priority 7)
# ═══════════════════════════════════════════════════════════════════════════
def generate_page4_issue_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#f8f9fa')
    fig.suptitle("DEFECT TAXONOMY & SENTENCE-LEVEL ASPECT SENTIMENT", fontsize=16, fontweight='bold', y=0.98, color='#1a252f')

    # Aspect Sentiment Ratio (Sentence-level extraction from Priority 7)
    aspect_pivot = pd.crosstab(aspect_df['Aspect'], aspect_df['Aspect_Sentiment'], normalize='index') * 100
    if 'Negative' in aspect_pivot.columns and 'Positive' in aspect_pivot.columns:
        aspect_pivot = aspect_pivot[['Negative', 'Neutral', 'Positive']] if 'Neutral' in aspect_pivot.columns else aspect_pivot[['Negative', 'Positive']]
        aspect_pivot.plot(kind='barh', stacked=True, ax=ax1, color=['#e74c3c', '#bdc3c7', '#2ecc71'])
        ax1.set_title("Sentence-Level Aspect Sentiment Distribution (%)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Percentage of Mentions")
        ax1.legend(loc='lower right')

    # Defect Regret Severity Hierarchy
    issue_rsi = df[df['Issue'] != 'none'].groupby('Issue')['RSI'].agg(['mean', 'count']).sort_values('mean', ascending=True)
    bars = ax2.barh(issue_rsi.index, issue_rsi['mean'], color='#3498db', height=0.6)
    ax2.set_title("Regret Severity Index (RSI) Impact by Defect Type", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Average RSI When Defect is Tagged")
    for bar in bars:
        ax2.text(bar.get_width() + 0.005, bar.get_y() + 0.15, f"{bar.get_width():.3f}", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(POWERBI_DIR, "page4_issue_analysis.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated {out_path}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: PLATFORM COMPARATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def generate_page5_platform_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='#f8f9fa')
    fig.suptitle("CROSS-PLATFORM PERFORMANCE & DEFECT PARITY", fontsize=16, fontweight='bold', y=0.98, color='#1a252f')

    # Amazon vs Flipkart Key Metrics
    plat_summary = df.groupby('Platform').agg(
        Avg_Rating=('Rating', 'mean'),
        Avg_RSI=('RSI', 'mean'),
        Neg_Ratio=('Sentiment_score', lambda s: (s < -0.05).mean() * 100),
        High_Risk_Pct=('RSI', lambda s: (s > 0.5).mean() * 100)
    ).reset_index()

    x = np.arange(len(plat_summary))
    width = 0.2
    ax1.bar(x - width*1.5, plat_summary['Avg_Rating'], width, label='Avg Rating (1-5)', color='#f39c12')
    ax1.bar(x - width*0.5, plat_summary['Avg_RSI'] * 5, width, label='Avg RSI (Scaled ×5)', color='#e74c3c')
    ax1.bar(x + width*0.5, plat_summary['Neg_Ratio'] / 10, width, label='Negative Reviews (/10%)', color='#9b59b6')
    ax1.bar(x + width*1.5, plat_summary['High_Risk_Pct'] / 10, width, label='High Risk Reviews (/10%)', color='#c0392b')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plat_summary['Platform'], fontsize=11, fontweight='bold')
    ax1.set_title("Platform KPI Parity Comparison", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # RSI Distribution Boxplot
    sns.boxplot(x='Platform', y='RSI', data=df, ax=ax2, palette=['#3498db', '#f1c40f'])
    ax2.set_title("Customer Regret Severity (RSI) Distribution", fontsize=12, fontweight='bold')
    ax2.set_ylabel("RSI Score")

    plt.tight_layout()
    out_path = os.path.join(POWERBI_DIR, "page5_platform_analysis.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated {out_path}")

if __name__ == "__main__":
    generate_page1_overview()
    generate_page2_product_risk()
    generate_page3_temporal()
    generate_page4_issue_analysis()
    generate_page5_platform_analysis()
    print("\n🎉 All 5 Power BI Dashboard pages rendered successfully in outputs/powerbi/!")
