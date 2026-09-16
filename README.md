# Samsung Regret Intelligence System

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-Dashboard_Ready-F2C811.svg)](outputs/powerbi/)
[![Data Warehouse](https://img.shields.io/badge/Schema-Star_Schema_OLAP-success.svg)](data/warehouse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise data warehousing, NLP, predictive intelligence, and business analytics system that converts unstructured multi-platform e-commerce customer reviews into actionable product risk metrics for Samsung consumer electronics.

---

## 📌 1. Overview & Problem Statement

Standard sentiment analysis and raw star ratings often fail to capture genuine consumer remorse:
* A **3-star review** may conceal a critical hardware defect (e.g., display flickering or battery degradation).
* A **1-star review** may reflect an isolated logistics or delivery issue rather than a defect in the product itself.

The **Samsung Regret Intelligence System** implements a **Star Schema Data Warehouse**, engineers a domain-specific metric called the **Regret Severity Index (RSI)**, and features a **Temporal Risk Scoring Engine** coupled with a **Power BI Executive Dashboard**. The system processes **7,236 multi-platform customer reviews** across **50 product lines**, enabling proactive defect detection and product risk prioritization.

---

## 🔄 2. End-to-End System Pipeline

```
┌─────────────┐
│ 11,000 Raw  │
│   Reviews   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Data Quality Validation Engine      │
│  • Duplicates removal (3,206 dupes)  │
│  • Missing review / product removal  │
│  • Rating bounds check (1 to 5)      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  ETL & Natural Language Processing   │
│  • Text Regex Normalization          │
│  • NLTK VADER Compound Polarity      │
│  • Sentence-Level Aspect Extraction  │
│    (Battery → Neg, Camera → Pos)     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Regret Severity Index (RSI) Engine  │
│  • Normalized Rating Penalty (35%)   │
│  • Inverted Negative Polarity (35%)  │
│  • Hardware Defect Penalty (15%)     │
│  • Review Text Specificity (15%)     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Dimensional Data Warehouse          │
│  • Star Schema: fact_review          │
│  • Dimensions: product, date,        │
│    platform, issue                   │
│  • 6 Analytical SQL OLAP Queries     │
└──────────────────┬───────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐ ┌──────────────────┐
│ Predictive      │ │ Temporal         │
│ Data Mining     │ │ Intelligence     │
│ • Non-Circular  │ │ • Monthly RSI    │
│   Temporal Risk │ │ • 3-Mo Rolling   │
│   Predictor     │ │ • Trend Slopes   │
│ • KMeans (k=3)  │ │   (OLS linreg)   │
│ • Apriori Rules │ │ • Risk           │
│ • IsolationFrst │ │   Acceleration   │
└────────┬────────┘ └────────┬─────────┘
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Power BI Executive Dashboard        │
│  1. Overview & Core KPIs             │
│  2. Product Risk Prioritization      │
│  3. Temporal Trends & Acceleration   │
│  4. Defect Taxonomy & Aspects        │
│  5. Retail Platform Parity           │
└──────────────────────────────────────┘
```

---

## 🏗️ 3. Data Warehouse Architecture (Star Schema)

The dimensional model deconstructs denormalized customer reviews into a central fact table and 4 dimension tables optimized for OLAP aggregations and Power BI reporting:

```
                  ┌──────────────────────┐
                  │      dim_date        │
                  │  Date_ID (PK)        │
                  │  Year, Month, Quarter│
                  │  DayOfWeek           │
                  └──────────┬───────────┘
                             │ 1
                             │
                             │ *
┌──────────────────────┐   ┌─┴────────────────────┐   ┌──────────────────────┐
│     dim_product      │   │     fact_review      │   │     dim_platform     │
│  Product_ID (PK)     ├───┤  Review_ID (PK)      ├───┤  Platform_ID (PK)    │
│  Product_Name        │ 1 │  Product_ID (FK)     │ 1 │  Platform_Name       │
│  Category            │ * │  Platform_ID (FK)    │ * └──────────────────────┘
└──────────────────────┘   │  Issue_ID (FK)       │
                           │  Date_ID (FK)        │
                           │  Rating, Sent, RSI   │
                           └─┬────────────────────┘
                             │ *
                             │
                             │ 1
                  ┌──────────┴───────────┐
                  │      dim_issue       │
                  │  Issue_ID (PK)       │
                  │  Issue_Type          │
                  └──────────────────────┘
```

---

## 📂 4. Project Structure

```
Samsung-Regret-Intelligence-System/
├── src/                                    # Production-grade Python modules
│   ├── __init__.py                         # Package entrypoint
│   ├── data_quality.py                     # Data validation & quality scorecard
│   ├── preprocessing.py                    # Text cleaning, deduplication, rating validation
│   ├── sentiment.py                        # NLTK VADER sentiment scoring & classification
│   ├── issue_extraction.py                 # Keyword categorization & sentence-level aspect extraction
│   ├── rsi.py                              # RSI computation, monthly/rolling/trend slope/acceleration
│   ├── etl.py                              # End-to-end pipeline orchestrator
│   ├── warehouse.py                        # Star schema builder & 6 SQL-style analytical queries
│   ├── mining.py                           # Temporal predictor, KMeans, Apriori, IsolationForest
│   └── risk_analysis.py                    # Multi-factor risk scorecard & worsening detection
├── notebooks/                              # Self-contained Jupyter Notebooks
│   ├── 01_data_generation.ipynb            # Raw data ingestion & text normalization
│   ├── 02_etl_pipeline.ipynb               # Quality checks → VADER → Aspects → RSI
│   ├── 03_warehouse.ipynb                  # Star Schema dimensional modeling & SQL analytics
│   ├── 04_mining.ipynb                     # Non-circular temporal ML, clustering, association rules
│   ├── 05_analysis.ipynb                   # Temporal trends, rolling RSI, product risk scorecard
│   └── 06_satisfaction_analysis.ipynb      # Positive drivers & customer satisfaction
├── outputs/
│   ├── powerbi/                            # Power BI export datasets & dashboard assets
│   │   ├── POWER_BI_DASHBOARD_GUIDE.md     # DAX measures, relationship map, visual blueprint
│   │   ├── powerbi_fact_reviews.csv        # Fact table for Power BI import
│   │   ├── powerbi_dim_product.csv         # Product dimension table
│   │   ├── powerbi_dim_date.csv            # Date dimension table
│   │   ├── powerbi_dim_platform.csv        # Platform dimension table
│   │   ├── powerbi_dim_issue.csv           # Issue dimension table
│   │   ├── powerbi_product_risk_summary.csv# Pre-aggregated product scorecard
│   │   ├── powerbi_temporal_trends.csv     # Monthly & rolling temporal metrics
│   │   ├── powerbi_aspect_sentiments.csv   # Sentence-level aspect sentiment occurrences
│   │   ├── page1_overview.png              # Rendered Dashboard Page 1: Overview
│   │   ├── page2_product_risk.png          # Rendered Dashboard Page 2: Product Risk
│   │   ├── page3_temporal_intelligence.png # Rendered Dashboard Page 3: Temporal Trends
│   │   ├── page4_issue_analysis.png        # Rendered Dashboard Page 4: Defect Taxonomy
│   │   └── page5_platform_analysis.png     # Rendered Dashboard Page 5: Platform Parity
│   ├── models/                             # Trained model checkpoints (.pkl)
│   └── plots/                              # Analytical visualizations (.png)
├── data/
│   ├── raw/                                # Dirty & Clean raw review CSVs
│   ├── processed/                          # Processed dataset with NLP & RSI
│   └── warehouse/                          # Fact and dimension CSV tables
├── requirements.txt
└── README.md
```

---

## 🔬 5. Core Methodologies

### A. Regret Severity Index (RSI) Formula
Standard star ratings alone hide customer dissatisfaction. RSI computes true remorse by balancing four signals:

$$\text{RSI}_i = 0.35 \cdot \left(\frac{5 - \text{Rating}_i}{4}\right) + 0.35 \cdot \left(\frac{1 - \text{Compound}_i}{2}\right) + 0.15 \cdot \text{IssueSeverity}_i + 0.15 \cdot \text{Specificity}_i$$

* **Rating Penalty (35%)**: Divergence from a 5-star rating ($0.0 \rightarrow 1.0$).
* **Sentiment Remorse (35%)**: Inverts VADER compound score into normalized dissatisfaction ($0.0 \rightarrow 1.0$).
* **Hardware Defect Penalty (15%)**: Weighted flag when high-impact defects (display, battery) are tagged.
* **Review Specificity (15%)**: Weight proportional to detail depth.

### B. Rule-Based Sentence-Level Aspect Extraction (Priority 7)
Rather than tagging an entire review with a single label, the NLP engine breaks text into clauses across contrastive conjunctions (`but`, `however`, `although`) and evaluates sentiment per component:

```text
Input:  "Battery drains quickly but the camera is excellent."
Output: 
  • Aspect: Battery → Sentiment: Negative (Compound: -0.35)
  • Aspect: Camera  → Sentiment: Positive (Compound: +0.57)
```
Over **1,884 aspect-sentiment pairs** were extracted and exported to `outputs/powerbi/powerbi_aspect_sentiments.csv`.

### C. Non-Circular Temporal Risk Predictor (Priority 1)
* **The Flaw in Prior Work**: `Rating + Sentiment → RSI → High_Risk → LogReg(Rating, Sentiment)` caused circular data leakage.
* **The Solution**: Historical quarterly features (`previous_avg_rsi`, `neg_ratio`, `issue_density`, `review_volume`, `avg_rating`) predict whether a product's RSI will cross into high risk ($> 0.5$) in the **subsequent quarter**.
* **Split Strategy**: Chronological time-split (Train: Q1 2022–Q1 2024; Test: Q2–Q4 2024) with balanced class weights.

### D. Longitudinal Temporal Trajectory (Priority 2 & 3)
Replaces arbitrary endpoint deltas ($RSI_{last} - RSI_{first}$) with:
* **Monthly RSI Time-Series**: Base longitudinal monitoring.
* **3-Month Rolling Average**: Noise dampening across seasonal sale spikes.
* **OLS Linear Regression Slope ($m$)**: Mathematical trajectory velocity ($y = mx + c$).
* **Risk Acceleration**: Second derivative ($\Delta$ slope) identifying accelerating customer remorse.
* **Consistent Worsening Detection**: Flags products exhibiting 3+ consecutive months of rising RSI.

---

## 📊 6. Power BI Dashboard Suite (Priority 8)

The system includes a dedicated 5-page Power BI dashboard suite documented in [`POWER_BI_DASHBOARD_GUIDE.md`](outputs/powerbi/POWER_BI_DASHBOARD_GUIDE.md):

| Page | Key Visualizations | Purpose |
| :--- | :--- | :--- |
| **1. Executive Overview** | KPI Cards (Total Reviews, Avg Rating, Avg RSI, High-Risk %), Category Donut, Monthly Volume/RSI Trend | Executive high-level health monitoring |
| **2. Product Risk Prioritization** | Top 10 Critical Products Bar Chart, RSI vs Slope Matrix Scatter, Risk Scorecard Table | Pinpoint products needing immediate hardware/software intervention |
| **3. Temporal Intelligence** | Category 3-Mo Rolling Curves, Risk Acceleration Curve ($\Delta$ slope) | Longitudinal drift and acceleration tracking |
| **4. Defect Taxonomy & Aspects** | Sentence-Level Aspect Sentiment Distribution (Positive vs Negative), Defect Severity Hierarchy | Granular component-level quality feedback |
| **5. Platform Analysis** | Amazon vs Flipkart KPI Parity, Regret Severity Boxplots | Retail channel comparative quality assessment |

---

## 📈 7. Key Findings & Insights

1. **Top Critical Products**: Samsung Buds 2, Galaxy Tab A8, and Galaxy S22 exhibited the highest risk scores driven by defect density and positive trend slopes.
2. **Defect Severity Hierarchy**: Camera (Mean RSI: 0.573), Display (0.542), and Battery (0.538) carry the highest regret penalties.
3. **Sentence-Level Aspects**: Battery and Software complaints exhibit the highest negative-to-positive ratio ($> 78\%$ negative mentions), while Camera and Display receive strong dual-sentiment polarity.
4. **Platform Parity**: Amazon (Mean RSI: 0.356) and Flipkart (Mean RSI: 0.351) demonstrated statistically indistinguishable regret distributions.
5. **Predictive Drivers**: In the temporal risk model, **Negative Review Ratio** and **Issue Density** were the two strongest leading indicators of future product risk.

---

## 💻 8. Tech Stack

* **Language**: Python 3.9+ / 3.10+ / 3.14
* **Data Warehousing & OLAP**: Pandas, NumPy, Star Schema Dimensional Modeling, SQL Window Analytics
* **NLP & Text Mining**: NLTK (`SentimentIntensityAnalyzer`, VADER Lexicon), Regular Expressions
* **Machine Learning**: Scikit-learn (LogisticRegression, KMeans, IsolationForest), MLxtend (Apriori), SciPy (linregress), Joblib
* **Data Visualization & BI**: Matplotlib, Seaborn, Microsoft Power BI Desktop
* **Notebook Environments**: Jupyter Notebooks, VS Code Interactive, Google Colab

---

## ▶️ 9. How to Run

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Nithin1138/Samsung-Regret-Intelligence-System.git
cd Samsung-Regret-Intelligence-System
pip install -r requirements.txt
```

### 2. Run Notebooks Sequentially
Execute notebooks in `notebooks/` in chronological order:
1. `01_data_generation.ipynb` — Dataset normalization & category standardization
2. `02_etl_pipeline.ipynb` — Pre-validation, VADER sentiment, aspect extraction, RSI
3. `03_warehouse.ipynb` — Star Schema data warehouse construction & SQL analytics
4. `04_mining.ipynb` — Non-circular temporal risk model, clustering, association rules
5. `05_analysis.ipynb` — Longitudinal temporal trends, OLS slopes, executive risk scorecard
6. `06_satisfaction_analysis.ipynb` — Positive sentiment drivers & feature satisfaction

### 3. Generate Power BI Datasets & Visual Renders
```bash
# Export Star Schema CSVs and Aspect datasets for Power BI
python3 scripts/export_powerbi_data.py

# Render 5-page dashboard visualization images
python3 scripts/generate_powerbi_visuals.py
```

---

## 📄 10. License

This project is open-source and available under the [MIT License](LICENSE).
