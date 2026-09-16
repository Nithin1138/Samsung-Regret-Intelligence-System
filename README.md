# Samsung Regret Intelligence System

An end-to-end data warehousing, NLP, and predictive intelligence system that converts unstructured multi-platform e-commerce customer reviews into actionable product risk metrics for Samsung consumer electronics.

---

## 📌 1. Overview & Problem Statement

Standard sentiment analysis and raw star ratings often fail to capture genuine consumer remorse:
* A **3-star review** may conceal a critical hardware defect (e.g., display flickering or battery degradation).
* A **1-star review** may reflect an isolated logistics issue rather than a product defect.

This project implements a **Star Schema Data Warehouse** and engineers a domain-specific metric called the **Regret Severity Index (RSI)** alongside a **Temporal Risk Scoring Engine**. The pipeline processes **7,494 multi-platform customer reviews** across **50 product lines**, enabling proactive defect detection and product risk prioritization.

---

## 🏗️ 2. System Architecture

```
                Samsung Reviews (11,000 raw)
                         │
                         ▼
              ┌─────────────────────┐
              │  Data Quality       │
              │  Validation         │
              │  (Duplicates, NaN,  │
              │   Invalid Ratings)  │
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  ETL Pipeline       │
              │  • Text Cleaning    │
              │  • VADER Sentiment  │
              │  • Issue Extraction │
              │  • RSI Calculation  │
              └────────┬────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌──────────────┐        ┌──────────────────┐
   │ Star Schema  │        │ Data Mining       │
   │ Warehouse    │        │ • Temporal Risk   │
   │ • fact_review│        │   Prediction      │
   │ • dim_product│        │ • KMeans Clusters │
   │ • dim_date   │        │ • Apriori Rules   │
   │ • dim_issue  │        │ • IsolationForest │
   │ • dim_platfrm│        └────────┬──────────┘
   └──────┬───────┘                 │
          │                         │
          ▼                         ▼
   ┌──────────────┐        ┌──────────────────┐
   │ OLAP & SQL   │        │ Temporal          │
   │ Analytics    │        │ Intelligence      │
   │ • Roll-up    │        │ • Monthly RSI     │
   │ • Drill-down │        │ • Rolling 3-month │
   │ • Slice/Dice │        │ • Trend Slopes    │
   │ • Window Fns │        │ • Risk Accel.     │
   └──────┬───────┘        └────────┬──────────┘
          │                         │
          └────────────┬────────────┘
                       ▼
              ┌─────────────────────┐
              │  Product Risk       │
              │  Scorecard          │
              │  Dynamic Tiering    │
              │  (Critical/High/    │
              │   Moderate/Low)     │
              └─────────────────────┘
```

---

## 📂 3. Project Structure

```
Samsung-Regret-Intelligence-System/
├── src/                                    # Reusable Python modules
│   ├── __init__.py
│   ├── data_quality.py                     # Raw data validation & quality report
│   ├── preprocessing.py                    # Text cleaning, deduplication, rating fix
│   ├── sentiment.py                        # VADER sentiment scoring & classification
│   ├── issue_extraction.py                 # Rule-based keyword issue categorization
│   ├── rsi.py                              # RSI computation, monthly/rolling/trend/accel.
│   ├── etl.py                              # Full ETL pipeline orchestrator
│   ├── warehouse.py                        # Star schema builder & SQL-style queries
│   ├── mining.py                           # KMeans, Apriori, IsolationForest, temporal predictor
│   └── risk_analysis.py                    # Risk scorecard, trend-based scoring, worsening detection
├── data/
│   ├── raw/
│   │   ├── samsung_dirty_dataset.csv       # Raw uncleaned reviews (11,000 rows)
│   │   └── samsung_clean_dataset.csv       # Deduplicated reviews (7,494 rows)
│   ├── processed/
│   │   └── samsung_processed_dataset.csv   # Feature-engineered dataset with NLP & RSI
│   └── warehouse/
│       ├── fact_review.csv                 # Central fact table with surrogate keys
│       ├── dim_product.csv                 # Product dimension (50 products)
│       ├── dim_platform.csv                # Platform dimension (Amazon, Flipkart)
│       ├── dim_issue.csv                   # Issue dimension (11 categories)
│       └── dim_date.csv                    # Date dimension (Year, Month, Day, Quarter)
├── notebooks/
│   ├── 01_data_generation.ipynb            # Raw data ingestion & cleaning
│   ├── 02_etl_pipeline.ipynb               # Data quality → Sentiment → Issues → RSI
│   ├── 03_warehouse.ipynb                  # Star Schema + SQL-style analytics
│   ├── 04_mining.ipynb                     # Temporal prediction, clustering, rules, outliers
│   ├── 05_analysis.ipynb                   # Temporal intelligence, OLAP, risk scorecard
│   └── 06_satisfaction_analysis.ipynb      # Positive sentiment & satisfaction analysis
├── outputs/
│   ├── models/                             # Serialized ML models (.pkl)
│   └── plots/                              # Generated visualizations (.png)
├── requirements.txt
└── README.md
```

---

## 🔬 4. Core Methodologies

### A. Regret Severity Index (RSI)

RSI balances textual polarity, numerical rating deviation, defect presence, and text specificity into a normalized continuous score from `0.0` (low regret) to `1.0` (high regret):

```
RSI = (0.4 × Negative_Sentiment) + (0.3 × Rating_Deviation) + (0.2 × Keyword_Intensity) + (0.1 × Specificity)
```

Where:
* **`Negative_Sentiment`** = max(0, -SentimentCompoundScore): Isolates pure negative polarity from VADER compound scores (0.0 → 1.0).
* **`Rating_Deviation`** = |5 - Rating| / 4: Measures divergence from a 5-star rating (0.0 → 1.0).
* **`Keyword_Intensity`**: Binary flag (1 if a hardware/software issue keyword is detected, 0 if none).
* **`Specificity`**: Calibrated constant weight (0.6) reflecting review descriptive granularity.

### B. Composite Product Risk Score (Trend-Based)

Products are scored using **trend-based growth** (linear regression slope) rather than simple endpoint comparison:

```
Risk_Score = (0.35 × Avg_RSI_norm) + (0.25 × RSI_Slope_norm) + (0.20 × Issue_Density_norm)
           + (0.10 × Neg_Ratio_norm) + (0.10 × Risk_Acceleration_norm)
```

Where:
* **`RSI_Slope`**: Linear regression slope of monthly RSI — captures sustained worsening/improving trend.
* **`Risk_Acceleration`**: Second derivative (slope of slope) — identifies whether risk is accelerating or decelerating.
* All components are min-max normalized to [0, 1] before weighting.

### C. Dynamic Risk Tiering

Products are classified into tiers based on interquartile score distributions (Q1, Q2, Q3):
* **Low Risk:** Risk Score ≤ Q1
* **Moderate Risk:** Q1 < Risk Score ≤ Q2
* **High Risk:** Q2 < Risk Score ≤ Q3
* **Critical Risk:** Risk Score > Q3

---

## ⚙️ 5. Machine Learning & Analytical Modules

| Module | Technique | Details | Key Result |
| :--- | :--- | :--- | :--- |
| **Temporal Risk Prediction** | Logistic Regression (balanced) | Predicts **next-quarter** high risk from current-quarter features (avg RSI, neg ratio, issue density, avg rating, avg sentiment, review count). Time-based train/test split. | Correctly identifies at-risk products using non-circular features |
| **Unsupervised Clustering** | K-Means (k=3) | Segments customer reviews into Low/Medium/High risk bands based on RSI | 3 distinct risk clusters with clear separation |
| **Association Mining** | Apriori Algorithm | Mines co-occurrence patterns across Issue × Risk × Platform (min_support=0.01, min_confidence=0.3) | Multi-feature rules discovered linking issue types to risk levels |
| **Anomaly Detection** | Isolation Forest | Identifies extreme sentiment-rating discrepancies | 5% contamination rate |
| **OLAP Operations** | Multi-Index Aggregations | Roll-Up, Drill-Down, Slice, Dice across Category, Time, Platform, Product | Complete dimensional analysis |
| **SQL-Style Analytics** | Pandas (SQL equivalents) | RANK, LAG, running SUM window functions; CTEs via method chaining; HAVING/subquery patterns | 6 analytical queries with documented SQL equivalents |

### Temporal Intelligence Metrics

| Metric | Method | Purpose |
| :--- | :--- | :--- |
| Monthly RSI | groupby(Product, Month).RSI.mean() | Base time series per product |
| 3-Month Rolling RSI | rolling(3).mean() | Smoothed trend |
| RSI Trend Slope | scipy.stats.linregress | Direction & velocity of change |
| Review Volume Trend | linregress on monthly counts | Growing/shrinking attention |
| Negative Sentiment Trend | rolling(3).mean() of neg ratio | Tone deterioration |
| Risk Acceleration | Slope of slopes (2nd derivative) | Accelerating vs. decelerating |
| Consistent Worsening | Consecutive increase detection | Products with N+ months of rising RSI |

---

## 📊 6. Key Findings

1. **Top Critical Products:** Samsung Air Conditioner, Samsung Washing Machine, and Samsung Galaxy Buds2 exhibited the highest composite risk scores due to high issue density and worsening RSI trends.
2. **Defect Severity Hierarchy:** Camera, General Issues, Battery, Display, and Performance are the top 5 issue categories by mean RSI.
3. **Temporal Intelligence:** 25 products showed 3+ consecutive months of RSI increase, indicating sustained worsening trends.
4. **Platform Parity:** Customer dissatisfaction remained consistent across Amazon and Flipkart platforms.
5. **Predictive Features:** Negative sentiment ratio and issue density are the strongest predictors of future high risk.

---

## 💻 7. Tech Stack

* **Language:** Python 3.10+
* **Data Processing & Warehousing:** Pandas, NumPy, Regular Expressions, Star Schema Dimensional Modeling
* **Natural Language Processing:** NLTK (VADER Sentiment Intensity Analyzer)
* **Machine Learning & Mining:** Scikit-learn, MLxtend, SciPy, Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebooks

---

## ▶️ 8. How to Run

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Nithin1138/Samsung-Regret-Intelligence-System.git
cd Samsung-Regret-Intelligence-System
pip install -r requirements.txt
```

### 2. Execute Notebook Pipeline Sequentially
Run the notebooks in `/notebooks/` in order:

1. `01_data_generation.ipynb` — Cleans raw data and extracts standard categories.
2. `02_etl_pipeline.ipynb` — Data quality validation, VADER sentiment, issue extraction, RSI computation.
3. `03_warehouse.ipynb` — Star Schema construction + SQL-style analytical queries.
4. `04_mining.ipynb` — Temporal risk prediction, K-Means clustering, Apriori mining, Isolation Forest.
5. `05_analysis.ipynb` — Temporal intelligence (rolling RSI, trend slopes, acceleration), OLAP operations, risk scorecard.
6. `06_satisfaction_analysis.ipynb` — Positive sentiment drivers & satisfaction analysis.

---

## 📄 9. License

This project is open-source and available under the [MIT License](LICENSE).
