# Enterprise Data Warehouse & Temporal Regret Intelligence System

An end-to-end data warehousing, NLP, and predictive machine learning intelligence system that converts unstructured multi-platform e-commerce customer reviews into actionable product risk intelligence for Samsung consumer electronics and appliances.

---

## 📌 1. Overview & Problem Statement

Standard sentiment analysis and raw star ratings often fail to capture genuine consumer remorse:
* A **3-star review** may conceal a critical hardware defect (e.g., display flickering or battery degradation).
* A **1-star review** may reflect an isolated logistics issue rather than a product defect.

To address this dissonance, this project implements a **Star Schema Data Warehouse** and engineers a domain-specific metric called the **Regret Severity Index (RSI)** alongside a **Temporal Risk Scoring Engine**. The pipeline processes 7,494 multi-platform customer reviews across 25+ product lines, enabling proactive defect detection and product risk prioritization.

---

## 🏗️ 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. RAW DATA INGESTION                           │
│   11,000 Dirty Reviews -> Sanitized to 7,494 Records (Amazon, Flipkart)│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     2. ETL & NLP FEATURE PIPELINE                      │
│ • Text Normalization (Regex, URL removal, Lowercasing)                 │
│ • NLTK VADER Sentiment Intensity Analyzer -> Compound Polarity Score   │
│ • Multi-Class Issue Categorization (10 Hardware/Software Bins)         │
│ • Mathematical Regret Severity Index (RSI) Computation                 │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               3. DATA WAREHOUSE LAYER (STAR SCHEMA)                    │
│                                                                        │
│   ┌──────────────┐      ┌─────────────────┐      ┌─────────────────┐   │
│   │ dim_product  │◄────┤   fact_review   ├────►│  dim_platform   │   │
│   │ (Product_ID) │      │  (Review_ID FKs │      │  (Platform_ID)  │   │
│   └──────────────┘      │  Rating, Sent,  │      └─────────────────┘   │
│   ┌──────────────┐      │     RSI)        │      ┌─────────────────┐   │
│   │  dim_issue   │◄────┤                 ├────►│    dim_date     │   │
│   │  (Issue_ID)  │      └─────────────────┘      │    (Date_ID)    │   │
│   └──────────────┘                               └─────────────────┘   │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │                                 │
                   ▼                                 ▼
┌──────────────────────────────────────┐ ┌───────────────────────────────┐
│     4. DATA MINING & ML MODELS       │ │     5. OLAP ANALYTICS ENGINE  │
│ • Logistic Regression (91% Accuracy) │ │ • Roll-Up (Category Aggreg.)  │
│ • K-Means Clustering (3 Segments)    │ │ • Drill-Down (Year -> Month)  │
│ • Apriori Rule Mining (56 Rules)     │ │ • Slice (Smartphones only)    │
│ • Isolation Forest (Outlier Triage)  │ │ • Dice (Multi-Condition Sub)  │
└──────────────────┬───────────────────┘ └───────────────┬───────────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│         6. TEMPORAL RISK SCORING & DECISION INTELLIGENCE               │
│  Risk Score = (0.5 * Avg_RSI) + (0.3 * Growth_Rate) + (0.2 * Density)  │
│  Outputs: Dynamic Risk Tiers (Critical/High/Moderate/Low) & Visuals    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 3. Project Structure

```
DWDM_Project/
├── data/
│   ├── raw/
│   │   ├── samsung_dirty_dataset.csv       # Raw uncleaned review dataset (11,000 rows)
│   │   └── samsung_clean_dataset.csv       # Sanitized dataset (7,494 rows)
│   ├── processed/
│   │   └── samsung_processed_dataset.csv   # Feature-engineered dataset with NLP & RSI
│   └── warehouse/
│       ├── fact_review.csv                 # Central fact table with surrogate foreign keys
│       ├── dim_product.csv                 # Product dimension (50 product variants)
│       ├── dim_platform.csv                # Platform dimension (Amazon, Flipkart)
│       ├── dim_issue.csv                   # Defect category dimension (11 classes)
│       └── dim_date.csv                    # Temporal dimension (Date, Year, Month, Day)
├── notebooks/
│   ├── 01_data_generation.ipynb            # Ingestion, regex cleaning, category normalization
│   ├── 02_etl_pipeline.ipynb               # VADER sentiment, issue extraction, RSI calculation
│   ├── 03_warehouse.ipynb                  # Star Schema dimensional modeling & key mapping
│   ├── 04_mining.ipynb                     # ML Classification, Clustering, Apriori, Anomaly Detection
│   ├── 05_analysis.ipynb                   # OLAP cubes, temporal risk scoring engine, visualizations
│   └── 06_satisfaction_analysis.ipynb     # Positive sentiment drivers & satisfaction analysis
├── outputs/
│   ├── models/                             # Serialized models (logistic_model.pkl, kmeans_model.pkl)
│   ├── plots/                              # Visual analytics (trend lines, distributions, bar charts)
│   └── top_positive_products.csv           # Top satisfaction ranking outputs
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation
```

---

## 🔬 4. Core Methodologies & Mathematical Formulations

### A. Regret Severity Index (RSI)
RSI balances textual polarity, numerical rating deviation, defect presence, and text specificity into a normalized continuous score from `0.0` (low regret) to `1.0` (high regret):

```
RSI = (0.4 × Negative_Sentiment) + (0.3 × Rating_Deviation) + (0.2 × Keyword_Intensity) + (0.1 × Specificity)
```

Where:
* **`Negative_Sentiment`** $= \max(0, -\text{SentimentCompoundScore})$: Isolates pure negative polarity from VADER compound scores ($0.0 \to 1.0$).
* **`Rating_Deviation`** $= \frac{|5 - \text{Rating}|}{4}$: Measures divergence from a 5-star rating ($0.0 \to 1.0$).
* **`Keyword_Intensity`**: Binary flag ($1$ if a hardware/software issue keyword is detected, $0$ if none).
* **`Specificity`**: Calibrated constant weight ($0.6$) reflecting review descriptive granularity.

### B. Composite Product Risk Score
Calculated per product line to prioritize systemic defect trends over isolated reviews:

```
Risk_Score = (0.5 × Avg_RSI) + (0.3 × Growth_Rate) + (0.2 × Issue_Density)
```

Where:
* **`Avg_RSI`**: Mean Regret Severity Index for the product.
* **`Growth_Rate`**: Min-max normalized temporal trajectory ($RSI_{\text{latest}} - RSI_{\text{earliest}}$) representing defect acceleration.
* **`Issue_Density`**: Proportion of reviews containing identified defect keywords ($\frac{N_{\text{issues}}}{N_{\text{total}}}$).

### C. Dynamic Risk Tiering
Products are dynamically classified into tiers based on interquartile score distributions ($Q_1, Q_2, Q_3$):
* **Low Risk:** $\text{Risk Score} \le Q_1$ ($\le 0.269$)
* **Moderate Risk:** $Q_1 < \text{Risk Score} \le Q_2$ ($0.269 - 0.283$)
* **High Risk:** $Q_2 < \text{Risk Score} \le Q_3$ ($0.283 - 0.294$)
* **Critical Risk:** $\text{Risk Score} > Q_3$ ($> 0.294$)

---

## ⚙️ 5. Machine Learning & Analytical Modules

| Module | Technique | Implementation Details | Verified Metric / Result |
| :--- | :--- | :--- | :--- |
| **Supervised Classification** | Logistic Regression | Trained on `[Rating, Sentiment_score]` to predict `High_Risk` (`RSI > 0.5`) | **91% Accuracy**, 0.86 F1-Score (High Risk), 0.93 F1-Score (Low Risk) |
| **Unsupervised Clustering** | K-Means ($k=3$) | Segmented customer remorse distribution into distinct operational bands | **Cluster 0 (High):** Mean RSI 0.624<br>**Cluster 1 (Low):** Mean RSI 0.144<br>**Cluster 2 (Medium):** Mean RSI 0.438 |
| **Association Mining** | Apriori Algorithm | Mined multi-feature itemsets (`min_support=0.01`, `min_confidence=0.3`) | **56 Rules Discovered**<br>Top: `(Issue_general_issue, Flipkart) → High_Risk` (Conf: 85.8%, Lift: 2.59) |
| **Anomaly Detection** | Isolation Forest | Identified non-conforming reviews and extreme sentiment discrepancies | **Contamination Rate = 5%** |
| **Dimensional OLAP** | Multi-Index Aggregations | Evaluated data cubes across Category, Time, Platform, and Product | **Roll-Up, Drill-Down, Slice, Dice** |

---

## 📊 6. Key Findings & Insights

1. **Top Critical Products:** Samsung Crystal UHD TV (Score: 0.450), Samsung Washing Machine (Score: 0.384), and Samsung Air Conditioner (Score: 0.346) exhibited the highest risk due to strong issue density and rising temporal regret.
2. **Defect Severity Hierarchy:** Defect categories with the highest average regret impact:
   * **Camera:** Mean RSI = 0.573
   * **General Issues:** Mean RSI = 0.564
   * **Battery:** Mean RSI = 0.556
   * **Display:** Mean RSI = 0.539
   * **Performance:** Mean RSI = 0.525
3. **Feature Impact:** Model coefficients revealed that sentiment polarity (Weight = -9.88) had approximately **3.1× higher influence** on high-risk prediction than numerical star ratings (Weight = -3.15).
4. **Platform Parity:** Overall customer dissatisfaction remained consistent across retail platforms (Amazon: Mean RSI = 0.356 vs. Flipkart: Mean RSI = 0.351).

---

## 💻 7. Tech Stack

* **Language:** Python 3.10+
* **Data Processing & Warehousing:** Pandas, NumPy, Regular Expressions (`re`), Dimensional Modeling (Star Schema)
* **Natural Language Processing:** NLTK (`SentimentIntensityAnalyzer`, VADER Lexicon)
* **Machine Learning & Mining:** Scikit-learn, MLxtend, Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebooks, Google Colab

---

## ▶️ 8. How to Run

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Nithin1138/Samsung-Regret-Intelligence-System.git
cd DWDM_Project
pip install -r requirements.txt
```

### 2. Execute Notebook Pipeline Sequentially
Run the notebooks in `/notebooks/` in the following sequence:

1. `01_data_generation.ipynb` — Cleans raw data and extracts standard categories.
2. `02_etl_pipeline.ipynb` — Runs VADER sentiment analysis, issue extraction, and RSI computation.
3. `03_warehouse.ipynb` — Constructs the Star Schema fact and dimension tables.
4. `04_mining.ipynb` — Executes Logistic Regression, K-Means clustering, Apriori mining, and Isolation Forest.
5. `05_analysis.ipynb` — Executes OLAP operations, computes product risk scores, and generates analytical plots.
6. `06_satisfaction_analysis.ipynb` — Analyzes positive drivers, feature satisfaction, and top-performing products.

---

## 📄 9. License

This project is open-source and available under the [MIT License](LICENSE).
