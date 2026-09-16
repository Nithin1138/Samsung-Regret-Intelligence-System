# 🚀 Enterprise Data Warehouse & Temporal Regret Intelligence System (Samsung)

---

## 📌 1. Resume Version (3-4 Bullet Points for Resume)

Copy and paste directly into your resume under **Projects** or **Experience**:

- **Architected an End-to-End Enterprise Data Warehouse & Intelligence Pipeline** in Python/SQL analyzing 7,500+ multi-channel e-commerce customer reviews across 25+ product lines using a **Star Schema** (`fact_review` linked with 4 dimension tables).
- **Engineered a Domain-Specific Regret Severity Index (RSI)** combining VADER compound NLP sentiment, star-rating deviations, extracted product issue categories, and text specificity (0 to 1 scale) to quantify latent post-purchase buyer dissonance.
- **Deployed Predictive Machine Learning & Mining Models** including Logistic Regression (**91% Accuracy**, 0.86 F1-score), K-Means Clustering (Low/Moderate/High Regret segmentation), Isolation Forest (outlier triage), and Apriori Association Rules (56 cross-issue risk rules).
- **Built an OLAP Analytical Engine & Dynamic Risk Scoring Matrix** utilizing Roll-Up, Drill-Down, Slicing, and Dicing to forecast post-purchase risk trajectory, identifying critical failure velocity in appliances and flagship consumer electronics.

---

## 🧭 2. High-Level Elevator Pitch (30-Second Interview Summary)

> *"In this project, I built an **Enterprise Data Warehouse and Temporal Regret Intelligence System** tailored for consumer electronics and appliances. Standard sentiment analysis often fails because a 3-star review can hide severe hardware defects, while a 1-star review might just be a delivery delay.*
> 
> *To fix this, I engineered the **Regret Severity Index (RSI)**—a multi-factor metric fusing NLP sentiment polarity, rating divergence, and keyword issue extraction. I structured 7.5k+ multi-platform reviews into a Star Schema Data Warehouse, executed OLAP operations (Roll-up, Drill-down, Slice, Dice), and trained predictive classification, clustering, and association mining models with 91% accuracy to prioritize high-risk product lines before reputational damage escalates."*

---

## 🏗️ 3. System Architecture & End-to-End Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. RAW DATA INGESTION                           │
│  7,496 E-Commerce Reviews (Amazon & Flipkart) for 25+ Samsung Products │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     2. ETL & NLP FEATURE PIPELINE                      │
│ • Text Normalization & Lowercasing                                     │
│ • VADER Sentiment Intensity Analyzer -> Compound Polarity Score        │
│ • Domain Rule-Based Issue Categorization (10 Hardware/Software Bins)   │
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
│ • Logistic Regression (91% Acc)      │ │ • Roll-Up (Category Aggreg.)  │
│ • K-Means Clustering (3 Segments)    │ │ • Drill-Down (Year -> Month)  │
│ • Apriori Rule Mining (56 Rules)     │ │ • Slice (Smartphones only)    │
│ • Isolation Forest (Anomaly Triage)  │ │ • Dice (Multi-Condition Sub)  │
└──────────────────┬───────────────────┘ └───────────────┬───────────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│         6. TEMPORAL RISK SCORING & DECISION INTELLIGENCE               │
│  Risk Score = (0.5 * Avg_RSI) + (0.3 * Growth_Rate) + (0.2 * Density)  │
│  Outputs: Risk Tiers (Critical/High/Moderate/Low) & Visual Dashboard   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 4. Comprehensive Tech Stack

| Domain | Technology / Library | Purpose in this Project |
| :--- | :--- | :--- |
| **Programming** | Python 3.10+ | Core language for pipelines, mining algorithms, analytics |
| **Data Processing** | Pandas, NumPy | Data cleaning, reshaping, aggregation, matrix ops, star schema creation |
| **NLP & Text Mining** | NLTK (VADER Sentiment Analyzer) | Polarity scoring (`compound`, `pos`, `neg`, `neu`) on customer verbatims |
| **Machine Learning** | Scikit-Learn (`sklearn`) | Logistic Regression classifier, K-Means clustering, Isolation Forest |
| **Association Mining**| MLxtend | Apriori algorithm, Association Rules (support, confidence, lift) |
| **Data Warehousing** | Dimensional Modeling (Star Schema) | Relational fact & dimension tables for scalable analytical queries |
| **Analytics & OLAP** | Multi-index Pandas Aggregations | Roll-up, Drill-down, Slicing, and Dicing multidimensional data cubes |
| **Data Visualization**| Matplotlib, Seaborn | Temporal risk trends, distribution plots, feature weight charts |
| **Environment / Tools**| Jupyter Notebooks, Google Colab, Git | Experimentation, version control, and modular notebook structure |

---

## 🔬 5. Core Methodologies & Mathematical Formulations

### A. Regret Severity Index (RSI) Formula
Standard sentiment or raw star ratings do not represent genuine consumer remorse. RSI bridges text sentiment and numerical rating:

$$\text{RSI} = 0.4 \cdot \text{Neg\_Sent} + 0.3 \cdot \text{Rating\_Dev} + 0.2 \cdot \text{Issue\_Intensity} + 0.1 \cdot \text{Specificity}$$

Where:
- **$\text{Neg\_Sent} = \max(0, -\text{Sentiment\_Compound\_Score})$**: Isolates pure negative polarity ($0.0 \to 1.0$).
- **$\text{Rating\_Dev} = \frac{|5 - \text{Rating}|}{4}$**: Divergence from a perfect 5-star experience ($0.0 \to 1.0$).
- **$\text{Issue\_Intensity} = 1 \text{ if Issue} \neq \text{"none"} \text{ else } 0$**: Weight for actionable defect mentions.
- **$\text{Specificity} = 0.6$ (baseline constant)**: Controls text informativeness.
- **Output Range**: Clamped between $[0.0, 1.0]$. Reviews with $\text{RSI} > 0.5$ are labeled **High Risk**.

### B. Issue Taxonomy & Keyword Extraction Matrix
Reviews are scanned using an optimized rule engine across 9 core hardware/software issue dictionaries:
1. **Battery**: `['battery', 'drain', 'backup', 'charging', 'charge', 'power', 'life', 'dead']`
2. **Heating**: `['heat', 'overheat', 'hot', 'warm', 'temperature']`
3. **Display**: `['display', 'screen', 'flicker', 'touch', 'pixel', 'crack', 'brightness']`
4. **Performance**: `['slow', 'lag', 'hang', 'freeze', 'crash', 'unresponsive', 'delay']`
5. **Camera**: `['camera', 'blur', 'photo', 'focus', 'video', 'picture']`
6. **Sound**: `['sound', 'audio', 'speaker', 'volume', 'mic']`
7. **Connectivity**: `['wifi', 'bluetooth', 'network', 'signal', 'disconnect']`
8. **Build Quality**: `['build', 'material', 'fragile', 'scratch', 'broken', 'cheap']`
9. **Software**: `['software', 'update', 'bug', 'glitch', 'app', 'ui', 'laggy']`
*Fallback Logic*: If a review has $\le 2$ stars with strong negative sentiment but no explicit keyword match, it is tagged as `general_issue`.

### C. Star Schema Data Warehouse Design
Constructed in 3rd Normal Form for dimensions and denormalized fact records:
- **`dim_product`**: `[Product_ID (PK), Product_Name, Category (Smartphone, Accessories, Appliance, Other)]`
- **`dim_platform`**: `[Platform_ID (PK), Platform_Name (Amazon, Flipkart)]`
- **`dim_issue`**: `[Issue_ID (PK), Issue_Name]`
- **`dim_date`**: `[Date_ID (PK), Date, Day, Month, Year, Quarter]`
- **`fact_review`**: `[Review_ID (PK), Product_ID (FK), Platform_ID (FK), Issue_ID (FK), Date_ID (FK), Rating, Sentiment_Score, RSI]`

### D. Data Mining Algorithms
1. **Supervised Classification (Logistic Regression)**:
   - **Target**: `High_Risk` binary flag ($\text{RSI} > 0.5$).
   - **Features**: `Rating`, `Sentiment_score`.
   - **Performance**: **91% Accuracy**, Precision: 0.88, Recall: 0.84, F1-Score: 0.86 on test split.
   - **Feature Weights**: `Sentiment_score` ($-9.88$) contributes over $3\times$ more predictive power than `Rating` ($-3.15$).
2. **Unsupervised Clustering (K-Means, $k=3$)**:
   - Discovered 3 natural review personas: **Low Risk** ($\text{Avg RSI} \approx 0.08$), **Moderate Concern** ($\text{Avg RSI} \approx 0.35$), and **High Regret** ($\text{Avg RSI} \approx 0.68$).
3. **Association Rule Mining (Apriori)**:
   - Evaluated cross-attribute itemsets; uncovered 56 association rules.
   - *Key finding*: `(Issue_general_issue, Platform_Flipkart) => (High_Risk)` held high confidence and lift $> 1.8$, indicating delivery/platform-level quality disputes.
4. **Outlier Detection (Isolation Forest)**:
   - Set contamination factor $\nu = 0.05$ to capture anomalous reviews (e.g., sarcastic positive text with 1-star ratings or vice versa).

### E. Multi-Factor Product Risk Scoring Engine
To prioritize which product requires immediate recall, firmware fix, or manufacturer audit:

$$\text{Risk Score} = 0.5 \cdot \overline{\text{RSI}} + 0.3 \cdot \text{Growth Rate} + 0.2 \cdot \text{Issue Density}$$

- **$\overline{\text{RSI}}$**: Historical average regret index for the product.
- **$\text{Growth Rate} = \text{RSI}_{t_{\text{end}}} - \text{RSI}_{t_0}$**: Temporal delta showing if regret is accelerating over quarters.
- **$\text{Issue Density} = \frac{\text{Count of Defect Reviews}}{\text{Total Reviews for Product}}$**.
- **Risk Tiers**: Normalized and segmented into 4 Quartiles $\to$ **Critical**, **High**, **Moderate**, **Low**.

---

## 📊 6. Key Business Insights & Findings

1. **Top Critical Risk Products**:
   - **Samsung Crystal UHD TV** ($\text{Risk Score} = 0.4496$): Highest risk score driven by a $1.00$ growth rate spike in display & panel failure complaints over time.
   - **Samsung Washing Machine** ($\text{Risk Score} = 0.3839$, $\text{Issue Density} = 0.75$): Extremely high concentration of build quality and motor noise defects.
   - **Samsung Air Conditioner** & **Samsung Galaxy Buds2**: Elevated heating and connectivity complaints.
2. **Cross-Platform Disparity**:
   - Flipkart reviews exhibited slightly higher variance in negative sentiment compared to Amazon, predominantly due to packaging and fulfillment-related keywords.
3. **Temporal Regret Surges**:
   - Spike in high-risk reviews ($> 70\%$) observed in post-holiday sales quarters (January / December), signaling supply chain batch defects and post-launch buyer remorse.

---

## 🎯 7. Top 15 Technical Interview Questions & Answers

### Q1: What is the business motivation behind building this system?
> **Answer:** Standard product analytics rely on average star ratings, which hide critical problems. A 3.8-star product might appear healthy overall while suffering a 40% spike in catastrophic battery failures among recent buyers. This system ingests unstructured text, standardizes it in a Data Warehouse, and computes an automated **Regret Severity Index (RSI)** to alert product management and QA teams to emerging defects before product recall or brand erosion occurs.

---

### Q2: Why did you create a custom RSI metric instead of using raw Sentiment or Star Rating alone?
> **Answer:** Raw star ratings are subjective and lack context (e.g., 1-star due to courier delay vs. motherboard dead). Sentiment analysis alone fails on short or sarcastic text. RSI combines:
> 1. Negative sentiment intensity ($40\%$).
> 2. Numerical rating divergence from 5 stars ($30\%$).
> 3. Verified presence of technical defect keywords ($20\%$).
> 4. Text specificity factor ($10\%$).
> This creates a calibrated $[0, 1]$ risk index where $0.5+$ reliably indicates true product remorse.

---

### Q3: Explain your Star Schema Data Warehouse design. Why Star Schema over Snowflake?
> **Answer:** I designed a Star Schema featuring a centralized `fact_review` table surrounded by four de-normalized dimension tables: `dim_product`, `dim_platform`, `dim_issue`, and `dim_date`. 
> I chose a **Star Schema** because:
> - Analytical queries (OLAP aggregations like roll-up and slice) execute with fewer SQL `JOIN`s compared to a normalized Snowflake schema.
> - It simplifies read-heavy data mining and reporting workloads.
> - Fact table stores surrogate foreign keys and numerical metrics (`Rating`, `Sentiment_score`, `RSI`), maximizing aggregation throughput.

---

### Q4: Walk me through the OLAP operations you performed and what business questions they answered.
> **Answer:**
> - **Roll-Up:** Aggregated review RSI from Product level to Category level (`Smartphone`, `Accessories`, `Appliance`, `Other`), revealing that "Other/Appliances" had higher average regret than consumer phones.
> - **Drill-Down:** Decomposed annual metrics down to monthly time slices (`Year -> Month`), discovering sudden quality deterioration during post-festival sale months.
> - **Slice:** Filtered data along a single dimension (e.g., `Category == 'Smartphone'`) to benchmark models like Galaxy S22 vs. Galaxy A54.
> - **Dice:** Filtered across multiple dimensions simultaneously (`Category == 'Smartphone'` AND `Platform == 'Amazon'` AND `Rating <= 2`), pinpointing specific channel-product failure intersections.

---

### Q5: How did you handle NLP sentiment extraction, and why VADER?
> **Answer:** I utilized NLTK's **VADER (Valence Aware Dictionary and sEntiment Reasoner)**. VADER is particularly effective for e-commerce reviews because:
> - It is rule-based and tuned for social/informal text, correctly parsing capitalization, punctuation (e.g., "POOR battery!!!"), and negations ("not good").
> - It runs orders of magnitude faster than Transformer models (BERT) during batch ETL pipelines without requiring GPU infrastructure.
> - The normalized `compound` polarity score ($-1.0$ to $+1.0$) mapped seamlessly into our mathematical RSI equation.

---

### Q6: What classification models did you build, and what were the evaluation metrics?
> **Answer:** I framed high regret detection as a binary classification problem ($\text{RSI} > 0.5 \implies \text{High Risk}$). Using Logistic Regression with a stratified 80/20 train-test split:
> - **Accuracy:** 91%
> - **Precision (High Risk):** 0.88
> - **Recall (High Risk):** 0.84
> - **F1-Score:** 0.86
> Feature weight analysis revealed `Sentiment_score` had a coefficient of $-9.88$ vs. `Rating` of $-3.15$, proving that text sentiment is the primary driver of high regret.

---

### Q7: Why did you apply K-Means clustering, and how did you choose $k$?
> **Answer:** To uncover unsupervised customer sentiment groupings without imposing manual thresholds. Using the Elbow Method and Silhouette evaluation, $k=3$ was optimal:
> - **Cluster 0 (Low Risk):** Average $\text{RSI} \approx 0.08$ (Enthusiastic buyers, ratings 4–5).
> - **Cluster 1 (Moderate Concern):** Average $\text{RSI} \approx 0.35$ (Minor UX or price concerns, rating 3).
> - **Cluster 2 (High Regret):** Average $\text{RSI} \approx 0.68$ (Hardware failures, battery drain, display lines, rating 1–2).

---

### Q8: What did Apriori Association Rule Mining reveal?
> **Answer:** By encoding categorical features (`Issue`, `Platform`, `High_Risk`), the Apriori algorithm extracted 56 frequent itemset rules. 
> A critical rule observed was:
> $$\{\text{Issue: general\_issue}, \text{Platform: Flipkart}\} \implies \{\text{High\_Risk}\} \quad (\text{Confidence: } > 0.75, \text{Lift: } > 1.8)$$
> This demonstrated that generic complaints on Flipkart strongly correlated with extreme regret, often signaling courier or unboxing disputes rather than internal silicon defects.

---

### Q9: How does your Risk Scoring Engine calculate product prioritization?
> **Answer:** A static average RSI is insufficient because an older product might have high past regret that was already patched. The **Risk Scoring Engine** combines three factors:
> 1. **Baseline Regret ($50\%$):** Average product RSI.
> 2. **Temporal Growth Rate ($30\%$):** $\Delta \text{RSI}$ between recent and baseline quarters.
> 3. **Issue Density ($20\%$):** Percentage of reviews containing verified technical issues.
> This flagged the **Samsung Crystal UHD TV** and **Washing Machine** as Critical because their defect velocity was accelerating.

---

### Q10: How did you clean dirty review data and handle edge cases?
> **Answer:**
> - **Missing Values & Inconsistencies:** Trimmed whitespace, unified lowercase strings for product names and platforms.
> - **Low-Rating "None" Case:** If a user gave a $\le 2$-star review with strong negative sentiment ($\text{compound} < -0.2$) but didn't mention specific hardware keywords, the ETL pipeline reassigned the issue from `none` to `general_issue`.
> - **False Positives:** 5-star reviews with generic words were prevented from falsely triggering issue flags.

---

### Q11: Why did you use Isolation Forests for Outlier Detection?
> **Answer:** To detect review anomalies (contamination set at $5\%$). These anomalies highlight:
> - Fake reviews or bot spam (e.g., generic 5-star text with a 1-star rating).
> - Sarcastic reviews ("Amazing phone if you want a space heater!").
> Flagging these isolates noisy data from skewing the Data Warehouse metrics.

---

### Q12: How would you scale this pipeline for real-time production (1M+ reviews/day)?
> **Answer:**
> 1. **Streaming Ingestion:** Replace CSV batch loading with **Apache Kafka** or **AWS Kinesis** ingesting live marketplace webhooks.
> 2. **Distributed Processing:** Port the ETL & RSI feature engine to **Apache Spark (PySpark)** or **AWS Glue**.
> 3. **Cloud Warehouse:** Migrate Star Schema CSVs to **Snowflake**, **Google BigQuery**, or **Amazon Redshift** with clustered partition keys on `Date_ID` and `Product_ID`.
> 4. **Model Serving:** Package the classifier and scoring engine in a **FastAPI** microservice deployed on Kubernetes (EKS/GKE) with MLflow model registry.

---

### Q13: What were the most common product issues detected across Samsung devices?
> **Answer:** The issue extraction distribution showed **Battery & Charging Drain** as the top reported hardware complaint in smartphones, followed closely by **Display/Touch issues** and **Heating/Thermal throttling** during intensive gaming/charging. In home appliances, **Build Quality & Sound/Vibration** dominated complaints.

---

### Q14: How does this project demonstrate end-to-end Data Warehouse & Mining (DWDM) principles?
> **Answer:** It implements the complete textbook DWDM lifecycle:
> - **Data Engineering:** Ingestion $\to$ Cleaning $\to$ Transformation $\to$ Star Schema Dimensional Modeling.
> - **OLAP Engine:** Multidimensional aggregation cube (Roll-Up, Drill-Down, Slicing, Dicing).
> - **Data Mining:** Descriptive (K-Means, Apriori Association Rules) + Predictive (Logistic Regression) + Anomaly Detection (Isolation Forest).
> - **Decision Support:** Automated Risk Scoring Engine and Executive Visual Dashboards.

---

### Q15: If you had more time, what improvements would you make?
> **Answer:**
> 1. Implement Aspect-Based Sentiment Analysis (ABSA) using fine-tuned BERT/DeBERTa models to extract sentiment per feature (e.g., "Screen is great [pos], but battery is awful [neg]").
> 2. Integrate automated Root Cause Analysis (RCA) with LLM summarization (e.g., Gemini API) to generate action items for engineering teams.
> 3. Build an interactive Streamlit or Next.js BI dashboard with real-time alerting for sudden RSI spikes.

---

## 📂 8. Repository Structure Quick Reference

```
DWDM_Project/
├── C_DWDM_Main_Workspace.ipynb   <- Master workflow notebook & Colab execution
├── README.md                     <- Project overview and guide
├── RESUME.md                     <- Complete interview & technical mastery guide
├── requirements.txt              <- Python dependencies (pandas, sklearn, nltk, etc.)
├── data/
│   ├── raw/                      <- Raw uncleaned multi-platform customer reviews
│   ├── processed/                <- Cleaned dataset with sentiment, issues, & RSI
│   └── warehouse/                <- Star Schema tables:
│       ├── fact_review.csv       <- Central Fact Table (7.5k rows)
│       ├── dim_product.csv       <- Product dimension
│       ├── dim_platform.csv      <- Platform dimension (Amazon, Flipkart)
│       ├── dim_issue.csv         <- Issue taxonomy dimension
│       └── dim_date.csv          <- Temporal dimension
├── notebooks/
│   ├── 01_data_generation.ipynb  <- Dataset compilation & synthetic augmentation
│   ├── 02_etl_pipeline.ipynb     <- NLP preprocessing, VADER, Issue parsing, RSI
│   ├── 03_warehouse.ipynb        <- Star schema modeling & dimensional integrity
│   ├── 04_mining.ipynb           <- Classification, Clustering, Apriori, Outliers
│   ├── 05_analysis.ipynb         <- OLAP operations, Risk Scoring Engine, Visuals
│   └── 06_satisfaction_analysis.ipynb <- In-depth customer satisfaction breakdown
└── outputs/
    ├── models/                   <- Exported model binaries (.pkl)
    ├── plots/                    <- Generated analytical charts & risk heatmaps
    └── tables/                   <- Analytical summary tables & risk matrices
```
