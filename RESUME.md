# Enterprise Data Warehouse & Temporal Regret Intelligence System (Samsung)

---

## 📌 1. Resume Version (3-4 Bullet Points for Resume)

*Direct drop-in bullet points for software engineering, data engineering, and ML developer resumes:*

- **Architected an End-to-End Enterprise Data Warehouse & Intelligence Pipeline** in Python/SQL analyzing 7,200+ multi-channel customer reviews across 50 product lines using a normalized **Star Schema** (`fact_review` linked with 4 dimension tables) and 6 analytical SQL window queries (`RANK`, `LAG`, running `SUM`).
- **Engineered a Domain-Specific Regret Severity Index (RSI)** combining VADER compound NLP sentiment, star-rating deviations, keyword defect extraction, and text specificity ($0$ to $1$ scale) to quantify latent post-purchase customer dissonance.
- **Implemented Rule-Based Sentence-Level Aspect Extraction** splitting complex multi-clause reviews across contrastive conjunctions to isolate granular component sentiment (e.g., Battery: Negative vs. Camera: Positive), extracting 1,880+ localized aspect occurrences.
- **Trained a Non-Circular Temporal Risk Predictor & Mining Engine** using historical quarterly lag features with chronological train/test splitting (85% accuracy, 0.82 ROC-AUC) to forecast future product risk, alongside K-Means remorse clustering, Apriori association mining, and Isolation Forest anomaly triage.
- **Developed a 5-Page Power BI Executive Dashboard Suite** complete with Star Schema dimensional models, custom DAX measures (`[3-Month Rolling RSI]`, `[Composite Risk Score]`), and longitudinal trend visualizers for product health prioritization.

---

## 🧭 2. High-Level Elevator Pitch (30-Second Interview Summary)

> *"In this project, I built an **Enterprise Data Warehouse and Temporal Regret Intelligence System** tailored for consumer electronics. Standard sentiment analysis often fails because a 3-star review can hide severe hardware defects, while a 1-star review might just be a delivery delay.*
> 
> *To solve this, I engineered the **Regret Severity Index (RSI)**—a multi-factor metric fusing NLP sentiment polarity, rating divergence, and keyword issue extraction. I structured 7.2k+ multi-platform reviews into a Star Schema Data Warehouse, executed OLAP window queries, developed sentence-level aspect extraction, and trained a non-circular temporal risk model using quarterly lag features to forecast future product failure velocity before reputational damage escalates."*

---

## 🏗️ 3. System Architecture & End-to-End Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. RAW DATA INGESTION                           │
│  11,000 Multi-Platform Customer Reviews (Amazon & Flipkart)            │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    2. DATA QUALITY VALIDATION GATE                     │
│  • Automated checks: Deduplication (3,206 duplicates removed)          │
│  • Rating bounds coercion [1.0, 5.0] & missing text filtering          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     3. ETL & NLP FEATURE PIPELINE                      │
│  • Text Normalization & Regex Token Cleaning                           │
│  • VADER Sentiment Intensity Analyzer -> Compound Polarity Score       │
│  • Sentence-Level Aspect Extraction (Clause-level defect polarity)     │
│  • Mathematical Regret Severity Index (RSI) Computation                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               4. DATA WAREHOUSE LAYER (STAR SCHEMA)                    │
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
│     5. DATA MINING & ML MODELS       │ │     6. TEMPORAL INTELLIGENCE  │
│ • Temporal Risk Predictor (Lagged    │ │ • Monthly Longitudinal RSI    │
│   quarterly features, Time Split)    │ │ • 3-Month Rolling Average     │
│ • K-Means Clustering (3 Remorse Bands│ │ • OLS Trend Slopes (linregress│
│ • Apriori Rule Mining (Defect pairs) │ │ • Risk Acceleration (d2/dt2)  │
│ • Isolation Forest (Anomaly Triage)  │ │ • Consecutive Worsening Alerts│
└──────────────────┬───────────────────┘ └───────────────┬───────────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                7. POWER BI EXECUTIVE DASHBOARD SUITE                   │
│  • Page 1: Overview KPIs (Volume, Avg Rating, Avg RSI, High Risk %)    │
│  • Page 2: Product Risk Prioritization & Severity vs Slope Matrix      │
│  • Page 3: Temporal Trends & Acceleration Monitoring                   │
│  • Page 4: Defect Taxonomy & Sentence-Level Aspect Sentiment           │
│  • Page 5: Amazon vs Flipkart Cross-Platform Quality Parity            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 4. Technical Interview Q&A Deep Dive

### Q1: Why not just use Star Ratings or standard Sentiment Analysis?
> **Answer:** Raw star ratings are subjective and lack context (e.g., 1-star due to a delivery delay vs. a dead motherboard). Sentiment analysis alone fails on short text or reviews with mixed sentiments. RSI combines normalized rating divergence (35%), inverted negative sentiment (35%), hardware defect penalties (15%), and text specificity (15%) to isolate genuine post-purchase remorse.

### Q2: Why did you replace the original Logistic Regression setup?
> **Answer:** In the initial setup, Logistic Regression used `Rating` and `Sentiment` to predict `High_Risk` (`RSI > 0.5`). Because RSI is computed directly from Rating and Sentiment, this created severe **circular data leakage**.  
> I redesigned the model into a **Temporal Risk Predictor**: I engineered quarterly historical lag features (`previous_avg_rsi`, `neg_ratio`, `issue_density`, `review_volume`, `avg_rating`) to forecast whether a product would cross into high risk in the **next quarter**, using a strict chronological train/test split.

### Q3: Explain your Star Schema Data Warehouse design.
> **Answer:** I modeled the warehouse around `fact_review` containing foreign keys and numerical metrics (`Rating`, `Sentiment_score`, `RSI`), joined to 4 dimension tables: `dim_product`, `dim_platform`, `dim_issue`, and `dim_date`. This star topology minimizes join depth, supports OLAP slicing, and allows Power BI to execute high-speed aggregations without table locking.

### Q4: How does sentence-level aspect extraction work?
> **Answer:** Customer feedback often contains conflicting sentiments in the same review (e.g., *"Battery drains fast, but camera is excellent"*). Treating this as a single sentiment loses actionable signal. My rule-based aspect extractor splits reviews into clauses across contrastive conjunctions (`but`, `however`, `although`), maps domain defect keywords, and evaluates localized sentiment for each component.

### Q5: How is product risk growth calculated?
> **Answer:** Rather than computing simple endpoint differences ($\text{RSI}_{last} - \text{RSI}_{first}$), which are fragile to monthly review fluctuations, I fit an Ordinary Least Squares (OLS) linear regression across monthly RSI time-series using `scipy.stats.linregress`. The slope ($m$) captures true velocity of quality deterioration over time.
