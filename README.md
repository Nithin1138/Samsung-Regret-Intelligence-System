# Enterprise Data Warehouse & Temporal Regret Intelligence System

## 📌 Overview
This project transforms raw customer review data into structured product risk intelligence for Samsung products.

It analyzes reviews from multiple platforms and predicts product risk using a custom metric called **Regret Severity Index (RSI)**.

---

## 🎯 Objectives
- Build a centralized data warehouse
- Extract sentiment and issue patterns from reviews
- Compute RSI (Regret Severity Index)
- Predict high-risk products early
- Enable data-driven decision making

---

## 📂 Project Structure
DWDM_Project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── warehouse/
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_etl_pipeline.ipynb
│   ├── 03_warehouse.ipynb
│   ├── 04_mining.ipynb
│   └── 05_analysis.ipynb
│
├── scripts/
├── config/
├── outputs/
├── reports/
├── requirements.txt
└── README.md

---

## 🔄 Workflow
1. Raw Data Collection (10k Samsung reviews)
2. ETL Pipeline:
   - Text preprocessing
   - Sentiment analysis (VADER)
   - Issue extraction
   - RSI calculation
3. Data Warehouse (Star Schema)
4. Data Mining:
   - Classification
   - Clustering
   - Association rules
5. OLAP Analysis
6. Risk Scoring Engine

---

## 🧠 Core Concept: RSI
RSI = weighted combination of:
- Negative sentiment
- Rating deviation
- Issue keywords
- Specificity

Range: 0 (low regret) → 1 (high regret)

---

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- NLTK (VADER)
- Scikit-learn
- MLxtend
- Matplotlib, Seaborn
- Google Colab

---

## ▶️ How to Run
1. Upload dataset to `/data/raw/`
2. Run `02_etl_pipeline.ipynb`
3. Run `03_warehouse.ipynb`
4. Run `04_mining.ipynb`
5. Run `05_analysis.ipynb`

---

## 📊 Expected Outputs
- High-risk product identification
- Issue pattern detection
- Platform comparison insights
- Temporal regret trends

---

## 📌 Conclusion
This system converts unstructured review data into a predictive intelligence layer for product risk management.

