Enterprise Data Warehouse & Temporal Regret Intelligence System

📌 Overview

This project transforms raw customer review data into structured product risk intelligence for Samsung products.

It analyzes multi-platform reviews and predicts product risk using a custom metric called Regret Severity Index (RSI).

⸻

🎯 Objectives

* Build a centralized data warehouse
* Extract sentiment and issue patterns from reviews
* Compute RSI (Regret Severity Index)
* Predict high-risk products early
* Enable data-driven decision making

⸻

🧠 Core Concept: RSI

RSI is a composite score (0 to 1) based on:

* Negative sentiment
* Rating deviation
* Issue presence
* Review specificity

Higher RSI → Higher customer regret → Higher product risk

⸻

🔄 Workflow

1. Raw Data Collection (Samsung product reviews)
2. ETL Pipeline:
    * Text preprocessing
    * Sentiment analysis (VADER)
    * Issue extraction
    * RSI computation
3. Data Warehouse (Star Schema)
4. Data Mining:
    * Classification (High Risk Prediction)
    * Clustering (Risk Segmentation)
    * Association Rules (Issue Patterns)
    * Outlier Detection
5. OLAP Analysis
6. Risk Scoring Engine (Product Prioritization)

⸻

📂 Project Structure

DWDM_Project/
├── data/
├── notebooks/
├── outputs/
├── scripts/
├── config/
├── reports/
└── README.md

⸻

⚙️ Technologies Used

* Python
* Pandas, NumPy
* NLTK (VADER)
* Scikit-learn
* MLxtend
* Matplotlib / Seaborn
* Google Colab

⸻

▶️ How to Run

1. Place dataset in /data/raw/
2. Run ETL → 02_etl_pipeline.ipynb
3. Run Warehouse → 03_warehouse.ipynb
4. Run Mining → 04_mining.ipynb
5. Run Analysis → 05_analysis.ipynb

⸻

📊 Key Outputs

* High-risk product identification
* Issue pattern discovery
* Platform comparison insights
* Temporal regret trends
* Product risk ranking

⸻

📌 Conclusion

This system converts unstructured review data into a structured, predictive intelligence layer for product risk management and decision-making.
