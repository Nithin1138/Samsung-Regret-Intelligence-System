# Samsung Regret Intelligence — Power BI Dashboard Guide & Specification

This guide details the schema relationships, DAX measures, and visual configurations needed to connect Power BI Desktop to the processed datasets in `outputs/powerbi/`.

---

## 📊 1. Data Model Architecture (Star Schema)

Import the following CSV files from `outputs/powerbi/` into Power BI:

```
                  ┌──────────────────────┐
                  │      dim_date        │
                  │  Date_ID (PK)        │
                  └──────────┬───────────┘
                             │ 1
                             │
                             │ *
┌──────────────────────┐   ┌─┴────────────────────┐   ┌──────────────────────┐
│     dim_product      │   │     fact_reviews     │   │     dim_platform     │
│  Product_ID (PK)     ├───┤  Review_ID (PK)      ├───┤  Platform_ID (PK)    │
│  Product, Category   │ 1 │  Product_ID (FK)     │ 1 │  Platform            │
└──────────────────────┘ * │  Platform_ID (FK)    │ * └──────────────────────┘
                           │  Issue_ID (FK)       │
                           │  Date_ID (FK)        │
                           │  Rating, Sent, RSI   │
                           └─┬────────────────────┘
                             │ *
                             │
                             │ 1
                  ┌──────────┴───────────┐
                  │      dim_issue       │
                  │  Issue_ID (PK)       │
                  │  Issue               │
                  └──────────────────────┘
```

### Table Relationships (Set in Model View)
- `dim_product[Product_ID]` (1) $\rightarrow$ `fact_reviews[Product_ID]` (*) [Single direction]
- `dim_platform[Platform_ID]` (1) $\rightarrow$ `fact_reviews[Platform_ID]` (*) [Single direction]
- `dim_issue[Issue_ID]` (1) $\rightarrow$ `fact_reviews[Issue_ID]` (*) [Single direction]
- `dim_date[Date_ID]` (1) $\rightarrow$ `fact_reviews[Date_ID]` (*) [Single direction]

---

## 📐 2. Essential DAX Measures

Create a dedicated **`_Measures`** table in Power BI and add the following DAX calculations:

### Core KPI Measures
```dax
Total Reviews = COUNTROWS(fact_reviews)

Average Rating = AVERAGE(fact_reviews[Rating])

Average Sentiment = AVERAGE(fact_reviews[Sentiment_score])

Average RSI = AVERAGE(fact_reviews[RSI])

High Risk Review Count = 
CALCULATE(
    COUNTROWS(fact_reviews),
    fact_reviews[RSI] > 0.5
)

High Risk Percentage = 
DIVIDE([High Risk Review Count], [Total Reviews], 0) * 100

Negative Review Count = 
CALCULATE(
    COUNTROWS(fact_reviews),
    fact_reviews[Sentiment_score] < -0.05
)

Negative Review Percentage = 
DIVIDE([Negative Review Count], [Total Reviews], 0) * 100
```

### Temporal & Moving Average Measures
```dax
3-Month Rolling RSI = 
VAR CurrentDate = MAX(dim_date[Date])
VAR StartDate = EDATE(CurrentDate, -3)
RETURN
CALCULATE(
    AVERAGE(fact_reviews[RSI]),
    DATESBETWEEN(dim_date[Date], StartDate, CurrentDate)
)

Prior Quarter RSI = 
CALCULATE(
    [Average RSI],
    DATEADD(dim_date[Date], -1, QUARTER)
)

QoQ RSI Drift = 
[Average RSI] - [Prior Quarter RSI]
```

### Product Risk Score Measure
```dax
Composite Risk Score = 
VAR NormRSI = [Average RSI]
VAR NormNeg = [Negative Review Percentage] / 100
VAR NormIssues = DIVIDE(CALCULATE(COUNTROWS(fact_reviews), fact_reviews[Issue_ID] <> 1), [Total Reviews], 0)
RETURN
(0.40 * NormRSI) + (0.35 * NormNeg) + (0.25 * NormIssues)
```

---

## 🖥️ 3. Dashboard Page Blueprint

The dashboard consists of 5 focused analytical views:

### Page 1 — Executive Overview
- **Visuals**:
  - 4 Top KPI Cards: `Total Reviews`, `Average Rating`, `Average RSI`, `High Risk %`.
  - Donut Chart: `Total Reviews` by `dim_product[Category]`.
  - Line & Clustered Column Chart: X = `dim_date[Month]`, Column = `Total Reviews`, Line = `Average RSI`.
  - Slicers: `Platform` (Amazon / Flipkart), `Date Range` (2022–2024), `Category`.

### Page 2 — Product Risk Prioritization
- **Visuals**:
  - Horizontal Clustered Bar Chart: `Composite Risk Score` by `dim_product[Product]` (Filtered to Top 10).
  - Scatter Matrix: X = `Average RSI`, Y = `Trend Slope`, Size = `Total Reviews`, Color = `Category`.
  - Data Table: Product, Category, Reviews, Avg Rating, Avg RSI, Risk Status.

### Page 3 — Temporal Intelligence & Acceleration
- **Visuals**:
  - Multi-Line Trend Chart: X = `Year-Month`, Lines = `Category`, Y = `3-Month Rolling RSI`.
  - Risk Acceleration Indicator: Flag products with 3+ consecutive months of worsening RSI.
  - Slicer: Product Selector, Date Drill-Down (Year $\rightarrow$ Quarter $\rightarrow$ Month).

### Page 4 — Issue Taxonomy & Aspect Sentiment (Priority 7)
- **Visuals**:
  - 100% Stacked Bar Chart: X = `Aspect`, Y = `% Mentions`, Legend = `Aspect_Sentiment` (Positive, Negative, Neutral) using `powerbi_aspect_sentiments.csv`.
  - Bar Chart: Defect Severity Rank by `dim_issue[Issue]` sorted by `Average RSI`.
  - Clause Drill-Through Table: Raw sentence clauses tagged with negative aspect sentiment.

### Page 5 — Cross-Platform Performance & Retail Parity
- **Visuals**:
  - Clustered Bar Chart: Amazon vs Flipkart side-by-side on Rating, RSI, Negative %, and Volume.
  - Defect Distribution Heatmap: Platforms on Rows, Defect Types on Columns, Values = Review Volume.
