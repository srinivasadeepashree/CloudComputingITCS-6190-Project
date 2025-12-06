# 🛒 ShopSense: Cloud Computing for Data Analysis & Recommendation Engine

## Project Overview

**ShopSense** is a comprehensive Cloud Computing for Data Analysis pipeline designed for a modern e-commerce platform. This project demonstrates an end-to-end solution for processing large-scale customer data, deriving actionable business intelligence, monitoring real-time transaction streams, and deploying predictive machine learning models.

The system is built entirely on **Apache Spark**, leveraging its structured APIs for batch processing, streaming, and scalable machine learning.

### 🎯 Key Objectives

  * **Data Ingestion:** Robustly load and clean large customer transaction datasets.
  * **Business Intelligence:** Perform Exploratory Data Analysis (EDA) and execute complex analytical queries to uncover trends in demographics, purchase behavior, and customer loyalty.
  * **Real-Time Analytics:** Process live transaction streams to calculate Key Performance Indicators (KPIs) and detect anomalies on the fly.
  * **Predictive Modeling:** Train and deploy machine learning models to predict customer behavior (Promo Code usage, Seasonal purchases) and provide personalized product recommendations.
  * **Interactive Deployment:** Serve the insights and models via a user-friendly web application.

-----

## 🏗️ Architecture

The project is modularized into five core stages:

1.  **Ingestion Layer:**

      * Loads raw CSV data.
      * Performs schema validation and data quality checks.
      * Reduces feature dimensionality using Correlation Analysis.
      * Persists optimized data in Parquet format.

2.  **Analytics Layer:**

      * **EDA:** Statistical summaries and distribution analysis.
      * **Advanced SQL:** Uses Window Functions, CTEs, and Self-Joins for metrics like Customer Lifetime Value (CLV) segmentation and Market Basket Analysis.

3.  **Streaming Layer:**

      * Simulates a real-time transaction feed.
      * Processes streams using Spark Structured Streaming with watermarking and windowed aggregations.
      * Monitors data quality and flags anomalies in real-time.

4.  **Machine Learning Layer:**

      * **Seasonal Predictor:** Random Forest Classifier to predict item purchases based on season and user profile.
      * **Promo Code Estimator:** Classification model to predict the likelihood of promo code usage.
      * **Recommender System:** ALS Collaborative Filtering to recommend novel products to users.

5.  **Application Layer:**

      * **Gradio Web UI:** An interactive dashboard that loads pre-trained models to serve real-time predictions and recommendations to end-users.

-----

## 📂 Repository Structure

```text
/
├── data/                   # Data storage (Raw CSVs, Processed Parquet, Streaming Input)
├── model/                  # Persisted Machine Learning Models
├── docs/                   # Documentions
├── notebooks/              # For viewing the result of EDA, ML model accuracies and complex SQL Queries
├── output/                 # Stores the graphs of EDA
├── src/
│   ├── data_ingestion.py     # Data loading and cleaning
│   ├── eda_analysis.py       # Exploratory Data Analysis
│   ├── complex_queries.py    # Advanced SQL Analytics
│   ├── streaming.py          # Real-time Data Simulator
│   ├── streaming_pipeline.py # Spark Streaming Consumer
│   ├── season_model.py       # ML: Seasonal Predictor
│   ├── promo_code_model.py   # ML: Promo Code Classifier
│   ├── recommendation.py     # ML: ALS Recommender
│   └── app.py                # Gradio Web Application
├── checkpoints/            # Spark Streaming checkpoint directory
├── run.sh                  # Master execution script
├── Makefile                # Execution script
├── requirements.txt        # For installing the packages
└── README.md               # Project Documentation
```

-----

## 🚀 Getting Started

### Prerequisites

  * **Python 3.8+**
  * **Java 8 or 11** (for Apache Spark)
  * **Apache Spark**
  * **Pip** packages: `pyspark`, `gradio`, `pandas`, `matplotlib`, `seaborn`, `numpy`

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt 
    ```

### Execution

You can run the entire pipeline using the automated script:

```bash
make
```

Alternatively, follow the detailed **[Reproduction Guide](https://github.com/srinivasadeepashree/CloudComputingITCS-6190-Project/blob/main/docs/reproduction_guide.md)** to run each component step-by-step.

-----

## 📚 Documentation

  * **[Dataset Overview](https://github.com/srinivasadeepashree/CloudComputingITCS-6190-Project/blob/main/docs/dataset_overview.md):** Details on the schema, data types, and initial quality checks.
  * **[Methodology](https://github.com/srinivasadeepashree/CloudComputingITCS-6190-Project/blob/main/docs/methodology.md):** In-depth explanation of the algorithms, streaming logic, and analytical techniques used.
  * **[Results & Insights](https://github.com/srinivasadeepashree/CloudComputingITCS-6190-Project/blob/main/docs/results.md):** Summary of key business findings, model performance metrics, and visualizations.
  * **[Limitations](https://github.com/srinivasadeepashree/CloudComputingITCS-6190-Project/blob/main/docs/limitations.md):** Discussion on current constraints and potential future improvements.

-----

## 📊 Key Results Highlights

  * **Revenue Analysis:** Identified **Clothing** and **Accessories** as primary revenue drivers.
  * **Customer Segmentation:** Successfully segmented customers into **VIP**, **Loyal**, and **New** tiers using CLV metrics.
  * **Model Accuracy:**
      * **Seasonal Predictor:** Achieved \>83% accuracy in predicting seasonal purchase preferences.
      * **Recommender:** Delivered personalized suggestions with a low RMSE of 0.64.
      * **Promo Code Predictor:** Achieved \>60% accuracy in predicting Prome Code usage.
-----

## 👤 Authors

**Srinivasa Deepashree, Vasudha Maganti, Sasi Vadana Atluri, Sai Tarun Vedagiri, Manish Yendeti**

*ITCS 6190 - Cloud Computing for Data Analysis Project*