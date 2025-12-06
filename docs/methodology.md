# 🛠️ **Project Methodology: ShopSense Big Data Pipeline**

Our pipeline follows a **layered big-data architecture**, using **Apache Spark across all layers** for scalability, parallelism, and consistent handling of large datasets.

---

## **1. Data Engineering (`data_ingestion.py`)**

**Technology:** *PySpark Structured APIs*

### **Steps**

* **Loading**
  Reads the raw `shopping.csv` file into a Spark DataFrame.

* **Schema Definition**
  Applies explicit data types to ensure data quality and prevent implicit casting issues.

* **Feature Reduction**

  * Calculated a **Correlation Matrix** using Spark ML’s `VectorAssembler` and `Correlation` modules.
  * Numerical features analyzed:

    * `Age`
    * `Purchase Amount (USD)`
    * `Previous Purchases`

* **Persistence**
  Saves clean, processed data in **Parquet format** at
  `data/processed/shopping/`
  This provides highly efficient, columnar read/write performance.

---

## **2. Analytics & Business Intelligence**

### (`eda_analysis.py`, `complex_queries.py`)

**Technology:** *PySpark SQL & DataFrame API*

### **Exploratory Data Analysis (EDA) — `eda_analysis.py`**

Performed visual and statistical analyses, including:

* Age Distribution
* Gender Distribution
* Revenue by Category

**Findings:**
Clothing and Accessories ranked as the top categories by both **purchase count** and **total revenue**.

### **Advanced Analytics — `complex_queries.py`**

Used SQL techniques for deeper insights:

* **Window Functions**

  * `ROW_NUMBER()` for ranking categories by age group
  * `NTILE()` for Customer Lifetime Value (CLV) segmentation (Query 6)

* **Self-Joins & CTEs**

  * Performed **Market Basket Analysis (MBA)** (Query 9)
  * Calculated **Lift Scores** to detect frequently co-purchased product categories

### **Outputs**

Generated detailed reports on:

* Customer segmentation
* Promotional effectiveness
* Seasonal spending trends
* Geographic purchase patterns

---

## **3. Real-Time Streaming**

### (`streaming.py`, `streaming_pipeline.py`)

**Technology:** *PySpark Structured Streaming*

### **Simulation — `streaming.py`**

Creates a pseudo-real-time stream by writing timestamped mini-batches of CSV files into a streaming source directory.

### **Streaming Pipeline — `streaming_pipeline.py`**

* **Watermarking**
  Applied a **2-minute watermark** on `event_time` to handle late-arriving data gracefully.

* **Tumbling Windows**
  Aggregated data in **1-minute windows** to compute real-time KPIs.

* **Concurrent Queries**
  Ran **five simultaneous streaming jobs**, including:

  * **Anomaly Detection** (Query 4)
  * **Data Quality Monitoring** (Query 5)

Outputs were written to:

* Console
* In-memory SQL tables

---

## **4. Machine Learning & Prediction**

**Technology:** *PySpark MLlib (ALS, RandomForestClassifier), Pipeline API*

### **Promo Code Predictor — `promo_code_model.py`**

* Algorithm: **Random Forest Classifier**
* Task: Predict likelihood of promo code usage
* Metrics reported: **AUC-ROC**, **Accuracy**

### **Seasonal Item Predictor — `season_model.py`**

* Random Forest Classifier predicting top 3 items likely purchased per season
* Achieved **~83% accuracy** on test data

### **Product Recommender — `recommendation.py`**

* Algorithm: **ALS Collaborative Filtering**
* Steps included:

  * Indexing item names → numeric IDs
  * Performing an **anti-join** to remove items previously purchased by the user
* Model performance: **RMSE = 0.6481**

---

## **5. Application Layer (`app.py`)**

**Technology:** *Gradio + PySpark Pipelines*

### **Deployment**

Interactive web application with **three functional tabs**:

1. **Seasonal Item Predictor**
2. **Product Recommendations**
3. **Promo Code Propensity**

The app loads three saved **Spark ML `PipelineModel`** objects, enabling full ML inference through a user-friendly interface.
