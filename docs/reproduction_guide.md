# 📖 **ShopSense Reproduction Guide**

This guide explains how to set up the environment and execute the full **ShopSense Big Data Analytics Pipeline**, from data ingestion to real-time streaming to application deployment.

---

# 📋 **Prerequisites**

Ensure the following software is installed:

* **Python 3.8+**
* **Java 8 or 11** (required for Apache Spark)
* **Apache Spark (PySpark)**
* **Pip** (Python Package Manager)

---

## **Required Python Libraries**

Install all dependencies:

```bash
pip install pyspark gradio pandas matplotlib seaborn numpy
```

---

# 🚀 **Execution Steps**

You may run the entire pipeline using the automated script or manually run each module for better visibility.

---

##  **Option A: Automated Execution (Shell Script)**

The project includes a master script `run.sh` that attempts to execute the pipeline end-to-end.

> ⚠️ Note: Streaming and app modules are long-running processes.

```bash
make
```

---

## **Option B: Manual Execution**

Run each component separately for clearer logs, debugging, and control.

---

### **1. Data Ingestion & Engineering**

Cleans the raw CSV, applies schema validation, and saves Parquet files.

```bash
python3 src/data_ingestion.py
```

**Output:**
Processed Parquet files saved to:

```
data/processed/shopping/
```

---

### **2. Exploratory Data Analysis (EDA)**

Runs statistical summaries, distributions, and revenue analysis.

```bash
python3 src/eda_analysis.py
```

**Output:**
Distribution tables and summary statistics printed to console.

---

### **3. Advanced SQL Analytics**

Executes the complete set of 9 analytics queries (CTEs, window functions, MBA, segmentation).

```bash
python3 src/complex_queries.py
```

**Output:**
Results for:

* CLV segmentation
* Market Basket Analysis
* Age-group rankings
* Subscription impact
* Seasonal revenue patterns
  … and more.

---

### **4. Real-Time Streaming Simulation**

This step requires **two terminals**: one to generate batches, one to consume them.

---

#### **Terminal 1 — Data Generator**

```bash
python3 src/streaming.py
```

---

#### **Terminal 2 — Spark Streaming Pipeline**

```bash
python3 src/streaming_pipeline.py
```

**You will see:**

* Real-time KPIs
* Category performance
* Anomaly detection alerts
* Data quality warnings

---

### **5. Machine Learning Model Training**

This trains and persists all three ML models into the `model/` folder.

```bash
# Seasonal Predictor
python3 src/season_model.py

# Promo Code Classifier
python3 src/promo_code_model.py

# Recommendation Engine (ALS)
python3 src/recommendation.py
```

---

### **6. Application Deployment**

Launch the interactive **Gradio** application.

```bash
python3 src/app.py
```

**Access in browser:**

```
http://127.0.0.1:7860
```

#### **Application Features**

* **Tab 1:** Seasonal purchase prediction
* **Tab 2:** Product recommender (ALS)
* **Tab 3:** Promo code probability predictor

---