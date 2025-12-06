# 🛑 **Project Limitations and Future Work**

While the **ShopSense** pipeline provides a strong end-to-end big-data architecture, several limitations—both in the dataset and infrastructure—create opportunities for improvement in future versions.

---

## **1. Dataset Limitations**

### Lack of Temporal Data

* The core `shopping.csv` dataset is a **static snapshot**.
* Missing continuous, granular `purchase_date` values restricts:

  * Time-series forecasting
  * Seasonal decomposition
  * Cohort retention analyses
  * Trend detection

### Feature Engineering Bias

* As part of preprocessing, highly correlated numerical features (**r > 0.85**) were removed to prevent multicollinearity.
* While necessary for model fairness and stability, this:

  * Reduces dimensional richness
  * May eliminate subtle interaction patterns

---

## **2. Model & Algorithm Constraints**

### ALS Cold-Start Problem

* Collaborative Filtering (ALS) fails with:

  * **New users** with no purchase history
  * **New items** with no interactions
* ❗ *Current limitation in recommendation coverage.*

* **Future Work:** Implement a **Hybrid Recommender System** that merges:

* **ALS (Collaborative Filtering)**
* **Seasonal Item Predictor (Content-Based)**
  This hybrid approach would significantly reduce cold-start failures.

---

### Recommender Filtering Logic

* The pipeline uses a **Left Anti-Join** to exclude items a customer already purchased.
* This is production-ready logic, but:

  * May reduce variety for low-activity users
  * Limits novel or serendipitous recommendations

---

### Seasonal Model Performance

* The Seasonal Predictor achieved **83.53% accuracy** while predicting **1 of 25 possible items**.
* Excellent performance, but:

  * Accuracy may plateau without more granular features (weather, time of day, location metadata).

---

## 3. Architectural Limitations (Local Deployment)

### Local Spark Master

* Current setup uses **`local[*]`** mode.
* Limits:

  * Parallelism
  * Fault tolerance
  * Cluster-level benchmarking

**Future Work:**
Move to a **fully distributed cloud cluster**, 

e.g.:

* **AWS EMR**
* **GCP Dataproc**
* **Azure HDInsight**

This enables stress-testing horizontal scalability and resilience.

---

### Streaming Output Persistence

* Streaming queries write results to:

  * Console
  * In-memory tables (`category_performance`, `data_quality`, etc.)

These are **not persistent** and unsuitable for production dashboards.

**Future Work:**
Write streaming outputs to a durable, low-latency store such as:

* Apache Cassandra / ScyllaDB
* Delta Lake tables
* Kafka Sink / Kinesis Firehose
* Redis Streams
* Elasticsearch (for analytics dashboards)
