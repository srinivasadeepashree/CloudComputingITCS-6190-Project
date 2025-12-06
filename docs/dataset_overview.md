# Dataset Overview

- **Source**: Shopping Trends Dataset (shopping_trends.csv)
- **Size**: 27533 records × 18 columns
- **Description**: Customer shopping behavior data including demographics, purchase patterns, payment methods, and product preferences

**Dataset Columns**:
- Customer ID, Age, Gender
- Item Purchased, Category, Purchase Amount (USD)
- Location, Size, Color, Season
- Review Rating, Subscription Status
- Shipping Type, Discount Applied, Promo Code Used
- Previous Purchases, Payment Method, Frequency of Purchases

## 📝 Dataset Summary

| Feature | Value |
| :--- | :--- |
| **Number of Entries** | 27533 |
| **Number of Columns** | 18 |
| **Missing Values** | **None** (Data is complete) |
| **Data Types** | 13 Categorical (`object`), 5 Numerical (`int64`, `float64`) |

---
## Key Data Attributes

| Category             | Column Examples                                                       | Description                                           | Data Type                 |
|----------------------|-----------------------------------------------------------------------|-------------------------------------------------------|---------------------------|
| Customer/Demographic | Customer ID, Age, Gender, Location, Subscription Status               | Basic profiling and segmentation features.            | Integer, String           |
| Purchase Details     | Item Purchased, Category, Purchase Amount (USD), Review Rating        | Transaction-level details and customer satisfaction.  | String, Integer, Double   |
| Logistics/Marketing  | Payment Method, Shipping Type, Discount Applied, Promo Code Used      | Operational and promotional effectiveness data.       | String                    |
| Loyalty/Frequency    | Previous Purchases, Frequency of Purchases                            | Measures of customer loyalty and buying cadence.      | Integer, String           |

## Initial Data Load & Validation

**Total Records Loaded:** 27,533

---

### 🔢 Numerical Feature Highlights
| Feature                 | Mean   | Standard Deviation | Range          |
|-------------------------|--------|---------------------|----------------|
| Age                     | 44.11  | 15.18               | 18 to 70       |
| Purchase Amount (USD)   | $59.80 | 23.74               | $20 to $100    |
| Review Rating           | 3.75   | 0.71                | 2.5 to 5.0     |
| Previous Purchases      | 25.32  | 14.51               | 1 to 50        |

---

### 🏷️ Categorical Feature Highlights
| Feature              | Most Frequent Value (Top) | Frequency | Key Insight                                                |
|----------------------|----------------------------|-----------|------------------------------------------------------------|
| Gender               | Male                       | 18,886    | Male transactions are roughly 2.2:1 compared to Female.   |
| Category             | Clothing                   | 12,162    | Most popular product category.                            |
| Size                 | M                          | 12,353    | The most commonly purchased size.                         |
| Season               | Spring                     | 7,088     | The season with the highest number of purchases.          |
| Subscription Status  | No                         | 20,024    | Majority of customers do not have a subscription.         |
| Payment Method       | Cash                       | 4,870     | Most preferred payment method.                            |

---

### Initial Schema Inference
- Used `inferSchema=True` for the initial load.  
- Later replaced with **explicitly defined schemas** in ingestion scripts for robustness.

---

### Data Quality (from `data_ingestion.py`)
- **Duplicates:** Checked — total rows (27,533) matched distinct rows.  
- **Null Values:** Validated for each column.

---

### Core Objective
The dataset is used to build a **full-stack data pipeline**, demonstrating proficiency in:
- Distributed computing for analytics  
- Streaming data processing  
- Machine learning workflows  

