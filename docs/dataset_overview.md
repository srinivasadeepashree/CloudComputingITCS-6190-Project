# Dataset Overview

- **Source**: Shopping Trends Dataset (shopping_trends.csv)
- **Size**: 3,900 records × 18 columns
- **Description**: Customer shopping behavior data including demographics, purchase patterns, payment methods, and product preferences

**Dataset Columns**:
- Customer ID, Age, Gender
- Item Purchased, Category, Purchase Amount (USD)
- Location, Size, Color, Season
- Review Rating, Subscription Status
- Shipping Type, Discount Applied, Promo Code Used
- Previous Purchases, Payment Method, Frequency of Purchases

### 📝 Dataset Summary

| Feature | Value |
| :--- | :--- |
| **Number of Entries** | 3,900 |
| **Number of Columns** | 18 |
| **Missing Values** | **None** (Data is complete) |
| **Data Types** | 13 Categorical (`object`), 5 Numerical (`int64`, `float64`) |

---

### 🔢 Numerical Feature Highlights

| Feature | Mean | Standard Deviation | Range |
| :--- | :--- | :--- | :--- |
| **Age** | 44.07 | 15.21 | 18 to 70 |
| **Purchase Amount (USD)** | **\$59.76** | 23.69 | \$20 to \$100 |
| **Review Rating** | **3.75** | 0.72 | 2.5 to 5.0 |
| **Previous Purchases** | 25.35 | 14.45 | 1 to 50 |

---

### 🏷️ Categorical Feature Highlights

| Feature | Most Frequent Value (`Top`) | Frequency | Key Insight |
| :--- | :--- | :--- | :--- |
| **Gender** | **Male** | 2,652 | Significantly more transactions from Male customers (2:1 ratio). |
| **Category** | **Clothing** | 1,737 | Most popular product category. |
| **Size** | **M** (Medium) | 1,755 | The most commonly purchased size. |
| **Season** | **Spring** | 999 | The season with the highest number of purchases. |
| **Subscription Status** | **No** | 2,847 | Majority of customers do not have a subscription. |
| **Payment Method** | **PayPal** | 677 | Most preferred payment method. |