# 📈 **Key Results and Business Insights**

The ShopSense big-data pipeline successfully processed the full dataset and generated the following analytical results and business insights.

---

## **A. Core Metrics and EDA Summary**

| **Metric**                          | **Result**                                            |
| ----------------------------------- | ----------------------------------------------------- |
| **Total Records**                   | **27,533**                                            |
| **Total Revenue**                   | **$1,646,610 USD**                                    |
| **Average Transaction Value (ATV)** | **≈ $59.80 USD**                                      |
| **Top-Selling Category**            | **Clothing** (12,162 purchases, $727,778 USD revenue) |
| **Gender Distribution**             | **Male: 68.5%**    **Female: 31.5%**                  |
| **Top Age Groups**                  | **50–59** and **60+** (largest customer counts)       |

---

## **B. Advanced Analytics & Segmentation**

| **Query / Metric**                                  | **Key Finding**                                                                                     | **Actionable Insight**                                                                    |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Top Categories by Age** *(Query 1)*               | Identifies the top 3 categories purchased for each age range.                                       | Enables age-based targeting and generation-specific campaign design.                      |
| **CLV Segmentation** *(Query 6)*                    | Customers segmented by lifetime value, purchase frequency, and loyalty. VIP “Champions” identified. | Helps prioritize high-value customers and optimize retention incentives.                  |
| **Market Basket Analysis (Lift Score)** *(Query 9)* | Discovers category pairs frequently purchased together. Lift > 1 indicates positive association.    | Supports bundling strategies, cross-selling, and improved recommendation logic.           |
| **Subscription Impact** *(Query 2)*                 | Analyzes differences in purchase behavior between subscribers and non-subscribers.                  | Measures subscription ROI and highlights whether loyalty perks influence spending habits. |

---

## **C. Machine Learning Performance**

| **Model**                    | **Evaluation Metric** | **Result** |
| ---------------------------- | --------------------- | ---------- |
| **Promo Code Predictor**     | AUC-ROC               | **0.5855** |
|                              | Accuracy              | **58.77%** |
| **ALS Recommendation Model** | RMSE                  | **0.6481** |
| **Seasonal Item Predictor**  | Accuracy (Multiclass) | **83.53%** |

---