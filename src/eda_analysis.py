"""
Exploratory Data Analysis Module for ShopSense
ITCS 6190 - Cloud Computing for Data Analysis Project

This module performs comprehensive EDA using Spark SQL and DataFrame APIs
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, countDistinct, avg, sum, min, max, stddev,
    round as spark_round, desc, asc, when, expr
)
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class ShoppingTrendsEDA:
    """
    Exploratory Data Analysis class for Shopping Trends dataset
    """
    
    def __init__(self, spark, df):
        """
        Initialize EDA with Spark session and DataFrame
        
        Args:
            spark: SparkSession object
            df: Loaded Spark DataFrame
        """
        self.spark = spark
        self.df = df
        self.df.createOrReplaceTempView("shopping")
        print("✓ Temporary view 'shopping' created for Spark SQL queries")
    
    def overview_statistics(self):
        """
        Display overview statistics about the dataset
        """
        print(f"\n{'='*80}")
        print("DATASET OVERVIEW")
        print(f"{'='*80}")
        
        total_rows = self.df.count()
        print(f"\nTotal Records: {total_rows:,}")
        
        # Unique customers
        unique_customers = self.df.select("Customer ID").distinct().count()
        print(f"Unique Customers: {unique_customers:,}")
        
        # Date range if applicable
        print(f"\nColumns in dataset: {len(self.df.columns)}")
        print(f"Column names: {', '.join(self.df.columns)}")
    
    def customer_demographics_analysis(self):
        """
        Analyze customer demographics using Spark SQL
        """
        print(f"\n{'='*80}")
        print("CUSTOMER DEMOGRAPHICS ANALYSIS")
        print(f"{'='*80}")
        
        # Age distribution
        print("\n1. Age Distribution Statistics:")
        age_stats = self.spark.sql("""
            SELECT 
                ROUND(AVG(Age), 2) as avg_age,
                MIN(Age) as min_age,
                MAX(Age) as max_age,
                ROUND(STDDEV(Age), 2) as stddev_age
            FROM shopping
        """)
        age_stats.show()
        
        # Age groups
        print("\n2. Customer Count by Age Group:")
        age_groups = self.spark.sql("""
            SELECT 
                CASE 
                    WHEN Age < 25 THEN '18-24'
                    WHEN Age BETWEEN 25 AND 34 THEN '25-34'
                    WHEN Age BETWEEN 35 AND 44 THEN '35-44'
                    WHEN Age BETWEEN 45 AND 54 THEN '45-54'
                    WHEN Age >= 55 THEN '55+'
                END as age_group,
                COUNT(*) as customer_count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM shopping), 2) as percentage
            FROM shopping
            GROUP BY age_group
            ORDER BY age_group
        """)
        age_groups.show()
        
        # Gender distribution
        print("\n3. Gender Distribution:")
        gender_dist = self.spark.sql("""
            SELECT 
                Gender,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM shopping), 2) as percentage
            FROM shopping
            GROUP BY Gender
            ORDER BY count DESC
        """)
        gender_dist.show()
        
        # Location analysis
        print("\n4. Top 10 Locations by Customer Count:")
        location_dist = self.spark.sql("""
            SELECT 
                Location,
                COUNT(*) as customer_count
            FROM shopping
            GROUP BY Location
            ORDER BY customer_count DESC
            LIMIT 10
        """)
        location_dist.show(truncate=False)
        
        return {
            'age_stats': age_stats,
            'age_groups': age_groups,
            'gender_dist': gender_dist,
            'location_dist': location_dist
        }
    
    def purchase_behavior_analysis(self):
        """
        Analyze purchase behavior patterns
        """
        print(f"\n{'='*80}")
        print("PURCHASE BEHAVIOR ANALYSIS")
        print(f"{'='*80}")
        
        # Purchase amount statistics
        print("\n1. Purchase Amount Statistics:")
        purchase_stats = self.spark.sql("""
            SELECT 
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_purchase,
                ROUND(MIN(`Purchase Amount (USD)`), 2) as min_purchase,
                ROUND(MAX(`Purchase Amount (USD)`), 2) as max_purchase,
                ROUND(STDDEV(`Purchase Amount (USD)`), 2) as stddev_purchase,
                ROUND(SUM(`Purchase Amount (USD)`), 2) as total_revenue
            FROM shopping
        """)
        purchase_stats.show()
        
        # Category analysis
        print("\n2. Revenue by Product Category:")
        category_revenue = self.spark.sql("""
            SELECT 
                Category,
                COUNT(*) as purchase_count,
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_amount,
                ROUND(SUM(`Purchase Amount (USD)`), 2) as total_revenue
            FROM shopping
            GROUP BY Category
            ORDER BY total_revenue DESC
        """)
        category_revenue.show(truncate=False)
        
        # Top purchased items
        print("\n3. Top 15 Most Purchased Items:")
        top_items = self.spark.sql("""
            SELECT 
                `Item Purchased`,
                Category,
                COUNT(*) as purchase_count,
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_price
            FROM shopping
            GROUP BY `Item Purchased`, Category
            ORDER BY purchase_count DESC
            LIMIT 15
        """)
        top_items.show(truncate=False)
        
        # Frequency of purchases
        print("\n4. Distribution by Purchase Frequency:")
        freq_dist = self.spark.sql("""
            SELECT 
                `Frequency of Purchases`,
                COUNT(*) as customer_count,
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_purchase_amount
            FROM shopping
            GROUP BY `Frequency of Purchases`
            ORDER BY customer_count DESC
        """)
        freq_dist.show(truncate=False)
        
        return {
            'purchase_stats': purchase_stats,
            'category_revenue': category_revenue,
            'top_items': top_items,
            'freq_dist': freq_dist
        }
    
    def payment_shipping_analysis(self):
        """
        Analyze payment methods and shipping preferences
        """
        print(f"\n{'='*80}")
        print("PAYMENT & SHIPPING ANALYSIS")
        print(f"{'='*80}")
        
        # Payment method distribution
        print("\n1. Payment Method Distribution:")
        payment_dist = self.spark.sql("""
            SELECT 
                `Payment Method`,
                COUNT(*) as transaction_count,
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_amount,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM shopping), 2) as percentage
            FROM shopping
            GROUP BY `Payment Method`
            ORDER BY transaction_count DESC
        """)
        payment_dist.show()
        
        # Shipping type analysis
        print("\n2. Shipping Type Preferences:")
        shipping_dist = self.spark.sql("""
            SELECT 
                `Shipping Type`,
                COUNT(*) as order_count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM shopping), 2) as percentage
            FROM shopping
            GROUP BY `Shipping Type`
            ORDER BY order_count DESC
        """)
        shipping_dist.show()
        
        # Discount and promo code usage
        print("\n3. Discount and Promo Code Usage:")
        discount_stats = self.spark.sql("""
            SELECT 
                `Discount Applied`,
                `Promo Code Used`,
                COUNT(*) as count,
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_purchase
            FROM shopping
            GROUP BY `Discount Applied`, `Promo Code Used`
            ORDER BY count DESC
        """)
        discount_stats.show()
        
        return {
            'payment_dist': payment_dist,
            'shipping_dist': shipping_dist,
            'discount_stats': discount_stats
        }
    
    def customer_loyalty_analysis(self):
        """
        Analyze customer loyalty and subscription patterns
        """
        print(f"\n{'='*80}")
        print("CUSTOMER LOYALTY ANALYSIS")
        print(f"{'='*80}")
        
        # Subscription status
        print("\n1. Subscription Status Distribution:")
        subscription_dist = self.spark.sql("""
            SELECT 
                `Subscription Status`,
                COUNT(*) as customer_count,
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_purchase,
                ROUND(AVG(`Previous Purchases`), 2) as avg_previous_purchases
            FROM shopping
            GROUP BY `Subscription Status`
            ORDER BY customer_count DESC
        """)
        subscription_dist.show()
        
        # Previous purchases analysis
        print("\n2. Customer Segmentation by Previous Purchases:")
        loyalty_segments = self.spark.sql("""
            SELECT 
                CASE 
                    WHEN `Previous Purchases` = 0 THEN 'New Customer'
                    WHEN `Previous Purchases` BETWEEN 1 AND 10 THEN 'Regular (1-10)'
                    WHEN `Previous Purchases` BETWEEN 11 AND 25 THEN 'Loyal (11-25)'
                    WHEN `Previous Purchases` > 25 THEN 'VIP (25+)'
                END as customer_segment,
                COUNT(*) as customer_count,
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_purchase
            FROM shopping
            GROUP BY customer_segment
            ORDER BY 
                CASE customer_segment
                    WHEN 'New Customer' THEN 1
                    WHEN 'Regular (1-10)' THEN 2
                    WHEN 'Loyal (11-25)' THEN 3
                    WHEN 'VIP (25+)' THEN 4
                END
        """)
        loyalty_segments.show()
        
        # Review rating analysis
        print("\n3. Review Rating Distribution:")
        rating_dist = self.spark.sql("""
            SELECT 
                `Review Rating`,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM shopping), 2) as percentage
            FROM shopping
            GROUP BY `Review Rating`
            ORDER BY `Review Rating` DESC
        """)
        rating_dist.show()
        
        return {
            'subscription_dist': subscription_dist,
            'loyalty_segments': loyalty_segments,
            'rating_dist': rating_dist
        }
    
    def seasonal_trend_analysis(self):
        """
        Analyze seasonal purchasing trends
        """
        print(f"\n{'='*80}")
        print("SEASONAL TREND ANALYSIS")
        print(f"{'='*80}")
        
        # Sales by season
        print("\n1. Sales Performance by Season:")
        season_sales = self.spark.sql("""
            SELECT 
                Season,
                COUNT(*) as purchase_count,
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_purchase,
                ROUND(SUM(`Purchase Amount (USD)`), 2) as total_revenue
            FROM shopping
            GROUP BY Season
            ORDER BY total_revenue DESC
        """)
        season_sales.show()
        
        # Category preferences by season
        print("\n2. Top Categories by Season:")
        season_category = self.spark.sql("""
            SELECT 
                Season,
                Category,
                COUNT(*) as purchase_count
            FROM shopping
            GROUP BY Season, Category
            ORDER BY Season, purchase_count DESC
        """)
        season_category.show(20, truncate=False)
        
        return {
            'season_sales': season_sales,
            'season_category': season_category
        }
    
    def advanced_correlations(self):
        """
        Analyze correlations and advanced patterns using DataFrame API
        """
        print(f"\n{'='*80}")
        print("ADVANCED PATTERN ANALYSIS")
        print(f"{'='*80}")
        
        # Age vs Purchase Amount
        print("\n1. Average Purchase Amount by Age Group and Gender:")
        age_gender_purchase = self.df.withColumn(
            "age_group",
            when(col("Age") < 25, "18-24")
            .when((col("Age") >= 25) & (col("Age") < 35), "25-34")
            .when((col("Age") >= 35) & (col("Age") < 45), "35-44")
            .when((col("Age") >= 45) & (col("Age") < 55), "45-54")
            .otherwise("55+")
        ).groupBy("age_group", "Gender").agg(
            spark_round(avg("`Purchase Amount (USD)`"), 2).alias("avg_purchase"),
            count("*").alias("count")
        ).orderBy("age_group", "Gender")
        
        age_gender_purchase.show()
        
        # Color and size preferences
        print("\n2. Most Popular Color-Size Combinations:")
        color_size = self.df.groupBy("Color", "Size").agg(
            count("*").alias("count")
        ).orderBy(desc("count")).limit(15)
        
        color_size.show(truncate=False)
        
        # Subscription impact on spending
        print("\n3. Impact of Subscription on Purchase Behavior:")
        subscription_impact = self.df.groupBy("Subscription Status").agg(
            count("*").alias("customer_count"),
            spark_round(avg("`Purchase Amount (USD)`"), 2).alias("avg_purchase"),
            spark_round(avg("Previous Purchases"), 2).alias("avg_previous_purchases"),
            spark_round(avg("Review Rating"), 2).alias("avg_rating")
        )
        subscription_impact.show()
        
        return {
            'age_gender_purchase': age_gender_purchase,
            'color_size': color_size,
            'subscription_impact': subscription_impact
        }
    
    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        """
        print(f"\n{'='*80}")
        print("COMPREHENSIVE SUMMARY REPORT")
        print(f"{'='*80}")
        
        # Key metrics
        summary = self.spark.sql("""
            SELECT 
                COUNT(DISTINCT `Customer ID`) as total_customers,
                COUNT(*) as total_transactions,
                ROUND(SUM(`Purchase Amount (USD)`), 2) as total_revenue,
                ROUND(AVG(`Purchase Amount (USD)`), 2) as avg_transaction_value,
                ROUND(AVG(Age), 1) as avg_customer_age,
                ROUND(AVG(`Review Rating`), 2) as avg_rating,
                COUNT(DISTINCT Category) as total_categories,
                COUNT(DISTINCT `Item Purchased`) as total_unique_items
            FROM shopping
        """)
        
        print("\nKey Business Metrics:")
        summary.show(vertical=True)
        
        return summary


def export_for_visualization(spark_df, output_name):
    """
    Convert Spark DataFrame to Pandas for visualization
    
    Args:
        spark_df: Spark DataFrame
        output_name: Name for the output
    
    Returns:
        Pandas DataFrame
    """
    pandas_df = spark_df.toPandas()
    print(f"✓ Converted '{output_name}' to Pandas DataFrame for visualization")
    return pandas_df


def main():
    """
    Main execution function for EDA
    """
    # Create Spark session
    spark = SparkSession.builder \
        .appName("ShoppingTrendsEDA") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Load data (assumes data_ingestion has been run)
    try:
        # Load from parquet (processed data)
        df = spark.read.parquet("data/processed/shopping")
        print("✓ Loaded processed data from parquet")
    except:
        # Fallback to CSV
        df = spark.read.csv("data/shopping.csv", header=True, inferSchema=True)
        print("✓ Loaded data from CSV")
    
    # Initialize EDA class
    eda = ShoppingTrendsEDA(spark, df)
    
    # Run all analyses
    print("\n" + "="*80)
    print("STARTING EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    eda.overview_statistics()
    demo_results = eda.customer_demographics_analysis()
    purchase_results = eda.purchase_behavior_analysis()
    payment_results = eda.payment_shipping_analysis()
    loyalty_results = eda.customer_loyalty_analysis()
    seasonal_results = eda.seasonal_trend_analysis()
    correlation_results = eda.advanced_correlations()
    summary = eda.generate_summary_report()
    
    print("\n" + "="*80)
    print("EDA COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\n✓ All analyses completed")
    print("✓ Results ready for visualization and further analysis")
    
    return spark, df, eda


if __name__ == "__main__":
    spark, df, eda = main()
    
    # input("\nPress Enter to stop Spark session...")
    # spark.stop()
    # print("✓ Spark session stopped")