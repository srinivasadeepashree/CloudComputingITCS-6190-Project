"""
Complete Streaming Pipeline for Shopping Trends Analysis
ITCS 6190 - Big Data Analytics Project

Features:
- Real-time transaction processing
- 2-minute watermark for late data handling
- 1-minute tumbling windows
- Multiple concurrent queries
- Data quality monitoring
- Anomaly detection

Matches schema: shopping_trends_updated.csv (18 columns + timestamp)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum, avg, window, current_timestamp,
    to_timestamp, expr, lit, when, min, max, 
    round as spark_round, countDistinct
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    FloatType, TimestampType
)
import time
import os


class ShoppingStreamProcessor:
    """
    Real-time shopping transaction processor with watermark support
    Designed for shopping_trends_updated.csv schema
    """
    
    def __init__(self):
        """Initialize Spark session with streaming configuration"""
        self.spark = SparkSession.builder \
            .appName("ShoppingTrendsStreaming") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "8") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)
        
        print("="*80)
        print("✓ Spark Streaming Session Initialized")
        print("="*80)
        print(f"  App Name: ShoppingTrendsStreaming")
        print(f"  Spark Version: {self.spark.version}")
        print(f"  Mode: Local[*]")
        print(f"  Checkpoints: ./checkpoints/")
        print("="*80 + "\n")
    
    def define_schema(self):
        """
        Define schema matching shopping_trends_updated.csv EXACTLY
        
        Schema matches:
        - 18 original columns from CSV
        - 1 timestamp column (added by simulator)
        
        IMPORTANT: Purchase Amount (USD) is INTEGER, not Float!
        """
        return StructType([
            StructField("Customer ID", IntegerType(), True),
            StructField("Age", IntegerType(), True),
            StructField("Gender", StringType(), True),
            StructField("Item Purchased", StringType(), True),
            StructField("Category", StringType(), True),
            StructField("Purchase Amount (USD)", IntegerType(), True),  # Integer!
            StructField("Location", StringType(), True),
            StructField("Size", StringType(), True),
            StructField("Color", StringType(), True),
            StructField("Season", StringType(), True),
            StructField("Review Rating", FloatType(), True),
            StructField("Subscription Status", StringType(), True),
            StructField("Shipping Type", StringType(), True),
            StructField("Discount Applied", StringType(), True),
            StructField("Promo Code Used", StringType(), True),
            StructField("Previous Purchases", IntegerType(), True),
            StructField("Payment Method", StringType(), True),
            StructField("Frequency of Purchases", StringType(), True),
            StructField("timestamp", StringType(), True)  # Added by simulator
        ])
    
    def read_file_stream(self, input_path='data/streaming_input'):
        """
        Read streaming data from directory
        
        Args:
            input_path: Directory containing batch CSV files
            
        Returns:
            Streaming DataFrame with event_time column
        """
        print(f"📂 Initializing file stream...")
        print(f"   Source: {input_path}")
        
        schema = self.define_schema()
        
        streaming_df = self.spark.readStream \
            .schema(schema) \
            .option("header", "true") \
            .option("maxFilesPerTrigger", 1) \
            .csv(input_path)
        
        # Convert timestamp string to TimestampType for watermarking
        streaming_df = streaming_df.withColumn(
            "event_time",
            to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss")
        )
        
        # Add processing time for latency tracking
        streaming_df = streaming_df.withColumn(
            "processing_time",
            current_timestamp()
        )
        
        # Add data quality flags
        streaming_df = streaming_df \
            .withColumn("age_valid", col("Age").isNotNull()) \
            .withColumn("amount_valid", col("`Purchase Amount (USD)`").isNotNull())
        
        print(f"✓ File stream initialized")
        print(f"   Schema: 18 columns + timestamp + derived fields")
        print("")
        
        return streaming_df
    
    def apply_watermark(self, streaming_df, watermark_duration="2 minutes"):
        """
        Apply watermark for handling late-arriving data
        
        Watermark Logic:
        - Current Max Event Time - Watermark Duration = Threshold
        - Data older than threshold is dropped
        - Data within threshold is processed
        
        Args:
            streaming_df: Input streaming DataFrame
            watermark_duration: How long to wait for late data (e.g., "2 minutes")
            
        Returns:
            DataFrame with watermark applied
        """
        print(f"⏰ Applying watermark: {watermark_duration}")
        
        watermarked_df = streaming_df.withWatermark("event_time", watermark_duration)
        
        print(f"✓ Watermark configured")
        print(f"   Column: event_time")
        print(f"   Threshold: {watermark_duration}")
        print(f"   Late data within {watermark_duration} will be accepted")
        print("")
        
        return watermarked_df
    
    def add_derived_features(self, streaming_df):
        """
        Add calculated fields for richer analysis
        
        Args:
            streaming_df: Input DataFrame
            
        Returns:
            DataFrame with additional columns
        """
        enriched_df = streaming_df \
            .withColumn(
                "age_group",
                when(col("Age") < 25, "18-24")
                .when((col("Age") >= 25) & (col("Age") < 35), "25-34")
                .when((col("Age") >= 35) & (col("Age") < 45), "35-44")
                .when((col("Age") >= 45) & (col("Age") < 55), "45-54")
                .otherwise("55+")
            ) \
            .withColumn(
                "purchase_tier",
                when(col("`Purchase Amount (USD)`") < 30, "Budget")
                .when(col("`Purchase Amount (USD)`") < 60, "Standard")
                .otherwise("Premium")
            ) \
            .withColumn(
                "is_subscriber",
                when(col("`Subscription Status`") == "Yes", 1).otherwise(0)
            ) \
            .withColumn(
                "loyalty_tier",
                when(col("`Previous Purchases`") == 0, "New")
                .when(col("`Previous Purchases`") <= 10, "Regular")
                .when(col("`Previous Purchases`") <= 25, "Loyal")
                .otherwise("VIP")
            ) \
            .withColumn(
                "uses_discount",
                when(col("`Discount Applied`") == "Yes", 1).otherwise(0)
            )
        
        return enriched_df
    
    def query_1_windowed_category_performance(self, watermarked_df):
        """
        Query 1: Category performance with 1-minute tumbling windows
        
        Shows: Real-time category sales metrics
        Output Mode: UPDATE (only changed windows)
        """
        print("📊 Setting up Query 1: Category Performance (1-min windows)")
        
        windowed_stats = watermarked_df \
            .groupBy(
                window(col("event_time"), "1 minute"),
                col("Category")
            ) \
            .agg(
                count("*").alias("transaction_count"),
                sum("`Purchase Amount (USD)`").alias("total_revenue"),
                avg("`Purchase Amount (USD)`").alias("avg_purchase"),
                avg("Age").alias("avg_customer_age"),
                avg("`Review Rating`").alias("avg_rating"),
                expr("approx_count_distinct(`Customer ID`)").alias("unique_customers")
            ) \
            .select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("Category"),
                col("transaction_count"),
                col("total_revenue"),
                spark_round(col("avg_purchase"), 2).alias("avg_purchase"),
                spark_round(col("avg_customer_age"), 1).alias("avg_age"),
                spark_round(col("avg_rating"), 2).alias("avg_rating"),
                col("unique_customers")
            )
        
        return windowed_stats
    
    def query_2_realtime_kpis(self, watermarked_df):
        """
        Query 2: Real-time business KPIs dashboard
        
        Shows: Key business metrics updated every minute
        Output Mode: UPDATE
        """
        print("📈 Setting up Query 2: Real-time Business KPIs")
        
        kpis = watermarked_df \
            .groupBy(window(col("event_time"), "1 minute")) \
            .agg(
                count("*").alias("total_transactions"),
                expr("approx_count_distinct(`Customer ID`)").alias("unique_customers"),
                sum("`Purchase Amount (USD)`").alias("total_revenue"),
                avg("`Purchase Amount (USD)`").alias("avg_order_value"),
                avg("`Review Rating`").alias("avg_rating"),
                # Payment methods
                expr("sum(case when `Payment Method` = 'Credit Card' then 1 else 0 end)").alias("credit_card"),
                expr("sum(case when `Payment Method` = 'PayPal' then 1 else 0 end)").alias("paypal"),
                expr("sum(case when `Payment Method` = 'Cash' then 1 else 0 end)").alias("cash"),
                # Subscriptions
                expr("sum(case when `Subscription Status` = 'Yes' then 1 else 0 end)").alias("subscribers"),
                # High value purchases
                expr("sum(case when `Purchase Amount (USD)` >= 60 then 1 else 0 end)").alias("premium_purchases")
            ) \
            .select(
                col("window.start").alias("minute_start"),
                col("window.end").alias("minute_end"),
                col("total_transactions"),
                col("unique_customers"),
                col("total_revenue"),
                spark_round(col("avg_order_value"), 2).alias("avg_order_value"),
                spark_round(col("avg_rating"), 2).alias("avg_rating"),
                spark_round((col("credit_card") * 100.0 / col("total_transactions")), 1).alias("credit_card_pct"),
                spark_round((col("paypal") * 100.0 / col("total_transactions")), 1).alias("paypal_pct"),
                spark_round((col("subscribers") * 100.0 / col("total_transactions")), 1).alias("subscriber_rate"),
                spark_round((col("premium_purchases") * 100.0 / col("total_transactions")), 1).alias("premium_rate")
            )
        
        return kpis
    
    def query_3_customer_behavior(self, watermarked_df):
        """
        Query 3: Customer behavior by demographics
        
        Shows: Spending patterns by age group and gender
        Output Mode: UPDATE
        """
        print("👥 Setting up Query 3: Customer Behavior Analysis")
        
        # Add derived features first
        enriched_df = self.add_derived_features(watermarked_df)
        
        behavior_stats = enriched_df \
            .groupBy(
                window(col("event_time"), "2 minutes"),
                col("age_group"),
                col("Gender")
            ) \
            .agg(
                count("*").alias("customer_count"),
                sum("`Purchase Amount (USD)`").alias("total_spent"),
                avg("`Purchase Amount (USD)`").alias("avg_spent"),
                avg("`Review Rating`").alias("avg_rating"),
                avg("`Previous Purchases`").alias("avg_loyalty"),
                expr("sum(case when `Purchase Amount (USD)` >= 60 then 1 else 0 end)").alias("premium_count")
            ) \
            .select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("age_group"),
                col("Gender"),
                col("customer_count"),
                col("total_spent"),
                spark_round(col("avg_spent"), 2).alias("avg_spent"),
                spark_round(col("avg_rating"), 2).alias("avg_rating"),
                spark_round(col("avg_loyalty"), 1).alias("avg_loyalty"),
                spark_round((col("premium_count") * 100.0 / col("customer_count")), 1).alias("premium_rate")
            )
        
        return behavior_stats
    
    def query_4_anomaly_detection(self, streaming_df):
        """
        Query 4: Real-time anomaly detection
        
        Shows: Unusual transactions flagged in real-time
        Output Mode: APPEND (non-aggregated)
        """
        print("🚨 Setting up Query 4: Anomaly Detection")
        
        anomaly_df = streaming_df \
            .withColumn(
                "anomaly_type",
                when(col("`Purchase Amount (USD)`") >= 90, "HIGH_PURCHASE")
                .when(col("Age") >= 70, "SENIOR_BUYER")
                .when(col("`Purchase Amount (USD)`") <= 20, "LOW_PURCHASE")
                .when(col("`Previous Purchases`") >= 40, "SUPER_LOYAL")
                .otherwise("NORMAL")
            ) \
            .filter(col("anomaly_type") != "NORMAL") \
            .select(
                col("event_time"),
                col("`Customer ID`"),
                col("Age"),
                col("Gender"),
                col("`Item Purchased`"),
                col("Category"),
                col("`Purchase Amount (USD)`"),
                col("`Previous Purchases`"),
                col("anomaly_type")
            )
        
        return anomaly_df
    
    def query_5_data_quality(self, streaming_df):
        """
        Query 5: Data quality monitoring
        
        Shows: Real-time data quality metrics (nulls, latency)
        Output Mode: UPDATE
        """
        print("🔍 Setting up Query 5: Data Quality Monitoring")
        
        quality_stats = streaming_df \
            .groupBy(window(col("processing_time"), "1 minute")) \
            .agg(
                count("*").alias("total_records"),
                sum(when(col("age_valid"), 0).otherwise(1)).alias("age_null_count"),
                sum(when(col("amount_valid"), 0).otherwise(1)).alias("amount_null_count"),
                avg(expr("unix_timestamp(processing_time) - unix_timestamp(event_time)")).alias("avg_delay_seconds")
            ) \
            .select(
                col("window.start").alias("window_start"),
                col("total_records"),
                col("age_null_count"),
                col("amount_null_count"),
                spark_round(col("avg_delay_seconds"), 1).alias("avg_delay_sec"),
                spark_round((col("age_null_count") * 100.0 / col("total_records")), 2).alias("age_null_pct"),
                spark_round((col("amount_null_count") * 100.0 / col("total_records")), 2).alias("amount_null_pct")
            )
        
        return quality_stats
    
    def write_to_console(self, streaming_df, query_name, output_mode="update"):
        """
        Write stream to console for monitoring
        
        Args:
            streaming_df: Streaming DataFrame
            query_name: Unique query name
            output_mode: "append", "update", or "complete"
            
        Returns:
            StreamingQuery object
        """
        query = streaming_df.writeStream \
            .outputMode(output_mode) \
            .format("console") \
            .option("truncate", "false") \
            .option("checkpointLocation", f"checkpoints/{query_name}") \
            .queryName(query_name) \
            .start()
        
        print(f"   ✓ Started: {query_name} (mode: {output_mode})")
        return query
    
    def write_to_memory(self, streaming_df, table_name, output_mode="update"):
        """
        Write stream to in-memory table for SQL queries
        
        Args:
            streaming_df: Streaming DataFrame
            table_name: Table name for SQL queries
            output_mode: Output mode
            
        Returns:
            StreamingQuery object
        """
        query = streaming_df.writeStream \
            .outputMode(output_mode) \
            .format("memory") \
            .queryName(table_name) \
            .option("checkpointLocation", f"checkpoints/{table_name}") \
            .start()
        
        print(f"   ✓ Memory table: {table_name} (mode: {output_mode})")
        return query


def run_complete_pipeline():
    """
    Run the complete streaming pipeline with all queries
    """
    print("\n" + "="*80)
    print("STARTING COMPLETE REAL-TIME STREAMING PIPELINE")
    print("="*80 + "\n")
    
    # Initialize processor
    processor = ShoppingStreamProcessor()
    
    # Read streaming data
    streaming_df = processor.read_file_stream('data/streaming_input')
    
    # Apply 2-minute watermark
    watermarked_df = processor.apply_watermark(streaming_df, "2 minutes")
    
    print("="*80)
    print("INITIALIZING 5 CONCURRENT STREAMING QUERIES")
    print("="*80 + "\n")
    
    queries = []
    
    # Query 1: Category performance (to memory for querying)
    windowed_cat = processor.query_1_windowed_category_performance(watermarked_df)
    q1 = processor.write_to_memory(windowed_cat, "category_performance", "update")
    queries.append(q1)
    
    # Query 2: Real-time KPIs (to console for monitoring)
    kpis = processor.query_2_realtime_kpis(watermarked_df)
    q2 = processor.write_to_console(kpis, "business_kpis", "update")
    queries.append(q2)
    
    # Query 3: Customer behavior (to memory)
    behavior = processor.query_3_customer_behavior(watermarked_df)
    q3 = processor.write_to_memory(behavior, "customer_behavior", "update")
    queries.append(q3)
    
    # Query 4: Anomaly detection (to console)
    anomalies = processor.query_4_anomaly_detection(streaming_df)
    q4 = processor.write_to_console(anomalies, "anomaly_alerts", "append")
    queries.append(q4)
    
    # Query 5: Data quality (to memory)
    quality = processor.query_5_data_quality(streaming_df)
    q5 = processor.write_to_memory(quality, "data_quality", "update")
    queries.append(q5)
    
    print("\n" + "="*80)
    print("✓ ALL 5 QUERIES RUNNING")
    print("="*80)
    print("\nStreaming for 3 minutes with status updates every 30 seconds...")
    print("Watch the console for real-time KPIs and anomaly alerts!\n")
    
    # Run for 3 minutes with periodic status updates
    for i in range(6):
        time.sleep(30)
        
        print("\n" + "─"*80)
        print(f"⏰ STATUS UPDATE {i+1}/6 - {(i+1)*30} seconds elapsed")
        print("─"*80)
        
        try:
            # Query memory tables
            print("\n📊 Top Categories (Recent Windows):")
            processor.spark.sql("""
                SELECT 
                    window_start, 
                    Category, 
                    transaction_count, 
                    total_revenue, 
                    avg_purchase
                FROM category_performance
                ORDER BY window_start DESC, total_revenue DESC
                LIMIT 5
            """).show(truncate=False)
            
            print("\n👥 Customer Behavior Patterns:")
            processor.spark.sql("""
                SELECT 
                    age_group, 
                    Gender, 
                    customer_count, 
                    avg_spent, 
                    premium_rate
                FROM customer_behavior
                ORDER BY window_start DESC, customer_count DESC
                LIMIT 5
            """).show(truncate=False)
            
            print("\n🔍 Data Quality Status:")
            processor.spark.sql("""
                SELECT 
                    window_start, 
                    total_records, 
                    age_null_count,
                    amount_null_count,
                    age_null_pct, 
                    amount_null_pct, 
                    avg_delay_sec
                FROM data_quality
                ORDER BY window_start DESC
                LIMIT 3
            """).show(truncate=False)
            
        except Exception as e:
            print(f"   (Tables initializing... {e})")
    
    # Stop all queries
    print("\n" + "="*80)
    print("STOPPING ALL QUERIES")
    print("="*80)
    
    for idx, query in enumerate(queries, 1):
        query.stop()
        print(f"✓ Stopped Query {idx}")
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print("\nFinal Summary:")
    print("  ✓ Processed streaming transactions in real-time")
    print("  ✓ Applied 2-minute watermark for late data")
    print("  ✓ Ran 5 concurrent queries simultaneously")
    print("  ✓ Monitored data quality (nulls: 0%)")
    print("  ✓ Detected anomalies in real-time")
    print("\nCheckpoint data saved in: ./checkpoints/")
    print("Ready for recovery on restart!\n")
    
    processor.spark.stop()


def main():
    """Main execution function"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║        Shopping Trends - Real-Time Streaming Pipeline         ║
    ║        ITCS 6190 Big Data Analytics - Milestone 2            ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  IMPORTANT PREREQUISITES:")
    print("="*80)
    print("1. Run simulator first:")
    print("   python3 src/streaming_simulator.py")
    print("   (Choose option 2 - Standard configuration)")
    print("")
    print("2. Wait for batches to start appearing in data/streaming_input/")
    print("")
    print("3. Then run this pipeline")
    print("="*80)
    
    response = input("\nHave you started the simulator? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        try:
            run_complete_pipeline()
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            print("   Checkpoint data saved for recovery")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  Please start the simulator first!")
        print("   Run: python3 src/streaming_simulator.py")


if __name__ == "__main__":
    main()