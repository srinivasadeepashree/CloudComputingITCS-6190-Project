# Data ingestion script placeholder
"""
Data Ingestion Module for Shopping Trends Analysis
ITCS 6190 - Big Data Analytics Project

This module handles the initial data loading and basic validation
using Apache Spark Structured APIs.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, isnan, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation


import numpy as np
import os
import sys


def create_spark_session(app_name="ShoppingTrendsAnalysis"):
    """
    Create and configure Spark session for local execution
    
    Args:
        app_name: Name of the Spark application
    
    Returns:
        SparkSession object
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print(f"✓ Spark Session created: {app_name}")
    print(f"  Spark Version: {spark.version}")
    return spark


def define_schema():
    """
    Define the schema for shopping_trends.csv
    Adjust field names and types based on your actual dataset structure
    
    Returns:
        StructType schema
    """
    schema = StructType([
        StructField("Customer ID", IntegerType(), True),
        StructField("Age", IntegerType(), True),
        StructField("Gender", StringType(), True),
        StructField("Item Purchased", StringType(), True),
        StructField("Category", StringType(), True),
        StructField("Purchase Amount (USD)", FloatType(), True),
        StructField("Location", StringType(), True),
        StructField("Size", StringType(), True),
        StructField("Color", StringType(), True),
        StructField("Season", StringType(), True),
        StructField("Review Rating", FloatType(), True),
        StructField("Subscription Status", StringType(), True),
        StructField("Payment Method", StringType(), True),
        StructField("Shipping Type", StringType(), True),
        StructField("Discount Applied", StringType(), True),
        StructField("Promo Code Used", StringType(), True),
        StructField("Previous Purchases", IntegerType(), True),
        StructField("Preferred Payment Method", StringType(), True),
        StructField("Frequency of Purchases", StringType(), True)
    ])
    return schema


def load_data(spark, file_path, schema=None):
    """
    Load CSV data into Spark DataFrame
    
    Args:
        spark: SparkSession object
        file_path: Path to the CSV file
        schema: Optional StructType schema
    
    Returns:
        DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    print(f"\n{'='*60}")
    print(f"Loading data from: {file_path}")
    print(f"{'='*60}")
    
    if schema:
        df = spark.read.csv(file_path, header=True, schema=schema)
    else:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    row_count = df.count()
    col_count = len(df.columns)
    
    print(f"✓ Data loaded successfully")
    print(f"  Rows: {row_count:,}")
    print(f"  Columns: {col_count}")
    
    return df


def validate_data(df):
    """
    Perform basic data quality checks
    
    Args:
        df: Spark DataFrame
    
    Returns:
        dict: Validation results
    """
    print(f"\n{'='*60}")
    print("DATA VALIDATION")
    print(f"{'='*60}")
    
    results = {}
    
    # Check for duplicates
    total_rows = df.count()
    distinct_rows = df.distinct().count()
    duplicates = total_rows - distinct_rows
    results['duplicates'] = duplicates
    
    print(f"\nDuplicate Check:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Distinct rows: {distinct_rows:,}")
    print(f"  Duplicates: {duplicates:,}")
    
    # Check for null values in each column
    print(f"\nNull Value Check:")
    null_counts = []
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        if null_count > 0:
            null_counts.append((column, null_count, f"{(null_count/total_rows)*100:.2f}%"))
            print(f"  {column}: {null_count} ({(null_count/total_rows)*100:.2f}%)")
    
    if not null_counts:
        print("  ✓ No null values found")
    
    results['null_counts'] = null_counts
    
    return results


def display_schema_info(df):
    """
    Display schema information
    
    Args:
        df: Spark DataFrame
    """
    print(f"\n{'='*60}")
    print("SCHEMA INFORMATION")
    print(f"{'='*60}\n")
    df.printSchema()


def display_sample_data(df, n=5):
    """
    Display sample records
    
    Args:
        df: Spark DataFrame
        n: Number of rows to display
    """
    print(f"\n{'='*60}")
    print(f"SAMPLE DATA (First {n} rows)")
    print(f"{'='*60}\n")
    df.show(n, truncate=False)


def get_basic_statistics(df):
    """
    Calculate basic statistics for numerical columns
    
    Args:
        df: Spark DataFrame
    
    Returns:
        DataFrame with statistics
    """
    print(f"\n{'='*60}")
    print("BASIC STATISTICS")
    print(f"{'='*60}\n")
    
    stats_df = df.describe()
    stats_df.show(truncate=False)
    
    return stats_df

 #------------------------------------------------------------------------------
# 4️⃣ COMPUTE CORRELATION MATRIX + REMOVE REDUNDANT COLUMNS
# ------------------------------------------------------------------------------
def reduce_features_by_correlation(df, threshold=0.85):
    print("\n🔍 Detecting numeric columns...")

    numeric_cols = [
        field.name for field in df.schema.fields
        if isinstance(field.dataType, (IntegerType, FloatType, DoubleType))
    ]

    print(f"Numeric Columns Found: {numeric_cols}")

    if len(numeric_cols) < 2:
        print("⚠ Not enough numeric columns for correlation analysis.")
        return df

    print("\n📊 Building feature vector for correlation...")
    assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="features"
    )
    vector_df = assembler.transform(df).select("features")

    print("📈 Computing correlation matrix...")
    corr_matrix = Correlation.corr(vector_df, "features").head()[0].toArray()
    print("\nCorrelation Matrix:")
    print(np.round(corr_matrix, 3))

    print("\n🔎 Identifying highly correlated pairs...")
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix[i][j]) > threshold:
                high_corr_pairs.append(
                    (numeric_cols[i], numeric_cols[j], corr_matrix[i][j])
                )

    if not high_corr_pairs:
        print("✔ No highly correlated columns found!")
        return df

    print("\n⚠ Highly correlated column pairs detected:")
    for col1, col2, corr_value in high_corr_pairs:
        print(f"  - ({col1}) <--> ({col2}) | Corr = {corr_value:.3f}")

    print("\n🗑 Removing redundant columns...")
    columns_to_drop = list({pair[1] for pair in high_corr_pairs})
    print(f"Columns Dropped: {columns_to_drop}")

    df_cleaned = df.drop(*columns_to_drop)
    print(f"✔ Feature Reduction Complete | Remaining Columns: {df_cleaned.columns}")

    return df_cleaned



def save_processed_data(df, output_path, format="parquet"):
    """
    Save processed DataFrame for downstream analysis
    
    Args:
        df: Spark DataFrame
        output_path: Path to save the data
        format: Output format (parquet, csv, etc.)
    """
    print(f"\n{'='*60}")
    print(f"Saving processed data to: {output_path}")
    print(f"{'='*60}")
    
    if format == "parquet":
        df.write.mode("overwrite").parquet(output_path)
    elif format == "csv":
        df.write.mode("overwrite").option("header", "true").csv(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"✓ Data saved successfully as {format}")


def main():
    """
    Main execution function for data ingestion
    """
    # Configuration
    DATA_PATH = "data/shopping.csv"
    OUTPUT_PATH = "data/processed/shopping_clean"
    
    try:
        # Create Spark session
        spark = create_spark_session()
        
        # Define schema (optional - can use inferSchema=True instead)
        # schema = define_schema()
        
        # Load data
        df = load_data(spark, DATA_PATH, schema=None)
        
        # Display schema
        display_schema_info(df)
        
        # Display sample data
        display_sample_data(df, n=10)

        print("\n📈 Starting Feature Correlation Reduction...")
        df_cleaned = reduce_features_by_correlation(df)

        print("\n💾 Saving cleaned DataFrame for ML Training...")
        save_processed_data(df_cleaned, OUTPUT_PATH, format="parquet")
        
        # Validate data
        validation_results = validate_data(df)
        
        # Get basic statistics
        stats_df = get_basic_statistics(df)
        
        # Cache DataFrame for repeated operations
        df.cache()
        print("\n✓ DataFrame cached for downstream operations")
        
        # Save processed data
        save_processed_data(df, OUTPUT_PATH, format="parquet")
        
        print(f"\n{'='*60}")
        print("DATA INGESTION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")
        
        # Keep Spark session alive for notebook usage
        return spark, df
        
    except Exception as e:
        print(f"\n✗ Error during data ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    spark, df = main()
    
    # Keep the session running for interactive exploration
    input("\nPress Enter to stop Spark session...")
    spark.stop()
    print("✓ Spark session stopped")