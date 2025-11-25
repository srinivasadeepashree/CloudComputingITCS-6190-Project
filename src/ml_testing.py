"""
Machine Learning Testing and Prediction Pipeline
ITCS 6190 - Big Data Analytics Project 

Tests trained ML models and makes predictions on new data
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, expr, udf, round as spark_round, avg
from pyspark.ml import PipelineModel
from pyspark.sql.types import StringType, FloatType
from pyspark.ml.functions import vector_to_array
import os


class MLModelTester:
    """
    Test and evaluate trained ML models
    """
    
    def __init__(self):
        """Initialize Spark session"""
        self.spark = SparkSession.builder \
            .appName("ShoppingTrendsMLTesting") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        print("="*80)
        print("✓ ML Testing Pipeline Initialized")
        print("="*80 + "\n")
    
    def load_models(self):
        """
        Load trained models from disk
        
        Returns:
            Dictionary of loaded models
        """
        print("📂 Loading trained models...")
        
        models = {}
        
        # Load pricing model
        pricing_path = "models/pricing_model"
        if os.path.exists(pricing_path):
            models['pricing'] = PipelineModel.load(pricing_path)
            print(f"  ✓ Pricing model loaded from: {pricing_path}")
        else:
            print(f"  ❌ Pricing model not found at: {pricing_path}")
            print(f"     Run ml_training.py first!")
        
        # Load category model
        category_path = "models/category_model"
        if os.path.exists(category_path):
            models['category'] = PipelineModel.load(category_path)
            print(f"  ✓ Category model loaded from: {category_path}")
        else:
            print(f"  ❌ Category model not found at: {category_path}")
            print(f"     Run ml_training.py first!")
        
        print("")
        return models
    
    def load_test_data(self, filepath="data/shopping.csv"):
        """
        Load test data (can be same or new data)
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame
        """
        print(f"📂 Loading test data from: {filepath}")
        
        df = self.spark.read.csv(filepath, header=True, inferSchema=True)
        
        print(f"✓ Test data loaded: {df.count():,} records")
        print("")
        
        return df
    
    def prepare_pricing_features(self, df):
        """
        Prepare features for pricing model (same as training)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with features
        """
        # Same feature engineering as training
        df = df.withColumn(
            'is_subscriber',
            when(col('`Subscription Status`') == 'Yes', 1.0).otherwise(0.0)
        )
        
        df = df.withColumn(
            'used_promo',
            when(col('`Promo Code Used`') == 'Yes', 1.0).otherwise(0.0)
        )
        
        df = df.withColumn(
            'is_new_customer',
            when(col('`Previous Purchases`') <= 5, 1.0).otherwise(0.0)
        )
        
        df = df.withColumn(
            'is_loyal_customer',
            when(col('`Previous Purchases`') >= 20, 1.0).otherwise(0.0)
        )
        
        df = df.withColumn(
            'high_value_purchase',
            when(col('`Purchase Amount (USD)`') >= 60, 1.0).otherwise(0.0)
        )
        
        df = df.withColumn(
            'purchases_per_age',
            col('`Previous Purchases`') / col('Age')
        )
        
        return df
    
    def prepare_category_features(self, df):
        """
        Prepare features for category model (same as training)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with features
        """
        df = df.withColumn(
            'customer_tier',
            when(col('`Previous Purchases`') == 0, 'New')
            .when(col('`Previous Purchases`') <= 10, 'Regular')
            .when(col('`Previous Purchases`') <= 25, 'Loyal')
            .otherwise('VIP')
        )
        
        df = df.withColumn(
            'age_group',
            when(col('Age') < 25, '18-24')
            .when((col('Age') >= 25) & (col('Age') < 35), '25-34')
            .when((col('Age') >= 35) & (col('Age') < 45), '35-44')
            .when((col('Age') >= 45) & (col('Age') < 55), '45-54')
            .otherwise('55+')
        )
        
        df = df.withColumn(
            'is_frequent_buyer',
            when(col('`Frequency of Purchases`').isin(['Weekly', 'Fortnightly']), 1.0)
            .otherwise(0.0)
        )
        
        return df
    
    def test_pricing_model(self, model, df):
        """
        Test Dynamic Pricing Model
        
        Args:
            model: Trained pricing model
            df: Test DataFrame
            
        Returns:
            Predictions DataFrame
        """
        print("="*80)
        print("TESTING MODEL 1: DYNAMIC PRICING (PRICE SENSITIVITY)")
        print("="*80 + "\n")
        
        # Prepare features
        df = self.prepare_pricing_features(df)
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions = model.transform(df)
        
        # Extract probability of needing discount (class 1)
        # NEW:
        predictions = predictions.withColumn(
            'probability_array',
            vector_to_array('probability')
        )
        predictions = predictions.withColumn(
            'price_sensitivity_score',
             col('probability_array')[1]
        )
        
        # Add business recommendations
        predictions = predictions.withColumn(
            'recommended_discount',
            when(col('price_sensitivity_score') >= 0.7, '20-25%')
            .when(col('price_sensitivity_score') >= 0.5, '15-20%')
            .when(col('price_sensitivity_score') >= 0.3, '10-15%')
            .otherwise('0-10%')
        )
        
        predictions = predictions.withColumn(
            'customer_segment',
            when(col('price_sensitivity_score') >= 0.7, 'High Sensitivity')
            .when(col('price_sensitivity_score') >= 0.5, 'Medium Sensitivity')
            .when(col('price_sensitivity_score') >= 0.3, 'Low Sensitivity')
            .otherwise('Price Insensitive')
        )
        
        # Show results
        print("\n📊 PREDICTION RESULTS:")
        print("-"*80)
        
        # Distribution by segment
        print("\nCustomer Segmentation:")
        predictions.groupBy('customer_segment').count().orderBy(col('count').desc()).show()
        
        # Sample predictions
        print("\n🔍 Sample Predictions (10 customers):")
        predictions.select(
            col('`Customer ID`').alias('Customer_ID'),
            'Age',
            col('`Previous Purchases`').alias('Loyalty'),
            col('`Subscription Status`').alias('Subscriber'),
            spark_round('price_sensitivity_score', 3).alias('Sensitivity_Score'),
            'recommended_discount',
            'customer_segment'
        ).show(10, truncate=False)
        
        # Business insights
        print("\n💡 BUSINESS INSIGHTS:")
        print("-"*80)
        
        high_sensitivity = predictions.filter(col('price_sensitivity_score') >= 0.7).count()
        low_sensitivity = predictions.filter(col('price_sensitivity_score') < 0.3).count()
        total = predictions.count()
        
        print(f"  High Sensitivity (>0.7): {high_sensitivity} customers ({high_sensitivity/total*100:.1f}%)")
        print(f"    → Recommendation: Offer 20-25% discount")
        print(f"  Low Sensitivity (<0.3): {low_sensitivity} customers ({low_sensitivity/total*100:.1f}%)")
        print(f"    → Recommendation: Minimal or no discount needed")
        print(f"\n  Potential Savings: ${low_sensitivity * 60 * 0.20:,.2f}")
        print(f"    (By not giving unnecessary 20% discounts to {low_sensitivity} price-insensitive customers)")
        
        print("\n✓ Pricing model testing complete")
        print("")
        
        return predictions
    
    def test_category_model(self, model, df):
        """
        Test Next Category Prediction Model
        
        Args:
            model: Trained category model
            df: Test DataFrame
            
        Returns:
            Predictions DataFrame
        """
        print("="*80)
        print("TESTING MODEL 2: NEXT CATEGORY PREDICTION")
        print("="*80 + "\n")
        
        # Prepare features
        df = self.prepare_category_features(df)
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions = model.transform(df)
        
        # Extract top 3 category probabilities
        @udf(returnType=StringType())
        def get_top_categories(probability, label_map={'0.0': 'Accessories', '1.0': 'Clothing', 
                                                       '2.0': 'Footwear', '3.0': 'Outerwear'}):
            """Extract top 3 predicted categories"""
            if probability is None:
                return "Unknown"
            
            # Get indices sorted by probability
            probs = list(probability)
            indexed_probs = [(i, p) for i, p in enumerate(probs)]
            sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
            
            # Get top 3
            top_3 = sorted_probs[:3]
            result = [f"{label_map.get(str(float(idx)), 'Unknown')}({prob:.2f})" 
                     for idx, prob in top_3]
            
            return ", ".join(result)
        
        predictions = predictions.withColumn(
            'top_3_predictions',
            get_top_categories(col('probability'))
        )
        
        # Add confidence score (probability of top prediction)
        @udf(returnType=FloatType())
        def get_confidence(probability):
            """Get confidence of top prediction"""
            if probability is None:
                return 0.0
            return float(max(list(probability)))
        
        predictions = predictions.withColumn(
            'confidence_score',
            get_confidence(col('probability'))
        )
        
        # Show results
        print("\n📊 PREDICTION RESULTS:")
        print("-"*80)
        
        # Sample predictions
        print("\n🔍 Sample Predictions (15 customers):")
        predictions.select(
            col('`Customer ID`').alias('Customer_ID'),
            'Age',
            'Gender',
            'Season',
            col('`Previous Purchases`').alias('Loyalty'),
            'Category',
            spark_round('confidence_score', 3).alias('Confidence'),
            'top_3_predictions'
        ).show(15, truncate=False)
        
        # Confidence distribution
        print("\n📊 Prediction Confidence Distribution:")
        predictions.withColumn(
            'confidence_level',
            when(col('confidence_score') >= 0.7, 'High (>70%)')
            .when(col('confidence_score') >= 0.5, 'Medium (50-70%)')
            .otherwise('Low (<50%)')
        ).groupBy('confidence_level').count().orderBy(col('count').desc()).show()
        
        # Category prediction distribution
        print("\n📊 Predicted Category Distribution:")
        predictions.groupBy('prediction').count().orderBy(col('count').desc()).show()
        
        # Business insights
        print("\n💡 BUSINESS INSIGHTS:")
        print("-"*80)
        
        high_conf = predictions.filter(col('confidence_score') >= 0.7).count()
        total = predictions.count()
        
        print(f"  High Confidence Predictions: {high_conf} ({high_conf/total*100:.1f}%)")
        print(f"    → Use for: Personalized product recommendations")
        print(f"    → Use for: Targeted email campaigns")
        print(f"    → Use for: Inventory planning by customer segment")
        
        print("\n  Example Use Cases:")
        print("    - Customer predicted to buy Clothing → Show clothing collection")
        print("    - Customer predicted to buy Footwear → Send shoe catalog")
        print("    - High confidence predictions → Primary recommendations")
        print("    - Lower confidence → Show diverse product mix")
        
        print("\n✓ Category model testing complete")
        print("")
        
        return predictions
    
    def create_customer_profile(self, pricing_pred, category_pred):
        """
        Create comprehensive customer profile combining both models
        
        Args:
            pricing_pred: Predictions from pricing model
            category_pred: Predictions from category model
            
        Returns:
            Combined customer profile
        """
        print("="*80)
        print("CREATING COMPREHENSIVE CUSTOMER PROFILES")
        print("="*80 + "\n")
        
        # Join predictions on Customer ID
        profile = pricing_pred.select(
            col('`Customer ID`').alias('Customer_ID'),
            'Age',
            'Gender',
            col('`Previous Purchases`').alias('Loyalty_Score'),
            col('`Subscription Status`').alias('Subscriber'),
            'price_sensitivity_score',
            'recommended_discount',
            'customer_segment'
        ).join(
            category_pred.select(
                col('`Customer ID`').alias('Customer_ID_cat'),
                'confidence_score',
                'top_3_predictions'
            ),
            col('Customer_ID') == col('Customer_ID_cat'),
            'inner'
        ).drop('Customer_ID_cat')
        
        # Add overall customer value score
        profile = profile.withColumn(
            'customer_value_score',
            spark_round(
                (1 - col('price_sensitivity_score')) * col('Loyalty_Score') / 50.0,
                3
            )
        )
        
        profile = profile.withColumn(
            'customer_tier',
            when(col('customer_value_score') >= 0.7, 'VIP')
            .when(col('customer_value_score') >= 0.5, 'High Value')
            .when(col('customer_value_score') >= 0.3, 'Medium Value')
            .otherwise('Standard')
        )
        
        # Show comprehensive profiles
        print("🎯 COMPREHENSIVE CUSTOMER PROFILES:")
        print("-"*80 + "\n")
        
        profile.select(
            'Customer_ID',
            'Age',
            'Gender',
            'Subscriber',
            'Loyalty_Score',
            spark_round('price_sensitivity_score', 2).alias('Price_Sens'),
            'recommended_discount',
            spark_round('confidence_score', 2).alias('Cat_Conf'),
            'top_3_predictions',
            'customer_tier'
        ).show(20, truncate=False)
        
        # Tier distribution
        print("\n📊 Customer Tier Distribution:")
        profile.groupBy('customer_tier').count().orderBy(col('count').desc()).show()
        
        # Business recommendations
        print("\n💼 BUSINESS RECOMMENDATIONS BY TIER:")
        print("-"*80)
        
        for tier in ['VIP', 'High Value', 'Medium Value', 'Standard']:
            tier_customers = profile.filter(col('customer_tier') == tier)
            count = tier_customers.count()
            avg_discount = tier_customers.select(avg('price_sensitivity_score')).first()[0]
            
            if count > 0:
                print(f"\n  {tier} ({count} customers):")
                if tier == 'VIP':
                    print(f"    - Minimal discounts needed (avg sensitivity: {avg_discount:.2f})")
                    print(f"    - Personalized service and early access")
                    print(f"    - High-confidence product recommendations")
                elif tier == 'High Value':
                    print(f"    - Moderate discount strategy (avg sensitivity: {avg_discount:.2f})")
                    print(f"    - Loyalty program incentives")
                    print(f"    - Category-based recommendations")
                elif tier == 'Medium Value':
                    print(f"    - Standard discount offers (avg sensitivity: {avg_discount:.2f})")
                    print(f"    - Engagement campaigns")
                    print(f"    - Cross-sell opportunities")
                else:
                    print(f"    - Competitive discounts (avg sensitivity: {avg_discount:.2f})")
                    print(f"    - Conversion-focused campaigns")
                    print(f"    - Broad product recommendations")
        
        print("\n✓ Customer profiles created")
        print("")
        
        return profile
    
    def generate_summary_report(self):
        """
        Generate final summary report
        """
        print("\n" + "="*80)
        print("TESTING COMPLETE - SUMMARY REPORT")
        print("="*80)
        
        print("\n✓ Both models tested successfully")
        print("\n📊 Model Outputs:")
        print("  1. Price Sensitivity Scores (0-1)")
        print("  2. Recommended Discount Levels")
        print("  3. Next Category Predictions")
        print("  4. Prediction Confidence Scores")
        print("  5. Comprehensive Customer Profiles")
        
        print("\n💡 Business Applications:")
        print("  ✓ Personalized pricing strategies")
        print("  ✓ Optimized discount allocation")
        print("  ✓ Targeted product recommendations")
        print("  ✓ Customer segmentation and tiering")
        print("  ✓ Marketing campaign optimization")
        
        print("\n🚀 Next Steps:")
        print("  1. Integrate models with streaming pipeline")
        print("  2. Deploy for real-time predictions")
        print("  3. Monitor model performance")
        print("  4. Retrain periodically with new data")
        
        print("\n" + "="*80)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")


def main():
    """
    Main execution function
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║        Machine Learning Testing Pipeline                       ║
    ║        ITCS 6190 Big Data Analytics Project                    ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize tester
    tester = MLModelTester()
    
    # Load trained models
    models = tester.load_models()
    
    if len(models) < 2:
        print("\n❌ ERROR: Models not found!")
        print("   Please run: python3 src/ml_training.py first")
        return
    
    # Load test data
    test_df = tester.load_test_data("data/shopping.csv")
    
    input("\nPress Enter to test Model 1 (Dynamic Pricing)...")
    
    # Test pricing model
    pricing_predictions = tester.test_pricing_model(models['pricing'], test_df)
    
    input("\nPress Enter to test Model 2 (Next Category Prediction)...")
    
    # Test category model
    category_predictions = tester.test_category_model(models['category'], test_df)
    
    input("\nPress Enter to create comprehensive customer profiles...")
    
    # Create customer profiles
    customer_profiles = tester.create_customer_profile(
        pricing_predictions, 
        category_predictions
    )
    
    # Generate summary
    tester.generate_summary_report()
    
    # Optionally save results
    print("💾 Saving results...")
    customer_profiles.write.mode("overwrite").parquet("output/customer_profiles")
    print("✓ Customer profiles saved to: output/customer_profiles/")
    print("")
    
    # Stop Spark
    tester.spark.stop()


if __name__ == "__main__":
    main()