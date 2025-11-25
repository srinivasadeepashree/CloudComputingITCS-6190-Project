"""
Machine Learning Training Pipeline
ITCS 6190 - Big Data Analytics Project

Trains two ML models:
1. Dynamic Pricing Model (Price Sensitivity Prediction)
2. Next Category Prediction Model

Uses Spark MLlib for scalable machine learning
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, expr, count, avg, lit
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder, 
    StandardScaler, MinMaxScaler
)
from pyspark.ml.classification import (
    RandomForestClassifier, 
    GBTClassifier,
    LogisticRegression,
    DecisionTreeClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os
import time


class MLModelTrainer:
    """
    Train and evaluate ML models for shopping trends
    """
    
    def __init__(self):
        """Initialize Spark session"""
        self.spark = SparkSession.builder \
            .appName("ShoppingTrendsML") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
        
        print("="*80)
        print("✓ ML Training Pipeline Initialized")
        print("="*80)
        print(f"  Spark Version: {self.spark.version}")
        print(f"  Model Directory: ./models/")
        print("="*80 + "\n")
    
    def load_data(self, filepath="data/shopping.csv"):
        """
        Load and validate dataset
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Spark DataFrame
        """
        print(f"📂 Loading data from: {filepath}")
        
        df = self.spark.read.csv(filepath, header=True, inferSchema=True)
        
        total_rows = df.count()
        total_cols = len(df.columns)
        
        print(f"✓ Data loaded successfully")
        print(f"  Rows: {total_rows:,}")
        print(f"  Columns: {total_cols}")
        print("")
        
        # Cache for performance
        df.cache()
        
        return df
    
    def prepare_pricing_model_data(self, df):
        """
        Prepare features for Dynamic Pricing Model (Option 2)
        
        Target: Price sensitivity (needs discount or not)
        Features: All customer attributes
        
        Args:
            df: Input DataFrame
            
        Returns:
            Prepared DataFrame with features and target
        """
        print("="*80)
        print("PREPARING DATA FOR DYNAMIC PRICING MODEL")
        print("="*80)
        
        # Create target variable: 1 = needs discount, 0 = doesn't need
        df = df.withColumn(
            'price_sensitive',
            when(col('`Discount Applied`') == 'Yes', 1.0).otherwise(0.0)
        )
        
        # Feature engineering
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
        
        # Display class distribution
        print("\n📊 Target Variable Distribution:")
        df.groupBy('price_sensitive').count().orderBy('price_sensitive').show()
        
        class_counts = df.groupBy('price_sensitive').count().collect()
        total = sum([row['count'] for row in class_counts])
        for row in class_counts:
            label = "Needs Discount" if row['price_sensitive'] == 1.0 else "No Discount"
            pct = (row['count'] / total) * 100
            print(f"  {label}: {row['count']} ({pct:.1f}%)")
        
        print("\n✓ Pricing model data prepared")
        print(f"  Total samples: {df.count():,}")
        print(f"  Features: Numerical + Categorical")
        print("")
        
        return df
    
    def prepare_category_model_data(self, df):
        """
        Prepare features for Next Category Prediction Model (Option 3)
        
        Target: Category (4 classes)
        Features: Customer demographics and behavior
        
        Args:
            df: Input DataFrame
            
        Returns:
            Prepared DataFrame with features and target
        """
        print("="*80)
        print("PREPARING DATA FOR NEXT CATEGORY PREDICTION MODEL")
        print("="*80)
        
        # Target is already there: Category
        # Just create additional features
        
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
        
        # Display category distribution
        print("\n📊 Target Variable Distribution (Categories):")
        df.groupBy('Category').count().orderBy(col('count').desc()).show()
        
        print("\n✓ Category model data prepared")
        print(f"  Total samples: {df.count():,}")
        print(f"  Target classes: 4 (Clothing, Footwear, Accessories, Outerwear)")
        print("")
        
        return df
    
    def train_pricing_model(self, df):
        """
        Train Dynamic Pricing Model (Binary Classification)
        
        Predicts: Will customer need discount? (Yes/No)
        
        Args:
            df: Prepared DataFrame
            
        Returns:
            Trained model and evaluation metrics
        """
        print("="*80)
        print("TRAINING MODEL 1: DYNAMIC PRICING (PRICE SENSITIVITY)")
        print("="*80)
        print("\nModel Type: Binary Classification")
        print("Algorithm: Gradient Boosting Trees")
        print("Target: price_sensitive (0=No discount, 1=Needs discount)")
        print("-"*80 + "\n")
        
        # Define feature columns
        categorical_cols = [
            'Gender', 'Category', 'Season', 'Location',
            'Payment Method', 'Frequency of Purchases',
            'Shipping Type'
        ]
        
        numerical_cols = [
            'Age', 'Purchase Amount (USD)', 'Previous Purchases',
            'Review Rating', 'is_subscriber', 'used_promo',
            'is_new_customer', 'is_loyal_customer', 'high_value_purchase',
            'purchases_per_age'
        ]
        
        # Build preprocessing pipeline
        print("🔧 Building feature preprocessing pipeline...")
        
        # Stage 1: String indexing for categorical variables
        indexers = [
            StringIndexer(
                inputCol=col, 
                outputCol=col + "_indexed",
                handleInvalid="keep"
            ) for col in categorical_cols
        ]
        
        # Stage 2: One-hot encoding
        encoders = [
            OneHotEncoder(
                inputCol=col + "_indexed",
                outputCol=col + "_encoded"
            ) for col in categorical_cols
        ]
        
        # Stage 3: Assemble all features
        assembler = VectorAssembler(
            inputCols=numerical_cols + [col + "_encoded" for col in categorical_cols],
            outputCol="features_raw"
        )
        
        # Stage 4: Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=False
        )
        
        # Stage 5: Classifier
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="price_sensitive",
            maxIter=50,
            maxDepth=5,
            stepSize=0.1,
            seed=8
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, gbt])
        
        print("✓ Pipeline created with stages:")
        print(f"  - String Indexing: {len(indexers)} categorical features")
        print(f"  - One-Hot Encoding: {len(encoders)} features")
        print(f"  - Feature Assembly: {len(numerical_cols)} numerical + {len(categorical_cols)} categorical")
        print(f"  - Feature Scaling: StandardScaler")
        print(f"  - Classifier: Gradient Boosting Trees")
        print("")
        
        # Split data
        print("📊 Splitting data...")
        train_df, test_df = df.randomSplit([0.7, 0.3], seed=8)
        
        train_count = train_df.count()
        test_count = test_df.count()
        
        print(f"  Training set: {train_count:,} samples ({train_count/(train_count+test_count)*100:.1f}%)")
        print(f"  Test set: {test_count:,} samples ({test_count/(train_count+test_count)*100:.1f}%)")
        print("")
        
        # Train model
        print("🚀 Training model...")
        start_time = time.time()
        
        model = pipeline.fit(train_df)
        
        training_time = time.time() - start_time
        print(f"✓ Model trained in {training_time:.2f} seconds")
        print("")
        
        # Make predictions
        print("🔮 Making predictions on test set...")
        predictions = model.transform(test_df)
        
        # Evaluate
        print("📈 Evaluating model performance...")
        print("-"*80)
        
        # Binary classification metrics
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol="price_sensitive",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="price_sensitive",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )
        
        auc = evaluator_auc.evaluate(predictions)
        pr = evaluator_pr.evaluate(predictions)
        
        # Multiclass metrics for detailed analysis
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="price_sensitive",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="price_sensitive",
            predictionCol="prediction",
            metricName="f1"
        )
        
        evaluator_precision = MulticlassClassificationEvaluator(
            labelCol="price_sensitive",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        
        evaluator_recall = MulticlassClassificationEvaluator(
            labelCol="price_sensitive",
            predictionCol="prediction",
            metricName="weightedRecall"
        )
        
        accuracy = evaluator_acc.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)
        precision = evaluator_precision.evaluate(predictions)
        recall = evaluator_recall.evaluate(predictions)
        
        print("\n📊 MODEL PERFORMANCE METRICS:")
        print("="*80)
        print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision:          {precision:.4f}")
        print(f"  Recall:             {recall:.4f}")
        print(f"  F1-Score:           {f1:.4f}")
        print(f"  AUC-ROC:            {auc:.4f}")
        print(f"  AUC-PR:             {pr:.4f}")
        print("="*80)
        
        # Confusion matrix
        print("\n📋 Confusion Matrix:")
        predictions.groupBy('price_sensitive', 'prediction').count().orderBy('price_sensitive', 'prediction').show()
        
        # Sample predictions
        print("\n🔍 Sample Predictions:")
        predictions.select(
            'Age', 'Previous Purchases', 'Subscription Status',
            'price_sensitive', 'prediction', 'probability'
        ).show(10, truncate=False)
        
        # Save model
        model_path = "models/pricing_model"
        print(f"\n💾 Saving model to: {model_path}")
        model.write().overwrite().save(model_path)
        print("✓ Model saved successfully")
        print("")
        
        return model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'auc_pr': pr
        }
    
    def train_category_model(self, df):
        """
        Train Next Category Prediction Model (Multi-class Classification)
        
        Predicts: Which category will customer buy? (4 classes)
        
        Args:
            df: Prepared DataFrame
            
        Returns:
            Trained model and evaluation metrics
        """
        print("="*80)
        print("TRAINING MODEL 2: NEXT CATEGORY PREDICTION")
        print("="*80)
        print("\nModel Type: Multi-class Classification")
        print("Algorithm: Random Forest")
        print("Target: Category (Clothing, Footwear, Accessories, Outerwear)")
        print("-"*80 + "\n")
        
        # Define feature columns
        categorical_cols = [
            'Gender', 'Season', 'Location'
        ]
        
        numerical_cols = [
            'Age', 'Purchase Amount (USD)',
            'Review Rating', 'is_frequent_buyer'
        ]
        
        # Build preprocessing pipeline
        print("🔧 Building feature preprocessing pipeline...")
        
        # Stage 1: String indexing for target
        label_indexer = StringIndexer(
            inputCol="Category",
            outputCol="label",
            handleInvalid="keep"
        )
        
        # Stage 2: String indexing for categorical features
        indexers = [
            StringIndexer(
                inputCol=col,
                outputCol=col + "_indexed",
                handleInvalid="keep"
            ) for col in categorical_cols
        ]
        
        # Stage 3: One-hot encoding
        encoders = [
            OneHotEncoder(
                inputCol=col + "_indexed",
                outputCol=col + "_encoded"
            ) for col in categorical_cols
        ]
        
        # Stage 4: Assemble features
        assembler = VectorAssembler(
            inputCols=numerical_cols + [col + "_encoded" for col in categorical_cols],
            outputCol="features"
        )
        
        # Stage 5: Classifier
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=10,
            # seed=8
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[label_indexer] + indexers + encoders + [assembler, rf])
        
        print("✓ Pipeline created with stages:")
        print(f"  - Target Indexing: Category → label")
        print(f"  - String Indexing: {len(indexers)} categorical features")
        print(f"  - One-Hot Encoding: {len(encoders)} features")
        print(f"  - Feature Assembly: {len(numerical_cols)} numerical + {len(categorical_cols)} categorical")
        print(f"  - Classifier: Random Forest (100 trees, max depth 10)")
        print("")
        
        # Split data
        print("📊 Splitting data...")
        train_df, test_df = df.randomSplit([0.70, 0.30])
        
        train_count = train_df.count()
        test_count = test_df.count()
        
        print(f"  Training set: {train_count:,} samples ({train_count/(train_count+test_count)*100:.1f}%)")
        print(f"  Test set: {test_count:,} samples ({test_count/(train_count+test_count)*100:.1f}%)")
        print("")
        
        # Train model
        print("🚀 Training model...")
        start_time = time.time()
        
        model = pipeline.fit(train_df)
        
        training_time = time.time() - start_time
        print(f"✓ Model trained in {training_time:.2f} seconds")
        print("")
        
        # Make predictions
        print("🔮 Making predictions on test set...")
        predictions = model.transform(test_df)
        
        # Evaluate
        print("📈 Evaluating model performance...")
        print("-"*80)
        
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        
        evaluator_precision = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        
        evaluator_recall = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )
        
        accuracy = evaluator_acc.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)
        precision = evaluator_precision.evaluate(predictions)
        recall = evaluator_recall.evaluate(predictions)
        
        print("\n📊 MODEL PERFORMANCE METRICS:")
        print("="*80)
        print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision:          {precision:.4f}")
        print(f"  Recall:             {recall:.4f}")
        print(f"  F1-Score:           {f1:.4f}")
        print("="*80)
        
        # Confusion matrix
        print("\n📋 Confusion Matrix:")
        predictions.groupBy('Category', 'prediction').count().orderBy('Category', 'prediction').show(20)
        
        # Per-class accuracy
        print("\n📊 Per-Category Performance:")
        for category_idx in range(4):
            category_predictions = predictions.filter(col('label') == category_idx)
            if category_predictions.count() > 0:
                category_accuracy = category_predictions.filter(col('label') == col('prediction')).count() / category_predictions.count()
                category_name = category_predictions.select('Category').first()[0]
                print(f"  {category_name}: {category_accuracy*100:.2f}%")
        
        # Sample predictions
        print("\n🔍 Sample Predictions:")
        predictions.select(
            'Age', 'Gender', 'Season', 'Previous Purchases',
            'Category', 'prediction', 'probability'
        ).show(10, truncate=False)
        
        # Save model
        model_path = "models/category_model"
        print(f"\n💾 Saving model to: {model_path}")
        model.write().overwrite().save(model_path)
        print("✓ Model saved successfully")
        print("")
        
        return model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def generate_model_summary(self, pricing_metrics, category_metrics):
        """
        Generate comprehensive summary report
        
        Args:
            pricing_metrics: Metrics from pricing model
            category_metrics: Metrics from category model
        """
        print("\n" + "="*80)
        print("TRAINING COMPLETE - MODEL SUMMARY")
        print("="*80)
        
        print("\n📊 MODEL 1: DYNAMIC PRICING (Price Sensitivity)")
        print("-"*80)
        print(f"  Type: Binary Classification")
        print(f"  Algorithm: Gradient Boosting Trees")
        print(f"  Accuracy: {pricing_metrics['accuracy']*100:.2f}%")
        print(f"  F1-Score: {pricing_metrics['f1']:.4f}")
        print(f"  AUC-ROC: {pricing_metrics['auc']:.4f}")
        print(f"  Status: ✓ Trained and Saved")
        print(f"  Location: ./models/pricing_model/")
        
        print("\n📊 MODEL 2: NEXT CATEGORY PREDICTION")
        print("-"*80)
        print(f"  Type: Multi-class Classification (4 classes)")
        print(f"  Algorithm: Random Forest")
        print(f"  Accuracy: {category_metrics['accuracy']*100:.2f}%")
        print(f"  F1-Score: {category_metrics['f1']:.4f}")
        print(f"  Status: ✓ Trained and Saved")
        print(f"  Location: ./models/category_model/")
        
        print("\n💡 BUSINESS IMPACT:")
        print("-"*80)
        print("  ✓ Personalized pricing strategy")
        print("  ✓ Optimized discount allocation")
        print("  ✓ Targeted product recommendations")
        print("  ✓ Increased conversion rates")
        print("  ✓ Reduced customer acquisition costs")
        
        print("\n🚀 NEXT STEPS:")
        print("-"*80)
        print("  1. Run testing pipeline: python3 src/ml_testing.py")
        print("  2. Integrate with streaming: python3 src/ml_streaming_integration.py")
        print("  3. Deploy models for real-time predictions")
        
        print("\n" + "="*80)
        print("✓ ALL MODELS TRAINED SUCCESSFULLY")
        print("="*80 + "\n")


def main():
    """
    Main execution function
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║        Machine Learning Training Pipeline                      ║
    ║        ITCS 6190 Big Data Analytics Project                    ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize trainer
    trainer = MLModelTrainer()
    
    # Load data
    df = trainer.load_data("data/shopping.csv")
    
    input("\n Press Enter to train Model 1 (Dynamic Pricing)...")
    
    # Train Model 1: Dynamic Pricing
    df_pricing = trainer.prepare_pricing_model_data(df)
    pricing_model, pricing_metrics = trainer.train_pricing_model(df_pricing)
    
    input("\nPress Enter to train Model 2 (Next Category Prediction)...")
    
    # Train Model 2: Category Prediction
    df_category = trainer.prepare_category_model_data(df)
    category_model, category_metrics = trainer.train_category_model(df_category)
    
    # Generate summary
    trainer.generate_model_summary(pricing_metrics, category_metrics)
    
    # Stop Spark
    trainer.spark.stop()


if __name__ == "__main__":
    main()