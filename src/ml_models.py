"""
Machine Learning Models for Shopping Trends Analysis
ITCS 6190 - Milestone 3

Implements 4 ML models:
1. Classification - Predict purchase category
2. Regression - Predict purchase amount
3. Clustering - Customer segmentation
4. Recommendation - Product recommendations
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder, 
    StandardScaler, MinMaxScaler
)
from pyspark.ml.classification import (
    RandomForestClassifier, LogisticRegression,
    DecisionTreeClassifier, GBTClassifier
)
from pyspark.ml.regression import (
    LinearRegression, RandomForestRegressor,
    GBTRegressor
)
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
    ClusteringEvaluator
)
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, when, count
import os
import pandas as pd


class ShoppingMLPipeline:
    """
    Complete ML pipeline for shopping trends prediction
    """
    
    def __init__(self, spark, df):
        self.spark = spark
        self.df = df
        self.models = {}
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        print("="*80)
        print("✓ Shopping ML Pipeline Initialized")
        print(f"  Dataset: {df.count():,} records")
        print(f"  Features: {len(df.columns)} columns")
        print("="*80 + "\n")
    
    def prepare_classification_data(self):
        """
        Prepare data for category classification
        Target: Category
        Features: Age, Gender, Previous Purchases, Season, etc.
        """
        print("\n📊 Preparing Classification Data...")
        
        # Select relevant features
        feature_cols = ['Age', 'Gender', 'Previous Purchases', 
                       'Subscription Status', 'Season', 'Review Rating',
                       'Discount Applied', 'Promo Code Used']
        
        ml_df = self.df.select(feature_cols + ['Category'])
        
        # Drop nulls
        ml_df = ml_df.dropna()
        
        # Index categorical features
        gender_indexer = StringIndexer(inputCol="Gender", outputCol="gender_idx")
        subscription_indexer = StringIndexer(inputCol="Subscription Status", outputCol="subscription_idx")
        season_indexer = StringIndexer(inputCol="Season", outputCol="season_idx")
        discount_indexer = StringIndexer(inputCol="Discount Applied", outputCol="discount_idx")
        promo_indexer = StringIndexer(inputCol="Promo Code Used", outputCol="promo_idx")
        
        # Index target
        label_indexer = StringIndexer(inputCol="Category", outputCol="label")
        
        # Apply indexers
        for indexer in [gender_indexer, subscription_indexer, season_indexer, 
                       discount_indexer, promo_indexer, label_indexer]:
            ml_df = indexer.fit(ml_df).transform(ml_df)
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=['Age', 'gender_idx', 'Previous Purchases', 
                      'subscription_idx', 'season_idx', 'Review Rating',
                      'discount_idx', 'promo_idx'],
            outputCol="features"
        )
        
        ml_df = assembler.transform(ml_df)
        
        # Split data
        train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
        
        print(f"✓ Classification data prepared")
        print(f"  Training: {train_df.count():,} records")
        print(f"  Testing: {test_df.count():,} records")
        print(f"  Features: {len(assembler.getInputCols())}")
        
        return train_df, test_df, label_indexer
    
    def train_classification_model(self):
        """
        Model 1: Random Forest Classification
        Predicts which category customer will purchase from
        """
        print("\n" + "="*80)
        print("MODEL 1: CATEGORY CLASSIFICATION (Random Forest)")
        print("="*80)
        
        # Prepare data
        train_df, test_df, label_indexer = self.prepare_classification_data()
        
        # Train Random Forest
        print("\n🔄 Training Random Forest Classifier...")
        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=100,
            maxDepth=10,
            seed=42
        )
        
        model = rf.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction"
        )
        
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        
        print(f"\n✓ Model Training Complete")
        print(f"\n📊 Classification Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Feature importance
        print(f"\n🔍 Top 5 Important Features:")
        feature_importance = list(zip(
            ['Age', 'Gender', 'Previous Purchases', 'Subscription', 
             'Season', 'Review Rating', 'Discount', 'Promo'],
            model.featureImportances.toArray()
        ))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for feat, importance in feature_importance[:5]:
            print(f"  {feat}: {importance:.4f}")
        
        # Save model
        model.write().overwrite().save("models/classification/random_forest")
        print(f"\n💾 Model saved: models/classification/random_forest")
        
        # Save metrics
        with open("results/classification_metrics.txt", "w") as f:
            f.write(f"Random Forest Classification Results\n")
            f.write(f"="*50 + "\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n")
        
        self.models['classification'] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': predictions
        }
        
        return model, predictions
    
    def prepare_regression_data(self):
        """
        Prepare data for purchase amount regression
        Target: Purchase Amount (USD)
        """
        print("\n📊 Preparing Regression Data...")
        
        feature_cols = ['Age', 'Gender', 'Category', 'Season', 
                       'Previous Purchases', 'Subscription Status',
                       'Review Rating', 'Discount Applied']
        
        ml_df = self.df.select(feature_cols + ['Purchase Amount (USD)'])
        ml_df = ml_df.dropna()
        
        # Rename target
        ml_df = ml_df.withColumnRenamed('Purchase Amount (USD)', 'label')
        
        # Index categorical features
        gender_indexer = StringIndexer(inputCol="Gender", outputCol="gender_idx")
        category_indexer = StringIndexer(inputCol="Category", outputCol="category_idx")
        season_indexer = StringIndexer(inputCol="Season", outputCol="season_idx")
        subscription_indexer = StringIndexer(inputCol="Subscription Status", outputCol="subscription_idx")
        discount_indexer = StringIndexer(inputCol="Discount Applied", outputCol="discount_idx")
        
        for indexer in [gender_indexer, category_indexer, season_indexer, 
                       subscription_indexer, discount_indexer]:
            ml_df = indexer.fit(ml_df).transform(ml_df)
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=['Age', 'gender_idx', 'category_idx', 'season_idx',
                      'Previous Purchases', 'subscription_idx', 'Review Rating',
                      'discount_idx'],
            outputCol="features"
        )
        
        ml_df = assembler.transform(ml_df)
        
        # Split data
        train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
        
        print(f"✓ Regression data prepared")
        print(f"  Training: {train_df.count():,} records")
        print(f"  Testing: {test_df.count():,} records")
        
        return train_df, test_df
    
    def train_regression_model(self):
        """
        Model 2: Linear Regression
        Predicts purchase amount
        """
        print("\n" + "="*80)
        print("MODEL 2: PURCHASE AMOUNT REGRESSION (Linear Regression)")
        print("="*80)
        
        # Prepare data
        train_df, test_df = self.prepare_regression_data()
        
        # Train Linear Regression
        print("\n🔄 Training Linear Regression Model...")
        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=0.1,
            elasticNetParam=0.5
        )
        
        model = lr.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
        
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
        
        print(f"\n✓ Model Training Complete")
        print(f"\n📊 Regression Metrics:")
        print(f"  RMSE (Root Mean Square Error): ${rmse:.2f}")
        print(f"  MAE (Mean Absolute Error):     ${mae:.2f}")
        print(f"  R² Score:                       {r2:.4f}")
        
        # Model coefficients
        print(f"\n🔍 Model Coefficients (top 5):")
        feature_names = ['Age', 'Gender', 'Category', 'Season',
                        'Previous Purchases', 'Subscription', 'Review Rating', 'Discount']
        coefficients = list(zip(feature_names, model.coefficients.toArray()))
        coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in coefficients[:5]:
            print(f"  {feat}: {coef:.4f}")
        
        # Save model
        model.write().overwrite().save("models/regression/linear_regression")
        print(f"\n💾 Model saved: models/regression/linear_regression")
        
        # Save metrics
        with open("results/regression_metrics.txt", "w") as f:
            f.write(f"Linear Regression Results\n")
            f.write(f"="*50 + "\n")
            f.write(f"RMSE: ${rmse:.2f}\n")
            f.write(f"MAE:  ${mae:.2f}\n")
            f.write(f"R²:   {r2:.4f}\n")
        
        self.models['regression'] = {
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions
        }
        
        return model, predictions
    
    def prepare_clustering_data(self):
        """
        Prepare data for customer clustering
        Features: Purchase behavior and demographics
        """
        print("\n📊 Preparing Clustering Data...")
        
        feature_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases',
                       'Review Rating']
        
        ml_df = self.df.select(feature_cols)
        ml_df = ml_df.dropna()
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="unscaled_features"
        )
        
        ml_df = assembler.transform(ml_df)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="unscaled_features",
            outputCol="features"
        )
        
        scaler_model = scaler.fit(ml_df)
        ml_df = scaler_model.transform(ml_df)
        
        print(f"✓ Clustering data prepared")
        print(f"  Records: {ml_df.count():,}")
        print(f"  Features: {len(feature_cols)} (scaled)")
        
        return ml_df
    
    def train_clustering_model(self, k=5):
        """
        Model 3: K-Means Clustering
        Segments customers into groups
        """
        print("\n" + "="*80)
        print(f"MODEL 3: CUSTOMER SEGMENTATION (K-Means, k={k})")
        print("="*80)
        
        # Prepare data
        ml_df = self.prepare_clustering_data()
        
        # Train K-Means
        print(f"\n🔄 Training K-Means with {k} clusters...")
        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=k,
            seed=42,
            maxIter=100
        )
        
        model = kmeans.fit(ml_df)
        
        # Make predictions
        predictions = model.transform(ml_df)
        
        # Evaluate
        evaluator = ClusteringEvaluator(
            featuresCol="features",
            predictionCol="cluster",
            metricName="silhouette"
        )
        
        silhouette = evaluator.evaluate(predictions)
        
        print(f"\n✓ Model Training Complete")
        print(f"\n📊 Clustering Metrics:")
        print(f"  Number of Clusters: {k}")
        print(f"  Silhouette Score:   {silhouette:.4f}")
        print(f"  Within Set Sum of Squared Errors: {model.summary.trainingCost:.2f}")
        
        # Cluster sizes
        print(f"\n🔍 Cluster Distribution:")
        cluster_counts = predictions.groupBy("cluster").count().orderBy("cluster")
        cluster_counts.show()
        
        # Cluster profiles
        print(f"\n📋 Cluster Profiles:")
        cluster_profiles = predictions.groupBy("cluster").agg({
            'Age': 'avg',
            'Purchase Amount (USD)': 'avg',
            'Previous Purchases': 'avg',
            'Review Rating': 'avg'
        }).orderBy("cluster")
        cluster_profiles.show()
        
        # Save model
        model.write().overwrite().save("models/clustering/kmeans")
        print(f"\n💾 Model saved: models/clustering/kmeans")
        
        # Save metrics
        with open("results/clustering_metrics.txt", "w") as f:
            f.write(f"K-Means Clustering Results\n")
            f.write(f"="*50 + "\n")
            f.write(f"Number of Clusters: {k}\n")
            f.write(f"Silhouette Score: {silhouette:.4f}\n")
        
        self.models['clustering'] = {
            'model': model,
            'silhouette': silhouette,
            'predictions': predictions
        }
        
        return model, predictions
    
    def prepare_recommendation_data(self):
        """
        Prepare data for product recommendations
        Uses Customer ID and Item Purchased
        """
        print("\n📊 Preparing Recommendation Data...")
        
        # Create customer-item matrix
        ml_df = self.df.select('Customer ID', 'Item Purchased')
        ml_df = ml_df.dropna()
        
        # Add implicit ratings (count of purchases)
        ml_df = ml_df.groupBy('Customer ID', 'Item Purchased').count()
        ml_df = ml_df.withColumnRenamed('count', 'rating')
        
        # Index items
        item_indexer = StringIndexer(inputCol="Item Purchased", outputCol="item_id")
        ml_df = item_indexer.fit(ml_df).transform(ml_df)
        
        # Rename for ALS
        ml_df = ml_df.withColumnRenamed('Customer ID', 'user_id')
        
        # Split data
        train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
        
        print(f"✓ Recommendation data prepared")
        print(f"  Training: {train_df.count():,} interactions")
        print(f"  Testing: {test_df.count():,} interactions")
        print(f"  Unique users: {ml_df.select('user_id').distinct().count():,}")
        print(f"  Unique items: {ml_df.select('item_id').distinct().count():,}")
        
        return train_df, test_df, item_indexer
    
    def train_recommendation_model(self):
        """
        Model 4: ALS Recommendation
        Recommends products to customers
        """
        print("\n" + "="*80)
        print("MODEL 4: PRODUCT RECOMMENDATIONS (ALS)")
        print("="*80)
        
        # Prepare data
        train_df, test_df, item_indexer = self.prepare_recommendation_data()
        
        # Train ALS
        print("\n🔄 Training ALS Model...")
        als = ALS(
            maxIter=10,
            regParam=0.1,
            userCol="user_id",
            itemCol="item_id",
            ratingCol="rating",
            coldStartStrategy="drop",
            implicitPrefs=True,
            seed=42
        )
        
        model = als.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Filter out NaN predictions before evaluation
        predictions_valid = predictions.filter(col("prediction").isNotNull())
        
        # Check if we have valid predictions
        valid_count = predictions_valid.count()
        total_count = predictions.count()
        
        print(f"\n📊 Prediction Coverage:")
        print(f"   Total test samples: {total_count}")
        print(f"   Valid predictions: {valid_count}")
        print(f"   Coverage: {(valid_count/total_count)*100:.2f}%")
        
        if valid_count == 0:
            print("\n⚠️  No valid predictions (cold start issue)")
            print("   Skipping RMSE evaluation")
            rmse = float('nan')
        else:
            # Evaluate only on valid predictions
            evaluator = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol="prediction"
            )
            
            rmse = evaluator.evaluate(predictions_valid)
            print(f"\n✓ Model Training Complete")
            print(f"\n📊 Recommendation Metrics:")
            print(f"  RMSE: {rmse:.4f}")
        
        # Generate top 5 recommendations for all users (this always works)
        print(f"\n🎁 Generating Top 5 Recommendations per User...")
        user_recs = model.recommendForAllUsers(5)
        
        print(f"\n📋 Sample Recommendations (First 5 customers):")
        user_recs.show(5, truncate=False)
        
        # Show recommendation statistics
        total_users_with_recs = user_recs.count()
        print(f"\n✓ Generated recommendations for {total_users_with_recs:,} users")
        
        # Save model
        model.write().overwrite().save("models/recommendation/als")
        print(f"\n💾 Model saved: models/recommendation/als")
        
        # Save metrics
        with open("results/recommendation_metrics.txt", "w") as f:
            f.write(f"ALS Recommendation Results\n")
            f.write(f"="*50 + "\n")
            f.write(f"RMSE: {rmse if not pd.isna(rmse) else 'N/A (cold start)'}\n")
            f.write(f"Coverage: {(valid_count/total_count)*100:.2f}%\n")
            f.write(f"Users with recommendations: {total_users_with_recs}\n")
        
        self.models['recommendation'] = {
            'model': model,
            'rmse': rmse,
            'predictions': predictions_valid if valid_count > 0 else predictions,
            'coverage': (valid_count/total_count)*100
        }
        
        return model, user_recs
    
    def train_all_models(self):
        """
        Train all 4 ML models sequentially
        """
        print("\n" + "="*80)
        print("TRAINING ALL ML MODELS")
        print("="*80)
        print("\nThis will train 4 models:")
        print("  1. Classification (Random Forest)")
        print("  2. Regression (Linear Regression)")
        print("  3. Clustering (K-Means)")
        print("  4. Recommendation (ALS)")
        print("\nEstimated time: 3-5 minutes\n")
        
        input("Press Enter to start training...")
        
        # Train each model
        self.train_classification_model()
        input("\n✓ Classification complete. Press Enter for Regression...")
        
        self.train_regression_model()
        input("\n✓ Regression complete. Press Enter for Clustering...")
        
        self.train_clustering_model(k=5)
        input("\n✓ Clustering complete. Press Enter for Recommendation...")
        
        self.train_recommendation_model()
        
        print("\n" + "="*80)
        print("✓ ALL MODELS TRAINED SUCCESSFULLY")
        print("="*80)
        
        # Summary
        print("\n📊 Model Performance Summary:")
        print("-"*80)
        if 'classification' in self.models:
            print(f"  Classification - Accuracy: {self.models['classification']['accuracy']:.2%}")
        if 'regression' in self.models:
            print(f"  Regression     - RMSE: ${self.models['regression']['rmse']:.2f}, R²: {self.models['regression']['r2']:.4f}")
        if 'clustering' in self.models:
            print(f"  Clustering     - Silhouette: {self.models['clustering']['silhouette']:.4f}")
        if 'recommendation' in self.models:
            print(f"  Recommendation - RMSE: {self.models['recommendation']['rmse']:.4f}")
        print("-"*80)
        
        print("\n💾 All models saved in: models/")
        print("📄 All metrics saved in: results/")
        
        return self.models


def main():
    """
    Execute ML pipeline
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║       Shopping Trends - Machine Learning Pipeline             ║
    ║       ITCS 6190 - Milestone 3                                 ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("ShoppingMLPipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Load data
    print("\n📂 Loading dataset...")
    df = spark.read.csv("data/shopping_trends.csv", header=True, inferSchema=True)
    print(f"✓ Loaded {df.count():,} records\n")
    
    # Create ML pipeline
    ml_pipeline = ShoppingMLPipeline(spark, df)
    
    # Train all models
    models = ml_pipeline.train_all_models()
    
    print("\n🎉 Machine Learning Pipeline Complete!")
    print("\nNext steps:")
    print("  1. Review results in results/ directory")
    print("  2. Load saved models for predictions")
    print("  3. Integrate with streaming pipeline")
    print("  4. Create visualization dashboard\n")
    
    spark.stop()


if __name__ == "__main__":
    main()