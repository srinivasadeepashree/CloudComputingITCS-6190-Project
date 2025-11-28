"""
Integrated Machine Learning Pipeline
ITCS 6190 - Big Data Analytics Project

This pipeline trains three distinct models to analyze customer behavior:
1. Subscription Status Prediction (Classification)
2. Spending Tier Prediction (Classification)
3. Purchase Amount Prediction (Regression)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, abs
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, 
    MulticlassClassificationEvaluator, 
    RegressionEvaluator
)
from pyspark.ml import Pipeline
import os
import sys

class IntegratedMLPipeline:
    def __init__(self):
        """Initialize Spark Session and Logging"""
        self.spark = SparkSession.builder \
            .appName("ShoppingTrendsIntegratedML") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("ERROR")
        os.makedirs("models", exist_ok=True)
        print("✓ Spark Session Initialized")

    def load_data(self, filepath="/workspaces/CloudComputingITCS-6190-Project/data/shopping.csv"):
        """Load and inspect data"""
        print(f"\n📂 Loading data from: {filepath}")
        if not os.path.exists(filepath):
            print(f"❌ Error: File '{filepath}' not found.")
            sys.exit(1)
            
        df = self.spark.read.csv(filepath, header=True, inferSchema=True)
        print(f"✓ Data Loaded: {df.count()} rows")
        return df

    # ==========================================
    # MODEL 1: SUBSCRIPTION PREDICTION
    # ==========================================
    def train_subscription_model(self, df):
        print("\n" + "="*60)
        print("MODEL 1: SUBSCRIPTION STATUS PREDICTION (Classification)")
        print("="*60)
        
        # 1. Prepare Data
        # Target: Subscription Status (Yes=1, No=0)
        # Features: Loyalty metrics (Previous Purchases, Frequency) + Demographics
        df_prep = df.withColumn(
            'label', 
            when(col('`Subscription Status`') == 'Yes', 1.0).otherwise(0.0)
        ).select(
            'label', 'Age', 'Gender', 'Previous Purchases', 
            'Frequency of Purchases', 'Payment Method', 'Review Rating'
        )
        
        print(f"   Target Distribution: {df_prep.groupBy('label').count().collect()}")

        # 2. Build Pipeline
        cat_cols = ['Gender', 'Frequency of Purchases', 'Payment Method']
        num_cols = ['Age', 'Previous Purchases', 'Review Rating']
        
        stages = []
        for c in cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            stages += [indexer, encoder]
            
        assembler = VectorAssembler(
            inputCols=[f"{c}_vec" for c in cat_cols] + num_cols, 
            outputCol="features"
        )
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50)
        pipeline = Pipeline(stages=stages + [assembler, rf])

        # 3. Train & Evaluate
        train_df, test_df = df_prep.randomSplit([0.8, 0.2], seed=42)
        print("🚀 Training Subscription Model...")
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        
        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        
        print(f"📊 Results - AUC-ROC: {auc:.4f}")
        
        # Save Model
        model_path = "models/subscription_model"
        model.write().overwrite().save(model_path)
        print(f"💾 Model saved to: {model_path}")
        return model

    # ==========================================
    # MODEL 2: SPENDING TIER PREDICTION
    # ==========================================
    def train_spending_tier_model(self, df):
        print("\n" + "="*60)
        print("MODEL 2: SPENDING TIER PREDICTION (Multi-class Classification)")
        print("="*60)
        
        # 1. Prepare Data
        # Create Tiers: Low (<$45), Medium ($45-$75), High (>$75)
        df_tiered = df.withColumn(
            'Spending_Tier',
            when(col('`Purchase Amount (USD)`') < 45, 'Low')
            .when((col('`Purchase Amount (USD)`') >= 45) & (col('`Purchase Amount (USD)`') <= 75), 'Medium')
            .otherwise('High')
        )
        
        # Select relevant features
        df_prep = df_tiered.select(
            'Spending_Tier', 'Age', 'Gender', 'Category', 
            'Season', 'Location', 'Subscription Status'
        )
        
        print("   Tier Distribution:")
        df_prep.groupBy('Spending_Tier').count().show()

        # 2. Build Pipeline
        label_indexer = StringIndexer(inputCol="Spending_Tier", outputCol="label")
        cat_cols = ['Gender', 'Category', 'Season', 'Location', 'Subscription Status']
        num_cols = ['Age']
        
        stages = [label_indexer]
        for c in cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            stages += [indexer, encoder]
            
        assembler = VectorAssembler(
            inputCols=[f"{c}_vec" for c in cat_cols] + num_cols, 
            outputCol="features"
        )
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50, maxDepth=10)
        pipeline = Pipeline(stages=stages + [assembler, rf])

        # 3. Train & Evaluate
        train_df, test_df = df_prep.randomSplit([0.8, 0.2], seed=42)
        print("🚀 Training Spending Tier Model...")
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        
        evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        
        print(f"📊 Results - Accuracy: {accuracy*100:.2f}%")
        
        model_path = "models/spending_tier_model"
        model.write().overwrite().save(model_path)
        print(f"💾 Model saved to: {model_path}")
        return model

    # ==========================================
    # MODEL 3: SPENDING AMOUNT PREDICTION
    # ==========================================
    def train_spending_amount_model(self, df):
        print("\n" + "="*60)
        print("MODEL 3: PURCHASE AMOUNT PREDICTION (Regression)")
        print("="*60)
        
        # 1. Prepare Data
        # Target: Purchase Amount (USD)
        df_prep = df.select(
            'Purchase Amount (USD)', 'Age', 'Gender', 'Category', 
            'Season', 'Review Rating', 'Previous Purchases', 'Subscription Status'
        )

        # 2. Build Pipeline
        cat_cols = ['Gender', 'Category', 'Season', 'Subscription Status']
        num_cols = ['Age', 'Review Rating', 'Previous Purchases']
        
        stages = []
        for c in cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            stages += [indexer, encoder]
            
        assembler = VectorAssembler(
            inputCols=[f"{c}_vec" for c in cat_cols] + num_cols, 
            outputCol="features"
        )
        
        # Random Forest Regressor
        rf = RandomForestRegressor(
            featuresCol="features", 
            labelCol="Purchase Amount (USD)",
            numTrees=50,
            maxDepth=10,
        )
        pipeline = Pipeline(stages=stages + [assembler, rf])

        # 3. Train & Evaluate
        train_df, test_df = df_prep.randomSplit([0.8, 0.2])
        print("🚀 Training Spending Amount Model...")
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        
        evaluator_rmse = RegressionEvaluator(labelCol="Purchase Amount (USD)", metricName="rmse")
        rmse = evaluator_rmse.evaluate(predictions)
        
        print(f"📊 Results - RMSE: ${rmse:.2f}")
        
        # Show specific predictions vs actuals
        print("   Sample Predictions:")
        predictions.withColumn("Diff", abs(col("Purchase Amount (USD)") - col("prediction"))) \
            .select("Category", "Purchase Amount (USD)", "prediction", "Diff") \
            .show(5)

        model_path = "models/spending_amount_model"
        model.write().overwrite().save(model_path)
        print(f"💾 Model saved to: {model_path}")
        return model

    def run(self):
        """Execute entire pipeline"""
        df = self.load_data()
        
        # Execute all three models sequentially
        self.train_subscription_model(df)
        self.train_spending_tier_model(df)
        self.train_spending_amount_model(df)
        
        print("\n" + "="*60)
        print("✓ ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
        print("="*60)
        self.spark.stop()

if __name__ == "__main__":
    pipeline = IntegratedMLPipeline()
    pipeline.run()


# """
# Optimized ML Pipeline: Subscription & Spending Tier Prediction
# ITCS 6190 - Big Data Analytics Project

# Models:
# 1. Subscription Status Predictor (Identify potential subscribers)
# 2. Customer Spending Tier Classifier (Segment customers by value)
# """

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when
# from pyspark.ml.feature import (
#     VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
# )
# from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
# from pyspark.ml import Pipeline
# import os
# import time

# class MLModelTrainer:
#     def __init__(self):
#         self.spark = SparkSession.builder \
#             .appName("ShoppingTrendsOptimized") \
#             .master("local[*]") \
#             .config("spark.driver.memory", "4g") \
#             .getOrCreate()
#         self.spark.sparkContext.setLogLevel("ERROR")
#         os.makedirs("models", exist_ok=True)
#         print("✓ Spark Session Initialized")

#     def load_data(self, filepath="/workspaces/CloudComputingITCS-6190-Project/data/shopping.csv"):
#         print(f"\n📂 Loading data from: {filepath}")
#         return self.spark.read.csv(filepath, header=True, inferSchema=True)

#     # ==========================================
#     # USE CASE 1: SUBSCRIPTION PREDICTION
#     # ==========================================
#     def prepare_subscription_data(self, df):
#         print("\n" + "="*50)
#         print("PREPARING DATA: SUBSCRIPTION PREDICTION")
#         print("="*50)
        
#         # Select ONLY required fields to reduce noise
#         # We hypothesize that loyalty (frequency, previous purchases) drives subscription
#         required_cols = [
#             'Subscription Status',    # Target
#             'Age',                    # Feature
#             'Gender',                 # Feature
#             'Previous Purchases',     # Feature
#             'Frequency of Purchases', # Feature
#             'Payment Method',         # Feature
#             'Review Rating'           # Feature
#         ]
        
#         df_selected = df.select(required_cols)
        
#         # Transform Target: Yes/No -> 1/0
#         df_prepared = df_selected.withColumn(
#             'label', 
#             when(col('`Subscription Status`') == 'Yes', 1.0).otherwise(0.0)
#         )
        
#         print(f"✓ Selected {len(required_cols)} relevant columns")
        
#         # Check Class Balance
#         print("Target Distribution:")
#         df_prepared.groupBy('label').count().show()
        
#         return df_prepared

#     def train_subscription_model(self, df):
#         print("🚀 Training Subscription Model (Random Forest)...")
        
#         # 1. Categorical Features to Index & Encode
#         cat_cols = ['Gender', 'Frequency of Purchases', 'Payment Method']
#         # 2. Numerical Features
#         num_cols = ['Age', 'Previous Purchases', 'Review Rating']
        
#         stages = []
        
#         # String Indexing & OHE for Categorical
#         for c in cat_cols:
#             indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
#             encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
#             stages += [indexer, encoder]
            
#         # Assemble Features
#         assembler_inputs = [f"{c}_vec" for c in cat_cols] + num_cols
#         assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
#         stages.append(assembler)
        
#         # Classifier - Random Forest handles imbalanced data reasonably well
#         rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50)
#         stages.append(rf)
        
#         # Pipeline
#         pipeline = Pipeline(stages=stages)
        
#         # Split
#         train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        
#         # Train
#         model = pipeline.fit(train_df)
#         predictions = model.transform(test_df)
        
#         # Evaluate
#         evaluator = BinaryClassificationEvaluator(labelCol="label")
#         auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
        
#         acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
#         accuracy = acc_evaluator.evaluate(predictions)
        
#         print(f"\n📊 Subscription Model Results:")
#         print(f"   Accuracy: {accuracy*100:.2f}%")
#         print(f"   AUC-ROC:  {auc:.4f}")
        
#         return model

#     # ==========================================
#     # USE CASE 2: SPENDING TIER PREDICTION
#     # ==========================================
#     def prepare_spending_data(self, df):
#         print("\n" + "="*50)
#         print("PREPARING DATA: SPENDING TIER PREDICTION")
#         print("="*50)
        
#         # Create Tiers from Purchase Amount
#         # Low: < $45, Medium: $45-$75, High: > $75
#         df_tiered = df.withColumn(
#             'Spending_Tier',
#             when(col('`Purchase Amount (USD)`') < 45, 'Low')
#             .when((col('`Purchase Amount (USD)`') >= 45) & (col('`Purchase Amount (USD)`') <= 75), 'Medium')
#             .otherwise('High')
#         )
        
#         # Select ONLY required fields
#         # Hypothesis: Category, Season, and Demographics drive spending power
#         required_cols = [
#             'Spending_Tier',        # Target
#             'Age',                  # Feature
#             'Gender',               # Feature
#             'Category',             # Feature (e.g. Footwear might cost more than Accessories)
#             'Season',               # Feature
#             'Location',             # Feature
#             'Subscription Status'   # Feature
#         ]
        
#         df_selected = df_tiered.select(required_cols)
#         print(f"✓ Created Spending Tiers (Low/Medium/High) and selected {len(required_cols)} columns")
        
#         df_selected.groupBy('Spending_Tier').count().show()
        
#         return df_selected

#     def train_spending_model(self, df):
#         print("🚀 Training Spending Tier Model (Decision Tree)...")
        
#         # Label Indexing for Target (High/Medium/Low -> 0/1/2)
#         label_indexer = StringIndexer(inputCol="Spending_Tier", outputCol="label")
        
#         # Categorical Features
#         cat_cols = ['Gender', 'Category', 'Season', 'Location', 'Subscription Status']
#         # Numerical Features
#         num_cols = ['Age']
        
#         stages = [label_indexer]
        
#         for c in cat_cols:
#             indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
#             encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
#             stages += [indexer, encoder]
            
#         assembler_inputs = [f"{c}_vec" for c in cat_cols] + num_cols
#         assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
#         stages.append(assembler)
        
#         # Decision Tree is good for clear "Cut-off" rules like spending tiers
#         # Using Random Forest for better generalization
#         rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50, maxDepth=10)
#         stages.append(rf)
        
#         pipeline = Pipeline(stages=stages)
        
#         train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        
#         model = pipeline.fit(train_df)
#         predictions = model.transform(test_df)
        
#         evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
#         accuracy = evaluator.evaluate(predictions)
#         f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        
#         print(f"\n📊 Spending Tier Model Results:")
#         print(f"   Accuracy: {accuracy*100:.2f}%")
#         print(f"   F1 Score: {f1:.4f}")
        
#         return model

#     def run(self):
#         df = self.load_data()
        
#         # Run Model 1
#         df_sub = self.prepare_subscription_data(df)
#         self.train_subscription_model(df_sub)
        
#         # Run Model 2
#         df_spend = self.prepare_spending_data(df)
#         self.train_spending_model(df_spend)
        
#         self.spark.stop()

# if __name__ == "__main__":
#     trainer = MLModelTrainer()
#     trainer.run()
