"""
Advanced Modeling Pipeline
ITCS 6190 - Big Data Analytics Project

1. Deep Learning (Multilayer Perceptron) for Subscription Prediction
2. Support Vector Machine (SVM) for Binary Classification
3. Seasonal Customer Segmentation (K-Means Clustering)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as _sum
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier, LinearSVC
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator, ClusteringEvaluator
from pyspark.ml import Pipeline
import os

class AdvancedModelTrainer:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("ShoppingAdvancedModels") \
            .master("local[*]") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        os.makedirs("models", exist_ok=True)
        print("✓ Spark Session Initialized")

    def load_data(self, filepath="/workspaces/CloudComputingITCS-6190-Project/data/shopping.csv"):
        return self.spark.read.csv(filepath, header=True, inferSchema=True)

    # ==========================================
    # MODEL A: DEEP LEARNING (MLP Classifier)
    # ==========================================
    def train_neural_network(self, df):
        print("\n" + "="*60)
        print("MODEL A: DEEP LEARNING (Multilayer Perceptron)")
        print("="*60)
        print("Goal: Predict Subscription Status using a Neural Network")
        
        # Prepare Data
        df_prep = df.withColumn('label', when(col('`Subscription Status`') == 'Yes', 1.0).otherwise(0.0))
        
        # Feature Engineering
        cat_cols = ['Gender', 'Category', 'Season', 'Payment Method']
        num_cols = ['Age', 'Previous Purchases', 'Review Rating']
        
        stages = []
        for c in cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            stages += [indexer, encoder]
            
        assembler = VectorAssembler(inputCols=[f"{c}_vec" for c in cat_cols] + num_cols, outputCol="features")
        stages.append(assembler)
        
        # Scaling is CRITICAL for Neural Networks
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        stages.append(scaler)
        
        # We need to run the pipeline partially to get input layer size
        partial_pipeline = Pipeline(stages=stages)
        train_df, test_df = df_prep.randomSplit([0.8, 0.2], seed=42)
        model_prep = partial_pipeline.fit(train_df)
        
        # Get input feature count
        input_size = model_prep.transform(train_df).select("scaled_features").head()[0].size
        print(f"✓ Input Layer Size: {input_size} neurons")
        
        # Define Neural Network Layers: [Input, Hidden1, Hidden2, Output]
        # Output is 2 because it's binary classification (0, 1)
        layers = [input_size, 20, 10, 2]
        
        # Classifier
        mlp = MultilayerPerceptronClassifier(
            layers=layers, 
            blockSize=128, 
            seed=42, 
            maxIter=100,
            featuresCol="scaled_features",
            labelCol="label"
        )
        
        # Train
        print("🚀 Training Neural Network...")
        mlp_model = mlp.fit(model_prep.transform(train_df))
        
        # Evaluate
        predictions = mlp_model.transform(model_prep.transform(test_df))
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        acc = evaluator.evaluate(predictions)
        print(f"📊 MLP Accuracy: {acc*100:.2f}%")
        
        return mlp_model

    # ==========================================
    # MODEL B: SUPPORT VECTOR MACHINE (SVM)
    # ==========================================
    def train_svm(self, df):
        print("\n" + "="*60)
        print("MODEL B: SUPPORT VECTOR MACHINE (Linear SVC)")
        print("="*60)
        print("Goal: Binary Classification with Hyperplane separation")
        
        df_prep = df.withColumn('label', when(col('`Subscription Status`') == 'Yes', 1.0).otherwise(0.0))
        
        # Use simpler features for SVM
        cat_cols = ['Gender', 'Frequency of Purchases']
        num_cols = ['Age', 'Previous Purchases']
        
        stages = []
        for c in cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            stages += [indexer, encoder]
            
        assembler = VectorAssembler(inputCols=[f"{c}_vec" for c in cat_cols] + num_cols, outputCol="features")
        stages.append(assembler)
        
        # Scaling is also important for SVM
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        stages.append(scaler)
        
        # SVM Classifier
        svm = LinearSVC(maxIter=10, regParam=0.1, featuresCol="scaled_features", labelCol="label")
        stages.append(svm)
        
        pipeline = Pipeline(stages=stages)
        train_df, test_df = df_prep.randomSplit([0.8, 0.2], seed=42)
        
        print("🚀 Training SVM...")
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        
        evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        print(f"📊 SVM AUC-ROC: {auc:.4f}")
        
        return model

    # ==========================================
    # MODEL C: SEASONAL CLUSTERING (K-Means)
    # ==========================================
    def perform_seasonal_segmentation(self, df):
        print("\n" + "="*60)
        print("MODEL C: SEASONAL CUSTOMER SEGMENTATION")
        print("="*60)
        print("Goal: Group users based on Seasonal Spending Habits")
        
        # 1. Pivot Data: Create a profile for each customer
        # Columns: Customer ID, Fall_Spend, Spring_Spend, Summer_Spend, Winter_Spend
        print("Processing customer seasonal profiles...")
        
        seasonal_profile = df.groupBy("Customer ID") \
            .pivot("Season") \
            .sum("Purchase Amount (USD)") \
            .na.fill(0)
            
        print("Sample Profiles:")
        seasonal_profile.show(5)
        
        # 2. Vectorize
        feature_cols = [c for c in seasonal_profile.columns if c != "Customer ID"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        # 3. K-Means Clustering
        # Let's find 4 clusters (e.g., maybe matching the 4 seasons, or Big Spenders vs Low Spenders)
        kmeans = KMeans(k=4, seed=1)
        
        pipeline = Pipeline(stages=[assembler, kmeans])
        
        print("🚀 Running K-Means Clustering...")
        model = pipeline.fit(seasonal_profile)
        predictions = model.transform(seasonal_profile)
        
        # Evaluate
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        print(f"📊 Cluster Silhouette Score: {silhouette:.4f} (Closeness of clusters)")
        
        # Analyze Clusters
        print("\n💡 Cluster Centers (Average Spend per Season):")
        centers = model.stages[-1].clusterCenters()
        print(f"{'Cluster':<10} {'Fall':<10} {'Spring':<10} {'Summer':<10} {'Winter':<10}")
        for i, center in enumerate(centers):
            # Note: The order of center values depends on feature_cols order. 
            # Ideally we map them back strictly, but typically they follow alpha order: Fall, Spring, Summer, Winter
            print(f"{i:<10} ${center[0]:<9.1f} ${center[1]:<9.1f} ${center[2]:<9.1f} ${center[3]:<9.1f}")
            
        print("\nInterpretation:")
        print("Look for clusters with high spend in specific columns (e.g., 'Winter Warriors')")
        print("or clusters with high spend everywhere ('VIPs').")

        return model

    def run(self):
        df = self.load_data()
        self.train_neural_network(df)
        self.train_svm(df)
        self.perform_seasonal_segmentation(df)
        self.spark.stop()

if __name__ == "__main__":
    trainer = AdvancedModelTrainer()
    trainer.run()