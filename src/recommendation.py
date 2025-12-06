"""
Machine Learning Pipeline: Product Recommendation Engine (Corrected & With Save)
ITCS 6190 - Cloud Computing for Data Analysis Project

Model 4: Collaborative Filtering Recommender (ALS)
Goal: Recommend products users have NOT bought yet but are likely to like.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, row_number
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import os

class RecommendationEngine:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("ShoppingRecommender") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        # Ensure the directory exists
        os.makedirs("model", exist_ok=True)
        print("✓ Spark Session Initialized")

    def load_data(self, filepath="/workspaces/CloudComputingITCS-6190-Project/data/shopping.csv"):
        print(f"\n📂 Loading data from: {filepath}")
        return self.spark.read.csv(filepath, header=True, inferSchema=True)

    def prepare_data(self, df):
        print("\n" + "="*50)
        print("PREPARING DATA FOR RECOMMENDATIONS")
        print("="*50)
        
        # 1. Select User, Item, and Rating
        data = df.select(
            col("`Customer ID`").alias("userId"),
            col("`Item Purchased`").alias("item_name"),
            col("`Review Rating`").alias("rating")
        )
        
        # 2. Convert Item Names to Integer IDs
        string_indexer = StringIndexer(inputCol="item_name", outputCol="itemId")
        indexer_model = string_indexer.fit(data)
        data_indexed = indexer_model.transform(data)
        
        # Cache this dataframe as we will use it for training AND filtering
        data_indexed.cache()
        
        print(f"✓ Data Prepared: {data_indexed.count()} interactions")
        print(f"✓ Unique Users: {data_indexed.select('userId').distinct().count()}")
        print(f"✓ Unique Items: {data_indexed.select('itemId').distinct().count()}")
        
        return data_indexed, indexer_model.labels

    def train_recommender(self, df):
        print("\n🚀 Training ALS Recommendation Model...")
        (training, test) = df.randomSplit([0.8, 0.2], seed=42)
        
        als = ALS(
            maxIter=10, 
            regParam=0.1, 
            userCol="userId", 
            itemCol="itemId", 
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )
        
        model = als.fit(training)
        
        predictions = model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        
        print(f"📊 Model Performance (RMSE): {rmse:.4f}")
        return model

    def save_model(self, model, path="model/als_recommender"):
        """Saves the trained ALS model to disk."""
        print(f"\n💾 Saving model to '{path}'...")
        try:
            # Overwrite ensures we don't crash if the folder exists
            model.write().overwrite().save(path)
            print("✓ Model saved successfully.")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def load_model(self, path="model/als_recommender"):
        """Loads a pre-trained ALS model."""
        print(f"\n📂 Loading model from '{path}'...")
        try:
            model = ALSModel.load(path)
            print("✓ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def generate_recommendations(self, model, df_history, item_labels):
        print("\n🔮 Generating recommendations (Filtering out purchased items)...")
        
        # 1. Get predictions for ALL items for every user
        raw_recs = model.recommendForAllUsers(25)
        
        # 2. Explode the array into individual rows [userId, itemId, rating]
        recs_exploded = raw_recs.select(
            col("userId"), 
            explode("recommendations").alias("rec")
        ).select(
            "userId", 
            col("rec.itemId").alias("itemId"), 
            col("rec.rating").alias("predicted_rating")
        )
        
        # 3. IDENTIFY HISTORY: Select just User and Item from original data
        user_history = df_history.select("userId", "itemId").distinct()
        
        # 4. ANTI-JOIN: Subtract History from Recommendations
        new_recs = recs_exploded.join(
            user_history, 
            on=["userId", "itemId"], 
            how="left_anti"
        )
        
        # 5. Rank the remaining items and take Top 3 per user
        windowSpec = Window.partitionBy("userId").orderBy(col("predicted_rating").desc())
        
        final_recs = new_recs.withColumn("rank", row_number().over(windowSpec)) \
            .filter(col("rank") <= 3) \
            .drop("rank")
        
        # 6. Convert Integer Item IDs back to String Names
        item_mapping = [(i, label) for i, label in enumerate(item_labels)]
        mapping_df = self.spark.createDataFrame(item_mapping, ["id", "name"])
        
        # Join to get names
        readable_recs = final_recs.join(mapping_df, final_recs.itemId == mapping_df.id) \
            .select("userId", "name", "predicted_rating") \
            .orderBy("userId", "predicted_rating", ascending=False)
            
        print("\n📋 Sample NEW Product Recommendations:")
        readable_recs.show(10, truncate=False)
        
        # Save Output
        save_path = "model/customer_recommendations_filtered"
        readable_recs.write.mode("overwrite").csv(save_path, header=True)
        print(f"💾 Saved recommendations to: {save_path}")
        
        return readable_recs

    def run(self):
        df = self.load_data()
        # We need the indexed dataframe (df_indexed) for the history filter later
        df_indexed, item_labels = self.prepare_data(df)
        
        # Train
        model = self.train_recommender(df_indexed)
        
        # Save Model
        # self.save_model(model)
        
        # Example of loading (optional usage check)
        # loaded_model = self.load_model()
        
        # Generate Recommendations
        self.generate_recommendations(model, df_indexed, item_labels)
        
        # self.spark.stop()

if __name__ == "__main__":
    engine = RecommendationEngine()
    engine.run()