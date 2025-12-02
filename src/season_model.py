"""
Machine Learning Pipeline: Seasonal Item Prediction (With Save/Load)
ITCS 6190 - Cloud Computing for Data Analysis Project

Model 5: Seasonal Item Context Predictor
Goal: Predict specific items a customer is most likely to buy in a specific season.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vector, DenseVector
from pyspark.sql.types import ArrayType, FloatType, StringType
import os

class SeasonalItemPredictor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("SeasonalItemPred") \
            .master("local[*]") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        # Ensure the directory exists
        os.makedirs("model", exist_ok=True)
        print("✓ Spark Session Initialized")

    def load_data(self, filepath="/workspaces/CloudComputingITCS-6190-Project/data/shopping.csv"):
        return self.spark.read.csv(filepath, header=True, inferSchema=True)

    def train_seasonal_model(self, df):
        print("\n" + "="*60)
        print("TRAINING SEASONAL ITEM PREDICTOR")
        print("="*60)
        
        # 1. Prepare Data
        # Target: Item Purchased
        # Features: Season (Context), Demographics (User)
        
        # Select relevant columns
        train_df = df.select(
            'Item Purchased', 'Season', 'Age', 'Gender', 
            'Location', 'Subscription Status', 'Previous Purchases'
        )
        
        # 2. Pipeline Stages
        stages = []
        
        # A. Index Target (Item Purchased)
        label_indexer = StringIndexer(
            inputCol="Item Purchased", 
            outputCol="label", 
            handleInvalid="keep"
        ).fit(train_df)
        stages.append(label_indexer)
        
        # B. Encode Categorical Features
        cat_cols = ['Season', 'Gender', 'Location', 'Subscription Status']
        for c in cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            stages += [indexer, encoder]
            
        # C. Assemble Features
        num_cols = ['Age', 'Previous Purchases']
        assembler = VectorAssembler(
            inputCols=[f"{c}_vec" for c in cat_cols] + num_cols, 
            outputCol="features"
        )
        stages.append(assembler)
        
        # D. Classifier (Random Forest)
        rf = RandomForestClassifier(
            labelCol="label", 
            featuresCol="features", 
            numTrees=100, 
            maxDepth=10
        )
        stages.append(rf)
        
        # E. Convert Predictions back to String Labels
        label_converter = IndexToString(
            inputCol="prediction", 
            outputCol="predicted_item", 
            labels=label_indexer.labels
        )
        stages.append(label_converter)
        
        # 3. Train Model
        print("🚀 Training model (predicting 1 of 25 items)...")
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(train_df)
        
        print("✓ Model Trained Successfully")
        return model, label_indexer.labels

    def save_model(self, model, path="model/seasonal_predictor"):
        """Saves the trained Pipeline model to disk."""
        print(f"\n💾 Saving model to '{path}'...")
        try:
            # Overwrite ensures we don't crash if the folder exists
            model.write().overwrite().save(path)
            print("✓ Model saved successfully.")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def load_model(self, path="model/seasonal_predictor"):
        """Loads a pre-trained Pipeline model."""
        print(f"\n📂 Loading model from '{path}'...")
        try:
            model = PipelineModel.load(path)
            print("✓ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def predict_for_customer(self, model, df, customer_id, target_season, item_labels):
        """
        Custom function to predict what a specific customer will buy in a specific season
        """
        print(f"\n🔮 Prediction for Customer #{customer_id} in {target_season.upper()}:")
        
        # 1. Get Customer's Profile (Static features)
        customer_profile = df.filter(col("`Customer ID`") == customer_id).limit(1)
        
        if customer_profile.count() == 0:
            print("❌ Customer not found.")
            return

        # 2. Force the 'Season' column to be our Target Season
        test_case = customer_profile.withColumn("Season", lit(target_season))
        
        # 3. Make Prediction
        prediction = model.transform(test_case)
        
        # 4. Extract Probabilities for all items
        row = prediction.select("probability", "predicted_item").collect()[0]
        probs = row['probability'].toArray()
        
        # Pair items with their probabilities and sort
        item_probs = list(zip(item_labels, probs))
        item_probs.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Display Top 3 Most Likely Items
        print(f"   Most likely purchases:")
        for i, (item, prob) in enumerate(item_probs[:3]):
            print(f"   {i+1}. {item}")

    def run(self):
        df = self.load_data()
        model, item_labels = self.train_seasonal_model(df)
        
        # Save Model
        self.save_model(model)

        # --- Interactive Examples ---
        # Predict for Customer 1 in Winter
        self.predict_for_customer(model, df, customer_id=1, target_season="Winter", item_labels=item_labels)
        
        # Predict for same Customer in Summer (Result should theoretically change)
        self.predict_for_customer(model, df, customer_id=1, target_season="Summer", item_labels=item_labels)
        
        # self.spark.stop()

if __name__ == "__main__":
    predictor = SeasonalItemPredictor()
    predictor.run()