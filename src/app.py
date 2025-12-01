import gradio as gr
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, explode, collect_set
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline, PipelineModel
import os
import shutil

# ==========================================
# BACKEND CLASS: Handles Spark & Models
# ==========================================
class ShoppingSystemBackend:
    def __init__(self):
        print("⚙️ Initializing Spark Session...")
        self.spark = SparkSession.builder \
            .appName("ShoppingGradioApp") \
            .master("local[*]") \
            .config("spark.ui.showConsoleProgress", "false") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # Paths
        self.data_path = "/workspaces/CloudComputingITCS-6190-Project/data/shopping.csv"
        self.model_dir = "model"
        
        # Load Data
        if not os.path.exists(self.data_path):
             # Fallback logic for different environments
            if os.path.exists("shopping.csv"): self.data_path = "shopping.csv"
            
        try:
            self.df = self.spark.read.csv(self.data_path, header=True, inferSchema=True).cache()
            print(f"✅ Data Loaded ({self.df.count()} rows)")
            
            # Initialize Models
            self.load_or_train_seasonal()
            self.load_or_train_promo()
            self.load_or_train_recommender()
            
        except Exception as e:
            print(f"❌ Critical Error: {e}")

    # ---------------------------------------------------------
    # 1. SEASONAL MODEL (Load or Train)
    # ---------------------------------------------------------
    def load_or_train_seasonal(self):
        path = f"{self.model_dir}/seasonal_predictor"
        if os.path.exists(path):
            print("📂 Loading saved Seasonal Model...")
            self.seasonal_pipeline = PipelineModel.load(path)
            # Extract labels from the IndexToString stage (last stage)
            self.item_labels = self.seasonal_pipeline.stages[-1].getLabels()
        else:
            print("🚀 Training Seasonal Model...")
            self._train_seasonal_logic(path)
            
    def _train_seasonal_logic(self, save_path):
        train_df = self.df.select('Item Purchased', 'Season', 'Age', 'Gender', 'Location', 'Subscription Status', 'Previous Purchases')
        stages = []
        
        # Target Indexer
        indexer = StringIndexer(inputCol="Item Purchased", outputCol="label", handleInvalid="keep").fit(train_df)
        self.item_labels = indexer.labels
        stages.append(indexer)
        
        # Features
        cat_cols = ['Season', 'Gender', 'Location', 'Subscription Status']
        for c in cat_cols:
            stages += [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"),
                       OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")]
            
        stages.append(VectorAssembler(inputCols=[f"{c}_vec" for c in cat_cols] + ['Age', 'Previous Purchases'], outputCol="features"))
        stages.append(RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50))
        stages.append(IndexToString(inputCol="prediction", outputCol="predicted_item", labels=indexer.labels))
        
        self.seasonal_pipeline = Pipeline(stages=stages).fit(train_df)
        self.seasonal_pipeline.write().overwrite().save(save_path)
        print("✅ Seasonal Model Trained & Saved")

    # ---------------------------------------------------------
    # 2. PROMO MODEL (Load or Train)
    # ---------------------------------------------------------
    def load_or_train_promo(self):
        path = f"{self.model_dir}/promo_code_model"
        if os.path.exists(path):
            print("📂 Loading saved Promo Model...")
            self.promo_pipeline = PipelineModel.load(path)
            # Extract labels from the IndexToString stage (we must ensure we added one during training)
            # If loaded model doesn't have IndexToString, we might fail. 
            # Note: The training logic below ADDS IndexToString to make this safe.
            self.promo_labels = self.promo_pipeline.stages[-1].getLabels()
        else:
            print("🚀 Training Promo Model...")
            self._train_promo_logic(path)

    def _train_promo_logic(self, save_path):
        stages = []
        # Target
        indexer = StringIndexer(inputCol="Promo Code Used", outputCol="label").fit(self.df)
        self.promo_labels = indexer.labels
        stages.append(indexer)
        
        # Features
        cat_cols = ['Gender', 'Payment Method', 'Subscription Status', 'Frequency of Purchases']
        for c in cat_cols:
            stages += [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"),
                       OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")]
            
        stages.append(VectorAssembler(inputCols=[f"{c}_vec" for c in cat_cols] + ['Age', 'Purchase Amount (USD)', 'Previous Purchases'], outputCol="features"))
        stages.append(RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50))
        # Add IndexToString so the pipeline is self-contained
        stages.append(IndexToString(inputCol="prediction", outputCol="predicted_label", labels=indexer.labels))
        
        self.promo_pipeline = Pipeline(stages=stages).fit(self.df)
        self.promo_pipeline.write().overwrite().save(save_path)
        print("✅ Promo Model Trained & Saved")

    # ---------------------------------------------------------
    # 3. RECOMMENDATION MODEL (ALS)
    # ---------------------------------------------------------
    def load_or_train_recommender(self):
        # 1. Always prepare the Indexers (Fast and needed for mapping names <-> IDs)
        data = self.df.select(col("`Customer ID`").alias("userId"), col("`Item Purchased`").alias("item_name"), col("`Review Rating`").alias("rating"))
        self.rec_item_indexer = StringIndexer(inputCol="item_name", outputCol="itemId").fit(data)
        data_indexed = self.rec_item_indexer.transform(data)
        self.rec_item_labels = self.rec_item_indexer.labels
        
        # 2. Load or Train ALS Model
        path = f"{self.model_dir}/als_recommender"
        if os.path.exists(path):
            print("📂 Loading saved ALS Model...")
            model = ALSModel.load(path)
        else:
            print("🚀 Training ALS Model...")
            als = ALS(maxIter=5, regParam=0.1, userCol="userId", itemCol="itemId", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)
            model = als.fit(data_indexed)
            model.write().overwrite().save(path)
            print("✅ ALS Model Trained & Saved")

        # 3. Pre-calculate Recs (Required for fast UI)
        print("⏳ caching recommendations...")
        # Get raw recs
        raw_recs = model.recommendForAllUsers(15)
        
        # Build lookup maps
        self.user_history_map = data_indexed.select("userId", "itemId").toPandas().groupby("userId")['itemId'].apply(set).to_dict()
        self.raw_recs_map = {}
        
        # Collect locally
        rows = raw_recs.collect()
        for row in rows:
            self.raw_recs_map[row['userId']] = [(r.itemId, r.rating) for r in row['recommendations']]
        print("✅ Recommender Ready")

    # ---------------------------------------------------------
    # UI PREDICTION FUNCTIONS
    # ---------------------------------------------------------
    def predict_seasonal_purchase(self, user_id, season):
        try:
            profile = self.df.filter(col("`Customer ID`") == int(user_id)).limit(1)
            if profile.count() == 0: return "❌ Customer not found."
            
            test_case = profile.withColumn("Season", lit(season))
            prediction = self.seasonal_pipeline.transform(test_case)
            
            # Get Probabilities
            probs = prediction.select("probability").collect()[0][0].toArray()
            item_probs = list(zip(self.item_labels, probs))
            item_probs.sort(key=lambda x: x[1], reverse=True)
            
            output = f"🎯 Predictions for User {user_id} in {season}:\n" + "-" * 30 + "\n"
            for i, (item, prob) in enumerate(item_probs[:3]):
                output += f"{i+1}. {item:<20} ({prob*100:.1f}%)\n"
            return output
        except Exception as e: return f"Error: {e}"

    def get_recommendations(self, user_id):
        try:
            uid = int(user_id)
            if uid not in self.raw_recs_map: return "⚠️ New user or no history."
            
            candidates = self.raw_recs_map[uid]
            bought = self.user_history_map.get(uid, set())
            final_recs = []
            
            for item_id, rating in candidates:
                if item_id not in bought:
                    final_recs.append((self.rec_item_labels[int(item_id)], rating))
                    if len(final_recs) >= 3: break
            
            output = f"🛍️ Top Picks for User {uid}:\n" + "-" * 30 + "\n"
            for i, (name, rating) in enumerate(final_recs):
                output += f"{i+1}. {name:<20} ({rating:.1f}★)\n"
            return output
        except Exception as e: return f"Error: {e}"

    def predict_promo_code(self, age, gender, spend, payment, sub_status, freq, prev_purchases):
        try:
            data = [(int(age), gender, float(spend), payment, sub_status, freq, int(prev_purchases))]
            cols = ['Age', 'Gender', 'Purchase Amount (USD)', 'Payment Method', 'Subscription Status', 'Frequency of Purchases', 'Previous Purchases']
            input_df = self.spark.createDataFrame(data, cols).withColumn("Promo Code Used", lit("No"))
            
            prediction = self.promo_pipeline.transform(input_df)
            row = prediction.select("probability", "predicted_label").collect()[0]
            
            # Find index of "Yes" in self.promo_labels to get specific confidence
            try:
                yes_index = self.promo_labels.index("Yes")
                confidence = row['probability'][yes_index] * 100
            except:
                confidence = row['probability'][1] * 100 # Fallback
            
            result_label = row['predicted_label']
            
            output = "📊 Promo Analysis:\n" + "-" * 30 + "\n"
            output += f"{'✅ YES' if result_label == 'Yes' else '❌ NO'}, likely to use promo.\n"
            output += f"Probability (Yes): {confidence:.1f}%"
            return output
        except Exception as e: return f"Error: {e}"

# ==========================================
# GRADIO INTERFACE
# ==========================================
backend = ShoppingSystemBackend()

with gr.Blocks(title="ShopSense", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛒 ShopSense")
    
    with gr.Tab("🍂 Seasonal Prediction"):
        with gr.Row():
            season_uid = gr.Number(label="Customer ID", value=1)
            season_in = gr.Dropdown(["Spring", "Summer", "Fall", "Winter"], label="Season", value="Winter")
        season_btn = gr.Button("Predict", variant="primary")
        season_out = gr.Textbox(label="Result", lines=4)
        season_btn.click(backend.predict_seasonal_purchase, inputs=[season_uid, season_in], outputs=season_out)

    with gr.Tab("⭐ Recommendations"):
        with gr.Row():
            rec_uid = gr.Number(label="Customer ID", value=1)
        rec_btn = gr.Button("Get Recommendations", variant="primary")
        rec_out = gr.Textbox(label="Result", lines=4)
        rec_btn.click(backend.get_recommendations, inputs=[rec_uid], outputs=rec_out)
        
    with gr.Tab("🎟️ Promo Propensity"):
        with gr.Row():
            p_age = gr.Number(label="Age", value=25)
            p_gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Female")
            p_spend = gr.Number(label="Spend ($)", value=50)
            p_pay = gr.Dropdown(["Credit Card", "PayPal", "Cash", "Venmo"], label="Payment", value="Credit Card")
        with gr.Row():
            p_sub = gr.Dropdown(["Yes", "No"], label="Subscriber?", value="No")
            p_freq = gr.Dropdown(["Weekly", "Monthly", "Annually"], label="Frequency", value="Monthly")
            p_prev = gr.Number(label="Prev Purchases", value=5)
        p_btn = gr.Button("Analyze", variant="primary")
        p_out = gr.Textbox(label="Result", lines=3)
        p_btn.click(backend.predict_promo_code, inputs=[p_age, p_gender, p_spend, p_pay, p_sub, p_freq, p_prev], outputs=p_out)

if __name__ == "__main__":
    demo.launch(share=True)


# import gradio as gr
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, lit, explode, collect_set
# from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, IndexToString
# from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.recommendation import ALS
# from pyspark.ml import Pipeline
# import pandas as pd
# import os

# # ==========================================
# # BACKEND CLASS: Handles Spark & Models
# # ==========================================
# class ShoppingSystemBackend:
#     def __init__(self):
#         print("⚙️ Initializing Spark Session...")
#         self.spark = SparkSession.builder \
#             .appName("ShoppingGradioApp") \
#             .master("local[*]") \
#             .config("spark.ui.showConsoleProgress", "false") \
#             .getOrCreate()
#         self.spark.sparkContext.setLogLevel("ERROR")
        
#         # Load Data
#         self.data_path = "/workspaces/CloudComputingITCS-6190-Project/data/shopping.csv"
#         if not os.path.exists(self.data_path):
#             raise FileNotFoundError(f"Could not find {self.data_path}")
            
#         self.df = self.spark.read.csv(self.data_path, header=True, inferSchema=True).cache()
#         print("✅ Data Loaded")

#         # Train Models immediately
#         self.train_seasonal_model()
#         self.train_recommendation_model()
        
#     # ---------------------------------------------------------
#     # 1. SEASONAL PREDICTION TRAINING
#     # ---------------------------------------------------------
#     def train_seasonal_model(self):
#         print("🚀 Training Seasonal Prediction Model...")
        
#         # Prepare Data
#         # We assume Age, Gender, etc. are static, but Season changes
#         train_df = self.df.select(
#             'Item Purchased', 'Season', 'Age', 'Gender', 
#             'Location', 'Subscription Status', 'Previous Purchases'
#         )
        
#         stages = []
        
#         # Target Indexer
#         self.item_indexer = StringIndexer(inputCol="Item Purchased", outputCol="label", handleInvalid="keep").fit(train_df)
#         self.item_labels = self.item_indexer.labels
#         stages.append(self.item_indexer)
        
#         # Feature Encoders
#         self.cat_cols = ['Season', 'Gender', 'Location', 'Subscription Status']
#         self.num_cols = ['Age', 'Previous Purchases']
        
#         for c in self.cat_cols:
#             indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
#             encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
#             stages += [indexer, encoder]
            
#         assembler = VectorAssembler(
#             inputCols=[f"{c}_vec" for c in self.cat_cols] + self.num_cols, 
#             outputCol="features"
#         )
#         stages.append(assembler)
        
#         # Classifier
#         rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50, maxDepth=10)
#         stages.append(rf)
        
#         # Prediction Converter
#         converter = IndexToString(inputCol="prediction", outputCol="predicted_item", labels=self.item_indexer.labels)
#         stages.append(converter)
        
#         self.seasonal_pipeline = Pipeline(stages=stages).fit(train_df)
#         print("✅ Seasonal Model Trained")

#     # ---------------------------------------------------------
#     # 2. RECOMMENDATION TRAINING (ALS)
#     # ---------------------------------------------------------
#     def train_recommendation_model(self):
#         print("🚀 Training Recommendation Model (ALS)...")
        
#         # Prepare Interactions
#         data = self.df.select(
#             col("`Customer ID`").alias("userId"),
#             col("`Item Purchased`").alias("item_name"),
#             col("`Review Rating`").alias("rating")
#         )
        
#         # Index Items
#         self.rec_item_indexer = StringIndexer(inputCol="item_name", outputCol="itemId").fit(data)
#         data_indexed = self.rec_item_indexer.transform(data)
#         self.rec_item_labels = self.rec_item_indexer.labels
        
#         # Train ALS
#         als = ALS(
#             maxIter=5, 
#             regParam=0.1, 
#             userCol="userId", 
#             itemCol="itemId", 
#             ratingCol="rating",
#             coldStartStrategy="drop",
#             nonnegative=True
#         )
#         model = als.fit(data_indexed)
        
#         # --- Pre-calculate Top 3 for ALL users to make UI fast ---
#         print("⏳ Pre-calculating recommendations (this takes a moment)...")
        
#         # 1. Get raw top 10 recs
#         raw_recs = model.recommendForAllUsers(15)
        
#         # 2. Get user purchase history to filter out what they already bought
#         user_history = data_indexed.groupBy("userId").agg(collect_set("itemId").alias("bought_items"))
        
#         # 3. Join and Filter (Python UDF for simplicity in filtering logic)
#         recs_exploded = raw_recs.select("userId", explode("recommendations").alias("rec"))
        
#         # Collect to Pandas for fast lookup in UI (Dataset is small enough: ~3900 users)
#         # In production, you would save this to a database (Redis/Postgres)
#         self.user_history_map = data_indexed.select("userId", "itemId").toPandas().groupby("userId")['itemId'].apply(set).to_dict()
        
#         # Store recommendations in a dictionary: {userId: [(itemId, rating), ...]}
#         # We perform the filtering at query time to keep startup faster
#         self.raw_recs_map = {}
#         rows = raw_recs.collect()
#         for row in rows:
#             self.raw_recs_map[row['userId']] = [(r.itemId, r.rating) for r in row['recommendations']]
            
#         print("✅ Recommendation System Ready")

#     # ---------------------------------------------------------
#     # PREDICTION FUNCTIONS FOR UI
#     # ---------------------------------------------------------
#     def predict_seasonal_purchase(self, user_id, season):
#         try:
#             user_id = int(user_id)
#             # Fetch user profile
#             profile = self.df.filter(col("`Customer ID`") == user_id).limit(1)
            
#             if profile.count() == 0:
#                 return "❌ Error: Customer ID not found in database."
            
#             # Create test case with injected Season
#             test_case = profile.withColumn("Season", lit(season))
            
#             # Predict
#             prediction = self.seasonal_pipeline.transform(test_case)
#             probs = prediction.select("probability").collect()[0][0].toArray()
            
#             # Sort items by confidence
#             item_probs = list(zip(self.item_labels, probs))
#             item_probs.sort(key=lambda x: x[1], reverse=True)
            
#             # Format Output
#             top_3 = item_probs[:3]
#             output = f"🎯 Seasonal Predictions for User {user_id} in {season}:\n"
#             output += "-" * 40 + "\n"
#             for i, (item, prob) in enumerate(top_3):
#                 output += f"{i+1}. {item:<20} \n"
            
#             return output
            
#         except Exception as e:
#             return f"Error processing request: {str(e)}"

#     def get_recommendations(self, user_id):
#         try:
#             user_id = int(user_id)
#             if user_id not in self.raw_recs_map:
#                 return "⚠️ New user or no history found. Showing popular items instead..."
            
#             # Get raw recommendations
#             candidates = self.raw_recs_map[user_id]
            
#             # Get items user already bought
#             bought = self.user_history_map.get(user_id, set())
            
#             # Filter
#             final_recs = []
#             for item_id, rating in candidates:
#                 if item_id not in bought:
#                     item_name = self.rec_item_labels[int(item_id)]
#                     final_recs.append((item_name, rating))
#                     if len(final_recs) >= 3:
#                         break
            
#             if not final_recs:
#                 return "ℹ️ User has bought all recommended items!"
            
#             # Format Output
#             output = f"🛍️ Recommended Products for User {user_id}:\n"
#             output += "-" * 40 + "\n"
#             for i, (name, rating) in enumerate(final_recs):
#                 output += f"{i+1}. {name:<20} (Predicted Rating: {rating:.1f} ★)\n"
                
#             return output
            
#         except Exception as e:
#             return f"Error: {str(e)}"

# # ==========================================
# # GRADIO INTERFACE
# # ==========================================

# # 1. Initialize Backend
# backend = ShoppingSystemBackend()

# # 2. Define UI Layout
# with gr.Blocks(title="Retail AI Dashboard") as demo:
#     gr.Markdown(
#         """
#         # 🛒 Retail AI Analytics Dashboard
#         Interact with the PySpark machine learning models.
#         """
#     )
    
#     with gr.Tab("🍂 Seasonal Prediction"):
#         gr.Markdown("### Predict what a customer will buy in a specific season")
#         with gr.Row():
#             season_uid_input = gr.Number(label="Customer ID", value=1, precision=0)
#             season_input = gr.Dropdown(choices=["Spring", "Summer", "Fall", "Winter"], label="Select Season", value="Winter")
        
#         season_btn = gr.Button("Predict Purchase", variant="primary")
#         season_output = gr.Textbox(label="Prediction Results", lines=5)
        
#         season_btn.click(
#             fn=backend.predict_seasonal_purchase,
#             inputs=[season_uid_input, season_input],
#             outputs=season_output
#         )

#     with gr.Tab("⭐ Item Recommendations"):
#         gr.Markdown("### Get personalized product recommendations (Collaborative Filtering)")
#         with gr.Row():
#             rec_uid_input = gr.Number(label="Customer ID", value=1, precision=0)
        
#         rec_btn = gr.Button("Get Recommendations", variant="primary")
#         rec_output = gr.Textbox(label="Top 3 Recommendations", lines=5)
        
#         rec_btn.click(
#             fn=backend.get_recommendations,
#             inputs=[rec_uid_input],
#             outputs=rec_output
#         )

# # 3. Launch
# if __name__ == "__main__":
#     demo.launch(share=True)