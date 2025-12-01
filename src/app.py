import gradio as gr
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, explode, collect_set
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
import pandas as pd
import os

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
        
        # Load Data
        self.data_path = "/workspaces/CloudComputingITCS-6190-Project/data/shopping.csv"
        # Safety check for file existence
        if not os.path.exists(self.data_path):
            # Fallback for local testing (remove if not needed)
            if os.path.exists("combined_df.csv"):
                self.data_path = "combined_df.csv"
            elif os.path.exists("shopping.csv"):
                self.data_path = "shopping.csv"
            else:
                print(f"⚠️ Warning: File not found at {self.data_path}")

        try:
            self.df = self.spark.read.csv(self.data_path, header=True, inferSchema=True).cache()
            print("✅ Data Loaded")
            
            # Train Models immediately
            self.train_seasonal_model()
            self.train_recommendation_model()
            self.train_promo_model()  # <--- NEW MODEL TRAINING
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    # ---------------------------------------------------------
    # 1. SEASONAL PREDICTION TRAINING
    # ---------------------------------------------------------
    def train_seasonal_model(self):
        print("🚀 Training Seasonal Prediction Model...")
        train_df = self.df.select(
            'Item Purchased', 'Season', 'Age', 'Gender', 
            'Location', 'Subscription Status', 'Previous Purchases'
        )
        stages = []
        
        # Target Indexer
        self.item_indexer = StringIndexer(inputCol="Item Purchased", outputCol="label", handleInvalid="keep").fit(train_df)
        self.item_labels = self.item_indexer.labels
        stages.append(self.item_indexer)
        
        # Features
        self.cat_cols = ['Season', 'Gender', 'Location', 'Subscription Status']
        self.num_cols = ['Age', 'Previous Purchases']
        for c in self.cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            stages += [indexer, encoder]
            
        assembler = VectorAssembler(inputCols=[f"{c}_vec" for c in self.cat_cols] + self.num_cols, outputCol="features")
        stages.append(assembler)
        
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50, maxDepth=10)
        stages.append(rf)
        
        converter = IndexToString(inputCol="prediction", outputCol="predicted_item", labels=self.item_indexer.labels)
        stages.append(converter)
        
        self.seasonal_pipeline = Pipeline(stages=stages).fit(train_df)
        print("✅ Seasonal Model Trained")

    # ---------------------------------------------------------
    # 2. RECOMMENDATION TRAINING (ALS)
    # ---------------------------------------------------------
    def train_recommendation_model(self):
        print("🚀 Training Recommendation Model (ALS)...")
        data = self.df.select(
            col("`Customer ID`").alias("userId"),
            col("`Item Purchased`").alias("item_name"),
            col("`Review Rating`").alias("rating")
        )
        self.rec_item_indexer = StringIndexer(inputCol="item_name", outputCol="itemId").fit(data)
        data_indexed = self.rec_item_indexer.transform(data)
        self.rec_item_labels = self.rec_item_indexer.labels
        
        als = ALS(maxIter=5, regParam=0.1, userCol="userId", itemCol="itemId", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)
        model = als.fit(data_indexed)
        
        print("⏳ Pre-calculating recommendations (this takes a moment)...")
        raw_recs = model.recommendForAllUsers(15)
        self.user_history_map = data_indexed.select("userId", "itemId").toPandas().groupby("userId")['itemId'].apply(set).to_dict()
        self.raw_recs_map = {}
        rows = raw_recs.collect()
        for row in rows:
            self.raw_recs_map[row['userId']] = [(r.itemId, r.rating) for r in row['recommendations']]
        print("✅ Recommendation System Ready")

    # ---------------------------------------------------------
    # 3. PROMO CODE MODEL TRAINING (NEW)
    # ---------------------------------------------------------
    def train_promo_model(self):
        print("🚀 Training Promo Code Prediction Model...")
        stages = []
        
        # Target: Promo Code Used
        self.promo_label_indexer = StringIndexer(inputCol="Promo Code Used", outputCol="label").fit(self.df)
        stages.append(self.promo_label_indexer)
        
        # Features
        cat_cols = ['Gender', 'Payment Method', 'Subscription Status', 'Frequency of Purchases']
        for c in cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            stages += [indexer, encoder]
            
        num_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases']
        assembler = VectorAssembler(inputCols=[f"{c}_vec" for c in cat_cols] + num_cols, outputCol="features")
        stages.append(assembler)
        
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50)
        stages.append(rf)
        
        self.promo_pipeline = Pipeline(stages=stages).fit(self.df)
        print("✅ Promo Code Model Trained")

    # ---------------------------------------------------------
    # UI PREDICTION FUNCTIONS
    # ---------------------------------------------------------
    def predict_seasonal_purchase(self, user_id, season):
        try:
            user_id = int(user_id)
            profile = self.df.filter(col("`Customer ID`") == user_id).limit(1)
            if profile.count() == 0: return "❌ Error: Customer ID not found in database."
            
            test_case = profile.withColumn("Season", lit(season))
            prediction = self.seasonal_pipeline.transform(test_case)
            probs = prediction.select("probability").collect()[0][0].toArray()
            
            item_probs = list(zip(self.item_labels, probs))
            item_probs.sort(key=lambda x: x[1], reverse=True)
            
            output = f"🎯 Seasonal Predictions for User {user_id} in {season}:\n" + "-" * 40 + "\n"
            for i, (item, prob) in enumerate(item_probs[:3]):
                output += f"{i+1}. {item:<20} \n"
            return output
        except Exception as e: return f"Error: {str(e)}"

    def get_recommendations(self, user_id):
        try:
            user_id = int(user_id)
            if user_id not in self.raw_recs_map: return "⚠️ New user or no history found. Showing popular items..."
            candidates = self.raw_recs_map[user_id]
            bought = self.user_history_map.get(user_id, set())
            final_recs = []
            for item_id, rating in candidates:
                if item_id not in bought:
                    item_name = self.rec_item_labels[int(item_id)]
                    final_recs.append((item_name, rating))
                    if len(final_recs) >= 3: break
            
            if not final_recs: return "ℹ️ User has bought everything!"
            output = f"🛍️ Recommendations for User {user_id}:\n" + "-" * 40 + "\n"
            for i, (name, rating) in enumerate(final_recs):
                output += f"{i+1}. {name:<20} (Rating: {rating:.1f} ★)\n"
            return output
        except Exception as e: return f"Error: {str(e)}"

    # NEW FUNCTION for Promo Code Prediction
    def predict_promo_code(self, age, gender, spend, payment, sub_status, freq, prev_purchases):
        try:
            # Create DataFrame from manual inputs
            data = [(int(age), gender, float(spend), payment, sub_status, freq, int(prev_purchases))]
            columns = ['Age', 'Gender', 'Purchase Amount (USD)', 'Payment Method', 
                       'Subscription Status', 'Frequency of Purchases', 'Previous Purchases']
            
            input_df = self.spark.createDataFrame(data, columns)
            # Add dummy target col for pipeline compatibility
            input_df = input_df.withColumn("Promo Code Used", lit("No"))
            
            prediction = self.promo_pipeline.transform(input_df)
            result = prediction.select("probability", "prediction").collect()[0]
            
            # Map index to label
            labels = self.promo_label_indexer.labels
            predicted_label = labels[int(result['prediction'])]
            
            # Get confidence
            # Assuming 'Yes' is one of the labels
            yes_idx = labels.index("Yes") if "Yes" in labels else 1
            confidence = result['probability'][yes_idx] * 100
            
            output = "📊 Prediction Result:\n" + "-" * 30 + "\n"
            if predicted_label == "Yes":
                output += "✅ YES, Likely to use a Promo Code.\n"
            else:
                output += "❌ NO, Unlikely to use a Promo Code.\n"
            output += f"Confidence (Yes): {confidence:.1f}%"
            return output
        except Exception as e: return f"Error: {str(e)}"

# ==========================================
# GRADIO INTERFACE
# ==========================================

backend = ShoppingSystemBackend()

with gr.Blocks(title="Retail AI Dashboard") as demo:
    gr.Markdown("# 🛒 Retail AI Analytics Dashboard")
    
    with gr.Tab("🍂 Seasonal Prediction"):
        gr.Markdown("### Predict what a customer will buy in a specific season")
        with gr.Row():
            season_uid_input = gr.Number(label="Customer ID", value=1, precision=0)
            season_input = gr.Dropdown(choices=["Spring", "Summer", "Fall", "Winter"], label="Select Season", value="Winter")
        season_btn = gr.Button("Predict Purchase", variant="primary")
        season_output = gr.Textbox(label="Prediction Results", lines=5)
        season_btn.click(backend.predict_seasonal_purchase, inputs=[season_uid_input, season_input], outputs=season_output)

    with gr.Tab("⭐ Item Recommendations"):
        gr.Markdown("### Get personalized product recommendations")
        with gr.Row():
            rec_uid_input = gr.Number(label="Customer ID", value=1, precision=0)
        rec_btn = gr.Button("Get Recommendations", variant="primary")
        rec_output = gr.Textbox(label="Top 3 Recommendations", lines=5)
        rec_btn.click(backend.get_recommendations, inputs=[rec_uid_input], outputs=rec_output)
        
    with gr.Tab("🎟️ Promo Code Prediction"):
        gr.Markdown("### Predict if a new user will use a Promo Code")
        with gr.Row():
            pc_age = gr.Number(label="Age", value=25)
            pc_gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Female")
            pc_spend = gr.Number(label="Purchase Amount ($)", value=50)
            pc_payment = gr.Dropdown(["Credit Card", "PayPal", "Cash", "Venmo", "Debit Card", "Bank Transfer"], label="Payment Method", value="Credit Card")
        with gr.Row():
            pc_sub = gr.Dropdown(["Yes", "No"], label="Subscriber?", value="No")
            pc_freq = gr.Dropdown(["Weekly", "Bi-Weekly", "Fortnightly", "Monthly", "Quarterly", "Every 3 Months", "Annually"], label="Frequency", value="Monthly")
            pc_prev = gr.Number(label="Previous Purchases", value=5)
            
        pc_btn = gr.Button("Predict Propensity", variant="primary")
        pc_output = gr.Textbox(label="Result", lines=4)
        
        pc_btn.click(
            backend.predict_promo_code,
            inputs=[pc_age, pc_gender, pc_spend, pc_payment, pc_sub, pc_freq, pc_prev],
            outputs=pc_output
        )

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