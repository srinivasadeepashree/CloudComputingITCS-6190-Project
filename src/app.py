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
            try:
                self.seasonal_pipeline = PipelineModel.load(path)
                # Robust check: Ensure the last stage is IndexToString (has getLabels)
                last_stage = self.seasonal_pipeline.stages[-1]
                if hasattr(last_stage, 'getLabels'):
                    self.item_labels = last_stage.getLabels()
                else:
                    raise AttributeError("Last stage is not IndexToString (missing labels)")
            except Exception as e:
                print(f"⚠️ Warning: Saved Seasonal Model is incompatible ({e}). Re-training...")
                self._train_seasonal_logic(path)
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
            try:
                self.promo_pipeline = PipelineModel.load(path)
                # Robust check: Ensure the last stage is IndexToString
                last_stage = self.promo_pipeline.stages[-1]
                if hasattr(last_stage, 'getLabels'):
                    self.promo_labels = last_stage.getLabels()
                else:
                    raise AttributeError("Last stage is not IndexToString (missing labels)")
            except Exception as e:
                print(f"⚠️ Warning: Saved Promo Model is incompatible ({e}). Re-training...")
                self._train_promo_logic(path)
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

with gr.Blocks(title="ShopSense") as demo:
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