"""
Machine Learning Model: Promo Code Propensity Predictor (With Evaluation)
ITCS 6190 - Big Data Analytics Project
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

class PromoModelApp:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("PromoCodePredictor") \
            .master("local[*]") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        print("✓ Spark Session Initialized")

    def train(self, filepath="data/shopping.csv"):
        print("🚀 Loading Data...")
        df = self.spark.read.csv(filepath, header=True, inferSchema=True)
        
        # 1. SPLIT DATA (80% Train, 20% Test)
        # We assume the data is shuffled randomly
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        
        print(f"📊 Training Set: {train_df.count()} rows")
        print(f"📊 Test Set:     {test_df.count()} rows")

        # 2. Pipeline Definition
        stages = []
        
        # Target Indexer
        label_indexer = StringIndexer(inputCol="Promo Code Used", outputCol="label").fit(df)
        stages.append(label_indexer)
        
        # Feature Encoders
        cat_cols = ['Gender', 'Payment Method', 'Subscription Status', 'Frequency of Purchases']
        for c in cat_cols:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            stages += [indexer, encoder]
            
        # Feature Assembler
        num_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases']
        assembler = VectorAssembler(
            inputCols=[f"{c}_vec" for c in cat_cols] + num_cols, 
            outputCol="features"
        )
        stages.append(assembler)
        
        # Classifier
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50)
        stages.append(rf)
        
        # 3. TRAIN on Training Set
        print("\n🚀 Training Model on 80% of data...")
        self.pipeline_model = Pipeline(stages=stages).fit(train_df)
        
        # 4. EVALUATE on Test Set
        print("🔍 Evaluating on Test Data...")
        predictions = self.pipeline_model.transform(test_df)
        
        # Calculate Metrics
        # AUC (Area Under ROC) - Good for binary classification
        binary_eval = BinaryClassificationEvaluator(labelCol="label")
        auc = binary_eval.evaluate(predictions)
        
        # Accuracy, Precision, Recall, F1
        multi_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
        accuracy = multi_eval.evaluate(predictions)
        
        multi_eval.setMetricName("weightedPrecision")
        precision = multi_eval.evaluate(predictions)
        
        multi_eval.setMetricName("weightedRecall")
        recall = multi_eval.evaluate(predictions)
        
        multi_eval.setMetricName("f1")
        f1 = multi_eval.evaluate(predictions)
        
        print("\n" + "="*40)
        print("📈 TEST DATA METRICS")
        print("="*40)
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print("="*40 + "\n")
        
    def predict_user_input(self):
        # ... (Keep this function exactly the same as before) ...
        # Import local to avoid scope issues
        from pyspark.sql.functions import lit
        
        print("\n" + "="*40)
        print("🔮 INTERACTIVE PREDICTION TOOL")
        print("="*40)
        print("Enter customer details to predict Promo Code usage:")
        
        try:
            # 1. Collect Inputs
            age = int(input("Enter Age (e.g., 25): "))
            gender = input("Enter Gender (Male/Female): ")
            spend = float(input("Enter Purchase Amount ($): "))
            payment = input("Enter Payment Method (Credit Card, PayPal, Cash, Venmo): ")
            sub_status = input("Is Subscriber? (Yes/No): ")
            freq = input("Frequency (Weekly, Monthly, Annually): ")
            prev_purchases = int(input("Previous Purchases Count (e.g., 10): "))
            
            # 2. Create DataFrame
            data = [(age, gender, spend, payment, sub_status, freq, prev_purchases)]
            columns = ['Age', 'Gender', 'Purchase Amount (USD)', 'Payment Method', 
                       'Subscription Status', 'Frequency of Purchases', 'Previous Purchases']
            
            input_df = self.spark.createDataFrame(data, columns)
            input_df = input_df.withColumn("Promo Code Used", lit("No")) # Dummy target
            
            # 3. Predict
            prediction = self.pipeline_model.transform(input_df)
            
            # 4. Result
            result = prediction.select("probability", "prediction").collect()[0]
            probs = result['probability']
            pred_class = result['prediction']
            
            # Check which index is "Yes" (usually index 1 in binary)
            likelihood = probs[1] * 100
            
            print("\n" + "-"*40)
            print("📊 PREDICTION RESULT")
            print("-" * 40)
            if pred_class == 1.0: # Assuming 1.0 = Yes
                print(f"✅ YES, likely to use Promo Code.")
            else:
                print(f"❌ NO, unlikely to use Promo Code.")
            print(f"Confidence: {likelihood:.1f}%")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    app = PromoModelApp()
    app.train()
    
    # Optional loop for manual testing
    # while True:
    #     app.predict_user_input()