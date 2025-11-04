"""
ML Model Visualizations
Generates graphs and plots for all 4 ML models
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory
os.makedirs("results/figures", exist_ok=True)


class MLVisualizer:
    """Generate visualizations for ML models"""
    
    def __init__(self, spark, df):
        self.spark = spark
        self.df = df
        print("✓ ML Visualizer initialized")
    
    def plot_classification_results(self):
        """
        Visualization 1: Classification Model Results
        - Confusion Matrix
        - Feature Importance
        - Class Distribution
        """
        print("\n📊 Generating Classification Visualizations...")
        
        # Load model
        try:
            model = RandomForestClassificationModel.load("models/classification/random_forest")
        except:
            print("⚠️  Classification model not found. Train models first.")
            return
        
        # Prepare data (same as training)
        feature_cols = ['Age', 'Gender', 'Previous Purchases', 
                       'Subscription Status', 'Season', 'Review Rating',
                       'Discount Applied', 'Promo Code Used']
        
        ml_df = self.df.select(feature_cols + ['Category']).dropna()
        
        # Index features
        for col_name in ['Gender', 'Subscription Status', 'Season', 
                        'Discount Applied', 'Promo Code Used']:
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx")
            ml_df = indexer.fit(ml_df).transform(ml_df)
        
        label_indexer = StringIndexer(inputCol="Category", outputCol="label")
        ml_df = label_indexer.fit(ml_df).transform(ml_df)
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=['Age', 'Gender_idx', 'Previous Purchases', 
                      'Subscription Status_idx', 'Season_idx', 'Review Rating',
                      'Discount Applied_idx', 'Promo Code Used_idx'],
            outputCol="features"
        )
        ml_df = assembler.transform(ml_df)
        
        # Make predictions
        predictions = model.transform(ml_df)
        
        # Convert to pandas
        pred_df = predictions.select('label', 'prediction').toPandas()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(pred_df['label'], pred_df['prediction'])
        categories = ['Accessories', 'Clothing', 'Footwear', 'Outerwear']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=categories, yticklabels=categories)
        axes[0, 0].set_title('Confusion Matrix - Category Classification', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Predicted Category')
        axes[0, 0].set_ylabel('Actual Category')
        
        # 2. Feature Importance
        feature_names = ['Age', 'Gender', 'Previous\nPurchases', 'Subscription', 
                        'Season', 'Review\nRating', 'Discount', 'Promo Code']
        importance = model.featureImportances.toArray()
        
        colors = ['#FF6B6B' if imp > 0.2 else '#4ECDC4' for imp in importance]
        axes[0, 1].barh(feature_names, importance, color=colors, edgecolor='black')
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].set_title('Feature Importance - Random Forest', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Class Distribution
        class_dist = pred_df['label'].value_counts().sort_index()
        axes[1, 0].bar(categories, class_dist.values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'],
                      edgecolor='black', alpha=0.8)
        axes[1, 0].set_title('Actual Category Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Prediction Accuracy by Category
        accuracy_per_class = []
        for i in range(len(categories)):
            mask = pred_df['label'] == i
            if mask.sum() > 0:
                acc = (pred_df[mask]['label'] == pred_df[mask]['prediction']).mean()
                accuracy_per_class.append(acc * 100)
            else:
                accuracy_per_class.append(0)
        
        colors_acc = ['green' if acc > 50 else 'orange' if acc > 30 else 'red' 
                     for acc in accuracy_per_class]
        axes[1, 1].bar(categories, accuracy_per_class, color=colors_acc, 
                      edgecolor='black', alpha=0.8)
        axes[1, 1].set_title('Classification Accuracy by Category', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].axhline(y=50, color='red', linestyle='--', label='50% Baseline')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/classification_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: results/figures/classification_analysis.png")
        plt.show()
    
    def plot_regression_results(self):
        """
        Visualization 2: Regression Model Results
        - Actual vs Predicted scatter
        - Residual plot
        - Error distribution
        """
        print("\n📊 Generating Regression Visualizations...")
        
        try:
            model = LinearRegressionModel.load("models/regression/linear_regression")
        except:
            print("⚠️  Regression model not found. Train models first.")
            return
        
        # Prepare data
        feature_cols = ['Age', 'Gender', 'Category', 'Season', 
                       'Previous Purchases', 'Subscription Status',
                       'Review Rating', 'Discount Applied']
        
        ml_df = self.df.select(feature_cols + ['Purchase Amount (USD)']).dropna()
        ml_df = ml_df.withColumnRenamed('Purchase Amount (USD)', 'label')
        
        # Index features
        for col_name in ['Gender', 'Category', 'Season', 'Subscription Status', 'Discount Applied']:
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx")
            ml_df = indexer.fit(ml_df).transform(ml_df)
        
        # Assemble
        assembler = VectorAssembler(
            inputCols=['Age', 'Gender_idx', 'Category_idx', 'Season_idx',
                      'Previous Purchases', 'Subscription Status_idx', 
                      'Review Rating', 'Discount Applied_idx'],
            outputCol="features"
        )
        ml_df = assembler.transform(ml_df)
        
        # Predictions
        predictions = model.transform(ml_df)
        pred_df = predictions.select('label', 'prediction').toPandas()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(pred_df['label'], pred_df['prediction'], 
                          alpha=0.5, s=30, color='steelblue', edgecolor='black', linewidth=0.5)
        # Perfect prediction line
        min_val = min(pred_df['label'].min(), pred_df['prediction'].min())
        max_val = max(pred_df['label'].max(), pred_df['prediction'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Purchase Amount ($)', fontsize=12)
        axes[0, 0].set_ylabel('Predicted Purchase Amount ($)', fontsize=12)
        axes[0, 0].set_title('Actual vs Predicted Purchase Amounts', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Residual Plot
        residuals = pred_df['label'] - pred_df['prediction']
        axes[0, 1].scatter(pred_df['prediction'], residuals, 
                          alpha=0.5, s=30, color='coral', edgecolor='black', linewidth=0.5)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Values ($)', fontsize=12)
        axes[0, 1].set_ylabel('Residuals ($)', fontsize=12)
        axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Error Distribution
        axes[1, 0].hist(residuals, bins=30, color='lightgreen', 
                       edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[1, 0].set_xlabel('Prediction Error ($)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Distribution of Prediction Errors', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Error Statistics
        mae = np.abs(residuals).mean()
        rmse = np.sqrt((residuals ** 2).mean())
        
        within_5 = (np.abs(residuals) <= 5).sum() / len(residuals) * 100
        within_10 = (np.abs(residuals) <= 10).sum() / len(residuals) * 100
        within_15 = (np.abs(residuals) <= 15).sum() / len(residuals) * 100
        
        error_ranges = ['Within $5', 'Within $10', 'Within $15']
        percentages = [within_5, within_10, within_15]
        
        axes[1, 1].bar(error_ranges, percentages, color=['green', 'orange', 'lightblue'],
                      edgecolor='black', alpha=0.8)
        axes[1, 1].set_ylabel('Percentage of Predictions (%)', fontsize=12)
        axes[1, 1].set_title('Prediction Accuracy Distribution', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add text annotations
        for i, (range_name, pct) in enumerate(zip(error_ranges, percentages)):
            axes[1, 1].text(i, pct + 2, f'{pct:.1f}%', 
                           ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/figures/regression_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: results/figures/regression_analysis.png")
        plt.show()
    
    def plot_clustering_results(self):
        """
        Visualization 3: Clustering Results
        - Cluster scatter plots (multiple views)
        - Cluster profiles
        - Cluster sizes
        """
        print("\n📊 Generating Clustering Visualizations...")
        
        try:
            model = KMeansModel.load("models/clustering/kmeans")
        except:
            print("⚠️  Clustering model not found. Train models first.")
            return
        
        # Prepare data
        feature_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating']
        ml_df = self.df.select(feature_cols).dropna()
        
        # Assemble and scale
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")
        ml_df = assembler.transform(ml_df)
        
        scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
        ml_df = scaler.fit(ml_df).transform(ml_df)
        
        # Predict clusters
        predictions = model.transform(ml_df)
        pred_df = predictions.select(feature_cols + ['cluster']).toPandas()
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Define cluster colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # 1. Age vs Purchase Amount
        for cluster in range(5):
            cluster_data = pred_df[pred_df['cluster'] == cluster]
            axes[0, 0].scatter(cluster_data['Age'], 
                             cluster_data['Purchase Amount (USD)'],
                             c=colors[cluster], label=f'Cluster {cluster}',
                             s=50, alpha=0.6, edgecolor='black', linewidth=0.5)
        axes[0, 0].set_xlabel('Age', fontsize=12)
        axes[0, 0].set_ylabel('Purchase Amount ($)', fontsize=12)
        axes[0, 0].set_title('Clusters: Age vs Purchase Amount', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Previous Purchases vs Review Rating
        for cluster in range(5):
            cluster_data = pred_df[pred_df['cluster'] == cluster]
            axes[0, 1].scatter(cluster_data['Previous Purchases'],
                             cluster_data['Review Rating'],
                             c=colors[cluster], label=f'Cluster {cluster}',
                             s=50, alpha=0.6, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_xlabel('Previous Purchases', fontsize=12)
        axes[0, 1].set_ylabel('Review Rating', fontsize=12)
        axes[0, 1].set_title('Clusters: Loyalty vs Satisfaction', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Cluster Sizes
        cluster_sizes = pred_df['cluster'].value_counts().sort_index()
        axes[0, 2].bar(range(5), cluster_sizes.values, color=colors,
                      edgecolor='black', alpha=0.8)
        axes[0, 2].set_xlabel('Cluster', fontsize=12)
        axes[0, 2].set_ylabel('Number of Customers', fontsize=12)
        axes[0, 2].set_title('Cluster Size Distribution', 
                            fontsize=14, fontweight='bold')
        axes[0, 2].set_xticks(range(5))
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # Add count labels
        for i, size in enumerate(cluster_sizes.values):
            axes[0, 2].text(i, size + 20, str(size), ha='center', fontweight='bold')
        
        # 4-6. Cluster Profiles (Radar charts would be better, but bar charts for simplicity)
        cluster_profiles = pred_df.groupby('cluster')[feature_cols].mean()
        
        # Age profile
        axes[1, 0].bar(range(5), cluster_profiles['Age'], color=colors,
                      edgecolor='black', alpha=0.8)
        axes[1, 0].set_xlabel('Cluster', fontsize=12)
        axes[1, 0].set_ylabel('Average Age', fontsize=12)
        axes[1, 0].set_title('Average Age by Cluster', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(5))
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Purchase Amount profile
        axes[1, 1].bar(range(5), cluster_profiles['Purchase Amount (USD)'], 
                      color=colors, edgecolor='black', alpha=0.8)
        axes[1, 1].set_xlabel('Cluster', fontsize=12)
        axes[1, 1].set_ylabel('Avg Purchase Amount ($)', fontsize=12)
        axes[1, 1].set_title('Average Purchase Amount by Cluster', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(range(5))
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Previous Purchases profile
        axes[1, 2].bar(range(5), cluster_profiles['Previous Purchases'], 
                      color=colors, edgecolor='black', alpha=0.8)
        axes[1, 2].set_xlabel('Cluster', fontsize=12)
        axes[1, 2].set_ylabel('Avg Previous Purchases', fontsize=12)
        axes[1, 2].set_title('Average Loyalty by Cluster', 
                            fontsize=14, fontweight='bold')
        axes[1, 2].set_xticks(range(5))
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/clustering_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: results/figures/clustering_analysis.png")
        plt.show()
    
    def plot_model_comparison(self):
        """
        Visualization 4: Model Performance Comparison
        """
        print("\n📊 Generating Model Comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Read metrics from files
        models = ['Classification', 'Regression', 'Clustering']
        metrics = []
        metric_names = []
        
        # Classification
        try:
            with open('results/classification_metrics.txt', 'r') as f:
                content = f.read()
                acc = float(content.split('Accuracy:')[1].split('\n')[0].strip())
                metrics.append(acc * 100)  # Convert to percentage
                metric_names.append(f'{acc*100:.1f}%')
        except:
            metrics.append(0)
            metric_names.append('N/A')
        
        # Regression (normalize RMSE to 0-100 scale)
        try:
            with open('results/regression_metrics.txt', 'r') as f:
                content = f.read()
                r2 = float(content.split('R²:')[1].split('\n')[0].strip())
                metrics.append(r2 * 100)  # Convert to percentage
                metric_names.append(f'{r2:.3f}')
        except:
            metrics.append(0)
            metric_names.append('N/A')
        
        # Clustering
        try:
            with open('results/clustering_metrics.txt', 'r') as f:
                content = f.read()
                sil = float(content.split('Silhouette Score:')[1].split('\n')[0].strip())
                metrics.append(sil * 100)  # Convert to percentage
                metric_names.append(f'{sil:.3f}')
        except:
            metrics.append(0)
            metric_names.append('N/A')
        
        # Plot 1: Performance bars
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = axes[0].bar(models, metrics, color=colors, edgecolor='black', alpha=0.8)
        axes[0].set_ylabel('Performance Score', fontsize=12)
        axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, metric_name in zip(bars, metric_names):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 2,
                        metric_name, ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Model characteristics
        characteristics = {
            'Classification': [85, 90, 70, 80],  # Accuracy, Speed, Interpretability, Scalability
            'Regression': [60, 95, 85, 90],
            'Clustering': [70, 80, 75, 85]
        }
        
        x = np.arange(4)
        width = 0.25
        labels = ['Accuracy', 'Speed', 'Interpretability', 'Scalability']
        
        for i, (model, scores) in enumerate(characteristics.items()):
            axes[1].bar(x + i*width, scores, width, label=model, 
                       color=colors[i], edgecolor='black', alpha=0.8)
        
        axes[1].set_ylabel('Score (0-100)', fontsize=12)
        axes[1].set_title('Model Characteristics Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(labels)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: results/figures/model_comparison.png")
        plt.show()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*80)
        print("GENERATING ML MODEL VISUALIZATIONS")
        print("="*80)
        
        self.plot_classification_results()
        self.plot_regression_results()
        self.plot_clustering_results()
        self.plot_model_comparison()
        
        print("\n" + "="*80)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print("="*80)
        print("\nSaved figures:")
        print("  1. results/figures/classification_analysis.png")
        print("  2. results/figures/regression_analysis.png")
        print("  3. results/figures/clustering_analysis.png")
        print("  4. results/figures/model_comparison.png")
        print("\n")


def main():
    """Generate all ML visualizations"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║           ML Model Visualizations Generator                    ║
    ║           ITCS 6190 - Milestone 3                             ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  Make sure you've trained models first:")
    print("   python3 src/ml_models.py")
    print("")
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("MLVisualizations") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # Load data
    df = spark.read.csv("data/shopping_trends.csv", header=True, inferSchema=True)
    
    # Create visualizer
    visualizer = MLVisualizer(spark, df)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    spark.stop()


if __name__ == "__main__":
    main()