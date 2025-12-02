#!/bin/bash

# 1. Stop the script immediately if any command fails
set -e

echo "🚀 Starting the Pipeline..."

# 2. Run your files in order
echo "--- Step 1: Running Data Preparation ---"
python3 src/data_ingestion.py

echo "--- Step 2: Running EDA Analysis ---"
python3 src/eda_analysis.py

echo "--- Step 3: Complex Queries ---"
python3 src/complex_queries.py

echo "--- Step 4: Streaming Data ---"
python3 src/streaming.py
python3 src/streaming_pipeline.py

echo "--- Step 5: Training the Models and Saving them ---"
python3 src/promo_code_model.py
python3 src/season_model.py
python3 src/recommendation.py

echo "--- Step 6: App ---"
python3 src/app.py

echo "✅ Pipeline finished successfully!"