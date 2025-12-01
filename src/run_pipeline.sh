#!/bin/bash

# 1. Stop the script immediately if any command fails
set -e

echo "🚀 Starting the Pipeline..."

# # 2. Run your files in order
# echo "--- Step 1: Running Data Preparation ---"
# python3 data_ingestion.py

# echo 
# echo "--- Step 2: Training the Model ---"
# python3 train_model.py

# echo "--- Step 3: Generating Recommendations ---"
# python3 recommend.py

echo "--- Step : App ---"
python3 src/app.py

echo "✅ Pipeline finished successfully!"