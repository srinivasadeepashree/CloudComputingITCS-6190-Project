**ITCS 6190 Course Project**  
**Team Members**: Srinivasa Deepashree, Vasudha Maganti, Sasi Vadana Atluri, Sai Tarun Vedagiri, Manish Yendeti

## Project Overview

This project implements a complete big data analytics pipeline using Apache Spark to analyze shopping trends data. The pipeline integrates Spark's Structured APIs, Spark SQL, Streaming capabilities, and MLlib to process, analyze, and model customer purchasing behaviors.

## Project Structure


```
shopping-trends-analysis/
├── data/
│   ├── shopping_trends.csv             # Raw dataset (sample)
│   └── sample/                         # Sample data (Parquet)
│   └── processed/                      # Processed data (Parquet)
├── src/
│   ├── data_ingestion.py               # Data loading and validation
│   └──  eda_analysis.py                # Exploratory Data Analysis 
|            
├── notebooks/
│   └── 01_data_exploration.ipynb       # Interactive EDA
├── tests/
├── docs/
├── output/
│   └── figures/                        # EDA plots
├── requirements.txt                    # Python dependencies
├── run.sh                              # Main execution script
└── README.md                           # This file
```
## Milestone 1: Data Ingestion + EDA ✓

### Completed Deliverables

#### 1. Data Ingestion Module (`src/data_ingestion.py`)
- ✅ Spark session configuration for local execution
- ✅ Schema definition with proper data types
- ✅ CSV data loading with validation
- ✅ Data quality checks (duplicates, nulls)
- ✅ Basic statistics computation
- ✅ Processed data export to Parquet format

#### 2. Exploratory Data Analysis (`src/eda_analysis.py`)
- ✅ Customer Demographics Analysis
- ✅ Purchase Distribution by Category
- ✅ Revenue Analysis

#### 3. Key Findings from EDA

**Dataset Overview**:
- Total Records: 3,900 transactions
- Unique Customers: ~3,900
- Total Revenue: 233081
- Average Transaction Value: 59.76 

**Customer Demographics**:
- Age Range: 18-70 years
- Gender Distribution: Male/Female
- Geographic Coverage: Multiple locations

**Purchase Patterns**:
- Categories: Clothing, Accessories, Footwear, Outerwear
- Seasonal Trends: Winter, Spring, Summer, Fall
  

**Revenue Metrics**:
- total revenue obtained
- average transactions made
- Able to figure out the no.of unique customers
- Customer Demographic Analysis

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd shopping-trends-analysis
```

2. **Create virtual environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up data directory**
```bash
mkdir -p data/processed
# Place shopping_trends_updated.csv in data/ directory
```

5. **Verify Java installation** (required for Spark)
```bash
java -version  # Should show Java 8 or 11
```

## Running the Pipeline

### Quick Start (Automated)

Run the complete pipeline for Milestone 1:

```bash
chmod +x run.sh
./run.sh
```

This will:
1. Validate environment and dependencies
2. Run data ingestion and validation
3. Execute comprehensive EDA
4. Generate analysis reports
5. Save processed data

### Manual Execution

**Step 1: Data Ingestion**
```bash
python3 src/data_ingestion.py
```

**Step 2: Exploratory Data Analysis**
```bash
python3 src/eda_analysis.py
```

**Step 3: Interactive Exploration (Optional)**
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

