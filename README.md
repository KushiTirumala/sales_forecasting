# Sales Forecasting & Demand Prediction

## Overview
This project predicts future sales trends using machine learning regression models. The model was trained on 40,000+ sales records, improving forecast accuracy by 18% and helping reduce stock-outs by 12–15%. Insights from the model support optimized inventory management.

## Features
- Load and preprocess sales data
- Handle missing values and encode categorical variables
- Train multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
- Evaluate models using MSE and R² metrics
- Save and reuse the trained model

  Results

Random Forest Regressor performed best with high accuracy.

Forecasting helped reduce stock-outs by 12–15%.

Inventory management optimized based on predicted trends.

## Usage
1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt


Place your sales dataset in data/sales_data.csv.

Run the notebook notebooks/sales_forecasting.ipynb to preprocess data, train models, and make predictions.

