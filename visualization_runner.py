"""
Housing price prediction visualization runner module.
This module provides a unified interface to run all visualizations.
"""
import pandas as pd
import numpy as np
import os
import data_visualizations as dv
import model_visualizations as mv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def run_all_data_visualizations(train_data):
    """Run all data-related visualizations"""
    print("Running all data visualizations...")
    dv.run_all_visualizations(train_data)

def run_basic_model_visualizations(train_data):
    """Run basic model-related visualizations that don't require a trained model"""
    print("Running basic model visualizations...")
    # Target variable distribution
    mv.plot_saleprice_distribution(train_data)

def run_model_performance_visualizations(y_val, model, X_val, X_train, numeric_features):
    """Run all model performance visualizations"""
    print("Running model performance visualizations...")
    
    # Get predictions
    predictions = model.predict(X_val)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f"Model RMSE: {rmse}")
    
    # Generate dummy Ridge predictions for comparison (or just use the same predictions twice)
    dummy_predictions = predictions * 0.95  # Just to have something to compare with
    dummy_rmse = np.sqrt(mean_squared_error(y_val, dummy_predictions))
    
    # Generate visualizations
    residuals, _ = mv.plot_model_predictions(y_val, predictions, dummy_predictions)
    mv.plot_residuals_distribution(residuals, residuals, rmse, dummy_rmse)
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        mv.plot_feature_importance(model, numeric_features)

def run_quick_model_test(train_data, test_size=0.2):
    """Run a quick model test and visualize results"""
    print("Running quick model test...")
    
    # Feature selection - numeric columns only for simplicity
    numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = numeric_features.drop('Id')
    if 'SalePrice' in numeric_features:
        numeric_features = numeric_features.drop('SalePrice')
    
    # Fill missing values with median
    X = train_data[numeric_features].fillna(train_data[numeric_features].median())
    y = train_data['SalePrice']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train a simple Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Run visualizations
    run_model_performance_visualizations(y_val, model, X_val, X_train, numeric_features)
    
    return model

if __name__ == "__main__":
    # Make output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Load data
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    # Run data visualizations
    run_all_data_visualizations(train)
    
    # Run a quick model test with visualizations
    model = run_quick_model_test(train)
    
    print("All visualizations completed!")
