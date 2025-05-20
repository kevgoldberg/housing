"""
Model visualization module for housing price prediction models.
This module provides functions to visualize model performance, predictions, and feature importance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import os

# Make output directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

def plot_saleprice_distribution(train_data):
    """Plot the distribution of the target variable (SalePrice)"""
    print("Generating SalePrice distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data['SalePrice'], kde=True)
    plt.title('SalePrice Distribution')
    plt.savefig('saleprice_distribution.png')
    plt.close()
    print("SalePrice distribution plot saved")

def plot_model_predictions(y_val, rf_val_predictions, ridge_val_predictions):
    """Create comparison plots of model predictions vs actual values"""
    print("Generating model prediction comparison plots...")
    plt.figure(figsize=(10, 8))

    # Subplot 1: Random Forest predictions vs actual values
    plt.subplot(2, 2, 1)
    plt.scatter(y_val, rf_val_predictions, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.title('Random Forest: Actual vs Predicted')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.tight_layout()

    # Subplot 2: Ridge predictions vs actual values
    plt.subplot(2, 2, 2)
    plt.scatter(y_val, ridge_val_predictions, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.title('Ridge Regression: Actual vs Predicted')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.tight_layout()

    # Subplot 3: Random Forest Residuals
    plt.subplot(2, 2, 3)
    rf_residuals = y_val - rf_val_predictions
    plt.scatter(rf_val_predictions, rf_residuals, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Random Forest: Residuals')
    plt.xlabel('Predicted Sale Price')
    plt.ylabel('Residuals')
    plt.tight_layout()

    # Subplot 4: Ridge Residuals
    plt.subplot(2, 2, 4)
    ridge_residuals = y_val - ridge_val_predictions
    plt.scatter(ridge_val_predictions, ridge_residuals, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Ridge Regression: Residuals')
    plt.xlabel('Predicted Sale Price')
    plt.ylabel('Residuals')
    plt.tight_layout()

    plt.savefig('visualizations/prediction_comparison.png')
    plt.close()
    print("Model prediction comparison plots saved to visualizations/prediction_comparison.png")
    
    return rf_residuals, ridge_residuals

def plot_residuals_distribution(rf_residuals, ridge_residuals, rf_val_rmse, ridge_val_rmse):
    """Create a histogram of residuals for the better model"""
    print("Generating residuals distribution plot...")
    plt.figure(figsize=(10, 6))
    if ridge_val_rmse < rf_val_rmse:
        plt.hist(ridge_residuals, bins=50, alpha=0.7)
        plt.title('Ridge Regression: Distribution of Residuals')
    else:
        plt.hist(rf_residuals, bins=50, alpha=0.7)
        plt.title('Random Forest: Distribution of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.savefig('visualizations/residuals_distribution.png')
    plt.close()
    print("Residuals distribution plot saved to visualizations/residuals_distribution.png")

def plot_feature_importance(model, numeric_features):
    """Plot feature importance for tree-based models like Random Forest"""
    print("Generating feature importance plot...")
    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Create feature importance visualization
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 Features by Importance')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()
    print("Feature importance plot saved to visualizations/feature_importance.png")
    
    return feature_importance

