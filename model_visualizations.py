"""
Model visualization module for housing price prediction models.
This module provides functions to visualize model performance, predictions, and feature importance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
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

def plot_outlier_analysis(y_val, predictions):
    """Analyze and visualize prediction outliers"""
    print("Generating outlier visualization...")
    # Calculate residuals and absolute residuals
    residuals = y_val - predictions
    abs_residuals = np.abs(residuals)
    
    # Get indices of top outliers
    outlier_indices = np.argsort(abs_residuals)[-20:]  # Top 20 outliers
    
    # Create dataframe of outliers with actual and predicted values
    outlier_df = pd.DataFrame({
        'Actual': y_val.iloc[outlier_indices],
        'Predicted': predictions[outlier_indices],
        'Residual': residuals.iloc[outlier_indices],
    })
    
    # Sort by absolute residual
    outlier_df = outlier_df.sort_values('Residual', key=abs)
    
    # Plot
    plt.figure(figsize=(12, 8))
    indices = np.arange(len(outlier_df))
    width = 0.4
    
    plt.bar(indices - width/2, outlier_df['Actual'], width, label='Actual')
    plt.bar(indices + width/2, outlier_df['Predicted'], width, label='Predicted')
    
    plt.title('Top 20 Prediction Outliers: Actual vs Predicted Values', fontsize=16)
    plt.xlabel('Outlier Index')
    plt.ylabel('Sale Price ($)')
    plt.legend()
    plt.xticks([])  # Hide x-axis labels for cleaner look
    plt.tight_layout()
    plt.savefig('visualizations/prediction_outliers.png')
    plt.close()
    print("Outlier visualization saved to visualizations/prediction_outliers.png")

def plot_partial_dependence(model, X_train, numeric_features):
    """Create partial dependence plots to show relationships learned by the model"""
    print("Generating partial dependence plots...")
    try:
        # Get feature importances
        feature_importances = pd.Series(model.feature_importances_, index=numeric_features)
        top_features = feature_importances.sort_values(ascending=False)[:6].index.tolist()
        
        # Create partial dependence plots directly using feature names
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        PartialDependenceDisplay.from_estimator(
            model, X_train, features=top_features[:6], 
            ax=ax.flatten()
        )
        plt.tight_layout()
        plt.savefig('visualizations/partial_dependence_plots.png')
        plt.close()
        print("Partial dependence plots saved to visualizations/partial_dependence_plots.png")
    except Exception as e:
        print(f"Error creating partial dependence plots: {e}")
