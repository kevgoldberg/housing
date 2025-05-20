"""
Enhanced visualizations for housing price prediction model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import os

# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Make output directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Load trained model and predictions (if available)
try:
    # Try to import trained model from main script
    from main import X_train, y_train, X_val, y_val, rf_model, numeric_features, rf_val_predictions
    model_loaded = True
    print("Successfully loaded model from main script")
except:
    model_loaded = False
    print("Could not load model from main script, will only show data visualizations")

# 1. Correlation Heatmap
def create_correlation_heatmap():
    print("Generating correlation heatmap...")
    numeric_data = train.select_dtypes(include=['int64', 'float64'])
    # Select subset of numeric columns with highest correlation to SalePrice
    corr_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    top_corr_features = corr_with_target.index[:15]  # Top 15 correlated features
    
    # Create correlation matrix for these features
    corr_matrix = train[top_corr_features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap of Top Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()
    print("Correlation heatmap saved to visualizations/correlation_heatmap.png")

# 2. Top Feature Scatter Plots
def create_feature_scatter_plots():
    print("Generating feature scatter plots...")
    # Get top 6 correlated features with SalePrice
    numeric_data = train.select_dtypes(include=['int64', 'float64'])
    corr_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    top_features = corr_with_target.index[1:7]  # Skip SalePrice itself
    
    # Create a 3x2 grid of scatter plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        sns.regplot(x=feature, y='SalePrice', data=train, ax=axes[i], 
                   scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        axes[i].set_title(f'SalePrice vs {feature}', fontsize=14)
        
    plt.tight_layout()
    plt.savefig('visualizations/top_feature_scatter_plots.png')
    plt.close()
    print("Feature scatter plots saved to visualizations/top_feature_scatter_plots.png")

# 3. Feature Distribution Plots
def create_feature_distribution_plots():
    print("Generating feature distribution plots...")
    # Get top 6 correlated features with SalePrice
    numeric_data = train.select_dtypes(include=['int64', 'float64'])
    corr_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    top_features = corr_with_target.index[1:7]  # Skip SalePrice itself
    
    # Create a 3x2 grid of histograms
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        sns.histplot(train[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}', fontsize=14)
        
    plt.tight_layout()
    plt.savefig('visualizations/feature_distributions.png')
    plt.close()
    print("Feature distribution plots saved to visualizations/feature_distributions.png")

# 4. Outlier Detection
def create_outlier_visualization():
    print("Generating outlier visualization...")
    # Looking at the most significant outliers
    if model_loaded:
        # Calculate residuals and absolute residuals
        residuals = y_val - rf_val_predictions
        abs_residuals = np.abs(residuals)
        
        # Get indices of top outliers
        outlier_indices = np.argsort(abs_residuals)[-20:]  # Top 20 outliers
        
        # Create dataframe of outliers with actual and predicted values
        outlier_df = pd.DataFrame({
            'Actual': y_val.iloc[outlier_indices],
            'Predicted': rf_val_predictions[outlier_indices],
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

# 5. Partial Dependence Plots
def create_partial_dependence_plots():
    if not model_loaded:
        print("Model not loaded, skipping partial dependence plots")
        return
    
    print("Generating partial dependence plots...")
    try:
        # Get feature importances
        feature_importances = pd.Series(rf_model.feature_importances_, index=numeric_features)
        top_features = feature_importances.sort_values(ascending=False)[:6].index.tolist()
        
        # Create partial dependence plots directly using feature names
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        PartialDependenceDisplay.from_estimator(
            rf_model, X_train, features=top_features[:6], 
            ax=ax.flatten()
        )
        plt.tight_layout()
        plt.savefig('visualizations/partial_dependence_plots.png')
        plt.close()
        print("Partial dependence plots saved to visualizations/partial_dependence_plots.png")
    except Exception as e:
        print(f"Error creating partial dependence plots: {e}")
        
        # Alternative approach: create individual feature importance plots
        print("Creating alternative feature importance visualization...")
        feature_importances = pd.Series(rf_model.feature_importances_, index=numeric_features)
        top_features = feature_importances.sort_values(ascending=False)[:10]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title('Top 10 Features by Importance', fontsize=16)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance_alt.png')
        plt.close()
        print("Alternative feature importance visualization saved to visualizations/feature_importance_alt.png")

# 6. Neighborhood comparison
def create_neighborhood_comparison():
    print("Generating neighborhood comparison...")
    if 'Neighborhood' in train.columns:
        # Average price by neighborhood
        neighborhood_prices = train.groupby('Neighborhood')['SalePrice'].agg(['mean', 'count', 'std'])
        neighborhood_prices = neighborhood_prices.sort_values('mean', ascending=False)
        
        # Only include neighborhoods with at least 10 samples
        neighborhood_prices = neighborhood_prices[neighborhood_prices['count'] >= 10]
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=neighborhood_prices.index, y='mean', data=neighborhood_prices, 
                   ci=None, palette='viridis')
        
        # Add error bars representing standard deviation
        plt.errorbar(
            x=np.arange(len(neighborhood_prices)), 
            y=neighborhood_prices['mean'], 
            yerr=neighborhood_prices['std'], 
            fmt='none', ecolor='black', capsize=5
        )
        
        plt.title('Average House Price by Neighborhood', fontsize=16)
        plt.xlabel('Neighborhood')
        plt.ylabel('Average Price ($)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('visualizations/neighborhood_price_comparison.png')
        plt.close()
        print("Neighborhood comparison saved to visualizations/neighborhood_price_comparison.png")
    else:
        print("Neighborhood column not found in data")

# Run all visualizations
if __name__ == "__main__":
    create_correlation_heatmap()
    create_feature_scatter_plots()
    create_feature_distribution_plots()
    create_outlier_visualization()
    create_partial_dependence_plots()
    create_neighborhood_comparison()
    
    print("\nAll visualizations completed and saved to the 'visualizations' folder")
