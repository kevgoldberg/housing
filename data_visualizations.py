"""
Data visualization module for exploring housing price dataset.
This module provides functions to visualize data relationships, distributions, and patterns.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Make output directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

def create_correlation_heatmap(train_data):
    """Generate correlation heatmap of numeric features"""
    print("Generating correlation heatmap...")
    numeric_data = train_data.select_dtypes(include=['int64', 'float64'])
    # Select subset of numeric columns with highest correlation to SalePrice
    corr_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    top_corr_features = corr_with_target.index[:15]  # Top 15 correlated features
    
    # Create correlation matrix for these features
    corr_matrix = train_data[top_corr_features].corr()
    
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

def create_feature_scatter_plots(train_data):
    """Create scatter plots of top correlated features with SalePrice"""
    print("Generating feature scatter plots...")
    # Get top 6 correlated features with SalePrice
    numeric_data = train_data.select_dtypes(include=['int64', 'float64'])
    corr_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    top_features = corr_with_target.index[1:7]  # Skip SalePrice itself
    
    # Create a 3x2 grid of scatter plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        sns.regplot(x=feature, y='SalePrice', data=train_data, ax=axes[i], 
                   scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        axes[i].set_title(f'SalePrice vs {feature}', fontsize=14)
        
    plt.tight_layout()
    plt.savefig('visualizations/top_feature_scatter_plots.png')
    plt.close()
    print("Feature scatter plots saved to visualizations/top_feature_scatter_plots.png")

def create_feature_distribution_plots(train_data):
    """Create distribution plots of top correlated features"""
    print("Generating feature distribution plots...")
    # Get top 6 correlated features with SalePrice
    numeric_data = train_data.select_dtypes(include=['int64', 'float64'])
    corr_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    top_features = corr_with_target.index[1:7]  # Skip SalePrice itself
    
    # Create a 3x2 grid of histograms
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        sns.histplot(train_data[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}', fontsize=14)
        
    plt.tight_layout()
    plt.savefig('visualizations/feature_distributions.png')
    plt.close()
    print("Feature distribution plots saved to visualizations/feature_distributions.png")

def create_neighborhood_comparison(train_data):
    """Create bar chart of average price by neighborhood"""
    print("Generating neighborhood comparison...")
    if 'Neighborhood' in train_data.columns:
        # Average price by neighborhood
        neighborhood_prices = train_data.groupby('Neighborhood')['SalePrice'].agg(['mean', 'count', 'std'])
        neighborhood_prices = neighborhood_prices.sort_values('mean', ascending=False)
        
        # Only include neighborhoods with at least 10 samples
        neighborhood_prices = neighborhood_prices[neighborhood_prices['count'] >= 10]
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=neighborhood_prices.index, y='mean', data=neighborhood_prices)
        
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

def run_all_visualizations(train_data):
    """Run all data visualizations"""
    create_correlation_heatmap(train_data)
    create_feature_scatter_plots(train_data)
    create_feature_distribution_plots(train_data)
    create_neighborhood_comparison(train_data)
    
    print("\nAll data visualizations completed and saved to the 'visualizations' folder")

if __name__ == "__main__":
    # Load the data
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    run_all_visualizations(train)
