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

def create_price_by_age(train_data):
    """Create boxplot of price by house age groups"""
    print("Generating price by house age plot...")
    if 'YearBuilt' in train_data.columns:
        # Create age bands
        current_year = 2025  # Using current year 
        age_data = train_data.copy()
        age_data['HouseAge'] = current_year - age_data['YearBuilt']
        
        age_bins = [0, 10, 20, 40, 60, 80, 100, 150]
        age_labels = ['0-10', '11-20', '21-40', '41-60', '61-80', '81-100', '100+']
        
        age_data['AgeGroup'] = pd.cut(age_data['HouseAge'], bins=age_bins, labels=age_labels)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='AgeGroup', y='SalePrice', data=age_data)
        plt.title('House Price Distribution by Age Group', fontsize=16)
        plt.xlabel('House Age (years)')
        plt.ylabel('Sale Price ($)')
        plt.tight_layout()
        plt.savefig('visualizations/price_by_age.png')
        plt.close()
        print("Price by house age plot saved to visualizations/price_by_age.png")
    else:
        print("YearBuilt column not found in data")

def create_quality_condition_plot(train_data):
    """Create heatmap of price by quality and condition ratings"""
    print("Generating quality and condition plot...")
    if 'OverallQual' in train_data.columns and 'OverallCond' in train_data.columns:
        # Create a pivot table of average price by quality and condition
        pivot = train_data.pivot_table(
            values='SalePrice', 
            index='OverallQual', 
            columns='OverallCond', 
            aggfunc='mean'
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
        plt.title('Average House Price by Quality and Condition', fontsize=16)
        plt.xlabel('Overall Condition (1-10)')
        plt.ylabel('Overall Quality (1-10)')
        plt.tight_layout()
        plt.savefig('visualizations/price_by_quality_condition.png')
        plt.close()
        print("Quality and condition plot saved to visualizations/price_by_quality_condition.png")
    else:
        print("OverallQual or OverallCond columns not found in data")

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

def create_area_price_by_neighborhood(train_data):
    """Create scatter plot of living area vs price by neighborhood"""
    print("Generating living area vs price by neighborhood plot...")
    if 'GrLivArea' in train_data.columns and 'Neighborhood' in train_data.columns:
        # Get top 9 neighborhoods by count
        top_neighborhoods = train_data['Neighborhood'].value_counts().head(9).index
        
        # Filter data
        filtered_data = train_data[train_data['Neighborhood'].isin(top_neighborhoods)]
        
        # Plot
        plt.figure(figsize=(14, 10))
        sns.scatterplot(
            x='GrLivArea', 
            y='SalePrice', 
            hue='Neighborhood', 
            data=filtered_data,
            palette='tab10',
            alpha=0.7
        )
        plt.title('Sale Price vs Living Area by Neighborhood', fontsize=16)
        plt.xlabel('Above Ground Living Area (square feet)')
        plt.ylabel('Sale Price ($)')
        plt.legend(title='Neighborhood')
        plt.tight_layout()
        plt.savefig('visualizations/price_vs_area_by_neighborhood.png')
        plt.close()
        print("Living area vs price by neighborhood plot saved to visualizations/price_vs_area_by_neighborhood.png")
    else:
        print("GrLivArea or Neighborhood columns not found in data")

def create_price_by_building_type(train_data):
    """Create boxplot of price by building type"""
    print("Generating price by building type plot...")
    if 'BldgType' in train_data.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='BldgType', y='SalePrice', data=train_data)
        plt.title('House Price Distribution by Building Type', fontsize=16)
        plt.xlabel('Building Type')
        plt.ylabel('Sale Price ($)')
        plt.tight_layout()
        plt.savefig('visualizations/price_by_building_type.png')
        plt.close()
        print("Price by building type plot saved to visualizations/price_by_building_type.png")
    else:
        print("BldgType column not found in data")

def run_all_visualizations(train_data):
    """Run all data visualizations"""
    create_correlation_heatmap(train_data)
    create_feature_scatter_plots(train_data)
    create_feature_distribution_plots(train_data)
    create_price_by_age(train_data)
    create_quality_condition_plot(train_data)
    create_neighborhood_comparison(train_data)
    create_area_price_by_neighborhood(train_data)
    create_price_by_building_type(train_data)
    
    print("\nAll data visualizations completed and saved to the 'visualizations' folder")

if __name__ == "__main__":
    # Load the data
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    run_all_visualizations(train)
