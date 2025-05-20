"""
Data visualization functions for housing price project.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from housing.config import VIS_DIR

# Ensure visualization directory exists
os.makedirs(VIS_DIR, exist_ok=True)

def create_correlation_heatmap(train_data):
    """Generate and save a correlation heatmap of top numeric features."""
    numeric_data = train_data.select_dtypes(include=['int64', 'float64'])
    corr_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    top_features = corr_with_target.index[:15]
    corr_matrix = train_data[top_features].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5, cbar_kws={'shrink':0.8})
    plt.title('Correlation Heatmap of Top Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'correlation_heatmap.png'))
    plt.close()

# ...existing code for other functions...
def create_feature_scatter_plots(train_data):
    """Scatter plots of top correlated features vs SalePrice."""
    numeric_data = train_data.select_dtypes(include=['int64', 'float64'])
    corr_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    top_feats = corr_with_target.index[1:7]
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()
    for i, feat in enumerate(top_feats):
        sns.regplot(x=feat, y='SalePrice', data=train_data, ax=axes[i], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        axes[i].set_title(f'SalePrice vs {feat}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'top_feature_scatter_plots.png'))
    plt.close()

def create_feature_distribution_plots(train_data):
    """Histograms of top correlated numeric features."""
    numeric_data = train_data.select_dtypes(include=['int64', 'float64'])
    corr_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    top_feats = corr_with_target.index[1:7]
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()
    for i, feat in enumerate(top_feats):
        sns.histplot(train_data[feat], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feat}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'feature_distributions.png'))
    plt.close()

def run_all_visualizations(train_data):
    """Run all data visualizations and save to disk."""
    create_correlation_heatmap(train_data)
    create_feature_scatter_plots(train_data)
    create_feature_distribution_plots(train_data)
    # other visualizations can be added here
    print(f"All data visualizations saved to '{VIS_DIR}'")
