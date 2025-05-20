"""
Model performance visualization functions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from config import VIS_DIR

os.makedirs(VIS_DIR, exist_ok=True)

def plot_saleprice_distribution(train_data):
    plt.figure(figsize=(10,6))
    sns.histplot(train_data['SalePrice'], kde=True)
    plt.title('SalePrice Distribution')
    plt.savefig(os.path.join(VIS_DIR, 'saleprice_distribution.png'))
    plt.close()

def plot_prediction_comparison(y_val, rf_pred, ridge_pred):
    fig, axes = plt.subplots(2,2,figsize=(10,8))
    axes = axes.flatten()
    axes[0].scatter(y_val, rf_pred, alpha=0.5)
    axes[0].plot([y_val.min(), y_val.max()],[y_val.min(), y_val.max()],'k--')
    axes[0].set_title('RF: Actual vs Predicted')
    axes[1].scatter(y_val, ridge_pred, alpha=0.5)
    axes[1].plot([y_val.min(), y_val.max()],[y_val.min(), y_val.max()],'k--')
    axes[1].set_title('Ridge: Actual vs Predicted')
    rf_res = y_val - rf_pred
    axes[2].scatter(rf_pred, rf_res, alpha=0.5)
    axes[2].axhline(0, linestyle='--')
    axes[2].set_title('RF Residuals')
    ridge_res = y_val - ridge_pred
    axes[3].scatter(ridge_pred, ridge_res, alpha=0.5)
    axes[3].axhline(0, linestyle='--')
    axes[3].set_title('Ridge Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR,'prediction_comparison.png'))
    plt.close()
    return rf_res, ridge_res

def plot_residuals_distribution(rf_res, ridge_res, rf_rmse, ridge_rmse):
    plt.figure(figsize=(10,6))
    data, title = (ridge_res, 'Ridge') if ridge_rmse<rf_rmse else (rf_res, 'RandomForest')
    plt.hist(data, bins=50, alpha=0.7)
    plt.title(f'{title} Residuals Distribution')
    plt.axvline(0, linestyle='--')
    plt.savefig(os.path.join(VIS_DIR,'residuals_distribution.png'))
    plt.close()

def plot_feature_importance(model, features):
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(12,8))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR,'feature_importance.png'))
    plt.close()
    return importances

def plot_partial_dependence(model, X_train, features):
    top = plot_feature_importance(model, features).index[:6]
    fig, ax = plt.subplots(2,3,figsize=(18,10))
    PartialDependenceDisplay.from_estimator(model, X_train, top, ax=ax.flatten())
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR,'partial_dependence.png'))
    plt.close()

def plot_outliers(y_val, pred):
    res = y_val - pred
    idx = np.argsort(np.abs(res))[-20:]
    df = pd.DataFrame({'Actual': y_val.iloc[idx], 'Pred': pred[idx], 'Res': res.iloc[idx]})
    df = df.sort_values('Res', key=abs)
    plt.figure(figsize=(12,8))
    i = np.arange(len(df))
    plt.bar(i-0.2, df['Actual'], width=0.4, label='Actual')
    plt.bar(i+0.2, df['Pred'], width=0.4, label='Pred')
    plt.legend()
    plt.title('Top 20 Outliers')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR,'outliers.png'))
    plt.close()
