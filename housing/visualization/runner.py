"""
CLI entrypoint for data & model visualizations
"""
import click
import pandas as pd
from housing.data import load_data
from housing.visualization.data import run_all_visualizations
from housing.visualization.model import (
    plot_saleprice_distribution,
    plot_prediction_comparison,
    plot_residuals_distribution,
    plot_feature_importance,
    plot_partial_dependence,
    plot_outliers,
)


def run_model_visualizations(
    y_val,
    rf_pred,
    ridge_pred,
    rf_rmse,
    ridge_rmse,
    rf_model,
    ridge_model,
    X_train,
    numeric_features,
):
    """Generate and save all model-related visualizations."""

    # Base visualizations comparing both models
    plot_saleprice_distribution(pd.DataFrame({"SalePrice": y_val}))
    rf_res, ridge_res = plot_prediction_comparison(y_val, rf_pred, ridge_pred)
    plot_residuals_distribution(rf_res, ridge_res, rf_rmse, ridge_rmse)

    # Determine which model performed better
    best_model = ridge_model if ridge_rmse < rf_rmse else rf_model
    best_pred = ridge_pred if ridge_rmse < rf_rmse else rf_pred

    # Feature importance and PDP only make sense for models with the attribute
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, numeric_features)
        plot_partial_dependence(best_model, X_train, numeric_features)

    # Outlier analysis for the chosen model
    plot_outliers(y_val, best_pred)

@click.group()
@click.pass_context
def main(ctx):
    """Visualization CLI for housing project"""
    ctx.ensure_object(dict)

@main.command()
@click.option('--all', 'do_all', is_flag=True, help='Run all data visualizations')
def data(do_all):
    """Run data visualizations"""
    train, _ = load_data()
    run_all_visualizations(train)

@main.command()
@click.option('--rf', 'rf_pred', help='RF predictions file (npy)')
@click.option('--ridge', 'ridge_pred', help='Ridge predictions file (npy)')
@click.option('--yval', 'y_val', help='Actual values file (npy)')
@click.pass_context
def model(ctx, rf_pred, ridge_pred, y_val):
    """Run model visualizations"""
    import numpy as np
    rf = np.load(rf_pred)
    ridge = np.load(ridge_pred)
    y = np.load(y_val)
    plot_saleprice_distribution(pd.DataFrame({'SalePrice': y}))
    rf_res, ridge_res = plot_prediction_comparison(y, rf, ridge)
    plot_residuals_distribution(rf_res, ridge_res, None, None)
    # further plots...
