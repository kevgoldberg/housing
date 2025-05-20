"""
Model training, evaluation, and inference
"""
import numpy as np
from sklearn.model_selection import train_test_split
from config import SEED
from housing.data import load_data, preprocess_numeric
from housing.visualization.runner import run_model_visualizations
from housing.models.random_forest import train_and_evaluate_rf
from housing.models.ridge import train_and_evaluate_ridge


def train_and_evaluate(test_size=0.2):
    train, test = load_data()
    X_train_all, y_train_all, X_test, numeric_features = preprocess_numeric(train, test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=test_size, random_state=SEED
    )

    # Random Forest baseline
    rf, rf_pred, rf_rmse = train_and_evaluate_rf(X_train, y_train, X_val, y_val)

    # Ridge with normalization & grid search
    best_ridge, ridge_pred, ridge_rmse = train_and_evaluate_ridge(X_train, y_train, X_val, y_val)

    # Visualize
    run_model_visualizations(
        y_val,
        rf_pred,
        ridge_pred,
        rf_rmse,
        ridge_rmse,
        rf,
        best_ridge,  # Use the best ridge estimator
        X_train_all,
        numeric_features
    )

    # Choose final model
    final = best_ridge if ridge_rmse < rf_rmse else rf
    return final, X_test


def main():
    model, X_test = train_and_evaluate()
    preds = model.predict(X_test)
    import pandas as pd
    from config import TEST_PATH
    test_df = pd.read_csv(TEST_PATH)
    submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': preds})
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")


if __name__ == '__main__':
    main()
