import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import SEED

def train_and_evaluate_rf(X_train, y_train, X_val, y_val):
    """
    Trains a RandomForestRegressor model and evaluates it.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        A tuple containing the trained model, predictions on validation set, and RMSE.
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
    return rf, rf_pred, rf_rmse
