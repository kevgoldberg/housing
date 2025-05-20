import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from config import SEED


def train_and_evaluate_gb(X_train, y_train, X_val, y_val):
    """Train a GradientBoostingRegressor and evaluate it on validation data."""
    gb = GradientBoostingRegressor(random_state=SEED)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_val)
    gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
    return gb, gb_pred, gb_rmse
