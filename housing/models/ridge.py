import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from housing.config import SEED, ALPHA_GRID

def train_and_evaluate_ridge(X_train, y_train, X_val, y_val):
    """
    Trains a Ridge regression model with StandardScaler and GridSearchCV and evaluates it.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        A tuple containing the best trained model, predictions on validation set, and RMSE.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(random_state=SEED))
    ])
    grid = GridSearchCV(
        pipeline,
        {'ridge__alpha': ALPHA_GRID},
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_
    ridge_pred = best_estimator.predict(X_val)
    ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))
    return best_estimator, ridge_pred, ridge_rmse
