"""
Data loading and preprocessing functions
"""
import pandas as pd
from config import TRAIN_PATH, TEST_PATH

def load_data():
    """Load training and test datasets from CSV files."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def preprocess_numeric(train, test):
    """
    Select numeric features, drop highly‐collinear ones, impute missing values with median,
    and split into X_train, y_train, X_test.
    Returns features list for modeling.
    """
    import numpy as np
    # initial numeric feature selection
    numeric = train.select_dtypes(include=['int64', 'float64']) \
                   .columns.drop(['Id', 'SalePrice'], errors='ignore')
    # copy for processing
    X_train = train[numeric].copy()
    y_train = train['SalePrice']
    X_test = test[numeric].copy()

    # drop multicollinear features (|r| > 0.9)
    corr = pd.concat([X_train, y_train], axis=1).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    if to_drop:
        print(f"Dropping {len(to_drop)} features with |r|>0.9: {to_drop}")
        X_train.drop(columns=to_drop, inplace=True)
        X_test.drop(columns=to_drop, inplace=True)
        numeric = [c for c in numeric if c not in to_drop]

    # impute missing values with median
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    return X_train, y_train, X_test, numeric
