"""Simple Tkinter GUI for predicting housing prices.

This version avoids external dependencies like ``pandas`` and
``scikit-learn`` so that it can run in minimal environments.
The model is a basic linear regression trained using only the
Python standard library.
"""

import csv
import os
import sys
import tkinter as tk
from tkinter import ttk

# Features to use for the simple model
FEATURES = ["OverallQual", "GrLivArea", "GarageCars"]

def train_model():
    """Train a linear regression model using only the standard library."""
    with open("train.csv", newline="") as f:
        reader = csv.DictReader(f)
        X = []
        y = []
        for row in reader:
            try:
                X.append(
                    [
                        1.0,
                        float(row["OverallQual"]),
                        float(row["GrLivArea"]),
                        float(row["GarageCars"]),
                    ]
                )
                y.append(float(row["SalePrice"]))
            except ValueError:
                # Skip rows with missing data
                pass

    n_features = len(X[0])
    # Compute X^T X and X^T y
    xtx = [[0.0] * n_features for _ in range(n_features)]
    xty = [0.0] * n_features
    for row, target in zip(X, y):
        for i in range(n_features):
            xty[i] += row[i] * target
            for j in range(n_features):
                xtx[i][j] += row[i] * row[j]

    coef = _solve_linear_system(xtx, xty)
    return coef


def _solve_linear_system(matrix, vector):
    """Solve matrix * x = vector for x using Gauss-Jordan elimination."""
    n = len(matrix)
    # Create augmented matrix
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        # Find pivot
        pivot_row = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError("Matrix is singular")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        # Normalize pivot row
        for c in range(col, n + 1):
            aug[col][c] /= pivot
        # Eliminate other rows
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]

    return [aug[i][n] for i in range(n)]

# Pre-compute model coefficients when the module is imported
MODEL_COEFS = train_model()

# Tkinter GUI setup
if not os.environ.get("DISPLAY"):
    sys.stderr.write(
        "Error: No display found. Set the DISPLAY environment variable to run the GUI.\n"
    )
    sys.exit(1)

root = tk.Tk()
root.title("Housing Price Predictor")

entries = {}
for feature in FEATURES:
    frame = ttk.Frame(root)
    frame.pack(padx=10, pady=5, fill='x')
    label = ttk.Label(frame, text=feature)
    label.pack(side='left')
    entry = ttk.Entry(frame)
    entry.pack(side='left', fill='x', expand=True)
    entries[feature] = entry

result_var = tk.StringVar()
result_label = ttk.Label(root, textvariable=result_var)
result_label.pack(pady=10)

def predict():
    try:
        values = [float(entries[f].get()) for f in FEATURES]
    except ValueError:
        result_var.set('Please enter valid numeric values.')
        return
    features_with_intercept = [1.0] + values
    pred = sum(c * v for c, v in zip(MODEL_COEFS, features_with_intercept))
    result_var.set(f'Predicted SalePrice: ${pred:,.2f}')

button = ttk.Button(root, text='Predict', command=predict)
button.pack(pady=5)

root.mainloop()
