import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk
from tkinter import ttk

# Features to use for the simple model
FEATURES = ['OverallQual', 'GrLivArea', 'GarageCars']

def train_model():
    """Train a simple model using selected features."""
    data = pd.read_csv('train.csv')
    X = data[FEATURES]
    y = data['SalePrice']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Tkinter GUI setup
root = tk.Tk()
root.title('Housing Price Predictor')

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
    sample = pd.DataFrame([values], columns=FEATURES)
    pred = model.predict(sample)[0]
    result_var.set(f'Predicted SalePrice: ${pred:,.2f}')

button = ttk.Button(root, text='Predict', command=predict)
button.pack(pady=5)

root.mainloop()
