import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import os

# Import visualization modules
import data_visualizations as dv
import model_visualizations as mv

# Make output directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Run data visualizations (uncomment to generate all visualizations)
# print("\nGenerating data visualizations...")
# dv.run_all_visualizations(train)

# Explore target variable (SalePrice)
print("\nSalePrice statistics:")
print(train['SalePrice'].describe())

# Plot the SalePrice distribution using the model_visualizations module
mv.plot_saleprice_distribution(train)

# Check for missing values
print("\nMissing values in train set:")
missing_train = train.isnull().sum().sort_values(ascending=False)
print(missing_train[missing_train > 0].head(10))

print("\nMissing values in test set:")
missing_test = test.isnull().sum().sort_values(ascending=False)
print(missing_test[missing_test > 0].head(10))

# Feature selection - start with numeric columns for simplicity
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
numeric_features = numeric_features.drop('Id')
if 'SalePrice' in numeric_features:
    numeric_features = numeric_features.drop('SalePrice')

print(f"\nSelected {len(numeric_features)} numeric features")

# Fill missing values with median
X_train = train[numeric_features].fillna(train[numeric_features].median())
y_train = train['SalePrice']
X_test = test[numeric_features].fillna(train[numeric_features].median())

# Feature correlation with target
correlation_data = train[list(numeric_features) + ['SalePrice']]
correlations = correlation_data.corr()['SalePrice'].sort_values(ascending=False)
print("\nTop 10 correlations with SalePrice:")
print(correlations.head(10))

# Split training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Train Random Forest model for comparison
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)

# Evaluate Random Forest on validation set
rf_val_predictions = rf_model.predict(X_val)
rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_predictions))
print(f"Random Forest Validation RMSE: {rf_val_rmse}")

# Create a pipeline with normalization and regularized model (Ridge Regression)
print("\nTraining Ridge Regression model with normalization...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalize features
    ('ridge', Ridge())             # Ridge regression (L2 regularization)
])

# Define hyperparameters to search
param_grid = {
    'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]  # Regularization strength
}

# Find the best hyperparameters using grid search and cross-validation
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, 
    scoring='neg_root_mean_squared_error',
    verbose=1
)
grid_search.fit(X_train_split, y_train_split)

# Print the best parameters
print(f"\nBest Ridge parameters: {grid_search.best_params_}")
print(f"Best cross-validation score (negative RMSE): {grid_search.best_score_}")

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on validation set
ridge_val_predictions = best_model.predict(X_val)
ridge_val_rmse = np.sqrt(mean_squared_error(y_val, ridge_val_predictions))
print(f"Ridge Regression Validation RMSE: {ridge_val_rmse}")

# Compare models
print("\nModel Comparison:")
print(f"Random Forest RMSE: {rf_val_rmse}")
print(f"Ridge Regression RMSE: {ridge_val_rmse}")

# Create visualizations of model predictions vs actual values
rf_residuals, ridge_residuals = mv.plot_model_predictions(y_val, rf_val_predictions, ridge_val_predictions)

# Create a histogram of residuals for the better model
mv.plot_residuals_distribution(rf_residuals, ridge_residuals, rf_val_rmse, ridge_val_rmse)

# Create feature importance visualization if using Random Forest
if ridge_val_rmse >= rf_val_rmse:
    # Feature importance is created below when choosing the model
    pass

# Choose the better model for final prediction
if ridge_val_rmse < rf_val_rmse:
    print("\nUsing Ridge Regression for final predictions (better performance)")
    final_model = best_model
    feature_importance = None  # Ridge doesn't have feature importances like RF
else:
    print("\nUsing Random Forest for final predictions (better performance)")
    final_model = rf_model
    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nTop 10 important features:")
    print(feature_importance.head(10))
    
    # Create feature importance visualization
    feature_importance = mv.plot_feature_importance(rf_model, numeric_features)

# Make predictions on test data using the selected model
test_predictions = final_model.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)

# Generate additional model visualizations (uncomment to generate)
# if ridge_val_rmse < rf_val_rmse:
#     model_for_viz = best_model
#     preds_for_viz = ridge_val_predictions
# else:
#     model_for_viz = rf_model
#     preds_for_viz = rf_val_predictions
#     mv.plot_partial_dependence(rf_model, X_train, numeric_features)
#     
# mv.plot_outlier_analysis(y_val, preds_for_viz)

print("\nSubmission file created!")
