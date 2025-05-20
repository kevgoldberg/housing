# Housing Price Prediction Project

This project implements machine learning models to predict housing prices based on various features of residential properties.

## Project Structure

The project is organized into the following modules:

- `main.py`: The main script that performs data loading, preprocessing, model training, evaluation, and inference.
- `data_visualizations.py`: Module containing functions for visualizing the housing data.
- `model_visualizations.py`: Module containing functions for visualizing model performance and predictions.
- `visualization_runner.py`: Unified interface to run all visualizations in one go.
- `enhanced_visualizations.py`: Additional complex visualizations (legacy).

## Getting Started

1. Install the required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Run the main script to train models and generate predictions:
```bash
python main.py
```

3. Alternatively, just run the visualizations:
```bash
python visualization_runner.py
```

## Data Visualization Functions

The `data_visualizations.py` module provides the following visualizations:

- Correlation heatmap of top features
- Scatter plots of top correlated features with the target variable
- Distribution plots of top features
- Price by house age
- Price by quality and condition ratings
- Neighborhood price comparison
- Living area vs. price by neighborhood
- Price by building type

## Model Visualization Functions

The `model_visualizations.py` module provides the following visualizations:

- Target variable (SalePrice) distribution
- Model predictions vs. actual values
- Residuals analysis
- Feature importance for tree-based models
- Outlier analysis
- Partial dependence plots

## Directory Structure

- `/`: Root directory containing Python scripts and CSV data files
- `/visualizations`: Directory containing generated visualization images

## Models Used

- Random Forest Regression
- Ridge Regression with hyperparameter tuning

