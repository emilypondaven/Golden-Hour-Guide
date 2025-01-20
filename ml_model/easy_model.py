# Machine Learning Pipeline for Sunrise Rating Prediction (Regression)

## Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt

filepath = 'ml_model/sunrise_data.csv'
features = ['low']
final = 'overall_rating'

## Step 2: Load and Inspect Data
def load_data(filepath):
    """Loads data from a CSV file."""
    df = pd.read_csv(filepath)
    print(df.head())
    print(df.info())
    return df

## Step 3: Preprocessing the Data
def preprocess_data(df):
    """Preprocesses the data by handling missing values and scaling features."""

    # Split features and target
    x = df[features]
    y = df[final]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    return X_scaled, y

## Step 4: Train-Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

## Step 5: Train Models
def train_linear_regression(X_train, y_train):
    """Trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def tune_random_forest(X_train, y_train):
    """Tunes Random Forest hyperparameters using RandomizedSearchCV."""
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 15),
        'min_samples_leaf': randint(1, 10)
    }
    rf = RandomForestRegressor(random_state=42)
    rand_search = RandomizedSearchCV(rf, param_distributions, n_iter=50, cv=5, scoring='r2', random_state=42, n_jobs=-1)
    rand_search.fit(X_train, y_train)
    print("Best Parameters:", rand_search.best_params_)
    return rand_search.best_estimator_


## Step 6: Evaluate Models
def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    y_pred = model.predict(X_test)
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
    print(f"R² Score: {r2_score(y_test, y_pred)}")
    return y_pred

## Step 7: Feature Importance
def plot_feature_importance(model, feature_names):
    """Plots feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        plt.barh(feature_names, importances)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.show()

## Step 8: Save and Load Model
def save_model(model, filepath):
    """Saves the model to a file."""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Loads the model from a file."""
    return joblib.load(filepath)

## Step 9: Main Pipeline
def main():

    # Load and preprocess data
    df = load_data(filepath)
    X, y = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    print("Training Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)

    print("Training Random Forest...")
    rf_model = tune_random_forest(X_train, y_train)

    # Evaluate models
    print("\nEvaluating Linear Regression:")
    evaluate_model(lr_model, X_test, y_test)

    print("\nEvaluating Random Forest:")
    evaluate_model(rf_model, X_test, y_test)

    # Feature Importance
    plot_feature_importance(rf_model, df[features].columns)

    # Save the best model
    save_model(rf_model, 'best_sunrise_model.pkl')
    print("Model saved as 'best_sunrise_model.pkl'.")

if __name__ == "__main__":
    main()