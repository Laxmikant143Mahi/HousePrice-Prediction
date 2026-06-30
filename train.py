"""
California House Price Prediction API - Model Training (train.py)

This script trains a RandomForestRegressor model using the California Housing dataset.
It performs the following steps:
1. Loads the dataset.
2. Splits the data into training (80%) and testing (20%) sets.
3. Fits a RandomForestRegressor model with n_estimators=100.
4. Evaluates the model using Mean Absolute Error (MAE) and R-squared (R2) score.
5. Saves the trained model, feature names list, and training metrics for API consumption.
"""

import os

# Workaround: Avoid PermissionError if the environment has SSLKEYLOGFILE set to a system directory (e.g., C:\sslkeys.txt)
if "SSLKEYLOGFILE" in os.environ:
    del os.environ["SSLKEYLOGFILE"]

import json
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    print("=== Model Training Pipeline Starting ===")
    
    # -------------------------------------------------------------------------
    # STEP 1: Load and Prepare the Dataset
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading California Housing Dataset...")
    housing_data = fetch_california_housing()
    
    # Create pandas DataFrame for the features
    X = pd.DataFrame(data=housing_data.data, columns=housing_data.feature_names)
    # Get the target column
    y = housing_data.target
    
    # Keep track of the exact feature list we are training on
    feature_names = list(housing_data.feature_names)
    print(f"Features: {feature_names}")
    print(f"Target: Median House Value (MedHouseVal)")

    # -------------------------------------------------------------------------
    # STEP 2: Train-Test Split (80% Train, 20% Test)
    # -------------------------------------------------------------------------
    print("\n[2/5] Splitting data into train and test sets (80-20 split)...")
    # Setting random_state=42 ensures that the split is reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # -------------------------------------------------------------------------
    # STEP 3: Train the RandomForestRegressor Model
    # -------------------------------------------------------------------------
    print("\n[3/5] Training RandomForestRegressor (n_estimators=100, random_state=42)...")
    # n_estimators=100 means we train 100 individual decision trees
    # max_depth=15 limits the tree depth so the saved model file fits under GitHub's 100MB file size limit (approx 57MB)
    # random_state=42 ensures the forest is built identically if run again
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    
    # Train the model on training data
    model.fit(X_train, y_train)
    print("Model training complete!")

    # -------------------------------------------------------------------------
    # STEP 4: Evaluate the Model
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating model performance on test set...")
    # Predict values for the test set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Absolute Error (MAE)
    # MAE represents the average absolute difference between predicted and actual values.
    # E.g., an MAE of 0.32 means on average the predictions are off by 0.32 units ($32,000 USD).
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate R-squared (R2) score
    # R2 measures the proportion of variance in the target variable that is predictable from features.
    # An R2 score closer to 1.0 indicates a better fit (e.g., 0.80 means the model explains 80% of variance).
    r2 = r2_score(y_test, y_pred)
    
    # Print the evaluation metrics neatly
    print("---------------------------------------------")
    print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Test Mean Absolute Error (in USD): ${mae * 100000:,.2f}")
    print(f"Test R² Score (Coefficient of Det.): {r2:.4f}")
    print("---------------------------------------------")

    # -------------------------------------------------------------------------
    # STEP 5: Serialize and Save Artifacts
    # -------------------------------------------------------------------------
    print("\n[5/5] Saving model and feature metadata...")
    
    # Ensure the model output directory exists
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Define file paths
    model_path = os.path.join(model_dir, "house_model.joblib")
    features_path = os.path.join(model_dir, "feature_names.joblib")
    metrics_path = os.path.join(model_dir, "metrics.json")
    
    # Save the trained model binary using joblib
    joblib.dump(model, model_path)
    print(f"Saved trained model to: {model_path}")
    
    # Save the exact feature names list to ensure consistent input column order
    joblib.dump(feature_names, features_path)
    print(f"Saved feature names list to: {features_path}")
    
    # Save training metrics into a JSON file for the API `/health` endpoint to read
    metrics_data = {
        "model_name": "RandomForestRegressor",
        "n_estimators": 100,
        "random_state": 42,
        "features": feature_names,
        "metrics": {
            "mean_absolute_error": float(mae),
            "mean_absolute_error_usd": float(mae * 100000),
            "r2_score": float(r2)
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved model evaluation metrics to: {metrics_path}")
    
    print("\n=== Model Training Pipeline Completed Successfully ===")

if __name__ == "__main__":
    train_model()
