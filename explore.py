"""
California House Price Prediction API - Data Exploration (explore.py)

This script loads the California Housing dataset from scikit-learn,
converts it into a Pandas DataFrame, and performs basic exploratory analysis.
Use this script to understand the dataset structure, features, and target variable.
"""
import os

# Workaround: Avoid PermissionError if the environment has SSLKEYLOGFILE set to a system directory (e.g., C:\sslkeys.txt)
if "SSLKEYLOGFILE" in os.environ:
    del os.environ["SSLKEYLOGFILE"]

import pandas as pd
from sklearn.datasets import fetch_california_housing

def explore_dataset():
    # -------------------------------------------------------------------------
    # STEP 1: Load the California Housing dataset from scikit-learn
    # -------------------------------------------------------------------------
    print("=== STEP 1: Loading California Housing Dataset ===")
    
    # fetch_california_housing() returns a dictionary-like "Bunch" object containing:
    # - data: The feature matrix (input columns)
    # - target: The target vector (median house value in units of $100,000)
    # - feature_names: Names of the feature columns
    # - DESCR: A full text description of the dataset
    housing_data = fetch_california_housing()
    
    # -------------------------------------------------------------------------
    # STEP 2: Convert the raw dataset into a Pandas DataFrame for analysis
    # -------------------------------------------------------------------------
    print("\n=== STEP 2: Converting to Pandas DataFrame ===")
    
    # Create the DataFrame with the features (inputs)
    df = pd.DataFrame(data=housing_data.data, columns=housing_data.feature_names)
    
    # Add the target column (what we want to predict).
    # NOTE: The target variable 'MedHouseVal' is expressed in hundreds of thousands of dollars.
    # For example, a value of 4.5 represents $450,000.
    df['MedHouseVal'] = housing_data.target
    
    print("DataFrame successfully created.")

    # -------------------------------------------------------------------------
    # STEP 3: Display the shape of the dataset
    # -------------------------------------------------------------------------
    print("\n=== STEP 3: Dataset Shape ===")
    # shape returns a tuple (number_of_rows, number_of_columns)
    print(f"Number of rows (samples): {df.shape[0]}")
    print(f"Number of columns (features + target): {df.shape[1]}")

    # -------------------------------------------------------------------------
    # STEP 4: Display the feature names
    # -------------------------------------------------------------------------
    print("\n=== STEP 4: Feature Names ===")
    # The columns used as inputs for the model
    print("Input Features:", list(housing_data.feature_names))
    print("Target Column: 'MedHouseVal' (Median House Value in $100,000s)")

    # -------------------------------------------------------------------------
    # STEP 5: Display the first 5 rows
    # -------------------------------------------------------------------------
    print("\n=== STEP 5: First 5 Rows of the Dataset ===")
    # head() is used to look at the top rows of a DataFrame to get a quick visual check
    print(df.head())

    # -------------------------------------------------------------------------
    # STEP 6: Display dataset information (column types and missing values)
    # -------------------------------------------------------------------------
    print("\n=== STEP 6: Dataset Information ===")
    # info() shows the data type of each column and checks for non-null count.
    # This helps verify if there are any missing (NaN) values that need handling.
    df.info()

    # -------------------------------------------------------------------------
    # STEP 7: Display summary statistics
    # -------------------------------------------------------------------------
    print("\n=== STEP 7: Summary Statistics ===")
    # describe() provides descriptive statistics including mean, standard deviation,
    # min, max, and quantiles for each numeric column.
    print(df.describe())

if __name__ == "__main__":
    explore_dataset()
