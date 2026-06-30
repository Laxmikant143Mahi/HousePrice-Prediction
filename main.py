"""
California House Price Prediction API (main.py)

This is a production-quality, beginner-friendly FastAPI application that serves
our trained RandomForestRegressor model. It handles:
1. Loading the serialized model, feature names list, and training metrics on startup.
2. Validating input data using Pydantic schemas.
3. Predicting house prices for single JSON requests.
4. Performing batch predictions on uploaded CSV files and returning results as a downloadable CSV.
"""

import io
import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# -------------------------------------------------------------------------
# STEP 1: Model Loading & Initialization
# -------------------------------------------------------------------------

# Define paths to model artifacts
MODEL_PATH = os.path.join("model", "house_model.joblib")
FEATURES_PATH = os.path.join("model", "feature_names.joblib")
METRICS_PATH = os.path.join("model", "metrics.json")

# Ensure the model has been trained before starting the API
if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH)):
    raise RuntimeError(
        "Model or features file not found! Please run 'python train.py' first to train the model."
    )

# Load the trained RandomForestRegressor model using joblib
model = joblib.load(MODEL_PATH)
# Force single-threaded inference to prevent multiprocessing deadlocks on Windows/Uvicorn
model.n_jobs = 1

# Load the list of expected input features in the exact order trained
feature_names = joblib.load(FEATURES_PATH)

# Load saved metrics if they exist
metrics_data = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)

# Extract Mean Absolute Error (MAE) from metrics for building prediction bounds.
# Default to 0.32 if metrics.json is not found.
model_mae = metrics_data.get("metrics", {}).get("mean_absolute_error", 0.3275)

# Initialize the FastAPI App
app = FastAPI(
    title="California House Price Prediction API",
    description="A simple, production-quality Machine Learning API to predict median house prices in California.",
    version="1.0.0"
)

# -------------------------------------------------------------------------
# STEP 2: Input Validation Schema (Pydantic Model)
# -------------------------------------------------------------------------

class HouseFeatures(BaseModel):
    """
    Input schema for the '/predict' endpoint.
    Includes validation constraints (e.g. gt=0 for positive values)
    and example inputs to make Swagger documentation easy to use.
    """
    MedInc: float = Field(
        ...,
        gt=0,
        description="Median income in block group (in tens of thousands of USD, e.g., 8.3 represents $83,000)",
        examples=[8.3252]
    )
    HouseAge: float = Field(
        ...,
        gt=0,
        description="Median house age in block group (in years)",
        examples=[41.0]
    )
    AveRooms: float = Field(
        ...,
        gt=0,
        description="Average number of rooms per household",
        examples=[6.9841]
    )
    AveBedrms: float = Field(
        ...,
        gt=0,
        description="Average number of bedrooms per household",
        examples=[1.0238]
    )
    Population: float = Field(
        ...,
        gt=0,
        description="Block group population count",
        examples=[322.0]
    )
    AveOccup: float = Field(
        ...,
        gt=0,
        description="Average number of household members (household size)",
        examples=[2.5556]
    )
    Latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Geographic latitude coordinate of the block group",
        examples=[37.88]
    )
    Longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Geographic longitude coordinate of the block group",
        examples=[-122.23]
    )

# -------------------------------------------------------------------------
# STEP 3: API Endpoint Implementations
# -------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["General"])
def read_root():
    """
    Root endpoint serving the interactive HTML dashboard from index.html.
    """
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    raise HTTPException(
        status_code=404,
        detail="Frontend interface template 'index.html' not found on server."
    )


@app.get("/download_sample", tags=["General"])
def download_sample():
    """
    Download the sample CSV input file for batch predictions.
    """
    sample_path = "sample_input.csv"
    if os.path.exists(sample_path):
        return FileResponse(
            sample_path,
            media_type="text/csv",
            filename="sample_housing_input.csv"
        )
    raise HTTPException(
        status_code=404,
        detail="Sample CSV file not found on server."
    )


@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Health check endpoint displaying model status and training diagnostics.
    """
    return {
        "status": "healthy",
        "model_name": metrics_data.get("model_name", "RandomForestRegressor"),
        "feature_names": feature_names,
        "average_error_mae_raw": model_mae,
        "average_error_mae_usd": model_mae * 100000
    }


@app.post("/predict", tags=["Prediction"])
def predict_price(features: HouseFeatures):
    """
    Predict the median house price for a single block group.
    
    Accepts JSON input, runs inference, converts units from $100k
    to normal USD, and returns predicted price along with prediction bounds.
    """
    try:
        # Convert Pydantic request object to a dictionary
        input_data = features.model_dump()
        
        # Convert dictionary to a Pandas DataFrame
        # We place values in a list (e.g. {key: [value]}) to create a 1-row DataFrame
        df_input = pd.DataFrame({key: [val] for key, val in input_data.items()})
        
        # VERY IMPORTANT: Reorder the DataFrame columns to match the training feature names.
        # Scikit-learn models will output incorrect predictions or throw errors if columns are out of order.
        df_input = df_input[feature_names]
        
        # Perform prediction (result is returned as a numpy array, e.g., [4.526])
        prediction = model.predict(df_input)
        
        # Convert raw prediction (which is in $100k units) to absolute USD
        predicted_price_usd = float(prediction[0]) * 100000
        
        # Calculate price bounds using Mean Absolute Error (MAE)
        mae_usd = model_mae * 100000
        price_range_low = max(0.0, predicted_price_usd - mae_usd)
        price_range_high = predicted_price_usd + mae_usd
        
        return {
            "predicted_price": round(predicted_price_usd, 2),
            "price_range_low": round(price_range_low, 2),
            "price_range_high": round(price_range_high, 2)
        }
        
    except Exception as e:
        # If any unexpected exception occurs, return a HTTP 500 error instead of crashing
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(e)}"
        )


@app.post("/predict_file", tags=["Prediction"])
async def predict_file(file: UploadFile = File(...)):
    """
    Accepts a CSV file upload, processes it row-by-row, computes predictions,
    adds a 'PredictedPriceUSD' column, and returns the annotated file as a download.
    """
    # 1. Validate file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV files are supported."
        )
    
    # 2. Read file contents asynchronously
    try:
        contents = await file.read()
        # Decode bytes to a string and wrap in StringIO for pandas
        csv_string = contents.decode("utf-8")
        df_input = pd.read_csv(io.StringIO(csv_string))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {str(e)}"
        )
    
    # 3. Validate that all required features exist in the uploaded file
    missing_columns = [col for col in feature_names if col not in df_input.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required feature columns in CSV: {missing_columns}"
        )
    
    # 4. Perform batch predictions
    try:
        # Extract features in the correct order
        features_only = df_input[feature_names]
        
        # Run prediction on the entire batch
        predictions = model.predict(features_only)
        
        # Convert predictions to USD and append new column to DataFrame
        df_input["PredictedPriceUSD"] = (predictions * 100000).round(2)
        
        # 5. Convert DataFrame back to CSV string
        output_csv = df_input.to_csv(index=False)
        
        # Stream the CSV back to the client as a downloadable file
        response_stream = io.BytesIO(output_csv.encode("utf-8"))
        return StreamingResponse(
            response_stream,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predicted_{file.filename}"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing batch prediction: {str(e)}"
        )
