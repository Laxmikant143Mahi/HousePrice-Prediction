# California House Price Prediction API

A production-ready, interview-friendly Machine Learning API that predicts median home values in California block groups. This project serves as an excellent reference for integrating scikit-learn models with FastAPI, featuring strict Pydantic V2 validation, complete Swagger UI integration, a modern mobile-responsive light-theme dashboard, and downloadable batch CSV prediction pipelines.

---

## Project Overview

This project loads the classical **California Housing Dataset** from `scikit-learn` to train a **RandomForestRegressor** model. The trained model is then serialized and exposed via a robust **FastAPI** web service. 

It is designed to demonstrate:
1. **Machine Learning Pipeline**: Data loading, exploration, train-test splitting (80-20), model training, evaluation (MAE, R² score), and model serialization using `joblib`.
2. **Interactive UI Dashboard**: A sleek, mobile-friendly light-theme dashboard built with Vanilla HTML, CSS, and JS, served dynamically at the root endpoint.
3. **REST API Design**: Endpoints for server status, health diagnostics (loading metrics dynamically), single-item predictions, and bulk CSV file prediction outputs.
4. **Data Validation**: Strict type enforcement and business-logic boundary validation using Pydantic V2 (`Field` constraints like positive-only inputs and coordinate limit bounds).
5. **Reliability**: Structured exception handling (returning standard `HTTPException` codes) and zero runtime code warnings.

---

## Features

- **Minimal Responsive UI (`index.html`)**: Beautiful, mobile-friendly frontend served dynamically on `GET /` with sliders for single prediction and drag-and-drop file uploader for bulk batches.
- **Data Exploration Script (`explore.py`)**: Quick Exploratory Data Analysis (EDA) showcasing data shape, features, and descriptive statistics.
- **Model Training Pipeline (`train.py`)**: Automated model building and serialization. Saves metrics like MAE and R² dynamically to power the API dashboard.
- **FastAPI Endpoints**:
  - `GET /`: Health status and interactive dashboard.
  - `GET /health`: Comprehensive status, returning loaded feature names, active model algorithm, and validation error ranges (MAE in USD).
  - `GET /download_sample`: Downloads a sample input template CSV file for batch testing.
  - `POST /predict`: Real-time JSON inference returning estimated price along with low/high bounds based on testing errors.
  - `POST /predict_file`: Bulk predictions from an uploaded CSV file, returning the original table appended with a `PredictedPriceUSD` column as a downloadable file.
- **Interactive API Documentation**: Runs automatically on `/docs` using Swagger UI.

---

## Folder Structure

```text
California-House-Price-Prediction-API/
│
├── data/                    # Placeholder for dataset files
│   └── .gitkeep
│
├── model/                   # Serialized binaries and evaluation metrics
│   ├── house_model.joblib   # Trained RandomForestRegressor model (~57MB)
│   ├── feature_names.joblib # Feature columns list (maintains ordering)
│   └── metrics.json         # MAE and R2 scores generated from training
│
├── explore.py               # Dataset exploration / EDA script
├── train.py                 # Training, evaluation, and serialization script
├── main.py                  # FastAPI application code
├── index.html               # Minimal responsive light-theme dashboard UI
├── requirements.txt         # Package dependencies
├── sample_input.csv         # Sample batch CSV file for testing
├── vercel.json              # Vercel serverless deployment configuration
├── .gitignore               # Git untracked files definitions
└── README.md                # Project documentation
```

---

## Technologies Used

- **Python**: Core programming language.
- **FastAPI**: Modern, high-performance web framework for building APIs.
- **Uvicorn**: Lightning-fast ASGI server implementation for running FastAPI.
- **Scikit-Learn**: Machine learning library used to load the dataset and train the `RandomForestRegressor`.
- **Pandas**: Structured data manipulation and analysis library.
- **Pydantic**: Data validation and settings management using python type annotations.
- **Joblib**: Efficient serialization of large Python objects/numpy arrays.
- **python-multipart**: Parses incoming multi-part form data (required for CSV file uploads).

---

## Installation & Setup

Follow these steps to run the project locally on your machine.

### 1. Clone the Repository & Navigate to Project Directory
```bash
git clone https://github.com/Laxmikant143Mahi/HousePrice-Prediction.git
cd "HousePrice-Prediction"
```

### 2. Create a Virtual Environment (Recommended)
Creating a virtual environment ensures that the project dependencies do not conflict with other Python projects on your machine.

On Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Train the Machine Learning Model
Before starting the API server, you must train the model and generate the serialization files (`.joblib` and `.json`).

Run:
```bash
python train.py
```

**Expected Output:**
```text
=== Model Training Pipeline Starting ===

[1/5] Loading California Housing Dataset...
Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
Target: Median House Value (MedHouseVal)

[2/5] Splitting data into train and test sets (80-20 split)...
Training set shape: (16512, 8)
Test set shape: (4128, 8)

[3/5] Training RandomForestRegressor (n_estimators=100, random_state=42)...
Model training complete!

[4/5] Evaluating model performance on test set...
---------------------------------------------
Test Mean Absolute Error (MAE): 0.3332
Test Mean Absolute Error (in USD): $33,322.09
Test R² Score (Coefficient of Det.): 0.8004
---------------------------------------------

[5/5] Saving model and feature metadata...
Saved trained model to: model\house_model.joblib
Saved feature names list to: model\feature_names.joblib
Saved model evaluation metrics to: model\metrics.json

=== Model Training Pipeline Completed Successfully ===
```

### Step 2: Start the FastAPI Server
Launch the API using the `uvicorn` development server:

```bash
uvicorn main:app --reload
```

- `--reload` enables auto-reload, meaning the server will restart automatically when code files are edited.
- By default, the application runs on: **http://127.0.0.1:8000**

---

## Testing Using Swagger UI

FastAPI automatically generates interactive documentation for your API.
1. Open your browser and go to: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
2. You will see a list of endpoints categorised under **General**, **Monitoring**, and **Prediction**.
3. You can click on any endpoint, select **"Try it out"**, fill in the parameters, and click **"Execute"** to see live responses.

---

## Example API Usage

### 1. Single House Prediction (`POST /predict`)

#### Example JSON Request
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984127,
  "AveBedrms": 1.023810,
  "Population": 322.0,
  "AveOccup": 2.555556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

#### Example JSON Response (Status Code: 200 OK)
```json
{
  "predicted_price": 426579.30,
  "price_range_low": 393257.21,
  "price_range_high": 459901.39
}
```
*Note: The target value is scaled from unit fractions in the dataset back to actual USD currency. The low and high bounds are dynamically computed using the model's test MAE (`$33,322.09`).*

---

### 2. Batch Predictions (`POST /predict_file`)

To predict values for multiple houses at once, upload a CSV file with headers matching our features.

#### Batch Prediction Steps:
1. Locate the [sample_input.csv](sample_input.csv) file inside the project directory (or download it from the dashboard UI).
2. In the Swagger UI `/docs`, expand the `POST /predict_file` endpoint.
3. Click **"Try it out"**.
4. Click **"Choose File"** and upload `sample_input.csv`.
5. Click **"Execute"**.
6. The API will respond with a downloadable CSV containing all original columns, plus an appended column named `PredictedPriceUSD` containing predictions for each row.

---

## Model Features & Descriptions

Here is a summary of the 8 features used to predict the house price:

| Feature Name | Description | Validation Constraints |
|---|---|---|
| **MedInc** | Median income in the block group (expressed in tens of thousands of USD). | `> 0` |
| **HouseAge** | Median age of houses in the block group (in years). | `> 0` |
| **AveRooms** | Average number of rooms per household. | `> 0` |
| **AveBedrms** | Average number of bedrooms per household. | `> 0` |
| **Population** | Total population in the block group. | `> 0` |
| **AveOccup** | Average number of household members. | `> 0` |
| **Latitude** | Latitude coordinate (degree north). | `-90` to `90` |
| **Longitude** | Longitude coordinate (degree east). | `-180` to `180` |

---

## How to Deploy (For Your Resume)

Having a live deployment URL on your resume makes a massive difference! Here are the easiest ways to deploy this API for free:

### Option A: Deploying on Vercel (Free Serverless Tier)
[Vercel](https://vercel.com/) is a serverless hosting platform that offers free hosting for hobby projects:
1. **Commit Model Binaries**: Because Vercel's serverless environment is **read-only** at runtime, the application cannot run training scripts upon boot. **You must train the model locally first** and commit the `model/` directory containing `house_model.joblib`, `feature_names.joblib`, and `metrics.json` to GitHub.
2. **Push to GitHub**: Initialize a Git repository, commit all files (including `vercel.json` and the trained `model/` folder), and push them to your GitHub repository.
3. **Import to Vercel**:
   - Go to your Vercel Dashboard and click **Add New > Project**.
   - Select your connected GitHub repository and click **Import**.
   - Vercel automatically detects the `vercel.json` file and handles the Python environment builds.
4. **Deploy**: Click **Deploy**. Your FastAPI dashboard and API routes will be live 24/7 at `https://your-project-name.vercel.app`!

### Option B: Deploying on Render (Free/Hobby Tier)
[Render](https://render.com/) is a very popular cloud hosting service. You can deploy this FastAPI service in minutes:
1. **Push to GitHub**: Initialize a Git repository, commit all files, and push them to a public GitHub repository.
2. **Create a Web Service on Render**:
   - Log in to Render and click **New > Web Service**.
   - Connect your GitHub account and select your repository.
3. **Configure Settings**:
   - **Environment**: Select `Python`.
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Deploy**: Click **Create Web Service**. Your app will be live at `https://your-app-name.onrender.com`.
*Note: Render's free tier spins down your application when it hasn't received traffic for 15 minutes. The first request after a sleep period might take ~1 minute to spin up.*

### Option C: Deploying on Hugging Face Spaces (Free Tier)
[Hugging Face Spaces](https://huggingface.co/spaces) is the absolute best place to showcase Machine Learning portfolios. 
1. Create a free Hugging Face account and navigate to Spaces > **Create new Space**.
2. Name your space, select **Docker** as the SDK, and choose the **Blank** template.
3. In your repository, create a file named `Dockerfile` with the following content:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY . .
   RUN pip install --no-cache-dir -r requirements.txt
   RUN python train.py
   EXPOSE 7860
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
4. Push your local files (including the Dockerfile) to the Hugging Face Space Git repository.
5. Hugging Face will automatically build and run your Docker container. Your API and its interactive UI dashboard will be live 24/7!

---

## Future Improvements

While this project is fully complete and production-grade, the following steps could be explored in future extensions:
1. **Feature Engineering**: Creating additional features like room-to-bedroom ratios or proximity to geographic hubs.
2. **Hyperparameter Tuning**: Implementing GridSearchCV or RandomizedSearchCV to find optimal parameters for the RandomForest model.
3. **Alternative Models**: Training gradient boosting algorithms (e.g., XGBoost, LightGBM) to compare performance metrics against Random Forest.
4. **CI/CD Pipeline**: Adding Github Actions to automatically test endpoints and lint code upon code updates.
