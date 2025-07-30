# Predictive Maintenance MLOps Solution

A comprehensive machine learning solution for predicting equipment failures on a time window using the Microsoft Azure Predictive Maintenance dataset.

## Project Overview

This project implements an end-to-end MLOps pipeline for predictive maintenance, focusing on clean engineering practices, reproducibility, and deployment readiness. The solution predicts machine failures within a 24-hour window based on telemetry data, error history, and maintenance records.

### Key Features

- **Reproducible Pipeline**: Fully automated training and deployment
- **MLOps Integration**: MLflow tracking, model versioning
- **Production API**: FastAPI deployment with automatic documentation
- **Time Handling**: Proper temporal splits and feature engineering for model training

## Dataset

**Source**: [Microsoft Azure Predictive Maintenance Dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance/data)

The dataset contains:
- **Telemetry**: 876K sensor readings (voltage, rotation, pressure, vibration)
- **Errors**: 3.9K error events across 5 error types
- **Failures**: 762 component failure events
- **Maintenance**: 3.2K maintenance records
- **Machines**: 100 machines with metadata (model, age)

## Project Structure

```
├── config/                  
│   └── config.yaml          # Configuration settings
├── data/                    # Raw CSV input data files
│   ├── features/            # Calculated features for model training
├── docs/                  
│   │   └── technical_desing.md     # Desing documentation 
│   │   └── technical_report.md     # Report documentation 
├── mlruns/                  # MLflow tracking data
├── models/                  # Trained model and performance plot
├── src/
│   ├── api/                
│   │   └── app.py               # FastAPI application
│   ├── data/                
│   │   └── data_loader.py       # Data loading utilities
│   ├── features/           
│   │   └── feature_engineer.py  # Feature engineering
│   └── models/             
│       └── trainer.py           # Model training and evaluation
├── docker-compose.yml       # Docker compose configuration file
├── Dockerfile               # Dockerfile to deploy api
├── README.md                # This file
├── requirements.txt         # Python libraries for this project
├── train_pipeline.py        # Python script for complete training pipeline
├── training.log             # Logs from the training pipeline
```

## Quick Start

### Option 1: Docker Deployment (Recommended for Production)

#### Prerequisites
- Docker and Docker Compose installed

#### Quick Deploy
```bash
# Clone and navigate to project
git clone https://github.com/VladimirSierra/MicrosoftAzurePredictiveMaintenance.git
cd MicrosoftAzurePredictiveMaintenance

# Deploy with Docker
docker compose up --build -d
```

To stop the service run:
```bash
# stop container
docker compose down
```

> **Note**: Depending on your version of Docker Compose, the command might be `docker compose` (v2) or `docker-compose` (v1).

#### Configuration
The API comes with default values configured directly in the Dockerfile:
- `HOST=0.0.0.0` - Host binding
- `PORT=8000` - Internal port of the container
- `LOG_LEVEL=info` - Logging level

If you want to modify these values, edit the `docker-compose.yml`.


#### Access API
- **API Endpoint**: `http://localhost:8000`
- **Interactive Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### Option 2: Local Development (not for production)

#### 1. Environment Setup
Requires **Python 3.9**.

```bash
# Clone and navigate to project
git clone https://github.com/VladimirSierra/MicrosoftAzurePredictiveMaintenance.git
cd MicrosoftAzurePredictiveMaintenance
```

```bash
# Create virtual environment
python3.9 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. **OPTIONAL:** Train the Model (a trained model is already saved and ready to use for predictions, use this script only if you want to retrain de model).
```bash
# Run the complete training pipeline
python3.9 train_pipeline.py
```
This will:
- Load and validate the data
- Train XGBoost model 
- Evaluate performance and create plots
- Save model 

#### 3. Run the API
```bash
# Start the FastAPI server
python3.9 -m uvicorn src.api.app:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive API documentation.

## Usage Examples

### Making Predictions

#### Single Prediction

Send a POST request to the `/predict` endpoint with the machine's feature data.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "machineID": 1,
  "volt_mean_3h": 175.4,
  "volt_std_3h": 10.2,
  "volt_mean_24h": 175.1,
  "volt_std_24h": 9.8,
  "volt_min_24h": 155.3,
  "volt_max_24h": 195.8,
  "rotate_mean_3h": 450.1,
  "rotate_std_3h": 25.5,
  "rotate_mean_24h": 449.8,
  "rotate_std_24h": 24.9,
  "rotate_min_24h": 400.2,
  "rotate_max_24h": 500.7,
  "pressure_mean_3h": 100.2,
  "pressure_std_3h": 5.1,
  "pressure_mean_24h": 100.0,
  "pressure_std_24h": 4.9,
  "pressure_min_24h": 90.3,
  "pressure_max_24h": 110.1,
  "vibration_mean_3h": 40.3,
  "vibration_std_3h": 3.2,
  "vibration_mean_24h": 40.1,
  "vibration_std_24h": 3.0,
  "vibration_min_24h": 35.1,
  "vibration_max_24h": 45.2,
  "error1_count_24h": 1,
  "error2_count_24h": 0,
  "error3_count_24h": 2,
  "error4_count_24h": 0,
  "error5_count_24h": 0,
  "comp1_days_since": 120,
  "comp2_days_since": 210,
  "comp3_days_since": 15,
  "comp4_days_since": 300,
  "age": 10,
  "model_model1": true,
  "model_model2": false,
  "model_model3": false,
  "model_model4": false
}'
```

**Example Response:**
```json
{
  "machineID": 1,
  "failure_probability": 0.8921,
  "risk_level": "HIGH",
  "recommendation": "High risk of failure detected. Schedule maintenance immediately.",
  "timestamp": "2025-07-29T18:10:00.123456"
}
```

#### Batch Predictions
Send a POST request to `/predict/batch` with a list of machines.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "machines": [
    {
      "machineID": 1,
      "volt_mean_3h": 0, "volt_std_3h": 0, "volt_mean_24h": 0, "volt_std_24h": 0, "volt_min_24h": 0, "volt_max_24h": 0,
      "rotate_mean_3h": 0, "rotate_std_3h": 0, "rotate_mean_24h": 0, "rotate_std_24h": 0, "rotate_min_24h": 0, "rotate_max_24h": 0,
      "pressure_mean_3h": 0, "pressure_std_3h": 0, "pressure_mean_24h": 0, "pressure_std_24h": 0, "pressure_min_24h": 0, "pressure_max_24h": 0,
      "vibration_mean_3h": 0, "vibration_std_3h": 0, "vibration_mean_24h": 0, "vibration_std_24h": 0, "vibration_min_24h": 0, "vibration_max_24h": 0,
      "error1_count_24h": 0, "error2_count_24h": 0, "error3_count_24h": 0, "error4_count_24h": 0, "error5_count_24h": 0,
      "comp1_days_since": 0, "comp2_days_since": 0, "comp3_days_since": 0, "comp4_days_since": 0,
      "age": 0, "model_model1": false, "model_model2": false, "model_model3": false, "model_model4": false
    },
    {
      "machineID": 2,
      "volt_mean_3h": 180, "volt_std_3h": 15, "volt_mean_24h": 178, "volt_std_24h": 14, "volt_min_24h": 160, "volt_max_24h": 200,
      "rotate_mean_3h": 480, "rotate_std_3h": 30, "rotate_mean_24h": 475, "rotate_std_24h": 28, "rotate_min_24h": 420, "rotate_max_24h": 520,
      "pressure_mean_3h": 110, "pressure_std_3h": 8, "pressure_mean_24h": 108, "pressure_std_24h": 7, "pressure_min_24h": 95, "pressure_max_24h": 120,
      "vibration_mean_3h": 50, "vibration_std_3h": 5, "vibration_mean_24h": 48, "vibration_std_24h": 4.5, "vibration_min_24h": 42, "vibration_max_24h": 55,
      "error1_count_24h": 3, "error2_count_24h": 1, "error3_count_24h": 0, "error4_count_24h": 0, "error5_count_24h": 1,
      "comp1_days_since": 20, "comp2_days_since": 180, "comp3_days_since": 250, "comp4_days_since": 90,
      "age": 15, "model_model1": false, "model_model2": true, "model_model3": false, "model_model4": false
    }
  ]
}'
```

**Example Response:**
```json
{
  "predictions": [
    {
      "machineID": 1,
      "failure_probability": 0.00038,
      "risk_level": "LOW",
      "recommendation": "Continue normal operations. Monitor regularly.",
      "timestamp": "2025-07-29T18:15:00.123456"
    },
    {
      "machineID": 2,
      "failure_probability": 0.9567,
      "risk_level": "HIGH",
      "recommendation": "High risk of failure detected. Schedule maintenance immediately.",
      "timestamp": "2025-07-29T18:15:00.123456"
    }
  ],
  "total_machines": 2,
  "high_risk_count": 1
}
```

### API Endpoints
- `GET /` - API information
- `GET /health` - Health api check
- `POST /predict` - Single machine prediction
- `POST /predict/batch` - Batch predictions

## Configuration

Edit `config/config.yaml` to customize:
- **Data paths**: Input and output directories
- **Feature engineering**: Time windows, aggregation methods
- **Model parameters**: XGBoost hyperparameters


## DEMO

The following video shows how to deploy the service with docker compose and test the endpoints.

https://github.com/user-attachments/assets/858a6bd5-c8e8-442d-91b1-4fe84c6208e9