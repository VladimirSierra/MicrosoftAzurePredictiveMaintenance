"""
FastAPI application for Predictive Maintenance model.
Provides endpoints for health checks and failure predictions on an specific time window (24 hours configured).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import yaml
from pathlib import Path
from typing import List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failures using telemetry. Predictions are given on a configured 24 hours window",
    version="1.0.0"
)

# Global model variable
model = None

class MachineData(BaseModel):
    """Request model for machine data (simplified for better performance)."""
    model_config = {"protected_namespaces": ()}

    # machine ID is not use for prediction, but is better to have tracability if you want to save information of requests
    machineID: int = Field(..., description="Machine identifier")
    
    # Telemetry features - must match trained model variables
    # volt features
    volt_mean_3h: float = Field(..., description="Average voltage in last 3 hours")
    volt_std_3h: float = Field(..., description="Voltage standard deviation in last 3 hours")
    volt_mean_24h: float = Field(..., description="Average voltage in last 24 hours")
    volt_std_24h: float = Field(..., description="Voltage standard deviation in last 24 hours")
    volt_min_24h: float = Field(..., description="Minimum voltage in last 24 hours")
    volt_max_24h: float = Field(..., description="Maximum voltage in last 24 hours")
    
    # rotate features
    rotate_mean_3h: float = Field(..., description="Average rotation in last 3 hours")
    rotate_std_3h: float = Field(..., description="Rotation standard deviation in last 3 hours")
    rotate_mean_24h: float = Field(..., description="Average rotation in last 24 hours")
    rotate_std_24h: float = Field(..., description="Rotation standard deviation in last 24 hours")
    rotate_min_24h: float = Field(..., description="Minimum rotation in last 24 hours")
    rotate_max_24h: float = Field(..., description="Maximum rotation in last 24 hours")
    
    # pressure features
    pressure_mean_3h: float = Field(..., description="Average pressure in last 3 hours")
    pressure_std_3h: float = Field(..., description="Pressure standard deviation in last 3 hours")
    pressure_mean_24h: float = Field(..., description="Average pressure in last 24 hours")
    pressure_std_24h: float = Field(..., description="Pressure standard deviation in last 24 hours")
    pressure_min_24h: float = Field(..., description="Minimum pressure in last 24 hours")
    pressure_max_24h: float = Field(..., description="Maximum pressure in last 24 hours")
    
    # vibration features
    vibration_mean_3h: float = Field(..., description="Average vibration in last 3 hours")
    vibration_std_3h: float = Field(..., description="Vibration standard deviation in last 3 hours")
    vibration_mean_24h: float = Field(..., description="Average vibration in last 24 hours")
    vibration_std_24h: float = Field(..., description="Vibration standard deviation in last 24 hours")
    vibration_min_24h: float = Field(..., description="Minimum vibration in last 24 hours")
    vibration_max_24h: float = Field(..., description="Maximum vibration in last 24 hours")
    
    # Error counts
    error1_count_24h: float = Field(0.0, description="Error1 count in last 24 hours")
    error2_count_24h: float = Field(0.0, description="Error2 count in last 24 hours")
    error3_count_24h: float = Field(0.0, description="Error3 count in last 24 hours")
    error4_count_24h: float = Field(0.0, description="Error4 count in last 24 hours")
    error5_count_24h: float = Field(0.0, description="Error5 count in last 24 hours")
    
    # Days since maintenance
    comp1_days_since: float = Field(0.0, description="Days since last comp1 maintenance")
    comp2_days_since: float = Field(0.0, description="Days since last comp2 maintenance")
    comp3_days_since: float = Field(0.0, description="Days since last comp3 maintenance")
    comp4_days_since: float = Field(0.0, description="Days since last comp4 maintenance")
    
    # Machine metadata
    age: int = Field(..., description="Machine age in years")
    model_model1: bool = Field(False, description="Is machine model1")
    model_model2: bool = Field(False, description="Is machine model2")
    model_model3: bool = Field(False, description="Is machine model3")
    model_model4: bool = Field(False, description="Is machine model4")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    machineID: int
    failure_probability: float = Field(..., description="Probability of failure in next 24 hours (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    recommendation: str = Field(..., description="Recommended action")
    # timestamp is for traceability, useful if you want to save information of requests
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    machines: List[MachineData]

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_machines: int
    high_risk_count: int

def load_model():
    """Load the trained model."""
    global model
    try:
        model_path = Path(config['data']['models_path']) / "predictive_maintenance_model.joblib"
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def get_risk_level_and_recommendation(probability: float) -> tuple:
    """Determine risk level and recommendation based on probability.
    
    Args:
        probability: Failure probability
        
    Returns:
        Tuple of (risk_level, recommendation)
    """
    if probability < 0.1:
        return "LOW", "Continue normal operations. Monitor regularly."
    elif probability < 0.3:
        return "MEDIUM", "Increase monitoring frequency. Schedule inspection."
    else:
        return "HIGH", "Urgent attention required. Schedule immediate maintenance."

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(machine_data: MachineData):
    """Predict failure for a single machine.
    
    Args:
        machine_data: Machine data for prediction
        
    Returns:
        Prediction response with failure probability and risk level
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = machine_data.dict()
        machineID = input_dict.pop('machineID')
        
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        probability = model.predict_proba(input_df)[0, 1]
        
        # Determine risk level and recommendation
        risk_level, recommendation = get_risk_level_and_recommendation(probability)
        
        return PredictionResponse(
            machineID=machineID,
            failure_probability=float(probability),
            risk_level=risk_level,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict failures for multiple machines.
    
    Args:
        request: Batch prediction request
        
    Returns:
        Batch prediction response
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        predictions = []
        high_risk_count = 0
        
        for machine_data in request.machines:
            # Convert input to DataFrame
            input_dict = machine_data.dict()
            machineID = input_dict.pop('machineID')
            
            input_df = pd.DataFrame([input_dict])
            
            # Make prediction
            probability = model.predict_proba(input_df)[0, 1]
            
            # Determine risk level and recommendation
            risk_level, recommendation = get_risk_level_and_recommendation(probability)
            
            if risk_level == "HIGH":
                high_risk_count += 1
            
            predictions.append(PredictionResponse(
                machineID=machineID,
                failure_probability=float(probability),
                risk_level=risk_level,
                recommendation=recommendation,
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_machines=len(request.machines),
            high_risk_count=high_risk_count
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown",
        "n_classes": model.n_classes_ if hasattr(model, 'n_classes_') else "Unknown",
        "prediction_window": f"{config['feature_engineering']['prediction_window']} hours"
    }
