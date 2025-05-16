import os
import sys
import mlflow
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager
sys.path.append('../')
from scripts.mlflow_utils import load_model

# Preloaded models dictionary
models = {}

# Application startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    print("Loading models...")
    # You can preload commonly used models here
    yield
    # Shutdown: Clean up resources
    print("Shutting down and cleaning up...")

# Initialize FastAPI app
app = FastAPI(title="ML Pipeline Model Serving", 
             description="API for serving machine learning models",
             version="1.0",
             lifespan=lifespan)

# Pydantic models for request validation
class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use for prediction")
    model_stage: str = Field("Production", description="Stage of the model (Production, Staging, etc.)")
    features: Dict[str, Any] = Field(..., description="Features as a dictionary of feature_name: value")

class BatchPredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use for prediction")
    model_stage: str = Field("Production", description="Stage of the model (Production, Staging, etc.)")
    features: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")

# API endpoints
@app.get("/")
async def root():
    return {"message": "ML Model Serving API"}

@app.get("/models")
async def list_registered_models():
    try:
        client = mlflow.tracking.MlflowClient()
        registered_models = client.list_registered_models()
        model_list = []
        
        for rm in registered_models:
            model_versions = client.get_latest_versions(rm.name)
            model_list.append({
                "name": rm.name,
                "versions": [
                    {"version": mv.version, "stage": mv.current_stage, "status": mv.status}
                    for mv in model_versions
                ]
            })
            
        return {"models": model_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.post("/predict")
async def predict(request: PredictionRequest):
    model_key = f"{request.model_name}_{request.model_stage}"
    
    try:
        # Load model if not already loaded
        if model_key not in models:
            models[model_key] = load_model(request.model_name, request.model_stage)
            
        if models[model_key] is None:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
        
        # Convert input features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = models[model_key].predict(features_df)
        
        # Get prediction probabilities if available (for classification)
        if hasattr(models[model_key], 'predict_proba'):
            probabilities = models[model_key].predict_proba(features_df).tolist()
            return {
                "prediction": prediction.tolist()[0],
                "probabilities": probabilities[0]
            }
        else:
            return {"prediction": prediction.tolist()[0]}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    model_key = f"{request.model_name}_{request.model_stage}"
    
    try:
        # Load model if not already loaded
        if model_key not in models:
            models[model_key] = load_model(request.model_name, request.model_stage)
            
        if models[model_key] is None:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
        
        # Convert input features to DataFrame
        features_df = pd.DataFrame(request.features)
        
        # Make prediction
        predictions = models[model_key].predict(features_df)
        
        # Get prediction probabilities if available (for classification)
        if hasattr(models[model_key], 'predict_proba'):
            probabilities = models[model_key].predict_proba(features_df).tolist()
            return {
                "predictions": predictions.tolist(),
                "probabilities": probabilities
            }
        else:
            return {"predictions": predictions.tolist()}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 