import os
import sys
import mlflow
import pandas as pd
import numpy as np
import json
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

# Add the current directory to the path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
        
        # Check if it's a PyTorch model
        if isinstance(models[model_key], torch.nn.Module):
            # PyTorch model inference
            import torch
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_df.values)
            
            # Set model to evaluation mode
            models[model_key].eval()
            
            # Make prediction
            with torch.no_grad():
                output = models[model_key](features_tensor)
                
                # Check model type
                if hasattr(models[model_key], 'layers') and isinstance(models[model_key].layers[-1], torch.nn.Linear):
                    output_dim = models[model_key].layers[-1].out_features
                    
                    if output_dim == 1:  # Binary classification or regression
                        # Check if it's classification (should have sigmoid in forward)
                        if "classifier" in model_key.lower() or "classification" in model_key.lower():
                            prediction = torch.sigmoid(output).item() > 0.5
                            return {
                                "prediction": int(prediction),
                                "confidence": float(torch.sigmoid(output).item())
                            }
                        else:  # Regression
                            return {"prediction": float(output.item())}
                    else:  # Multi-class classification
                        probabilities = torch.softmax(output, dim=1).tolist()[0]
                        prediction = int(torch.argmax(output, dim=1).item())
                        return {
                            "prediction": prediction,
                            "probabilities": probabilities
                        }
        else:
            # scikit-learn model inference
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
        
        # Check if it's a PyTorch model
        if isinstance(models[model_key], torch.nn.Module):
            # PyTorch model inference
            import torch
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_df.values)
            
            # Set model to evaluation mode
            models[model_key].eval()
            
            # Make prediction
            with torch.no_grad():
                outputs = models[model_key](features_tensor)
                
                # Check model type
                if hasattr(models[model_key], 'layers') and isinstance(models[model_key].layers[-1], torch.nn.Linear):
                    output_dim = models[model_key].layers[-1].out_features
                    
                    if output_dim == 1:  # Binary classification or regression
                        # Check if it's classification
                        if "classifier" in model_key.lower() or "classification" in model_key.lower():
                            predictions = (torch.sigmoid(outputs) > 0.5).int().tolist()
                            confidences = torch.sigmoid(outputs).tolist()
                            return {
                                "predictions": predictions,
                                "confidences": confidences
                            }
                        else:  # Regression
                            return {"predictions": outputs.flatten().tolist()}
                    else:  # Multi-class classification
                        probabilities = torch.softmax(outputs, dim=1).tolist()
                        predictions = torch.argmax(outputs, dim=1).tolist()
                        return {
                            "predictions": predictions,
                            "probabilities": probabilities
                        }
        else:
            # scikit-learn model inference
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