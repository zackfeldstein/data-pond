#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import time
import signal
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to the path
sys.path.append('../')

from scripts.minio_utils import get_minio_client, list_datasets, list_files_in_dataset, read_csv_from_minio
from scripts.feature_engineering import engineer_features, split_data
from scripts.model_training import train_sklearn_model, train_pytorch_model
from scripts.mlflow_utils import setup_mlflow
import mlflow

def start_mlflow_server():
    """Start MLflow tracking server"""
    print("Starting MLflow tracking server...")
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "s3://ml-datalake/mlflow/",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    # Start process in the background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Check if process is running
    if process.poll() is None:
        print("MLflow server started successfully")
        return process
    else:
        print("Failed to start MLflow server")
        print(process.stderr.read())
        return None

def start_model_serving():
    """Start FastAPI model serving"""
    print("Starting model serving with FastAPI...")
    cmd = ["uvicorn", "scripts.model_serving:app", "--host", "0.0.0.0", "--port", "8000"]
    
    # Start process in the background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Check if process is running
    if process.poll() is None:
        print("FastAPI server started successfully")
        return process
    else:
        print("Failed to start FastAPI server")
        print(process.stderr.read())
        return None

def run_pipeline(dataset_name, use_gpu=False, run_serving=True):
    """Run the complete ML pipeline"""
    print(f"\n{'='*50}\nRunning ML Pipeline for dataset: {dataset_name}\n{'='*50}")
    
    # 1. Check if dataset exists in MinIO
    datasets = list_datasets()
    if dataset_name not in datasets:
        print(f"Dataset '{dataset_name}' not found in MinIO. Available datasets: {datasets}")
        return False
    
    # 2. List files in the dataset
    files = list_files_in_dataset(dataset_name)
    if not files:
        print(f"No files found in dataset '{dataset_name}'")
        return False
    
    print(f"Files in dataset: {files}")
    
    # 3. Load the first file (adjust as needed for your use case)
    file_to_load = files[0]
    data = read_csv_from_minio(dataset_name, file_to_load)
    if data is None:
        print(f"Failed to load file {file_to_load}")
        return False
    
    print(f"Loaded {file_to_load} with shape: {data.shape}")
    print(data.head())
    
    # 4. Feature Engineering
    print("\n\nPerforming feature engineering...")
    
    # Assume the last column is the target (adjust as needed)
    target_col = data.columns[-1]
    print(f"Using {target_col} as the target column")
    
    processed_data, preprocessor = engineer_features(
        data, 
        target_col=target_col,
        dataset_name=dataset_name,
        save_to_minio=True
    )
    
    print(f"Processed data shape: {processed_data.shape}")
    
    # 5. Split data for modeling
    X_train, X_test, y_train, y_test = split_data(processed_data, target_col)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 6. Determine if it's a classification or regression problem
    if y_train.dtype == 'object' or len(np.unique(y_train)) < 10:
        problem_type = 'classification'
    else:
        problem_type = 'regression'
    
    print(f"Detected problem type: {problem_type}")
    
    # 7. Set up MLflow
    experiment_id = setup_mlflow()
    
    # 8. Train models
    print("\n\nTraining scikit-learn model...")
    sklearn_model = train_sklearn_model(
        X_train, y_train, X_test, y_test, 
        problem_type=problem_type,
        model_type='rf'
    )
    
    # If GPU is available and requested, train PyTorch model
    if use_gpu:
        print("\n\nTraining PyTorch model on GPU...")
        # Create a validation set from test set
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        
        pytorch_model, val_loss = train_pytorch_model(
            X_train, y_train, X_val, y_val,
            problem_type=problem_type,
            device='cuda:0',  # Use GPU
            epochs=50,
            batch_size=32
        )
    
    # 9. Register the model in MLflow Registry
    print("\n\nRegistering model in MLflow Model Registry...")
    model_name = f"{dataset_name}_{problem_type}_model"
    
    with mlflow.start_run(experiment_id=experiment_id):
        # Log scikit-learn model
        mlflow.sklearn.log_model(
            sklearn_model, 
            "model",
            registered_model_name=model_name
        )
    
    # 10. Promote model to Production
    print("\n\nPromoting model to Production...")
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=['None'])[0].version
    
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage='Production'
    )
    
    print(f"Model {model_name} version {latest_version} promoted to Production")
    
    # 11. Start model serving if requested
    serving_process = None
    if run_serving:
        serving_process = start_model_serving()
        if serving_process:
            print(f"Model serving API is running at http://localhost:8000")
            print("You can test it with a sample prediction:")
            
            # Sample data for prediction (first row of test set)
            sample_data = X_test.iloc[0].to_dict()
            request_data = {
                "model_name": model_name,
                "model_stage": "Production",
                "features": sample_data
            }
            
            print("\nAPI request example:")
            print(f"curl -X POST http://localhost:8000/predict \\
                -H \"Content-Type: application/json\" \\
                -d '{request_data}'")
    
    print("\n\nPipeline execution complete!")
    print(f"MLflow UI: http://localhost:5000")
    if serving_process:
        print(f"Model Serving API: http://localhost:8000")
        print("Press Ctrl+C to stop the servers")
        
        try:
            # Keep the script running to maintain the servers
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down servers...")
            serving_process.terminate()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run the ML Pipeline')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset in MinIO')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for PyTorch model training')
    parser.add_argument('--no-serving', action='store_true', help='Skip starting the model serving API')
    
    args = parser.parse_args()
    
    # Start MLflow server
    mlflow_process = start_mlflow_server()
    
    if mlflow_process is None:
        print("Failed to start MLflow server. Exiting.")
        return
    
    try:
        # Run the pipeline
        run_pipeline(args.dataset, use_gpu=args.gpu, run_serving=not args.no_serving)
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
    finally:
        # Clean up processes
        if mlflow_process:
            print("Stopping MLflow server...")
            mlflow_process.terminate()
    
    print("Done!")

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split  # Added here to avoid circular import
    main() 