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
import torch

# Force CUDA to be used at script startup
if torch.cuda.is_available():
    print("FORCING CUDA USAGE AT STARTUP")
    # Set PyTorch to use CUDA
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # Force all tensors to be created on GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Set PyTorch to use GPU by default
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Add the current directory to the path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
    
    # 7. Set up MLflow (optional)
    try:
        experiment_id = setup_mlflow()
        mlflow_available = experiment_id is not None
    except Exception as e:
        print(f"MLflow setup failed: {e}")
        print("Continuing without MLflow...")
        experiment_id = None
        mlflow_available = False
    
    # 8. Train PyTorch model
    print("\n\nTraining PyTorch model...")
    # Create a validation set from test set
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    # Forced GPU usage if requested
    if use_gpu:
        if torch.cuda.is_available():
            # Really make sure we're using CUDA
            print("FORCING CUDA FOR MODEL TRAINING")
            # Set environment variables
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            # Empty cache
            torch.cuda.empty_cache()
            # Create a CUDA device and set as default
            device = torch.device('cuda:0')
            # Set default tensor type to CUDA
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
            # Print GPU information
            print(f"GPU device being used: {torch.cuda.get_device_name(0)}")
            print(f"Is CUDA initialized: {torch.cuda.is_initialized()}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Current device: {torch.cuda.current_device()}")
            
            # Create a test tensor to verify GPU is working
            test_tensor = torch.tensor([1.0, 2.0, 3.0])
            print(f"Test tensor device: {test_tensor.device}")
            
            # Move test tensor to GPU explicitly
            test_tensor = test_tensor.cuda()
            print(f"Test tensor after .cuda(): {test_tensor.device}")
        else:
            print("WARNING: GPU requested but CUDA is not available. Using CPU instead.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    print(f"FINAL DEVICE SELECTION: {device}")
    
    # Convert the data to better format for PyTorch if needed
    if isinstance(X_train, pd.DataFrame):
        # Ensure all columns are numeric to avoid the "setting an array element with a sequence" error
        X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_val_numeric = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test_numeric = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    else:
        X_train_numeric = X_train
        X_val_numeric = X_val
        X_test_numeric = X_test
    
    pytorch_model, val_loss = train_pytorch_model(
        X_train_numeric, y_train, X_val_numeric, y_val,
        problem_type=problem_type,
        device=device,
        epochs=50,
        batch_size=32
    )
    
    # 9. Handle model registration if MLflow is available
    model_name = f"{dataset_name}_{problem_type}_pytorch_model"
    latest_version = None
    
    if mlflow_available:
        print("\n\nRegistering model in MLflow Model Registry...")
        
        try:
            with mlflow.start_run(experiment_id=experiment_id):
                # Log PyTorch model
                print(f"Logging PyTorch model as '{model_name}'")
                mlflow.pytorch.log_model(
                    pytorch_model, 
                    "model",
                    registered_model_name=model_name
                )
                print("Model logged successfully")
        except Exception as e:
            print(f"Error logging model to MLflow: {e}")
            print("Continuing with pipeline without MLflow model registration...")
            mlflow_available = False
    else:
        print("Skipping MLflow model registration (MLflow not available)")
    
    # 10. Promote model to Production if MLflow is available
    if mlflow_available:
        try:
            print("\n\nPromoting model to Production...")
            client = mlflow.tracking.MlflowClient()
            
            # Get latest model version
            try:
                latest_versions = client.get_latest_versions(model_name, stages=['None'])
                if latest_versions:
                    latest_version = latest_versions[0].version
                    
                    # Transition to production
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version,
                        stage='Production'
                    )
                    print(f"Model {model_name} version {latest_version} promoted to Production")
                else:
                    print(f"No versions found for model {model_name}")
            except Exception as e:
                print(f"Error retrieving model versions: {e}")
        except Exception as e:
            print(f"Error promoting model to production: {e}")
    else:
        print("Skipping model promotion (MLflow not available)")
        
    # Save model locally as backup if MLflow failed
    if not mlflow_available:
        try:
            local_model_path = f"models/{model_name}.pt"
            print(f"Saving model locally to {local_model_path}")
            torch.save(pytorch_model.state_dict(), local_model_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model locally: {e}")
    
    # 11. Start model serving if requested
    serving_process = None
    if run_serving:
        serving_process = start_model_serving()
        if serving_process:
            print(f"Model serving API is running at http://localhost:8000")
            print("You can test it with a sample prediction:")
            
            # Sample data for prediction (first row of test set)
            if isinstance(X_test_numeric, pd.DataFrame):
                sample_data = X_test_numeric.iloc[0].to_dict()
            else:
                # If not a DataFrame, convert to dict with feature indices as keys
                sample_data = {f"feature_{i}": val for i, val in enumerate(X_test_numeric[0])}
                
            request_data = {
                "model_name": model_name,
                "model_stage": "Production",
                "features": sample_data
            }
            
            print("\nAPI request example:")
            print(f"curl -X POST http://localhost:8000/predict -H \"Content-Type: application/json\" -d '{request_data}'")
    
    print("\n\nPipeline execution complete!")
    print(f"MLflow UI: http://localhost:8080")
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
    parser.add_argument('--start-mlflow', action='store_true', help='Start MLflow server (not needed if already running)')
    
    args = parser.parse_args()
    
    mlflow_process = None
    # Only start MLflow server if explicitly requested
    if args.start_mlflow:
        mlflow_process = start_mlflow_server()
        if mlflow_process is None:
            print("Failed to start MLflow server. Exiting.")
            return
    else:
        print("Using existing MLflow server at http://localhost:8080")
    
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