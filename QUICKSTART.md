# ML Pipeline Quick Start Guide

This guide will help you get started with the ML pipeline using MinIO, MLflow, and PyTorch/scikit-learn.

## Prerequisites

- Python 3.7+
- Access to MinIO server
- GPU machine (optional, for PyTorch acceleration)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure MinIO and MLflow

Update the MinIO and MLflow configuration files with your settings:

- `config/minio_config.py`: Update with your MinIO endpoint and credentials
- `config/mlflow_config.py`: Update with your MLflow tracking server settings

## Step 3: Create and Upload Sample Data

Create sample datasets and upload them to MinIO:

```bash
# Create and upload a retail dataset sample
python scripts/upload_data.py --dataset retail --create-sample --sample-rows 1000

# Create and upload an academic performance dataset sample
python scripts/upload_data.py --dataset academic-performance --create-sample --sample-rows 1000

# Create and upload a planetary dataset sample (good for regression)
python scripts/upload_data.py --dataset planets --create-sample --sample-rows 1000
```

You can also upload your own datasets:

```bash
# Upload a single file
python scripts/upload_data.py --dataset my_dataset --file /path/to/data.csv

# Upload all files in a directory
python scripts/upload_data.py --dataset my_dataset --directory /path/to/data_folder
```

## Step 4: Run the ML Pipeline

Run the complete ML pipeline from data loading to model serving:

```bash
# Run with default scikit-learn models
python scripts/run_pipeline.py --dataset academic-performance

# Run with GPU acceleration for PyTorch models
python scripts/run_pipeline.py --dataset planets --gpu

# Run without starting the model serving API
python scripts/run_pipeline.py --dataset retail --no-serving
```

The pipeline will:
1. Load data from MinIO
2. Perform feature engineering
3. Train models and log to MLflow
4. Register the best model
5. Start the model serving API

## Step 5: Explore in Jupyter

Start Jupyter to explore datasets and run the notebook interactively:

```bash
jupyter notebook notebooks/
```

Open `data_exploration.ipynb` and follow along with the steps.

## Step 6: Access MLflow UI

Access the MLflow UI to view experiment results:

```
http://localhost:5000
```

## Step 7: Make Predictions with the API

Once the model serving API is running, you can make predictions:

```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"model_name": "retail_classification_model", "model_stage": "Production", "features": {...}}'
```

## Troubleshooting

- **MinIO Connection Issues**: Verify your MinIO credentials and endpoint in `config/minio_config.py`
- **MLflow Tracking Errors**: Ensure the MLflow server is running and configured correctly
- **GPU Not Detected**: If using `--gpu` flag, ensure CUDA is properly installed and PyTorch can access the GPU

## Next Steps

1. Customize the feature engineering for your specific datasets
2. Add more models to the training pipeline
3. Implement hyperparameter tuning
4. Set up authentication for the FastAPI server 