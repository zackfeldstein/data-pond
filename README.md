# ML Pipeline with MinIO and MLflow

This project demonstrates a complete machine learning pipeline using MinIO as a data lake, Jupyter for exploration, MLflow for experiment tracking, and FastAPI for model serving.

## Pipeline Components

1. **MinIO Data Lake** - Stores raw and processed data
2. **Jupyter Notebooks** - For data exploration and analysis
3. **Feature Engineering Scripts** - Python utilities for data preprocessing
4. **MLflow Tracking** - For experiment tracking and model versioning
5. **PyTorch/scikit-learn** - For model training (with GPU support)
6. **MLflow Model Registry** - To manage model versions
7. **FastAPI** - For serving models via REST API

## Directory Structure

```
.
├── config/                 # Configuration files
│   ├── minio_config.py    # MinIO connection settings
│   └── mlflow_config.py   # MLflow configuration
├── data/                  # Local data directory
│   ├── raw/               # Raw data copies (optional)
│   └── processed/         # Processed data (optional)
├── models/                # Local model storage (optional)
├── notebooks/             # Jupyter notebooks
│   └── data_exploration.ipynb  # Example exploration notebook
├── scripts/               # Utility scripts
│   ├── minio_utils.py     # MinIO interaction utilities
│   ├── feature_engineering.py # Feature engineering functions
│   ├── model_training.py  # Model training utilities
│   ├── mlflow_utils.py    # MLflow interaction utilities
│   └── model_serving.py   # FastAPI model serving
└── requirements.txt       # Python dependencies
```

## Setup Instructions

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Configure MinIO Connection**

Update the configuration in `config/minio_config.py` with your MinIO endpoint and credentials.

3. **Set Up MLflow**

```bash
# Start MLflow tracking server (with MinIO as artifact store)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://ml-datalake/mlflow/ \
  --host 0.0.0.0 \
  --port 5000
```

4. **Run Jupyter for Exploration**

```bash
jupyter notebook notebooks/
```

5. **Train Models**

You can train models using the notebook or create a separate training script using the provided utilities.

6. **Serve Models**

```bash
python -m scripts.model_serving
```

## Step-by-Step Workflow

### 1. Data Exploration

- Open `notebooks/data_exploration.ipynb`
- Connect to MinIO and list available datasets
- Load and explore a dataset
- Perform basic data analysis

### 2. Feature Engineering

- Preprocess the data using `scripts/feature_engineering.py`
- Handle missing values, encode categorical features, and normalize data
- Save processed data back to MinIO

### 3. Model Training

- Train models using scikit-learn or PyTorch
- Log parameters, metrics, and artifacts to MLflow
- Compare model performance

### 4. Model Registration

- Register the best model in MLflow Model Registry
- Transition the model to "Production" stage

### 5. Model Serving

- Start the FastAPI application
- Make predictions using the REST API

## API Endpoints

- `GET /models` - List registered models
- `POST /predict` - Make a single prediction
- `POST /batch-predict` - Make batch predictions

## GPU Support

For running PyTorch models on GPU:

1. Ensure CUDA is installed on your system
2. Use the PyTorch training functions with `device='cuda:0'`

## Example:

```python
from scripts.model_training import train_pytorch_model

# Train a model on GPU
model, val_loss = train_pytorch_model(
    X_train, y_train, X_val, y_val,
    problem_type='classification',
    device='cuda:0'  # Use GPU
)
```

## MinIO Data Organization

- `ml-datalake/raw/` - Raw datasets
- `ml-datalake/processed/` - Processed datasets
- `ml-datalake/mlflow/` - MLflow artifacts
- `ml-datalake/models/` - Saved models

## Notes

- Update MinIO and MLflow connection parameters based on your setup
- For production use, consider adding authentication to the API
- You can extend the feature engineering functions for domain-specific use cases # data-pond
