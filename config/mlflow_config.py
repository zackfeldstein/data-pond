# MLflow configuration
MLFLOW_TRACKING_URI = 'http://localhost:5000'  # Local MLflow server
MLFLOW_EXPERIMENT_NAME = 'ml-pipeline-demo'

# Artifact storage - using MinIO
ARTIFACT_LOCATION = 's3://ml-datalake/mlflow/'

# S3/MinIO connection for MLflow
AWS_ACCESS_KEY_ID = 'minioadmin'  # Same as MinIO credentials
AWS_SECRET_ACCESS_KEY = 'minioadmin'
AWS_ENDPOINT_URL = 'http://192.168.1.200:9000' 