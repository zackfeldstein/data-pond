# MinIO configuration
MINIO_ENDPOINT = '192.168.1.161:32000'  # MinIO endpoint
MINIO_ACCESS_KEY = 'pipeline'  # Update with your access key
MINIO_SECRET_KEY = 'JlD2nlOi8Pe5fTUUfBBgwGPm6ZcTqyWl3z0D2PTh'  # Update with your secret key
MINIO_SECURE = False  # Set to True if using HTTPS
BUCKET_NAME = 'ml-datalake'

# Data paths
RAW_DATA_PREFIX = 'raw/'
PROCESSED_DATA_PREFIX = 'processed/'
MODEL_REGISTRY_PREFIX = 'models/' 