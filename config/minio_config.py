# MinIO configuration
MINIO_ENDPOINT = '192.168.1.200:9000'  # Update with your MinIO endpoint
MINIO_ACCESS_KEY = 'minioadmin'  # Update with your access key
MINIO_SECRET_KEY = 'minioadmin'  # Update with your secret key
MINIO_SECURE = False  # Set to True if using HTTPS
BUCKET_NAME = 'ml-datalake'

# Data paths
RAW_DATA_PREFIX = 'raw/'
PROCESSED_DATA_PREFIX = 'processed/'
MODEL_REGISTRY_PREFIX = 'models/' 