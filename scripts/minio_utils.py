from minio import Minio
import os
import pandas as pd
import io
import sys

# Add the current directory to the path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.minio_config import *

def get_minio_client():
    """Create and return a MinIO client"""
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

def list_datasets():
    """List available datasets in the raw data bucket"""
    client = get_minio_client()
    objects = client.list_objects(BUCKET_NAME, prefix=RAW_DATA_PREFIX, recursive=False)
    datasets = [obj.object_name.replace(RAW_DATA_PREFIX, '').rstrip('/') 
                for obj in objects if obj.object_name.endswith('/')]
    return datasets

def list_files_in_dataset(dataset_name):
    """List files in a specific dataset"""
    client = get_minio_client()
    prefix = f"{RAW_DATA_PREFIX}{dataset_name}/"
    objects = client.list_objects(BUCKET_NAME, prefix=prefix, recursive=True)
    files = [obj.object_name.replace(prefix, '') for obj in objects 
             if not obj.object_name.endswith('/')]
    return files

def read_csv_from_minio(dataset_name, file_name):
    """Read a CSV file from MinIO into a pandas DataFrame"""
    client = get_minio_client()
    object_name = f"{RAW_DATA_PREFIX}{dataset_name}/{file_name}"
    
    try:
        response = client.get_object(BUCKET_NAME, object_name)
        data = pd.read_csv(io.BytesIO(response.data))
        return data
    except Exception as e:
        print(f"Error reading {object_name}: {e}")
        return None
    finally:
        response.close()
        response.release_conn()
        
def save_df_to_minio(df, dataset_name, file_name, prefix=PROCESSED_DATA_PREFIX):
    """Save a pandas DataFrame to MinIO as a CSV file"""
    client = get_minio_client()
    object_name = f"{prefix}{dataset_name}/{file_name}"
    
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    csv_buffer = io.BytesIO(csv_bytes)
    
    try:
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=csv_buffer,
            length=len(csv_bytes),
            content_type='text/csv'
        )
        print(f"Successfully saved {object_name} to MinIO")
    except Exception as e:
        print(f"Error saving to {object_name}: {e}") 