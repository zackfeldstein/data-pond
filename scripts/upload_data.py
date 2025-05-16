#!/usr/bin/env python

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import io
import numpy as np

# Add the parent directory to the path
sys.path.append('../')
from scripts.minio_utils import get_minio_client
from config.minio_config import BUCKET_NAME, RAW_DATA_PREFIX

def ensure_bucket_exists(client, bucket_name):
    """Ensure that the bucket exists, creating it if necessary"""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Created bucket: {bucket_name}")
        return True
    except Exception as e:
        print(f"Error ensuring bucket exists: {e}")
        return False

def upload_file(client, dataset_name, file_path):
    """Upload a single file to MinIO"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False
        
        # Determine content type based on file extension
        content_type = "text/csv" if file_path.suffix.lower() == ".csv" else "application/octet-stream"
        
        # Determine object name in MinIO
        object_name = f"{RAW_DATA_PREFIX}{dataset_name}/{file_path.name}"
        
        # Upload the file
        client.fput_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            file_path=str(file_path),
            content_type=content_type
        )
        
        print(f"Uploaded {file_path} to {object_name}")
        return True
    except Exception as e:
        print(f"Error uploading file {file_path}: {e}")
        return False

def upload_directory(client, dataset_name, directory_path):
    """Upload all files in a directory to MinIO"""
    try:
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            print(f"Directory not found: {directory_path}")
            return False
        
        # Create the dataset directory in MinIO if it doesn't exist
        dataset_dir = f"{RAW_DATA_PREFIX}{dataset_name}/"
        try:
            client.put_object(
                bucket_name=BUCKET_NAME,
                object_name=dataset_dir,
                data=io.BytesIO(b""),
                length=0
            )
        except:
            pass  # Ignore errors if directory already exists
        
        # Upload all files in the directory
        success_count = 0
        file_count = 0
        
        for file_path in directory_path.glob("*"):
            if file_path.is_file():
                file_count += 1
                if upload_file(client, dataset_name, file_path):
                    success_count += 1
        
        print(f"Uploaded {success_count}/{file_count} files from {directory_path} to {dataset_dir}")
        return success_count > 0
    except Exception as e:
        print(f"Error uploading directory {directory_path}: {e}")
        return False

def create_sample_dataset(output_dir, dataset_name, rows=1000, cols=10):
    """Create a sample dataset for testing"""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / f"{dataset_name}_sample.csv"
        
        print(f"Creating sample dataset with {rows} rows and {cols} columns")
        
        # Generate sample data based on dataset type
        if dataset_name == "retail":
            # Retail sales data
            df = pd.DataFrame({
                'customer_id': [f"CUST{i:04d}" for i in range(1, rows + 1)],
                'transaction_date': pd.date_range(start='2023-01-01', periods=rows).astype(str),
                'product_id': [f"PROD{i:03d}" for i in range(1, rows + 1)],
                'product_category': pd.Series(pd.Categorical.from_codes(
                    np.random.randint(0, 5, size=rows),
                    categories=['Electronics', 'Clothing', 'Food', 'Home', 'Beauty']
                )),
                'quantity': np.random.randint(1, 10, size=rows),
                'unit_price': np.round(np.random.uniform(10, 1000, size=rows), 2),
                'total_amount': None,
                'payment_method': pd.Series(pd.Categorical.from_codes(
                    np.random.randint(0, 3, size=rows),
                    categories=['Credit Card', 'Cash', 'Mobile Payment']
                )),
                'store_id': [f"STORE{i:02d}" for i in np.random.randint(1, 20, size=rows)],
                'customer_satisfaction': np.random.randint(1, 6, size=rows)
            })
            # Calculate total amount
            df['total_amount'] = df['quantity'] * df['unit_price']
            
        elif dataset_name == "academic-performance":
            # Academic performance data
            df = pd.DataFrame({
                'student_id': [f"STUD{i:04d}" for i in range(1, rows + 1)],
                'gender': pd.Series(pd.Categorical.from_codes(
                    np.random.randint(0, 2, size=rows),
                    categories=['Male', 'Female']
                )),
                'age': np.random.randint(18, 30, size=rows),
                'major': pd.Series(pd.Categorical.from_codes(
                    np.random.randint(0, 5, size=rows),
                    categories=['Computer Science', 'Engineering', 'Mathematics', 'Physics', 'Business']
                )),
                'study_hours_per_week': np.random.randint(0, 40, size=rows),
                'sleep_hours_per_day': np.random.randint(4, 12, size=rows),
                'previous_gpa': np.round(np.random.uniform(1.0, 4.0, size=rows), 2),
                'attendance_rate': np.round(np.random.uniform(0.5, 1.0, size=rows), 2),
                'extracurricular_activities': np.random.randint(0, 4, size=rows),
                'final_grade': np.random.randint(40, 100, size=rows)
            })
            
        elif dataset_name == "planets":
            # Planetary data (for regression)
            df = pd.DataFrame({
                'planet_id': range(1, rows + 1),
                'mass_earth_units': np.round(np.random.uniform(0.1, 2000, size=rows), 3),
                'radius_earth_units': np.round(np.random.uniform(0.1, 95, size=rows), 3),
                'orbital_period_days': np.round(np.random.uniform(10, 30000, size=rows), 1),
                'distance_from_star_AU': np.round(np.random.uniform(0.1, 30, size=rows), 2),
                'star_mass_solar_units': np.round(np.random.uniform(0.1, 10, size=rows), 2),
                'star_temperature_K': np.round(np.random.uniform(2000, 12000, size=rows), 0),
                'metallicity': np.round(np.random.uniform(-1.5, 0.5, size=rows), 2),
                'eccentricity': np.round(np.random.uniform(0, 0.9, size=rows), 3),
                'surface_temperature_K': np.round(np.random.uniform(50, 700, size=rows), 1)
            })
        else:
            # Generic dataset with random values
            df = pd.DataFrame(
                np.random.randn(rows, cols),
                columns=[f'feature_{i}' for i in range(cols-1)] + ['target']
            )
        
        # Save to CSV file
        df.to_csv(file_path, index=False)
        print(f"Created sample dataset: {file_path}")
        
        return file_path
    except Exception as e:
        print(f"Error creating sample dataset: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Upload data to MinIO data lake')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset in MinIO')
    parser.add_argument('--file', type=str, help='Path to a single file to upload')
    parser.add_argument('--directory', type=str, help='Path to a directory containing files to upload')
    parser.add_argument('--create-sample', action='store_true', help='Create and upload a sample dataset')
    parser.add_argument('--sample-rows', type=int, default=1000, help='Number of rows for sample dataset')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory for sample dataset')
    
    args = parser.parse_args()
    
    if not any([args.file, args.directory, args.create_sample]):
        print("Error: Must specify one of --file, --directory, or --create-sample")
        parser.print_help()
        return
    
    # Connect to MinIO
    client = get_minio_client()
    if not client:
        print("Failed to connect to MinIO")
        return
    
    # Ensure bucket exists
    if not ensure_bucket_exists(client, BUCKET_NAME):
        print(f"Failed to ensure bucket {BUCKET_NAME} exists")
        return
    
    # Upload data based on provided arguments
    if args.create_sample:
        sample_file = create_sample_dataset(args.output_dir, args.dataset, rows=args.sample_rows)
        if sample_file:
            upload_file(client, args.dataset, sample_file)
    elif args.file:
        upload_file(client, args.dataset, args.file)
    elif args.directory:
        upload_directory(client, args.dataset, args.directory)

if __name__ == '__main__':
    main() 