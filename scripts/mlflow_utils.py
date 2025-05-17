import mlflow
import os
import sys

# Add the current directory to the path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.mlflow_config import *

def setup_mlflow():
    """Configure MLflow with the appropriate tracking URI and S3 credentials"""
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Configure S3/MinIO connection for artifact storage
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = AWS_ENDPOINT_URL
    
    # List available experiments to see what we're working with
    try:
        print("Available MLflow experiments:")
        experiments = mlflow.search_experiments()
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
    except Exception as e:
        print(f"Could not list experiments: {e}")
    
    # Try to get or create the experiment
    try:
        # First look for experiment by name
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        
        if experiment is None:
            # Try to create it if it doesn't exist
            print(f"Creating new experiment: {MLFLOW_EXPERIMENT_NAME}")
            experiment_id = mlflow.create_experiment(
                name=MLFLOW_EXPERIMENT_NAME, 
                artifact_location=ARTIFACT_LOCATION
            )
            print(f"Created experiment with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {MLFLOW_EXPERIMENT_NAME} (ID: {experiment_id})")
        
        return experiment_id
        
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        
        # Fallback - try to use default experiment
        try:
            print("Attempting to use default experiment")
            experiment_id = "0"  # Default experiment ID
            mlflow.set_experiment(experiment_id=experiment_id)
            return experiment_id
        except:
            print("Could not use default experiment. Trying to create a new one.")
            try:
                # Create experiment with a unique name based on timestamp
                import time
                new_exp_name = f"ml_pipeline_{int(time.time())}"
                experiment_id = mlflow.create_experiment(
                    name=new_exp_name,
                    artifact_location=ARTIFACT_LOCATION
                )
                print(f"Created new experiment: {new_exp_name} (ID: {experiment_id})")
                return experiment_id
            except Exception as e2:
                print(f"All attempts to set up MLflow failed: {e2}")
                return None

def log_model_metrics(model, X_test, y_test, model_name, params=None):
    """Log model metrics to MLflow"""
    experiment_id = setup_mlflow()
    
    if experiment_id is None:
        print("Failed to set up MLflow. Metrics will not be logged.")
        return
    
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        # Log metrics
        if hasattr(model, 'predict_proba'):
            # For classification models
            import sklearn.metrics as metrics
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            mlflow.log_metric("accuracy", metrics.accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0))
            mlflow.log_metric("recall", metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0))
            mlflow.log_metric("f1", metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            # Log confusion matrix as a figure
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            cm = metrics.confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # Save figure and log as artifact
            plt.savefig('confusion_matrix.png')
            mlflow.log_artifact('confusion_matrix.png')
            
        else:
            # For regression models
            import sklearn.metrics as metrics
            y_pred = model.predict(X_test)
            
            mlflow.log_metric("mean_squared_error", metrics.mean_squared_error(y_test, y_pred))
            mlflow.log_metric("mean_absolute_error", metrics.mean_absolute_error(y_test, y_pred))
            mlflow.log_metric("r2_score", metrics.r2_score(y_test, y_pred))

def load_model(model_name, stage='Production'):
    """Load a model from the MLflow model registry"""
    setup_mlflow()
    try:
        model_uri = f"models:/{model_name}/{stage}"
        
        # Check if it's a PyTorch model based on name
        if "pytorch" in model_name.lower():
            try:
                model = mlflow.pytorch.load_model(model_uri)
                print(f"Loaded PyTorch model: {model_name}")
                return model
            except Exception as e:
                print(f"Error loading PyTorch model, trying sklearn: {e}")
                model = mlflow.sklearn.load_model(model_uri)
                return model
        else:
            # Try sklearn first
            try:
                model = mlflow.sklearn.load_model(model_uri)
                print(f"Loaded scikit-learn model: {model_name}")
                return model
            except Exception as e:
                print(f"Error loading scikit-learn model, trying PyTorch: {e}")
                model = mlflow.pytorch.load_model(model_uri)
                return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None 