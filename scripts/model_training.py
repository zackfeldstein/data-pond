import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# Add the current directory to the path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.mlflow_utils import log_model_metrics, setup_mlflow
import mlflow

# PyTorch Dataset class
class TabularDataset(Dataset):
    def __init__(self, X, y):
        print(f"X type: {type(X)}")
        if isinstance(X, pd.DataFrame):
            print(f"DataFrame columns: {X.columns}")
            print(f"DataFrame dtypes: {X.dtypes}")
            
            # Simple and robust approach - convert entire DataFrame at once
            # Convert all columns to string first, then to numeric
            # This handles mixed types safely by converting everything to strings first
            
            try:
                # First attempt - try direct conversion with dtype option
                X_numeric = X.select_dtypes(include=['number']).copy()
                
                # For non-numeric columns, try to convert what we can
                for col in X.columns:
                    if col not in X_numeric.columns:
                        try:
                            X_numeric[col] = pd.to_numeric(X[col], errors='coerce')
                        except:
                            # If conversion fails, create dummy column of zeros
                            X_numeric[col] = 0
                
                # Fill any remaining NaN values
                X_numeric = X_numeric.fillna(0)
                
                # Convert to numpy array and then to tensor
                X_array = X_numeric.values.astype(np.float32)
                self.X = torch.FloatTensor(X_array)
                
            except Exception as e:
                print(f"Error in DataFrame conversion: {e}")
                
                # Ultimate fallback - create tensor of zeros
                X_zeros = np.zeros((len(X), len(X.columns)), dtype=np.float32)
                self.X = torch.FloatTensor(X_zeros)
                print("WARNING: Using zeros tensor as fallback due to conversion error")
                
        elif isinstance(X, np.ndarray):
            print(f"NumPy array shape: {X.shape}, dtype: {X.dtype}")
            
            try:
                # Handle different array types
                if X.dtype == np.object_ or X.dtype == 'O':
                    # Print examples for debugging
                    print(f"Working with object array. Sample: {X.flatten()[0] if X.size > 0 else 'empty'}")
                    
                    # Create a DataFrame for easier handling
                    X_df = pd.DataFrame(X)
                    # Convert all columns to numeric
                    X_numeric = X_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
                    X_numeric = X_numeric.astype(np.float32)
                    
                elif X.dtype.kind in ['i', 'f', 'u']:  # Integer, float, unsigned int
                    # Numeric arrays can be converted directly
                    X_numeric = X.astype(np.float32)
                else:
                    # For other array types (e.g., string arrays), create zeros
                    print(f"Unsupported array dtype: {X.dtype}")
                    X_numeric = np.zeros(X.shape, dtype=np.float32)
                
                # Create tensor from array
                self.X = torch.FloatTensor(X_numeric)
                
            except Exception as e:
                print(f"Error converting numpy array: {e}")
                # Emergency fallback
                X_numeric = np.zeros(X.shape, dtype=np.float32)
                self.X = torch.FloatTensor(X_numeric)
                print("WARNING: Using zeros tensor as fallback due to array conversion error")
        else:
            # For other types, convert to numpy first then to tensor
            try:
                np_array = np.array(X, dtype=np.float32)
                self.X = torch.FloatTensor(np_array)
            except Exception as e:
                print(f"Error converting to tensor: {e}")
                # Fallback to zeros
                X_zeros = np.zeros((len(X), 1), dtype=np.float32)
                self.X = torch.FloatTensor(X_zeros)
            
        print(f"y type: {type(y)}")
        
        try:
            # Convert y to numeric tensor
            if isinstance(y, pd.Series):
                print(f"y dtype: {y.dtype}")
                
                if pd.api.types.is_numeric_dtype(y):
                    # Already numeric
                    self.y = torch.FloatTensor(y.values)
                else:
                    # Categorical data - encode as integers
                    try:
                        # First try to convert to numeric directly in case it's actually numeric
                        numeric_y = pd.to_numeric(y, errors='coerce')
                        if numeric_y.isna().sum() == 0:  # All converted successfully
                            self.y = torch.FloatTensor(numeric_y.values)
                        else:
                            # Label encode categorical data
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y_encoded = le.fit_transform(y.astype(str))
                            self.y = torch.LongTensor(y_encoded)
                            print(f"Label encoded target with {len(le.classes_)} classes")
                    except Exception as e:
                        print(f"Error encoding target: {e}")
                        # Last resort - use zeros
                        self.y = torch.zeros(len(y), dtype=torch.long)
                        
            elif isinstance(y, np.ndarray):
                print(f"y shape: {y.shape}, dtype: {y.dtype}")
                
                if np.issubdtype(y.dtype, np.number):
                    # Already numeric
                    self.y = torch.FloatTensor(y)
                else:
                    # Try to encode categorical data
                    try:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y.astype(str).flatten())
                        self.y = torch.LongTensor(y_encoded)
                    except Exception as e:
                        print(f"Error encoding numpy target: {e}")
                        self.y = torch.zeros(len(y), dtype=torch.long)
            else:
                # Try to convert directly
                try:
                    y_array = np.array(y, dtype=np.float32)
                    self.y = torch.FloatTensor(y_array)
                except:
                    # Try as categorical
                    try:
                        y_list = list(y)
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_encoded = le.fit_transform([str(item) for item in y_list])
                        self.y = torch.LongTensor(y_encoded)
                    except:
                        # Last resort
                        self.y = torch.zeros(len(self.X), dtype=torch.long)
                        
        except Exception as e:
            print(f"Unexpected error processing target: {e}")
            # Ultimate fallback
            self.y = torch.zeros(len(self.X), dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Simple PyTorch MLP model for classification
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Check if CUDA is available and move model to GPU
        if torch.cuda.is_available():
            print("Moving MLPClassifier to CUDA device on initialization")
            self.cuda()
        
    def forward(self, x):
        # Make sure input is on the same device as model
        if next(self.parameters()).is_cuda and not x.is_cuda:
            x = x.cuda()
        return self.layers(x)

# Simple PyTorch MLP model for regression
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(MLPRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Check if CUDA is available and move model to GPU
        if torch.cuda.is_available():
            print("Moving MLPRegressor to CUDA device on initialization")
            self.cuda()
        
    def forward(self, x):
        # Make sure input is on the same device as model
        if next(self.parameters()).is_cuda and not x.is_cuda:
            x = x.cuda()
        return self.layers(x)

# Training function for PyTorch models
def train_pytorch_model(X_train, y_train, X_val, y_val, problem_type='classification', 
                       hidden_dim=64, learning_rate=0.001, batch_size=32, epochs=100, 
                       early_stopping_patience=10, device=None):
    
    # Force CUDA usage if available
    if torch.cuda.is_available() and (device is None or (isinstance(device, torch.device) and device.type.startswith('cuda'))):
        print("FORCING CUDA IN MODEL TRAINING FUNCTION")
        # Set up CUDA device
        cuda_device = 0  # Use first GPU
        torch.cuda.set_device(cuda_device)
        # Use CUDA tensors by default
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device(f'cuda:{cuda_device}')
        # Show CUDA memory usage
        print(f"Initial CUDA memory: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
        torch.cuda.empty_cache()
    else:
        # CPU fallback
        torch.set_default_tensor_type('torch.FloatTensor')
        if device is None:
            device = torch.device('cpu')
        elif not isinstance(device, torch.device):
            device = torch.device('cpu')
    
    # Make sure we have a device object
    if isinstance(device, str):
        device = torch.device(device)
        
    # Override device to force CUDA if available
    if torch.cuda.is_available() and not device.type.startswith('cuda'):
        print("DEVICE WAS CPU BUT CUDA IS AVAILABLE - FORCING CUDA")
        device = torch.device('cuda:0')
        
    # Print very explicitly what device we're using
    print(f"TRAINING MODEL ON: {device}")
    
    # Print detailed device information
    print(f"Training on device: {device}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
        torch.cuda.empty_cache()
    
    # Create datasets and dataloaders
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Determine input and output dimensions
    input_dim = X_train.shape[1]
    
    if problem_type == 'classification':
        num_classes = len(np.unique(y_train))
        output_dim = 1 if num_classes == 2 else num_classes
        
        # Create model and move to device
        model = MLPClassifier(input_dim, hidden_dim, output_dim)
        model = model.to(device)  # Explicitly move model to the device
        
        print(f"Model created and moved to {device}")
        
        # Check if model is on the correct device
        param_device = next(model.parameters()).device
        print(f"Model parameters are on: {param_device}")
        
        # Create loss function
        criterion = nn.BCEWithLogitsLoss() if output_dim == 1 else nn.CrossEntropyLoss()
        
    else:  # regression
        output_dim = 1
        # Create model and move to device
        model = MLPRegressor(input_dim, hidden_dim, output_dim)
        model = model.to(device)  # Explicitly move model to the device
        
        print(f"Model created and moved to {device}")
        
        # Check if model is on the correct device
        param_device = next(model.parameters()).device
        print(f"Model parameters are on: {param_device}")
        
        # Create loss function
        criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print memory usage after model creation
    if device.type == 'cuda':
        print(f"GPU memory after model creation: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Make MLflow tracking completely optional
    mlflow_active = False
    try:
        mlflow.set_tracking_uri("http://localhost:8080")
        run = mlflow.start_run()
        mlflow_active = True
        
        # Log parameters if MLflow is active
        mlflow.log_param("model_type", f"PyTorch {problem_type}")
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
    except Exception as e:
        print(f"MLflow tracking unavailable: {e}")
        print("Continuing without MLflow tracking...")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                # Move data to the correct device and convert to the right type
                X_batch = X_batch.to(device)
                
                # Convert target to the appropriate type based on the problem
                if problem_type == 'classification':
                    if output_dim == 1:  # Binary classification
                        y_batch = y_batch.float().to(device).view(-1, 1)
                    else:  # Multi-class
                        y_batch = y_batch.long().to(device)
                else:  # Regression
                    y_batch = y_batch.float().to(device)
                
                # Print device information for the first batch of the first epoch
                if epoch == 0 and train_loss == 0:
                    print(f"X_batch device: {X_batch.device}")
                    print(f"y_batch device: {y_batch.device}")
                    print(f"Model device: {next(model.parameters()).device}")
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                # Compute loss
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
                # Print memory usage for the first batch
                if epoch == 0 and train_loss == 0 and device.type == 'cuda':
                    print(f"GPU memory during training: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            
            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    # Reshape target for BCE loss if binary classification
                    if problem_type == 'classification' and output_dim == 1:
                        y_batch = y_batch.view(-1, 1)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                
                val_loss /= len(val_loader.dataset)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log metrics to MLflow if active
            if mlflow_active:
                try:
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                except Exception as e:
                    print(f"Error logging metrics to MLflow: {e}")
                    mlflow_active = False  # Disable for future iterations
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
                # Final evaluation
        model.eval()
        try:
            with torch.no_grad():
                # Convert validation data safely
                try:
                    # Handle different data types
                    if isinstance(X_val, pd.DataFrame):
                        # Convert DataFrame to numeric tensor
                        X_val_numeric = X_val.fillna(0).astype('float32', errors='ignore')
                        X_val_array = X_val_numeric.values.astype(np.float32)
                    elif isinstance(X_val, np.ndarray):
                        # Convert numpy array to numeric tensor
                        if X_val.dtype == np.object_ or X_val.dtype == 'O':
                            # Use zeros for object arrays
                            X_val_array = np.zeros(X_val.shape, dtype=np.float32)
                        else:
                            # Convert numeric arrays directly
                            X_val_array = X_val.astype(np.float32)
                    else:
                        # Handle other types
                        X_val_array = np.zeros((len(X_val), X_train.shape[1]), dtype=np.float32)
                    
                    # Create tensor and move to device
                    X_val_tensor = torch.tensor(X_val_array, dtype=torch.float32).to(device)
                    outputs = model(X_val_tensor)
                    
                    # Process outputs based on problem type
                    if problem_type == 'classification':
                        if output_dim == 1:
                            y_pred = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy().flatten()
                        else:
                            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_val, y_pred)
                        print(f"Validation Accuracy: {accuracy:.4f}")
                        print(classification_report(y_val, y_pred))
                        
                        # Log to MLflow if active
                        if mlflow_active:
                            try:
                                mlflow.log_metric("accuracy", accuracy)
                            except Exception as e:
                                print(f"Error logging metrics: {e}")
                                mlflow_active = False
                    else:
                        # Regression metrics
                        y_pred = outputs.cpu().numpy().flatten()
                        mse = mean_squared_error(y_val, y_pred)
                        print(f"Validation MSE: {mse:.4f}")
                        
                        # Log to MLflow if active
                        if mlflow_active:
                            try:
                                mlflow.log_metric("mean_squared_error", mse)
                            except Exception as e:
                                print(f"Error logging metrics: {e}")
                                mlflow_active = False
                except Exception as e:
                    print(f"Error during validation: {e}")
                    print("Skipping final validation step")
        except Exception as e:
            print(f"Error in evaluation block: {e}")
            print("Validation could not be completed")
        
        # Save the PyTorch model if MLflow is active
        if mlflow_active:
            try:
                mlflow.pytorch.log_model(model, "pytorch_model")
            except Exception as e:
                print(f"Error saving model to MLflow: {e}")
                mlflow_active = False
    
    # End the MLflow run if it was started
    if mlflow_active:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Error ending MLflow run: {e}")
    
    return model, best_val_loss

# Function to train scikit-learn models
def train_sklearn_model(X_train, y_train, X_test, y_test, problem_type='classification', model_type='rf'):
    """Train a scikit-learn model with MLflow tracking"""
    
    # Choose model based on problem type and model type
    if problem_type == 'classification':
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            params = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
        else:  # logistic regression
            model = LogisticRegression(max_iter=1000, random_state=42)
            params = {'C': 1.0, 'penalty': 'l2', 'max_iter': 1000, 'random_state': 42}
    else:  # regression
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            params = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
        else:  # linear regression
            model = LinearRegression()
            params = {}
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Log metrics with MLflow
    model_name = f"{model_type}_{problem_type}"
    log_model_metrics(model, X_test, y_test, model_name, params)
    
    return model 