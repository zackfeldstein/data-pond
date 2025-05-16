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
sys.path.append('../')
from scripts.mlflow_utils import log_model_metrics, setup_mlflow
import mlflow

# PyTorch Dataset class
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        self.y = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y)
        
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
        
    def forward(self, x):
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
        
    def forward(self, x):
        return self.layers(x)

# Training function for PyTorch models
def train_pytorch_model(X_train, y_train, X_val, y_val, problem_type='classification', 
                       hidden_dim=64, learning_rate=0.001, batch_size=32, epochs=100, 
                       early_stopping_patience=10, device=None):
    
    # Determine device if not specified
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
        model = MLPClassifier(input_dim, hidden_dim, output_dim).to(device)
        criterion = nn.BCEWithLogitsLoss() if output_dim == 1 else nn.CrossEntropyLoss()
    else:  # regression
        output_dim = 1
        model = MLPRegressor(input_dim, hidden_dim, output_dim).to(device)
        criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", f"PyTorch {problem_type}")
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Reshape target for BCE loss if binary classification
                if problem_type == 'classification' and output_dim == 1:
                    y_batch = y_batch.view(-1, 1)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                # Compute loss
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
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
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
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
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val).to(device)
            outputs = model(X_val_tensor)
            
            if problem_type == 'classification':
                if output_dim == 1:  # Binary classification
                    y_pred = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy().flatten()
                else:  # Multi-class classification
                    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                
                accuracy = accuracy_score(y_val, y_pred)
                mlflow.log_metric("accuracy", accuracy)
                print(f"Validation Accuracy: {accuracy:.4f}")
                print(classification_report(y_val, y_pred))
            else:  # Regression
                y_pred = outputs.cpu().numpy().flatten()
                mse = mean_squared_error(y_val, y_pred)
                mlflow.log_metric("mean_squared_error", mse)
                print(f"Validation MSE: {mse:.4f}")
        
        # Save the PyTorch model
        mlflow.pytorch.log_model(model, "pytorch_model")
    
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