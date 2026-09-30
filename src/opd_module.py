"""
Stage 4: OPD Patient Volume Prediction Module
===============================================
Project: Hospital Operations Decision Support System
Author: Health Informatics Student & AGY Pair Programmer

This module handles:
1. Data preprocessing & feature encoding for OPD predictions.
2. Chronological time-series train/test splitting.
3. Training and evaluating Scikit-learn Random Forest & Ridge models.
4. Saving the trained model artifact (models/opd_model.joblib).
5. Providing a clean prediction API function for the dashboard.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Feature column definitions for OPD Module
OPD_FEATURES = [
    'opd_appointments_scheduled',
    'opd_cancellation_rate',
    'opd_walkins_expected',
    'opd_doctors_on_duty',
    'opd_volume_lag1',
    'is_weekend',
    'is_public_holiday',
    'month',
    'temperature_c',
    'rainfall_mm'
]

OPD_TARGET = 'opd_total_volume'
MODEL_PATH = 'models/opd_model.joblib'

def train_opd_model(csv_path="data/master_hospital_operations.csv", model_save_path=MODEL_PATH):
    """
    Trains Scikit-learn RandomForest model to predict OPD total volume.
    
    Returns:
    --------
    tuple: (trained_model, metrics_dict)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Feature & Target Isolation
    X = df[OPD_FEATURES]
    y = df[OPD_TARGET]
    
    # 2. Chronological Train-Test Split (80% Train, 20% Test) to prevent time leakage
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"OPD Module - Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 3. Model Training (Random Forest Regressor)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4)
    }
    
    print(f"OPD Model Evaluation Results -> MAE: {metrics['MAE']} patients, RMSE: {metrics['RMSE']}, R²: {metrics['R2']}")
    
    # 5. Save Model Artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"OPD Model saved successfully to {model_save_path}")
    
    return model, metrics

def predict_opd_volume(input_features_dict, model_path=MODEL_PATH):
    """
    Predicts OPD patient volume for a single day given input features dictionary.
    """
    if not os.path.exists(model_path):
        train_opd_model(model_save_path=model_path)
        
    model = joblib.load(model_path)
    
    # Format input as 1-row DataFrame
    input_df = pd.DataFrame([input_features_dict])
    input_df = input_df[OPD_FEATURES]  # Enforce feature order
    
    prediction = model.predict(input_df)[0]
    return int(round(prediction))

if __name__ == '__main__':
    train_opd_model()
