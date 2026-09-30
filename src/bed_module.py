"""
Stage 6: Hospital Bed Occupancy Prediction Module
===================================================
Project: Hospital Operations Decision Support System
Author: Health Informatics Student & AGY Pair Programmer

This module handles:
1. Data preprocessing for Bed Occupancy forecasting.
2. Chronological time-series train/test splitting.
3. Training and evaluating Scikit-learn RandomForest model.
4. Saving the trained model artifact (models/bed_model.joblib).
5. Providing a clean prediction API function returning predicted census & occupancy rate.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BED_FEATURES = [
    'occupied_beds_lag1',
    'total_hospital_beds',
    'planned_surgeries_today',
    'expected_discharges_today',
    'ed_volume_lag1',
    'is_weekend',
    'month'
]

FLU_MAPPING = {'Low': 0, 'Medium': 1, 'High': 2}
MODEL_PATH = 'models/bed_model.joblib'

def preprocess_bed_data(df):
    """
    Preprocesses Bed Occupancy DataFrame by encoding flu levels.
    """
    df_proc = df.copy()
    if 'flu_outbreak_level' in df_proc.columns:
        df_proc['flu_level_encoded'] = df_proc['flu_outbreak_level'].map(FLU_MAPPING).fillna(0)
    features = BED_FEATURES + ['flu_level_encoded']
    return df_proc[features]

def train_bed_model(csv_path="data/master_hospital_operations.csv", model_save_path=MODEL_PATH):
    """
    Trains Scikit-learn RandomForest model to predict midnight bed census.
    
    Returns:
    --------
    tuple: (trained_model, metrics_dict)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    X = preprocess_bed_data(df)
    y = df['occupied_beds']
    
    # Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Bed Module - Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4)
    }
    
    print(f"Bed Model Evaluation Results -> MAE: {metrics['MAE']} beds, RMSE: {metrics['RMSE']}, R²: {metrics['R2']}")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Bed Model saved successfully to {model_save_path}")
    
    return model, metrics

def predict_bed_occupancy(input_features_dict, model_path=MODEL_PATH):
    """
    Predicts midnight occupied beds and occupancy rate.
    
    Returns:
    --------
    tuple: (predicted_occupied_beds: int, occupancy_rate: float)
    """
    if not os.path.exists(model_path):
        train_bed_model(model_save_path=model_path)
        
    model = joblib.load(model_path)
    
    input_df = pd.DataFrame([input_features_dict])
    processed_df = preprocess_bed_data(input_df)
    
    pred_beds = int(round(model.predict(processed_df)[0]))
    total_beds = input_features_dict.get('total_hospital_beds', 500)
    
    rate = round(pred_beds / total_beds, 4)
    return pred_beds, rate

if __name__ == '__main__':
    train_bed_model()
