"""
Stage 5: Emergency (ED) Patient Volume Prediction Module
=========================================================
Project: Hospital Operations Decision Support System
Author: Health Informatics Student & AGY Pair Programmer

This module handles:
1. Data preprocessing & feature encoding for Emergency predictions.
2. Chronological time-series train/test splitting.
3. Training and evaluating Scikit-learn Gradient Boosting Regressor model.
4. Saving the trained model artifact (models/ed_model.joblib).
5. Providing a clean prediction API function for the dashboard.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Feature column definitions for ED Module
ED_NUMERICAL_FEATURES = [
    'ed_volume_lag1',
    'ed_doctors_on_duty',
    'nursing_staff_on_duty',
    'is_weekend',
    'is_public_holiday',
    'month',
    'temperature_c',
    'rainfall_mm',
    'air_quality_index'
]

FLU_MAPPING = {'Low': 0, 'Medium': 1, 'High': 2}
MODEL_PATH = 'models/ed_model.joblib'

def preprocess_ed_data(df):
    """
    Preprocesses ED DataFrame by encoding categorical variables.
    """
    df_proc = df.copy()
    if 'flu_outbreak_level' in df_proc.columns:
        df_proc['flu_level_encoded'] = df_proc['flu_outbreak_level'].map(FLU_MAPPING).fillna(0)
    features = ED_NUMERICAL_FEATURES + ['flu_level_encoded']
    return df_proc[features]

def train_ed_model(csv_path="data/master_hospital_operations.csv", model_save_path=MODEL_PATH):
    """
    Trains Scikit-learn GradientBoostingRegressor to predict ED total volume.
    
    Returns:
    --------
    tuple: (trained_model, metrics_dict)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Preprocess features & target
    X = preprocess_ed_data(df)
    y = df['ed_total_volume']
    
    # Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"ED Module - Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Model Training
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=5,
        random_state=42
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
    
    print(f"ED Model Evaluation Results -> MAE: {metrics['MAE']} patients, RMSE: {metrics['RMSE']}, R²: {metrics['R2']}")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"ED Model saved successfully to {model_save_path}")
    
    return model, metrics

def predict_ed_volume(input_features_dict, model_path=MODEL_PATH):
    """
    Predicts Emergency patient volume for a single day.
    """
    if not os.path.exists(model_path):
        train_ed_model(model_save_path=model_path)
        
    model = joblib.load(model_path)
    
    input_df = pd.DataFrame([input_features_dict])
    processed_df = preprocess_ed_data(input_df)
    
    prediction = model.predict(processed_df)[0]
    return int(round(prediction))

if __name__ == '__main__':
    train_ed_model()
