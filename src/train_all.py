"""
Model Training Master Script
============================
Project: Hospital Operations Decision Support System
Author: Health Informatics Student & AGY Pair Programmer

Trains all 3 machine learning prediction models (OPD, ED, Bed Occupancy)
and outputs model evaluation metrics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.opd_module import train_opd_model
from src.ed_module import train_ed_model
from src.bed_module import train_bed_model

def train_all_modules():
    print("="*60)
    print("STARTING MODEL TRAINING FOR ALL 3 PREDICTION MODULES")
    print("="*60)
    
    print("\n--- 1. Training OPD Volume Model ---")
    opd_model, opd_metrics = train_opd_model()
    
    print("\n--- 2. Training Emergency Volume Model ---")
    ed_model, ed_metrics = train_ed_model()
    
    print("\n--- 3. Training Hospital Bed Occupancy Model ---")
    bed_model, bed_metrics = train_bed_model()
    
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY & EVALUATION METRICS")
    print("="*60)
    print(f"OPD Patient Volume Model    -> MAE: {opd_metrics['MAE']} patients | RMSE: {opd_metrics['RMSE']} | R² Score: {opd_metrics['R2']}")
    print(f"Emergency Patient Volume    -> MAE: {ed_metrics['MAE']} patients | RMSE: {ed_metrics['RMSE']} | R² Score: {ed_metrics['R2']}")
    print(f"Hospital Bed Occupancy Model -> MAE: {bed_metrics['MAE']} beds     | RMSE: {bed_metrics['RMSE']} | R² Score: {bed_metrics['R2']}")
    print("="*60)

if __name__ == '__main__':
    train_all_modules()
