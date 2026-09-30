"""
Stage 3: Exploratory Data Analysis (EDA) Module
================================================
Project: Hospital Operations Decision Support System
Author: Health Informatics Student & AGY Pair Programmer

This script performs EDA on the master hospital dataset:
1. Calculates key operational descriptive statistics.
2. Identifies correlations between weather, calendar, and operational targets.
3. Generates and saves visual charts (data/eda_charts.png).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(csv_path="data/master_hospital_operations.csv", output_plot_path="data/eda_charts.png"):
    """
    Loads master dataset, prints summary statistics, and generates visualization charts.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run generate_synthetic_data.py first.")
        
    print("Loading dataset for EDA...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print("\n" + "="*50)
    print("1. DATASET OVERVIEW")
    print("="*50)
    print(f"Total Daily Records: {len(df)}")
    print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    
    print("\n" + "="*50)
    print("2. TARGET VARIABLES DESCRIPTIVE STATISTICS")
    print("="*50)
    targets = ['opd_total_volume', 'ed_total_volume', 'occupied_beds', 'bed_occupancy_rate']
    stats = df[targets].describe().T[['mean', 'std', 'min', '50%', 'max']]
    stats.columns = ['Mean', 'Std Dev', 'Min', 'Median (50%)', 'Max']
    print(stats)
    
    # 3. Visualization Plots
    print("\n" + "="*50)
    print("3. GENERATING EDA CHARTS...")
    print("="*50)
    
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hospital Operations Exploratory Data Analysis (EDA)', fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: 3-Year Time Series Trends
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['opd_total_volume'], label='OPD Volume', color='#2b5c8f', alpha=0.7, linewidth=1)
    ax1.plot(df['date'], df['ed_total_volume'], label='Emergency (ED) Volume', color='#d9534f', alpha=0.8, linewidth=1)
    ax1.set_title('Daily Operational Volume Trends (2022-2024)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Patient Count')
    ax1.legend(loc='upper right')
    
    # Plot 2: OPD & ED Volume by Day of Week
    ax2 = axes[0, 1]
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_day = df.groupby('day_of_week')[['opd_total_volume', 'ed_total_volume']].mean().reindex(day_order)
    df_day.plot(kind='bar', ax=ax2, color=['#2b5c8f', '#d9534f'], width=0.7)
    ax2.set_title('Average Daily Patient Volume by Day of Week', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Patients')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Bed Occupancy Rate Distribution & Alert Thresholds
    ax3 = axes[1, 0]
    sns.histplot(df['bed_occupancy_rate'] * 100, kde=True, ax=ax3, color='#4b6b94', bins=30)
    ax3.axvline(85, color='orange', linestyle='--', linewidth=2, label='Warning Threshold (85%)')
    ax3.axvline(95, color='red', linestyle='--', linewidth=2, label='Critical Threshold (95%)')
    ax3.set_title('Hospital Bed Occupancy Rate (%) Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Bed Occupancy Rate (%)')
    ax3.set_ylabel('Frequency (Days)')
    ax3.legend(loc='upper left')
    
    # Plot 4: Key Feature Correlations Heatmap
    ax4 = axes[1, 1]
    corr_cols = [
        'opd_total_volume', 'ed_total_volume', 'occupied_beds',
        'opd_appointments_scheduled', 'opd_volume_lag1',
        'ed_volume_lag1', 'occupied_beds_lag1', 'temperature_c', 'rainfall_mm', 'air_quality_index'
    ]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax4, cbar=True, annot_kws={"size": 8})
    ax4.set_title('Operational Feature Correlation Matrix', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=90, labelsize=8)
    ax4.tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    
    print(f"SUCCESS: EDA plot saved to {output_plot_path}")

if __name__ == '__main__':
    run_eda()
