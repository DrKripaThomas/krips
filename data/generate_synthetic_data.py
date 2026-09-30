"""
Synthetic Hospital Dataset Generator
====================================
Project: Hospital Operations Decision Support System using Predictive Analytics
Author: Health Informatics Student & AGY Pair Programmer

This script generates a realistic, synthetic 3-year daily operational dataset 
for a 500-bed hospital. It models realistic relationships between calendar, 
environmental, OPD, Emergency, and Bed occupancy metrics while preventing 
data leakage.
"""

import os
import numpy as np
import pandas as pd

def generate_hospital_dataset(start_date="2022-01-01", end_date="2024-12-31", random_seed=42):
    """
    Generates a daily time-series dataset of hospital operational variables.
    
    Parameters:
    -----------
    start_date : str
        Start date for the dataset generation (YYYY-MM-DD).
    end_date : str
        End date for the dataset generation (YYYY-MM-DD).
    random_seed : int
        Seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame
        Complete master dataset.
    """
    # Set random seed for exact reproducibility
    np.random.seed(random_seed)
    
    # 1. Create Date Range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    df = pd.DataFrame({'date': date_range})
    
    # 2. Calendar Features
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    
    # Season mapping
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
            
    df['season'] = df['month'].apply(get_season)
    
    # Public holidays (synthetic ~10 random holidays per year)
    holiday_probs = np.random.binomial(1, 10/365, size=n_days)
    df['is_public_holiday'] = holiday_probs
    
    # 3. Environmental & Epidemiological Features
    # Temperature with seasonal sinusoidal pattern
    day_of_year = df['date'].dt.dayofyear
    base_temp = 20 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_noise = np.random.normal(0, 3, n_days)
    df['temperature_c'] = np.round(base_temp + temp_noise, 1)
    
    # Rainfall (mm) - mostly dry, occasional rain spikes
    rain = np.random.exponential(scale=3.0, size=n_days)
    rain[rain < 1.0] = 0.0  # Dry days
    df['rainfall_mm'] = np.round(rain, 1)
    
    # Air Quality Index (AQI) (range 30 to 250)
    aqi_base = np.random.normal(75, 25, n_days)
    # Higher AQI in Winter due to smog
    winter_mask = df['season'] == 'Winter'
    aqi_base[winter_mask] += np.random.uniform(20, 50, winter_mask.sum())
    df['air_quality_index'] = np.round(np.clip(aqi_base, 30, 250), 1)
    
    # Flu Outbreak Level (Low, Medium, High)
    flu_levels = []
    for m in df['month']:
        if m in [11, 12, 1, 2]:  # Peak flu months
            flu_levels.append(np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3]))
        elif m in [3, 10]:
            flu_levels.append(np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.4, 0.1]))
        else:
            flu_levels.append(np.random.choice(['Low', 'Medium', 'High'], p=[0.85, 0.14, 0.01]))
    df['flu_outbreak_level'] = flu_levels
    
    # 4. OPD Operational Variables
    # OPD scheduled appointments: High on weekdays, reduced on Sat, zero on Sun/Holidays
    scheduled_appts = []
    for _, row in df.iterrows():
        if row['day_of_week'] == 'Sunday' or row['is_public_holiday'] == 1:
            scheduled_appts.append(0)
        elif row['day_of_week'] == 'Saturday':
            scheduled_appts.append(int(np.random.normal(120, 15)))
        else:
            scheduled_appts.append(int(np.random.normal(350, 30)))
    df['opd_appointments_scheduled'] = np.maximum(0, scheduled_appts)
    
    # Cancellation rate (10% - 20%, higher on heavy rain days)
    base_cancel = np.random.uniform(0.08, 0.15, n_days)
    heavy_rain_mask = df['rainfall_mm'] > 15.0
    base_cancel[heavy_rain_mask] += np.random.uniform(0.05, 0.10, heavy_rain_mask.sum())
    df['opd_cancellation_rate'] = np.round(np.clip(base_cancel, 0.05, 0.35), 3)
    
    # Walk-ins expected/actual
    walkins = []
    for _, row in df.iterrows():
        if row['day_of_week'] == 'Sunday' or row['is_public_holiday'] == 1:
            walkins.append(0)
        elif row['day_of_week'] == 'Saturday':
            walkins.append(int(np.random.normal(30, 8)))
        else:
            walkins.append(int(np.random.normal(70, 15)))
    df['opd_walkins_expected'] = np.maximum(0, walkins)
    
    # OPD Staffing roster
    opd_docs = []
    for _, row in df.iterrows():
        if row['day_of_week'] == 'Sunday' or row['is_public_holiday'] == 1:
            opd_docs.append(0)
        elif row['day_of_week'] == 'Saturday':
            opd_docs.append(6)
        else:
            opd_docs.append(int(np.random.choice([16, 18, 20], p=[0.2, 0.6, 0.2])))
    df['opd_doctors_on_duty'] = opd_docs
    
    # TARGET 1: OPD Total Volume = Scheduled * (1 - cancellation_rate) + Walkins
    opd_actual = (df['opd_appointments_scheduled'] * (1 - df['opd_cancellation_rate']) + df['opd_walkins_expected']).astype(int)
    df['opd_total_volume'] = np.maximum(0, opd_actual)
    
    # 5. Emergency (ED) Operational Variables
    # ED baseline volume ~120 to 160 per day
    ed_base = np.random.normal(135, 15, n_days)
    
    # ED weekend surge (+15-25 cases)
    ed_base[df['is_weekend'] == 1] += np.random.uniform(15, 25, df['is_weekend'].sum())
    
    # ED flu impact (+15 for Medium, +35 for High)
    flu_med = df['flu_outbreak_level'] == 'Medium'
    flu_high = df['flu_outbreak_level'] == 'High'
    ed_base[flu_med] += np.random.uniform(10, 20, flu_med.sum())
    ed_base[flu_high] += np.random.uniform(25, 45, flu_high.sum())
    
    # ED poor air quality impact
    high_aqi = df['air_quality_index'] > 130
    ed_base[high_aqi] += np.random.uniform(10, 20, high_aqi.sum())
    
    # TARGET 2: ED Total Volume
    df['ed_total_volume'] = np.round(np.clip(ed_base, 80, 260)).astype(int)
    
    # ED Triage Split (High Acuity ~30%, Low Acuity ~70%)
    high_acuity = np.round(df['ed_total_volume'] * np.random.uniform(0.25, 0.35, n_days)).astype(int)
    df['ed_triage_high_acuity'] = high_acuity
    df['ed_triage_low_acuity'] = df['ed_total_volume'] - df['ed_triage_high_acuity']
    
    # ED Staffing roster
    df['ed_doctors_on_duty'] = np.random.choice([6, 8, 10], size=n_days, p=[0.2, 0.6, 0.2])
    df['nursing_staff_on_duty'] = np.random.choice([38, 42, 45], size=n_days, p=[0.25, 0.50, 0.25])
    
    # 6. Bed Occupancy & Capacity Variables
    df['total_hospital_beds'] = 500  # Fixed total capacity
    
    # Elective planned surgeries (lower on weekends/holidays)
    surgeries = []
    for _, row in df.iterrows():
        if row['is_weekend'] == 1 or row['is_public_holiday'] == 1:
            surgeries.append(int(np.random.normal(5, 2)))
        else:
            surgeries.append(int(np.random.normal(25, 4)))
    df['planned_surgeries_today'] = np.maximum(0, surgeries)
    
    # Simulation of daily bed census continuity:
    # Occupied_Beds(t) = Occupied_Beds(t-1) + New_Admissions(t) - Discharges(t)
    initial_occupied = 380
    occupied_list = []
    expected_discharges_list = []

    current_occupied = initial_occupied
    
    for i in range(n_days):
        # Emergency admissions (~60% of high acuity ED cases need admission)
        ed_admissions = int(df.loc[i, 'ed_triage_high_acuity'] * 0.60)
        
        # Surgical admissions (~80% of elective surgeries need inpatient bed)
        surg_admissions = int(df.loc[i, 'planned_surgeries_today'] * 0.80)
        
        total_admissions = ed_admissions + surg_admissions
        
        # Expected discharges based on current occupancy (~10-12% of occupied beds discharged daily)
        planned_discharges = int(current_occupied * np.random.uniform(0.09, 0.12))
        expected_discharges_list.append(planned_discharges)
        
        # Actual discharges (small variation around planned discharges)
        actual_discharges = planned_discharges + int(np.random.normal(0, 3))
        actual_discharges = max(5, actual_discharges)
        
        # Update midnight census
        current_occupied = current_occupied + total_admissions - actual_discharges
        # Constrain census between 250 beds and maximum 500 beds
        current_occupied = max(250, min(500, current_occupied))
        
        occupied_list.append(current_occupied)
        
    df['expected_discharges_today'] = expected_discharges_list
    
    # TARGET 3: Midnight Bed Occupancy
    df['occupied_beds'] = occupied_list
    
    # Derived Target: Bed Occupancy Rate
    df['bed_occupancy_rate'] = np.round(df['occupied_beds'] / df['total_hospital_beds'], 4)
    
    # 7. Lagged Features (t-1) to Prevent Data Leakage
    df['opd_volume_lag1'] = df['opd_total_volume'].shift(1).fillna(df['opd_total_volume'].iloc[0])
    df['ed_volume_lag1'] = df['ed_total_volume'].shift(1).fillna(df['ed_total_volume'].iloc[0])
    df['occupied_beds_lag1'] = df['occupied_beds'].shift(1).fillna(initial_occupied)
    
    return df

if __name__ == '__main__':
    print("Generating Master Synthetic Hospital Operations Dataset...")
    dataset = generate_hospital_dataset(start_date="2022-01-01", end_date="2024-12-31", random_seed=42)
    
    # Ensure data directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'master_hospital_operations.csv')
    dataset.to_csv(output_path, index=False)
    
    print(f"SUCCESS: Dataset generated with {len(dataset)} rows and {len(dataset.columns)} columns.")
    print(f"Saved to: {output_path}")
    print("\nDataset Summary Preview:")
    print(dataset.head(5)[['date', 'day_of_week', 'opd_total_volume', 'ed_total_volume', 'occupied_beds', 'bed_occupancy_rate']])
