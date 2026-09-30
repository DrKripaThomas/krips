"""
Stage 8: Hospital Operations Decision Support System - Gradio Dashboard
========================================================================
Project: Hospital Operations Decision Support System using Predictive Analytics
Author: Health Informatics Student & AGY Pair Programmer

This interactive web dashboard allows hospital administrators to:
1. Input operational variables for tomorrow's shift.
2. Generate ML forecasts for OPD Volume, Emergency Volume, and Bed Occupancy.
3. Compute Staffing Resource Needs (Physicians & Nurses).
4. Receive Automated Operational Alerts & Clinical Management Recommendations.
5. Review Model Analytics and Data Dictionary documentation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import gradio as gr
import pandas as pd
import numpy as np

from src.opd_module import predict_opd_volume
from src.ed_module import predict_ed_volume
from src.bed_module import predict_bed_occupancy
from src.resource_planning import analyze_resource_requirements
from src.decision_support import generate_operational_alerts

def run_full_decision_support_pipeline(
    day_name, is_weekend_flag, public_holiday_flag, month_num,
    temp_c, rain_mm, aqi_val, flu_lvl,
    scheduled_appts, cancel_rate, walkins_exp, scheduled_opd_docs,
    scheduled_ed_docs, scheduled_nurses,
    planned_surg, expected_discharges,
    opd_lag1, ed_lag1, bed_lag1
):
    """
    Master pipeline function called by Gradio UI.
    """
    # 1. Format inputs for OPD Model
    opd_input = {
        'opd_appointments_scheduled': int(scheduled_appts),
        'opd_cancellation_rate': float(cancel_rate),
        'opd_walkins_expected': int(walkins_exp),
        'opd_doctors_on_duty': int(scheduled_opd_docs),
        'opd_volume_lag1': int(opd_lag1),
        'is_weekend': 1 if is_weekend_flag else 0,
        'is_public_holiday': 1 if public_holiday_flag else 0,
        'month': int(month_num),
        'temperature_c': float(temp_c),
        'rainfall_mm': float(rain_mm)
    }
    
    # 2. Format inputs for ED Model
    ed_input = {
        'ed_volume_lag1': int(ed_lag1),
        'ed_doctors_on_duty': int(scheduled_ed_docs),
        'nursing_staff_on_duty': int(scheduled_nurses),
        'is_weekend': 1 if is_weekend_flag else 0,
        'is_public_holiday': 1 if public_holiday_flag else 0,
        'month': int(month_num),
        'temperature_c': float(temp_c),
        'rainfall_mm': float(rain_mm),
        'air_quality_index': float(aqi_val),
        'flu_outbreak_level': flu_lvl
    }
    
    # 3. Format inputs for Bed Model
    bed_input = {
        'occupied_beds_lag1': int(bed_lag1),
        'total_hospital_beds': 500,
        'planned_surgeries_today': int(planned_surg),
        'expected_discharges_today': int(expected_discharges),
        'ed_volume_lag1': int(ed_lag1),
        'is_weekend': 1 if is_weekend_flag else 0,
        'month': int(month_num),
        'flu_outbreak_level': flu_lvl
    }
    
    # 4. Generate Machine Learning Predictions
    pred_opd = predict_opd_volume(opd_input)
    pred_ed = predict_ed_volume(ed_input)
    pred_beds, occ_rate = predict_bed_occupancy(bed_input)
    
    # 5. Run Resource Planning Analysis
    resource_analysis = analyze_resource_requirements(
        pred_opd, pred_ed, pred_beds,
        int(scheduled_opd_docs), int(scheduled_ed_docs), int(scheduled_nurses)
    )
    
    # 6. Run Decision Support Alert Engine
    dse_output = generate_operational_alerts(
        pred_opd, pred_ed, pred_beds, 500, occ_rate, resource_analysis
    )
    
    # 7. Format UI Outputs
    # Summary Metrics DataFrame
    summary_df = pd.DataFrame([
        {'Prediction Module': 'OPD Patient Volume', 'Predicted Target Value': f'{pred_opd} Patients', 'Operational Baseline': '~350 (Weekday) / 0 (Sun)'},
        {'Prediction Module': 'Emergency (ED) Volume', 'Predicted Target Value': f'{pred_ed} Patients', 'Operational Baseline': '~135 - 160 / Day'},
        {'Prediction Module': 'Hospital Bed Occupancy', 'Predicted Target Value': f'{pred_beds} Beds / 500 Capacity', 'Operational Baseline': 'Capacity Limit: 500 Beds'},
        {'Prediction Module': 'Bed Occupancy Rate (%)', 'Predicted Target Value': f'{occ_rate*100:.1f}%', 'Operational Baseline': 'Warning: ≥85%, Critical: ≥95%'}
    ])
    
    # Staffing Table DataFrame
    staffing_df = pd.DataFrame([
        {'Role': 'OPD Physicians', 'Required Staff': resource_analysis['req_opd_docs'], 'Scheduled Roster': resource_analysis['scheduled_opd_docs'], 'Gap (Deficit/Surplus)': f"{resource_analysis['opd_doc_gap']:+d}"},
        {'Role': 'Emergency Physicians', 'Required Staff': resource_analysis['req_ed_docs'], 'Scheduled Roster': resource_analysis['scheduled_ed_docs'], 'Gap (Deficit/Surplus)': f"{resource_analysis['ed_doc_gap']:+d}"},
        {'Role': 'Floor Nursing Staff', 'Required Staff': resource_analysis['req_nurses'], 'Scheduled Roster': resource_analysis['scheduled_nurses'], 'Gap (Deficit/Surplus)': f"{resource_analysis['nurse_gap']:+d}"}
    ])
    
    # Build Alerts Markdown Text
    alert_md = f"### 🛡️ System Alert Status: **{dse_output['overall_status']}**\n\n"
    for alt in dse_output['alerts']:
        alert_md += f"#### {alt['title']}\n{alt['message']}\n\n"
        
    rec_md = "### 📋 Clinical & Operational Action Plan\n\n"
    for i, rec in enumerate(dse_output['recommendations'], 1):
        rec_md += f"- **{rec}**\n"
        
    return summary_df, staffing_df, alert_md, rec_md


# Construct Gradio Interface
def create_app():
    custom_css = """
    .main-header { text-align: center; color: #1e3a8a; margin-bottom: 20px; }
    .alert-box { border-radius: 8px; padding: 15px; background-color: #f8fafc; }
    """
    
    with gr.Blocks(title="Hospital Operations Decision Support System", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏥 Hospital Operations Decision Support System (HODSS)
            ### *Predictive Analytics & Resource Optimization Engine for Hospital Management*
            ---
            """
        )
        
        with gr.Tabs():
            # TAB 1: OPERATIONAL PREDICTION & DECISION SUPPORT
            with gr.TabItem("📊 Tomorrow's Operations Forecast & Decision Engine"):
                gr.Markdown("### ⚙️ Step 1: Input Operational Parameters for Tomorrow's Shift")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🗓️ Calendar & Weather Context")
                        day_name = gr.Dropdown(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], value='Friday', label='Day of Week')
                        is_weekend_flag = gr.Checkbox(value=False, label='Is Weekend?')
                        public_holiday_flag = gr.Checkbox(value=False, label='Is Public Holiday?')
                        month_num = gr.Slider(1, 12, value=3, step=1, label='Month of Year')
                        temp_c = gr.Slider(-5.0, 45.0, value=24.5, step=0.5, label='Forecast Temperature (°C)')
                        rain_mm = gr.Slider(0.0, 100.0, value=5.0, step=1.0, label='Forecast Rainfall (mm)')
                        aqi_val = gr.Slider(20.0, 300.0, value=95.0, step=5.0, label='Air Quality Index (AQI)')
                        flu_lvl = gr.Radio(['Low', 'Medium', 'High'], value='Medium', label='Flu Outbreak Risk Level')
                        
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🩺 OPD & Shift Staff Roster")
                        scheduled_appts = gr.Number(value=320, label='Pre-booked OPD Appointments')
                        cancel_rate = gr.Slider(0.0, 0.4, value=0.12, step=0.01, label='Expected Cancellation Rate')
                        walkins_exp = gr.Number(value=70, label='Expected Walk-in Patients')
                        scheduled_opd_docs = gr.Number(value=18, label='Scheduled OPD Doctors')
                        scheduled_ed_docs = gr.Number(value=8, label='Scheduled ED Doctors')
                        scheduled_nurses = gr.Number(value=42, label='Scheduled Floor Nurses')

                    with gr.Column(scale=1):
                        gr.Markdown("#### 🛌 Bed Management & Lag Inputs (t-1)")
                        planned_surg = gr.Number(value=25, label='Elective Surgeries Scheduled')
                        expected_discharges = gr.Number(value=40, label='Planned Discharges Today')
                        opd_lag1 = gr.Number(value=350, label="Yesterday's OPD Volume (t-1)")
                        ed_lag1 = gr.Number(value=145, label="Yesterday's ED Volume (t-1)")
                        bed_lag1 = gr.Number(value=410, label="Yesterday's Occupied Beds (t-1)")
                
                run_btn = gr.Button("🚀 Run Operational Decision Support Engine", variant="primary", size="lg")
                
                gr.Markdown("---")
                gr.Markdown("### 📈 Step 2: Predictive Analytics & Decision Support Output")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎯 Predicted Hospital Volumes")
                        summary_table = gr.Dataframe(headers=['Prediction Module', 'Predicted Target Value', 'Operational Baseline'], interactive=False)
                        
                        gr.Markdown("#### 👩‍⚕️ Staffing Capacity & Resource Planning Analysis")
                        staffing_table = gr.Dataframe(headers=['Role', 'Required Staff', 'Scheduled Roster', 'Gap (Deficit/Surplus)'], interactive=False)
                        
                    with gr.Column(scale=1):
                        alert_box = gr.Markdown("*(Run the engine above to generate live operational alerts)*")
                        rec_box = gr.Markdown("*(Run the engine above to view management action plans)*")
                
                run_btn.click(
                    fn=run_full_decision_support_pipeline,
                    inputs=[
                        day_name, is_weekend_flag, public_holiday_flag, month_num,
                        temp_c, rain_mm, aqi_val, flu_lvl,
                        scheduled_appts, cancel_rate, walkins_exp, scheduled_opd_docs,
                        scheduled_ed_docs, scheduled_nurses,
                        planned_surg, expected_discharges,
                        opd_lag1, ed_lag1, bed_lag1
                    ],
                    outputs=[summary_table, staffing_table, alert_box, rec_box]
                )
                
            # TAB 2: MODEL PERFORMANCE ANALYTICS
            with gr.TabItem("📈 Machine Learning Models & Metrics"):
                gr.Markdown("### 🧪 Scikit-learn Prediction Models Performance")
                gr.Markdown(
                    """
                    All models are trained on chronological 80/20 time-series splits using synthetic 3-year daily operational logs (1,096 days).
                    
                    | Prediction Module | Target Variable | Machine Learning Algorithm | Mean Absolute Error (MAE) | RMSE | R² Score |
                    |---|---|---|---|---|---|
                    | **Module 1: OPD Volume** | `opd_total_volume` | Scikit-learn RandomForestRegressor | **3.06 patients** | 4.99 | **0.9989** |
                    | **Module 2: Emergency (ED) Volume** | `ed_total_volume` | Scikit-learn GradientBoostingRegressor | **13.08 patients** | 16.44 | **0.4023** |
                    | **Module 3: Bed Occupancy** | `occupied_beds` | Scikit-learn RandomForestRegressor | **4.48 beds** | 5.78 | **0.9162** |
                    """
                )
                
            # TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
            with gr.TabItem("🔍 Historical EDA Charts"):
                gr.Markdown("### 📊 Exploratory Data Analysis & Trends")
                eda_img_path = os.path.join(os.path.dirname(__file__), 'data', 'eda_charts.png')
                if os.path.exists(eda_img_path):
                    gr.Image(value=eda_img_path, label="Historical Operational Trends & Correlations")
                else:
                    gr.Markdown("*(Run `python src/eda_analysis.py` to generate charts)*")
                    
            # TAB 4: SYSTEM ARCHITECTURE & DATA DICTIONARY
            with gr.TabItem("📖 System Architecture & Documentation"):
                gr.Markdown(
                    """
                    ### 🏗️ Master System Architecture
                    ```
                    Synthetic Daily Logs (1,096 Days)
                           │
                           ├──► Module 1: OPD Volume Predictor (Random Forest)
                           ├──► Module 2: Emergency Volume Predictor (Gradient Boosting)
                           └──► Module 3: Bed Occupancy Predictor (Random Forest)
                                   │
                                   ▼
                    Resource Planning Engine ──► Doctor & Nurse Ratio Calculations
                                   │
                                   ▼
                    Decision Support Engine ──► Rule-based Alert Escalation (Normal/Warning/Critical)
                                   │
                                   ▼
                    Interactive Gradio Dashboard (app.py)
                    ```
                    
                    ### 🛡️ Data Leakage Prevention Protocol
                    - **Strict Lagged Inputs**: Operational models for day $t$ only consume historical metrics up to day $t-1$ or pre-scheduled inputs.
                    - **No Clinical Data**: Zero patient personal identifiers, diagnoses, or treatment fields are used, ensuring 100% operational focus and privacy compliance.
                    """
                )
    return demo

if __name__ == '__main__':
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
