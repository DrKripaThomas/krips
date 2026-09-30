"""
Stage 7b: Decision Support Alert Engine Module
================================================
Project: Hospital Operations Decision Support System
Author: Health Informatics Student & AGY Pair Programmer

This module acts as the core Decision-Support Engine (DSE).
It evaluates predictions and resource planning gaps against clinical thresholds 
to generate structured, prioritized operational alerts & management recommendations.
"""

def generate_operational_alerts(pred_opd_vol, pred_ed_vol, pred_beds, total_beds, occupancy_rate, staffing_res):
    """
    Evaluates hospital predictions and returns list of operational alerts and recommended actions.
    
    Returns:
    --------
    dict with:
    - 'overall_status': 'CRITICAL', 'WARNING', or 'NORMAL'
    - 'alerts': list of dicts with alert_level, title, message
    - 'recommendations': list of action strings
    """
    alerts = []
    recommendations = []
    overall_status = 'NORMAL'
    
    # 1. Bed Occupancy Threshold Evaluation
    if occupancy_rate >= 0.95:
        overall_status = 'CRITICAL'
        alerts.append({
            'level': 'CRITICAL',
            'title': '🚨 CRITICAL BED CAPACITY SURGE (≥95%)',
            'message': f'Predicted Occupancy is {pred_beds}/{total_beds} beds ({occupancy_rate*100:.1f}%). Risk of ER boarding and admission blockage!'
        })
        recommendations.append('Action 1: Activate emergency bed discharge protocol & postpone non-urgent elective surgeries.')
        recommendations.append('Action 2: Convert step-down beds to acute inpatient beds.')
    elif occupancy_rate >= 0.85:
        if overall_status != 'CRITICAL':
            overall_status = 'WARNING'
        alerts.append({
            'level': 'WARNING',
            'title': '⚠️ HIGH BED OCCUPANCY WARNING (≥85%)',
            'message': f'Predicted Occupancy is {pred_beds}/{total_beds} beds ({occupancy_rate*100:.1f}%). Capacity pressure is elevated.'
        })
        recommendations.append('Action: Expedite morning discharge rounds to free up capacity before peak afternoon admissions.')

    # 2. Emergency Volume Surge Evaluation
    if pred_ed_vol >= 180:
        if overall_status != 'CRITICAL':
            overall_status = 'WARNING'
        alerts.append({
            'level': 'HIGH SURGE',
            'title': '🚑 HIGH EMERGENCY PATIENT SURGE EXPECTED',
            'message': f'Predicted Emergency volume is {pred_ed_vol} patients (High threshold: 180).'
        })
        recommendations.append('Action: Prepare Fast-Track triage lanes and notify on-call emergency physicians.')
    elif pred_ed_vol >= 150:
        alerts.append({
            'level': 'MODERATE SURGE',
            'title': '⚡ MODERATE EMERGENCY VOLUME EXPECTED',
            'message': f'Predicted Emergency volume is {pred_ed_vol} patients.'
        })

    # 3. Staffing Deficit Evaluation
    if staffing_res['nurse_gap'] < 0:
        shortage = abs(staffing_res['nurse_gap'])
        alerts.append({
            'level': 'WARNING',
            'title': '👩‍⚕️ NURSING STAFF DEFICIT ALERT',
            'message': f'Nursing deficit of {shortage} floor nurses (Scheduled: {staffing_res["scheduled_nurses"]}, Required: {staffing_res["req_nurses"]}).'
        })
        recommendations.append(f'Action: Call in {shortage} float-pool or shift-extension nurses to cover inpatient wards.')

    if staffing_res['ed_doc_gap'] < 0:
        shortage = abs(staffing_res['ed_doc_gap'])
        alerts.append({
            'level': 'WARNING',
            'title': '👨‍⚕️ EMERGENCY DOCTOR DEFICIT ALERT',
            'message': f'Emergency physician deficit of {shortage} doctors (Scheduled: {staffing_res["scheduled_ed_docs"]}, Required: {staffing_res["req_ed_docs"]}).'
        })
        recommendations.append(f'Action: Assign {shortage} float physician(s) to Emergency Triage.')

    if staffing_res['opd_doc_gap'] < 0:
        shortage = abs(staffing_res['opd_doc_gap'])
        alerts.append({
            'level': 'INFO',
            'title': '🩺 OPD PHYSICIAN SHORTAGE',
            'message': f'OPD physician deficit of {shortage} doctors.'
        })

    # Default Normal status recommendation if clean
    if not alerts:
        alerts.append({
            'level': 'NORMAL',
            'title': '✅ HOSPITAL OPERATIONS STABLE',
            'message': f'Bed occupancy ({occupancy_rate*100:.1f}%), ED volume ({pred_ed_vol}), and staffing levels are within normal parameters.'
        })
        recommendations.append('No emergency operational escalation required. Maintain standard shifts.')

    return {
        'overall_status': overall_status,
        'alerts': alerts,
        'recommendations': recommendations
    }
