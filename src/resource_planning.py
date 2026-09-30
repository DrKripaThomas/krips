"""
Stage 7a: Resource Planning Analysis Module
=============================================
Project: Hospital Operations Decision Support System
Author: Health Informatics Student & AGY Pair Programmer

This module performs clinical workload & resource planning calculations:
1. Translates predicted OPD and ED patient volumes into required physician shifts.
2. Translates predicted Bed Occupancy and ED volumes into required nursing shifts.
3. Computes staffing deficit / surplus against scheduled rosters.
"""

def analyze_resource_requirements(predicted_opd_vol, predicted_ed_vol, predicted_occupied_beds,
                                  scheduled_opd_docs, scheduled_ed_docs, scheduled_nurses):
    """
    Computes required staffing based on predicted workloads and compares against scheduled staff.
    
    Standard Staffing Ratios:
    - 1 OPD Doctor per 20 OPD patients
    - 1 ED Doctor per 18 ED patients
    - 1 Floor Nurse per 8 Occupied Beds + 1 Nurse per 12 ED patients
    
    Returns:
    --------
    dict containing recommended counts, scheduled counts, and deficit/surplus gaps.
    """
    # 1. Calculate Required Staff
    req_opd_docs = max(1, int(round(predicted_opd_vol / 20.0))) if predicted_opd_vol > 0 else 0
    req_ed_docs = max(2, int(round(predicted_ed_vol / 18.0)))
    
    nurse_bed_component = predicted_occupied_beds / 8.0
    nurse_ed_component = predicted_ed_vol / 12.0
    req_nurses = max(10, int(round(nurse_bed_component + nurse_ed_component)))
    
    # 2. Compute Gaps (Scheduled - Required)
    opd_doc_gap = scheduled_opd_docs - req_opd_docs
    ed_doc_gap = scheduled_ed_docs - req_ed_docs
    nurse_gap = scheduled_nurses - req_nurses
    
    return {
        'req_opd_docs': req_opd_docs,
        'scheduled_opd_docs': scheduled_opd_docs,
        'opd_doc_gap': opd_doc_gap,
        
        'req_ed_docs': req_ed_docs,
        'scheduled_ed_docs': scheduled_ed_docs,
        'ed_doc_gap': ed_doc_gap,
        
        'req_nurses': req_nurses,
        'scheduled_nurses': scheduled_nurses,
        'nurse_gap': nurse_gap
    }
