# Master Hospital Operations Dataset - Data Dictionary

This data dictionary outlines the columns, data types, variable roles, and data leakage mitigations for `master_hospital_operations.csv`.

| Column Name | Data Type | Description | Example Value | Role | Target Module | Data Leakage Mitigation |
|---|---|---|---|---|---|---|
| `date` | `datetime64[ns]` | Date of the operational record | `2024-03-15` | Metadata | All | Used for time indexing and chronological splits only. |
| `day_of_week` | `string` | Day of the week (Monday-Sunday) | `Friday` | Input | OPD, ED, Bed | Calendar attribute known in advance. |
| `is_weekend` | `int64` | Flag indicating weekend (1=Yes, 0=No) | `0` | Input | OPD, ED, Bed | Known in advance. |
| `month` | `int64` | Month of year (1-12) | `3` | Input | OPD, ED, Bed | Known in advance. |
| `season` | `string` | Season (Winter, Spring, Summer, Autumn) | `Spring` | Input | OPD, ED, Bed | Known in advance. |
| `is_public_holiday` | `int64` | Flag for public holiday (1=Yes, 0=No) | `0` | Input | OPD, ED, Bed | Known in advance. |
| `temperature_c` | `float64` | Daily average temperature in Celsius | `24.5` | Input | OPD, ED | Exogenous weather forecast feature. |
| `rainfall_mm` | `float64` | Daily precipitation in mm | `12.0` | Input | OPD, ED | Exogenous weather forecast feature. |
| `air_quality_index` | `float64` | Air Quality Index (AQI) | `145.0` | Input | ED, Bed | Exogenous environmental feature. |
| `flu_outbreak_level` | `string` | Public health flu risk (Low, Medium, High) | `Medium` | Input | OPD, ED, Bed | Public health surveillance feature. |
| `opd_appointments_scheduled` | `int64` | Pre-booked outpatient appointments | `320` | Input | OPD Volume | Known at start of day. |
| `opd_cancellation_rate` | `float64` | Expected cancellation percentage | `0.12` | Input | OPD Volume | Estimated from historical trends. |
| `opd_doctors_on_duty` | `int64` | Scheduled OPD physicians on shift | `18` | Input | OPD, Resource | Roster planned in advance. |
| `opd_total_volume` | `int64` | Actual total OPD patients seen at end of day | `365` | **TARGET 1** | OPD Volume | Target variable. Do NOT use as input for same day prediction. |
| `ed_triage_low_acuity` | `int64` | Minor emergency cases (Category 4-5) | `110` | Derived/Lag | ED Volume | Historical lag used as input feature. |
| `ed_triage_high_acuity` | `int64` | Severe emergency cases (Category 1-3) | `45` | Derived/Lag | ED Volume | Historical lag used as input feature. |
| `ed_doctors_on_duty` | `int64` | Scheduled Emergency physicians on shift | `8` | Input | ED, Resource | Roster planned in advance. |
| `nursing_staff_on_duty` | `int64` | Scheduled floor nurses on shift | `42` | Input | Resource | Roster planned in advance. |
| `ed_total_volume` | `int64` | Actual total Emergency visits at end of day | `155` | **TARGET 2** | ED Volume | Target variable. Do NOT use as input for same day prediction. |
| `total_hospital_beds` | `int64` | Total licensed operational bed capacity | `500` | Input | Bed, Resource | Static operational capacity. |
| `planned_surgeries_today` | `int64` | Elective surgeries requiring post-op bed | `25` | Input | Bed Occupancy | Scheduled in advance. |
| `expected_discharges_today` | `int64` | Planned discharges scheduled during rounds | `40` | Input | Bed Occupancy | Planned discharge target (avoids end-of-day actual discharge leakage). |
| `occupied_beds` | `int64` | Midnight bed census (occupied bed count) | `415` | **TARGET 3** | Bed Occupancy | Target variable. Do NOT use as input for same day prediction. |
| `bed_occupancy_rate` | `float64` | Ratio: `occupied_beds / total_hospital_beds` | `0.83` | Derived Target | Bed, Decision Support | Derived from Target 3 for operational alert engine. |
| `opd_volume_lag1` | `int64` | OPD patient volume from previous day (t-1) | `350` | Input (Lag) | OPD Volume | Valid input feature (lagged t-1). |
| `ed_volume_lag1` | `int64` | Emergency volume from previous day (t-1) | `148` | Input (Lag) | ED Volume | Valid input feature (lagged t-1). |
| `occupied_beds_lag1` | `int64` | Bed occupancy from previous day (t-1) | `410` | Input (Lag) | Bed Occupancy | Valid input feature (lagged t-1). |
