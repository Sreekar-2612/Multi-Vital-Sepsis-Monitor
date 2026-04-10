# Prompt for Anthropic API - Sepsis Detection Model Update
## WITH EXPLICIT FALLBACK STRATEGY FOR LOW CONFIDENCE

## Model Update Requirements

I need to modify my sepsis detection model's Stage 1 (Isolation Forest) to use a **multi-window confidence-based baseline establishment** followed by **continuous 40-second window monitoring**:

---

## **CRITICAL: Fallback Strategy When Confidence < 75%**

**THIS IS THE MOST IMPORTANT SECTION - READ CAREFULLY**

### **Scenario: After Window 5, Confidence Score is < 75% (e.g., 72%)**

**Answer: Use HYBRID MODE (Option A + C Combined)**

```
IF confidence_score >= 75%:
    → Use personalized Isolation Forest baseline (NORMAL MODE)
    → Begin Phase B Monitoring
    
ELSE IF confidence_score < 75% AND window_5_completed:
    → Proceed with monitoring BUT use HYBRID MODE
    → Do NOT wait for more windows
    → Do NOT discard the 5 windows
    → Use both approaches:
      A) Keep the personalized baseline (from 5 windows)
      B) Supplement with global thresholds
```

---

## **HYBRID MODE Specification (When Confidence 50-75%)**

When baseline confidence is between 50-75% (low but not zero), use BOTH:

### **1. Personalized Baseline (from 5 windows) - 60% weight**
```
personalized_anomaly = Isolation_Forest_score(8D_vital_vector)
Weight: 60%
```

### **2. Global Threshold Anomaly - 40% weight**
```
global_anomaly = calculate_deviation_from_global_ranges(vitals)

Global ranges (clinically validated):
├─ HR: 40-120 bpm
├─ RR: 8-30 breaths/min
├─ SpO2: 92-100%
├─ Temp: 35.5-39.5°C
├─ Movement: 0-50 (max acceptable)
├─ HRV: 20-200 ms
└─ RRV: 5-30 ms

For each vital out of range:
  deviation_score += (deviation_magnitude / vital_range) × 100
  
global_anomaly = sum_of_deviations / 8_vitals

Weight: 40%
```

### **Combined Anomaly Score (Hybrid Mode)**
```
IF confidence_score >= 75%:
    final_anomaly = personalized_anomaly (100%)
    
ELSE IF 50% <= confidence_score < 75%:
    final_anomaly = (0.60 × personalized_anomaly) 
                  + (0.40 × global_anomaly)
    confidence_level = "MODERATE_CAUTION"
    
ELSE IF confidence_score < 50%:
    final_anomaly = (0.30 × personalized_anomaly) 
                  + (0.70 × global_anomaly)
    confidence_level = "LOW_CAUTION"
    alert_clinician = TRUE
    recommendation = "Consider manual vitals assessment"
```

---

## **Critical Thresholds Explained**

```
Confidence >= 75%:
├─ Status: BASELINE_NORMAL
├─ Mode: Full Personalized (Isolation Forest only)
├─ Action: Begin normal Phase B monitoring
└─ Risk: Low - baseline is reliable

Confidence 50-75%:
├─ Status: BASELINE_ACCEPTABLE
├─ Mode: Hybrid (60% Personalized + 40% Global)
├─ Action: Begin Phase B with caution
├─ Alert: Moderate - suboptimal baseline
└─ Recommendation: Clinical assessment recommended

Confidence < 50%:
├─ Status: BASELINE_UNRELIABLE
├─ Mode: Hybrid (30% Personalized + 70% Global)
├─ Action: Begin Phase B with EXTREME caution
├─ Alert: CRITICAL - baseline quality very low
└─ Recommendation: URGENT - Manual vital signs assessment required
```

---

## **Reason for Hybrid Mode (Not Waiting or Discarding)**

1. **Waiting for more windows is clinically unsafe**
   - Patient needs monitoring NOW
   - Sepsis can progress rapidly
   - Delaying > 3.3 minutes risks missing deterioration

2. **Discarding 5 windows wastes data**
   - Even if confidence is low, 5 windows provide SOME information
   - Better to use imperfect personalized + global than global alone

3. **Hybrid mode balances safety + speed**
   - Still benefits from patient-specific baseline
   - Falls back to universal standards when uncertain
   - Dynamically adjusts weighting based on confidence

---

# Full Model Specification (With Hybrid Fallback Integrated)

### **Phase A: Baseline Establishment**
- **Duration**: 5 consecutive windows × 40 seconds = 200 seconds total (~3.3 minutes)
- **Data Collection**: Each 40-second window captures 8 vital parameters:
  1. Heart Rate (HR) - bpm
  2. Respiratory Rate (RR) - breaths/min
  3. Oxygen Saturation (SpO2) - percentage (%)
  4. Temperature (Temp) - °C
  5. Movement Score - activity/motion units (0-100)
  6. Heart Rate Variability (HRV) - milliseconds (ms)
  7. Respiratory Rate Variability (RRV) - milliseconds (ms)

### **Per-Window Analysis (Each 40-second window)**

For each window, calculate:
- **Mean** for each of 8 vitals
- **Standard Deviation** for each vital
- **Coefficient of Variation (CV)** = (std_dev / mean) × 100
- **Stability Flags**:
  - HR stable: CV < 5%
  - RR stable: CV < 8%
  - SpO2 stable: CV < 1%
  - Temp stable: CV < 0.5%
  - Movement low: mean < 20
  - HRV normal: within 20-200 ms range
  - RRV normal: within 5-30 ms range

### **Confidence Score Calculation (Across All 5 Baseline Windows)**

Calculate 4 component scores and combine with weighted formula:

**1. Stability Score (Weight: 40%)**
- Measure CV consistency across all 5 windows for each vital
- Formula: `stability_score = 1 - (average_CV / max_acceptable_CV) × 100`
- Higher if vitals show low variation across windows
- Range: 0-100%

**2. Consistency Score (Weight: 35%)**
- Count how many readings fall within expected clinical ranges:
  - HR: 40-120 bpm
  - RR: 8-30 breaths/min
  - SpO2: 92-100%
  - Temp: 35.5-39.5°C
  - Movement: 0-50 (low activity)
  - HRV: 20-200 ms
  - RRV: 5-30 ms
- Formula: `consistency_score = (readings_in_range / total_readings) × 100`
- Range: 0-100%

**3. Activity Quality Score (Weight: 15%)**
- Penalize high movement during baseline (motion artifacts)
- Formula: `activity_quality = 1 - (avg_movement / 100) × 100`
- Penalize windows where movement > 30
- Range: 0-100%

**4. Variability Quality Score (Weight: 10%)**
- Ensure HRV and RRV are healthy (not suppressed or excessive)
- `hrv_quality = 1.0 if HRV in range, 0.5 if borderline, 0.0 if abnormal`
- `rrv_quality = 1.0 if RRV in range, 0.5 if borderline, 0.0 if abnormal`
- Formula: `variability_quality = (hrv_quality + rrv_quality) / 2 × 100`
- Range: 0-100%

**Final Confidence Score:**
```
confidence_score = (0.40 × stability_score) 
                 + (0.35 × consistency_score) 
                 + (0.15 × activity_quality_score) 
                 + (0.10 × variability_quality_score)
```
- Range: 0-100%

### **Baseline Completion Decision (After Window 5)**

```
DECISION TREE:
│
├─ IF confidence_score >= 75%:
│  ├─ Status: BASELINE_NORMAL
│  ├─ Mode: Full Personalized (Isolation Forest 100%)
│  ├─ Action: Lock baseline, begin Phase B monitoring
│  └─ Alert level: NORMAL
│
├─ ELSE IF 50% <= confidence_score < 75%:
│  ├─ Status: BASELINE_ACCEPTABLE
│  ├─ Mode: Hybrid (60% Personalized + 40% Global)
│  ├─ Action: Lock baseline, begin Phase B with caution flag
│  ├─ Alert level: MODERATE CAUTION
│  └─ Recommendation: "Baseline quality is suboptimal. Consider manual assessment."
│
└─ ELSE IF confidence_score < 50%:
   ├─ Status: BASELINE_UNRELIABLE
   ├─ Mode: Hybrid (30% Personalized + 70% Global)
   ├─ Action: Lock baseline, begin Phase B with EXTREME caution flag
   ├─ Alert level: CRITICAL CAUTION
   └─ Recommendation: "URGENT - Baseline quality very low. Manual vital signs assessment REQUIRED."
```

### **Baseline Finalization Object (Regardless of Confidence)**

```json
{
  "baseline_metadata": {
    "timestamp_established": "ISO_8601_timestamp",
    "baseline_duration_seconds": 200,
    "total_windows": 5,
    "windows_duration_each": 40,
    "confidence_score": "float_0_to_100"
  },
  "baseline_quality": "NORMAL" | "ACCEPTABLE" | "UNRELIABLE",
  "baseline_mode": {
    "status": "Full Personalized" | "Hybrid" | "Hybrid (Extreme Caution)",
    "personalized_weight": "float (100%, 60%, or 30%)",
    "global_threshold_weight": "float (0%, 40%, or 70%)",
    "description": "human-readable explanation"
  },
  "confidence_breakdown": {
    "stability_score": "float_0_to_100",
    "consistency_score": "float_0_to_100",
    "activity_quality_score": "float_0_to_100",
    "variability_quality_score": "float_0_to_100",
    "overall_confidence": "float_0_to_100"
  },
  "vital_ranges": {
    "HR": {
      "mean": "float",
      "std": "float",
      "lower_bound": "mean - 2×std",
      "upper_bound": "mean + 2×std",
      "cv_percent": "float"
    },
    ...
  },
  "global_thresholds": {
    "HR": {"min": 40, "max": 120},
    "RR": {"min": 8, "max": 30},
    "SpO2": {"min": 92, "max": 100},
    "Temp": {"min": 35.5, "max": 39.5},
    "Movement": {"min": 0, "max": 50},
    "HRV": {"min": 20, "max": 200},
    "RRV": {"min": 5, "max": 30}
  },
  "window_history": [
    {
      "window_number": 1,
      "timestamp_start": "ISO_timestamp",
      "timestamp_end": "ISO_timestamp",
      "vitals_mean": {"HR": "..." , "...": "..."},
      "vitals_std": {"HR": "..." , "...": "..."},
      "stability_flags": {"HR": true, "...": false},
      "stable_vital_count": "int_out_of_7"
    },
    ... (windows 2-5)
  ],
  "clinical_notes": "string with recommendations based on baseline quality",
  "ready_for_monitoring": true
}
```

---

## **Phase B: Real-Time Monitoring**

**Timeline:**
- **After baseline locked at T=200s**: Begin continuous monitoring
- **Monitoring intervals**: Every 40 seconds (same as baseline window duration)
- **Duration**: Continuous as long as patient is monitored
- **Anomaly calculation**: Use appropriate mode (Normal, Hybrid, or Hybrid-Extreme)

### **For Each Monitoring Window (40 seconds)**

For every new 40-second monitoring window after baseline is locked:

1. **Collect 8 vital parameters** (same as baseline windows)

2. **Calculate window metrics**:
   - Mean for each vital
   - Std_dev for each vital
   - CV for each vital
   - Stability flags

3. **Calculate Z-scores** (deviation from baseline):
   ```
   z_score = (current_window_mean - baseline_mean) / baseline_std
   ```
   - For all 8 vitals individually

4. **Multivariate Anomaly Detection**:

   **IF baseline_mode = Full Personalized (confidence >= 75%):**
   ```
   anomaly_score = Isolation_Forest_score(8D_vital_vector)
   ```

   **ELSE IF baseline_mode = Hybrid (50-75% confidence):**
   ```
   personalized_anomaly = Isolation_Forest_score(8D_vital_vector)
   global_anomaly = calculate_global_threshold_deviation(vitals)
   anomaly_score = (0.60 × personalized_anomaly) + (0.40 × global_anomaly)
   confidence_indicator = "MODERATE_CAUTION"
   ```

   **ELSE IF baseline_mode = Hybrid-Extreme (< 50% confidence):**
   ```
   personalized_anomaly = Isolation_Forest_score(8D_vital_vector)
   global_anomaly = calculate_global_threshold_deviation(vitals)
   anomaly_score = (0.30 × personalized_anomaly) + (0.70 × global_anomaly)
   confidence_indicator = "EXTREME_CAUTION"
   alert_clinician = TRUE
   ```

5. **Calculate Final Score** using 50-30-20 weighting:
   ```
   final_score = (0.50 × rf_prob_severe) 
               + (0.30 × anomaly_score/100) 
               + (0.20 × qsofa_score/4)
   ```
   - Where:
     - `rf_prob_severe`: Random Forest probability of Severe Sepsis (0-1)
     - `anomaly_score`: Anomaly score (normalized to 0-1, depending on mode)
     - `qsofa_score`: Clinical qSOFA score (0-4)
   - Range: 0-1

6. **Generate Status Prediction**:
   - **Critical**: final_score > 0.65
   - **High-Risk**: final_score > 0.4 OR rf_prob_severe > 40%
   - **Mild Stress**: rf_prob_mild > 50% OR anomaly_score > 50%
   - **Normal**: otherwise

   **PLUS: Add confidence indicator to status**
   ```
   status_with_confidence = {
     "status": "NORMAL" | "MILD_STRESS" | "HIGH_RISK" | "CRITICAL",
     "baseline_confidence_level": "NORMAL" | "MODERATE_CAUTION" | "EXTREME_CAUTION",
     "recommendation": "string based on confidence level"
   }
   ```

---

## **Return Format**

### **During Baseline Establishment (Windows 1-5, every 40 seconds)**

```json
{
  "phase": "ESTABLISHING_BASELINE",
  "window_number": "int (1-5)",
  "timestamp_window_start": "ISO_timestamp",
  "timestamp_window_end": "ISO_timestamp",
  "duration_seconds": 40,
  "vitals_mean": {
    "HR": "float",
    "RR": "float",
    "SpO2": "float",
    "Temp": "float",
    "Movement": "float",
    "HRV": "float",
    "RRV": "float"
  },
  "vitals_std": {
    "HR": "float",
    "RR": "float",
    "SpO2": "float",
    "Temp": "float",
    "Movement": "float",
    "HRV": "float",
    "RRV": "float"
  },
  "stability_flags": {
    "HR": "bool",
    "RR": "bool",
    "SpO2": "bool",
    "Temp": "bool",
    "Movement": "bool",
    "HRV": "bool",
    "RRV": "bool"
  },
  "stable_vital_count": "int_out_of_7",
  "confidence_scores": {
    "stability_score": "float_0_to_100",
    "consistency_score": "float_0_to_100",
    "activity_quality_score": "float_0_to_100",
    "variability_quality_score": "float_0_to_100",
    "overall_confidence": "float_0_to_100"
  },
  "status": "ESTABLISHING" | "BASELINE_LOCKED",
  "progress": "Window X/5",
  "time_elapsed_seconds": "int",
  "time_remaining_seconds": "int"
}
```

### **Window 5 Output (Baseline Finalization)**

```json
{
  "phase": "ESTABLISHING_BASELINE",
  "window_number": 5,
  "timestamp_window_start": "ISO_timestamp",
  "timestamp_window_end": "ISO_timestamp",
  "duration_seconds": 40,
  "vitals_mean": {...},
  "vitals_std": {...},
  "stability_flags": {...},
  "confidence_scores": {
    "stability_score": "float",
    "consistency_score": "float",
    "activity_quality_score": "float",
    "variability_quality_score": "float",
    "overall_confidence": "87%"
  },
  "status": "BASELINE_LOCKED",
  "baseline_quality": "NORMAL" | "ACCEPTABLE" | "UNRELIABLE",
  "baseline_mode": {
    "name": "Full Personalized" | "Hybrid" | "Hybrid (Extreme Caution)",
    "personalized_weight": "100%" | "60%" | "30%",
    "global_threshold_weight": "0%" | "40%" | "70%"
  },
  "monitoring_ready": true,
  "clinical_alert": "NONE" | "MODERATE_CAUTION" | "EXTREME_CAUTION",
  "recommendation": "string describing next steps"
}
```

### **After Baseline Locked - Each 40-Second Monitoring Window**

```json
{
  "phase": "MONITORING",
  "window_number": "int (6, 7, 8, ...)",
  "timestamp_window_start": "ISO_timestamp",
  "timestamp_window_end": "ISO_timestamp",
  "duration_seconds": 40,
  "vitals_mean": {
    "HR": "float",
    "RR": "float",
    "SpO2": "float",
    "Temp": "float",
    "Movement": "float",
    "HRV": "float",
    "RRV": "float"
  },
  "vitals_std": {
    "HR": "float",
    "RR": "float",
    "SpO2": "float",
    "Temp": "float",
    "Movement": "float",
    "HRV": "float",
    "RRV": "float"
  },
  "deviations_from_baseline": {
    "HR_zscore": "float",
    "RR_zscore": "float",
    "SpO2_zscore": "float",
    "Temp_zscore": "float",
    "Movement_zscore": "float",
    "HRV_zscore": "float",
    "RRV_zscore": "float"
  },
  "anomaly_score": "float_0_to_100",
  "anomaly_interpretation": "string (Normal | Minor Deviation | Significant Deviation | Severe Anomaly)",
  "anomaly_calculation_method": "Full Personalized" | "Hybrid (60-40)" | "Hybrid (30-70)",
  "rf_components": {
    "rf_prob_normal": "float_0_to_1",
    "rf_prob_mild_infection": "float_0_to_1",
    "rf_prob_severe_sepsis": "float_0_to_1"
  },
  "qsofa_score": "float_0_to_4",
  "final_score": "float_0_to_1",
  "status": "NORMAL" | "MILD_STRESS" | "HIGH_RISK" | "CRITICAL",
  "baseline_info": {
    "baseline_established_at": "ISO_timestamp",
    "baseline_confidence": "float_0_to_100",
    "baseline_quality": "NORMAL" | "ACCEPTABLE" | "UNRELIABLE",
    "baseline_mode": "Full Personalized" | "Hybrid" | "Hybrid (Extreme Caution)",
    "windows_since_baseline_locked": "int"
  },
  "abnormal_vitals": [
    "list of vitals with significant deviations"
  ],
  "confidence_indicator": "NORMAL" | "MODERATE_CAUTION" | "EXTREME_CAUTION",
  "risk_explanation": "human-readable explanation of current status",
  "clinical_recommendation": "string with specific recommendations based on baseline quality + status"
}
```

---

## **Example Scenarios**

### **SCENARIO 1: High Confidence Baseline (87%)**

```
Window 5 Result:
├─ Confidence: 87%
├─ Status: BASELINE_LOCKED
├─ Quality: NORMAL
├─ Mode: Full Personalized (Isolation Forest 100%)
├─ Alert: NONE
└─ Recommendation: "Baseline quality is excellent. Begin normal monitoring."

Monitoring Window 6 (with 87% confidence baseline):
├─ HR: 82 bpm (z=+2.67)
├─ Anomaly Score: 62 (using 100% Isolation Forest)
├─ Baseline Confidence: NORMAL
├─ Status: HIGH_RISK
└─ Recommendation: "Monitor closely"
```

### **SCENARIO 2: Moderate Confidence Baseline (72%)**

```
Window 5 Result:
├─ Confidence: 72%
├─ Status: BASELINE_LOCKED
├─ Quality: ACCEPTABLE
├─ Mode: Hybrid (60% Personalized + 40% Global Thresholds)
├─ Alert: MODERATE_CAUTION
└─ Recommendation: "Baseline quality is suboptimal. Monitoring active with caution. 
                   Consider manual vital signs assessment if status escalates."

Monitoring Window 6 (with 72% confidence baseline):
├─ HR: 82 bpm (z=+2.67)
├─ Personalized anomaly (IF): 62
├─ Global anomaly (thresholds): 45
├─ Combined Anomaly: (0.60 × 62) + (0.40 × 45) = 37.2 + 18 = 55.2
├─ Baseline Confidence: MODERATE_CAUTION
├─ Status: HIGH_RISK (due to RF + qSOFA compensating for lower anomaly)
└─ Recommendation: "CAUTION - Baseline quality is suboptimal. Status is HIGH_RISK.
                   Recommend manual vital signs assessment and closer monitoring."
```

### **SCENARIO 3: Low Confidence Baseline (48%)**

```
Window 5 Result:
├─ Confidence: 48%
├─ Status: BASELINE_LOCKED
├─ Quality: UNRELIABLE
├─ Mode: Hybrid (30% Personalized + 70% Global Thresholds)
├─ Alert: EXTREME_CAUTION
└─ Recommendation: "URGENT - Baseline quality is very low. Manual vital signs 
                   assessment is REQUIRED immediately. Use global thresholds as primary."

Monitoring Window 6 (with 48% confidence baseline):
├─ HR: 82 bpm (z=+2.67)
├─ Personalized anomaly (IF): 62
├─ Global anomaly (thresholds): 45
├─ Combined Anomaly: (0.30 × 62) + (0.70 × 45) = 18.6 + 31.5 = 50.1
├─ Baseline Confidence: EXTREME_CAUTION
├─ Status: MILD_STRESS (lower anomaly score due to high global weight)
├─ Alert: CLINICAL ASSESSMENT REQUIRED
└─ Recommendation: "CRITICAL - Baseline reliability is very low. Escalate to clinician
                   immediately. Recommendation: Manual vital signs assessment REQUIRED.
                   Do not rely solely on model predictions."
```

---

## **Critical Handling Rules**

### **Rule 1: Never Discard 5 Windows**
- Always use data from all 5 baseline windows, even if confidence < 75%
- Partial information is better than no information

### **Rule 2: Never Wait for More Windows**
- After Window 5 completes, LOCK baseline immediately
- Clinical safety requires timely monitoring start
- Waiting > 3.3 minutes risks missing sepsis progression

### **Rule 3: Always Provide Clinical Context**
- In every monitoring output, include baseline_confidence indicator
- Include specific recommendations based on confidence level
- Guide clinician on how much to trust model predictions

### **Rule 4: Dynamic Weighting Based on Confidence**
- Higher confidence → Trust personalized baseline more (100%, 60%, or 30%)
- Lower confidence → Fall back to global thresholds (0%, 40%, or 70%)
- Always transparent about which method is being used

### **Rule 5: Escalate Low Confidence Cases**
- If confidence < 50%, send explicit clinical alert
- Recommend manual vital signs assessment
- This is a SIGNAL, not a stop

---

## **Key Requirements Summary**

| Requirement | Specification |
|-------------|-----------------|
| **Baseline Duration** | 5 windows × 40 seconds = 200 seconds total |
| **Window Size (Baseline)** | 40 seconds per window |
| **Window Size (Monitoring)** | 40 seconds per window |
| **Vital Parameters** | 8 (HR, RR, SpO2, Temp, Movement, HRV, RRV) |
| **Confidence Threshold** | ≥75% = Normal, 50-75% = Acceptable, <50% = Unreliable |
| **Fallback Strategy** | Hybrid weighting (not waiting, not discarding) |
| **Monitoring Start** | T=200s (immediately after Window 5, regardless of confidence) |
| **Monitoring Interval** | Every 40 seconds (same as baseline) |
| **Isolation Forest** | Trained on 8D vital space from 5 baseline windows |
| **Final Score Weighting** | 50% RF + 30% Anomaly + 20% qSOFA |
| **Status Levels** | NORMAL, MILD_STRESS, HIGH_RISK, CRITICAL |
| **Anomaly Calculation** | Mode-dependent (100% IF, 60-40% Hybrid, or 30-70% Hybrid) |
| **Clinical Alerts** | NONE, MODERATE_CAUTION, or EXTREME_CAUTION |
| **Continuous Operation** | Monitoring continues indefinitely, every 40 seconds |

---

## **Success Criteria**

✅ Baseline established in ~3.3 minutes (200 seconds, 5 windows)
✅ Confidence score accurately reflects baseline quality
✅ Monitoring begins IMMEDIATELY after Window 5 (never waits)
✅ All 5 windows are utilized (never discarded)
✅ Anomaly calculation adapts to baseline confidence level
✅ Final scores correctly weighted based on confidence
✅ Status predictions include confidence indicators
✅ Clinical alerts guide appropriate skepticism/trust
✅ Graceful degradation from high to low confidence
✅ Full audit trail for validation and debugging
