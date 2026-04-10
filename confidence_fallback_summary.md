# Quick Reference: Confidence Fallback Strategies

## The Critical Answer to Your Question

**If confidence stays <75% after window 5, what happens?**

### Answer: HYBRID MODE (Use both personalized + global together)

```
DO NOT:  ❌ Wait for more windows  (clinically unsafe)
DO NOT:  ❌ Discard the 5 windows   (wastes valuable data)
DO:      ✅ Proceed to monitoring    (patient needs protection NOW)
DO:      ✅ Use weighted blend       (best of both approaches)
```

---

## Three Confidence Scenarios

| Scenario | Confidence Score | Baseline Mode | IF Weight | Threshold Weight | Alert Level | Recommendation |
|----------|------------------|---------------|-----------|------------------|-------------|-----------------|
| **HIGH** | **≥ 75%** | Full Personalized | 100% | 0% | NORMAL | Begin normal monitoring. Baseline is reliable. |
| **MODERATE** | **50-75%** | Hybrid Blend | 60% | 40% | MODERATE_CAUTION | Begin monitoring with caution flag. Consider manual vital assessment if status escalates. |
| **LOW** | **< 50%** | Hybrid Extreme | 30% | 70% | EXTREME_CAUTION | Begin monitoring immediately. URGENT - Manual vital signs assessment REQUIRED. Do not rely solely on model predictions. |

---

## What This Means in Practice

### Scenario 1: High Confidence (87%)
```
Window 5 complete → Confidence = 87%
├─ Status: BASELINE_LOCKED
├─ Quality: NORMAL
├─ Isolation Forest: 100%
├─ Global Thresholds: 0%
├─ Alert: NONE
└─ Monitoring begins → Use personalized baseline only

Monitoring Window 6:
├─ Anomaly Score = Isolation_Forest(8D_vector)
├─ Final Score = (0.50 × RF) + (0.30 × Anomaly) + (0.20 × qSOFA)
├─ Clinician Confidence: HIGH ✅
└─ Recommendation: "Proceed with normal monitoring"
```

### Scenario 2: Moderate Confidence (72%)
```
Window 5 complete → Confidence = 72%
├─ Status: BASELINE_LOCKED
├─ Quality: ACCEPTABLE
├─ Isolation Forest: 60%
├─ Global Thresholds: 40%
├─ Alert: MODERATE_CAUTION
└─ Monitoring begins → Use hybrid blend

Monitoring Window 6:
├─ Personalized Anomaly = IF_score(8D_vector) = 62
├─ Global Anomaly = threshold_deviation(vitals) = 45
├─ Combined Anomaly = (0.60 × 62) + (0.40 × 45) = 55.2
├─ Final Score = (0.50 × RF) + (0.30 × 0.552) + (0.20 × qSOFA)
├─ Clinician Confidence: MODERATE ⚠️
└─ Recommendation: "Monitor with caution. Baseline quality suboptimal. 
                   Consider manual vital signs assessment."
```

### Scenario 3: Low Confidence (48%)
```
Window 5 complete → Confidence = 48%
├─ Status: BASELINE_LOCKED
├─ Quality: UNRELIABLE
├─ Isolation Forest: 30%
├─ Global Thresholds: 70%
├─ Alert: EXTREME_CAUTION
└─ Monitoring begins → Use extreme caution blend

Monitoring Window 6:
├─ Personalized Anomaly = IF_score(8D_vector) = 62
├─ Global Anomaly = threshold_deviation(vitals) = 45
├─ Combined Anomaly = (0.30 × 62) + (0.70 × 45) = 50.1
├─ Final Score = (0.50 × RF) + (0.30 × 0.501) + (0.20 × qSOFA)
├─ Clinician Confidence: LOW 🔴
└─ Recommendation: "CRITICAL - Baseline reliability very low. 
                   Manual vital signs assessment REQUIRED IMMEDIATELY. 
                   Do not rely solely on model predictions."
```

---

## Why This Hybrid Strategy?

### ❌ NOT: "Wait for more windows"
- **Problem**: Sepsis can deteriorate rapidly in minutes
- **Risk**: Delaying monitoring by even 3 minutes can miss critical progression
- **Clinical harm**: Patient unprotected during additional waiting period
- **Answer**: Always proceed at T=200s, no exceptions

### ❌ NOT: "Discard the 5 windows and use global only"
- **Problem**: Throwing away patient-specific baseline data
- **Loss**: No personalization at all = less sensitive detection
- **Answer**: Use that data even if confidence is low

### ✅ YES: "Hybrid Mode - Use weighted blend"
- **Benefit 1**: Patient gets monitoring immediately (safety)
- **Benefit 2**: Personalization still contributes (sensitivity)
- **Benefit 3**: Falls back to universal standards when uncertain (safety)
- **Benefit 4**: Transparent about confidence (clinician knows baseline quality)
- **Result**: Best of both worlds - speed AND robustness

---

## Weight Adjustments by Confidence

### High Confidence (≥75%)
```
Trust personalized baseline completely
Weight = 100% IF + 0% Global
Logic: Patient's 5-window baseline is stable and consistent
       → Isolation Forest learned good model
       → Use personalized detection exclusively
```

### Moderate Confidence (50-75%)
```
Blend both approaches
Weight = 60% IF + 40% Global
Logic: Patient's baseline has some variance
       → Isolation Forest partially reliable
       → Global thresholds provide safety net
       → Anomaly = 0.60×IF_score + 0.40×threshold_score
```

### Low Confidence (<50%)
```
Trust global thresholds more
Weight = 30% IF + 70% Global
Logic: Patient's baseline is very noisy
       → Isolation Forest less reliable
       → Fall back to universal clinical standards
       → Anomaly = 0.30×IF_score + 0.70×threshold_score
```

---

## Key Implementation Rules

### Rule 1: Never Wait
```javascript
if (window_5_completed) {
    lock_baseline()            // Even if confidence < 75%
    begin_monitoring()         // Immediately at T=200s
    never_wait_for_more_data() // Clinical safety requires it
}
```

### Rule 2: Never Discard
```javascript
// Use all 5 baseline windows regardless of confidence
baseline_mean = mean(window1, window2, window3, window4, window5)
baseline_std = std(window1, window2, window3, window4, window5)

// Both personalized AND global get trained/defined
isolation_forest.train(all_5_windows)
global_thresholds = {HR: (40,120), RR: (8,30), ...}

// Then blend them based on confidence
if confidence >= 75:
    use 100% IF
elif 50 <= confidence < 75:
    use 60% IF + 40% Global
else:
    use 30% IF + 70% Global
```

### Rule 3: Always Signal Confidence
```javascript
// In every monitoring window output
output = {
    anomaly_score: float,
    baseline_confidence: 48,  // 0-100
    baseline_mode: "Hybrid (Extreme Caution)",
    confidence_indicator: "EXTREME_CAUTION",
    
    // Clinician guidance based on confidence
    clinical_recommendation: "URGENT - Baseline reliability very low. 
                            Manual vital signs assessment REQUIRED."
}
```

---

## Monitoring Window Output Examples

### With High Confidence Baseline (87%)
```json
{
  "window_number": 6,
  "anomaly_score": 62,
  "anomaly_calculation_method": "Full Personalized (100% IF)",
  "baseline_confidence": 87,
  "baseline_mode": "Full Personalized",
  "confidence_indicator": "NORMAL",
  "status": "HIGH_RISK",
  "clinical_recommendation": "Monitor closely. Baseline quality is excellent."
}
```

### With Moderate Confidence Baseline (72%)
```json
{
  "window_number": 6,
  "anomaly_score": 55.2,
  "anomaly_calculation_method": "Hybrid (60% IF + 40% Global)",
  "baseline_confidence": 72,
  "baseline_mode": "Hybrid Blend",
  "confidence_indicator": "MODERATE_CAUTION",
  "status": "HIGH_RISK",
  "clinical_recommendation": "CAUTION - Baseline quality suboptimal. 
                            Status is HIGH_RISK. Consider manual vital 
                            signs assessment and closer monitoring."
}
```

### With Low Confidence Baseline (48%)
```json
{
  "window_number": 6,
  "anomaly_score": 50.1,
  "anomaly_calculation_method": "Hybrid (30% IF + 70% Global)",
  "baseline_confidence": 48,
  "baseline_mode": "Hybrid Extreme",
  "confidence_indicator": "EXTREME_CAUTION",
  "status": "MILD_STRESS",
  "clinical_recommendation": "CRITICAL - Baseline reliability very low. 
                            Manual vital signs assessment REQUIRED immediately. 
                            Do not rely solely on model predictions."
}
```

---

## Summary Decision Tree

```
Window 5 Completes
        ↓
Calculate Confidence Score
        ↓
    ┌───┴───┬──────────────┬───────────┐
    ↓       ↓              ↓           ↓
  ≥75%   50-75%          <50%     (doesn't happen)
    ↓       ↓              ↓
  LOCK    LOCK            LOCK
  BLEND  (60-40)       (30-70)
  100%IF HYBRID       EXTREME
    ↓       ↓              ↓
BEGIN   BEGIN            BEGIN
MON.    MON.             MON.
NORMAL  CAUTION       CRITICAL
  ✅      ⚠️            🔴
```

---

## Critical Points to Communicate to Anthropic

1. **Never wait** - Begin monitoring at T=200s regardless of confidence
2. **Never discard** - All 5 windows contribute to the calculation
3. **Always blend** - Use confidence-weighted combination of IF + Global
4. **Always signal** - Include confidence indicator in every monitoring output
5. **Guide clinician** - Provide specific recommendations based on confidence level
6. **Transparent weighting** - Tell clinician which method is being used (100% IF, 60-40, 30-70)

---

## Test Cases to Validate

### Test 1: High Confidence Path
```
Baseline: 5 stable windows with CV < 5% across all vitals
Expected: Confidence > 85%, Mode = Full Personalized (100%)
Validation: Monitoring uses IF-only, no global thresholds
```

### Test 2: Moderate Confidence Path
```
Baseline: 5 windows with some variance (CV 5-10%), movement > 20
Expected: 60% < Confidence < 75%, Mode = Hybrid (60-40)
Validation: Monitoring anomaly = 0.60×IF + 0.40×Global
```

### Test 3: Low Confidence Path
```
Baseline: 5 windows very noisy (CV > 12%), inconsistent readings
Expected: Confidence < 50%, Mode = Hybrid (30-70)
Validation: Monitoring anomaly = 0.30×IF + 0.70×Global
```

### Test 4: Transition to Monitoring
```
Verify: At T=200s, monitoring begins IMMEDIATELY
Verify: No pause, no waiting, no re-calculation
Verify: Window 6 output includes baseline_confidence and mode
```

### Test 5: Confidence in Output
```
Verify: Every monitoring window shows confidence_indicator
Verify: Every monitoring window includes clinical_recommendation
Verify: Recommendation changes based on confidence level
```
