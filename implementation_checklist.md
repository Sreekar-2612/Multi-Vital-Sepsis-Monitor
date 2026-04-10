# Implementation Checklist for Anthropic API

## Files Ready to Submit

✅ **File 1**: `prompt_with_fallback_strategy.md` (MAIN PROMPT)
   - Complete specification with hybrid fallback strategy
   - All 3 scenarios detailed
   - Full return formats
   - Implementation requirements

✅ **File 2**: `confidence_fallback_summary.md` (QUICK REFERENCE)
   - One-page summary of the 3 confidence scenarios
   - Comparison tables
   - Quick decision tree
   - Test cases

---

## What to Tell Anthropic

### Short Version (Copy-Paste Ready)

```
I need to build a sepsis detection model with the following specs:

PHASE A: BASELINE ESTABLISHMENT (5 windows × 40 seconds)
- Collect 8 vitals per window: HR, RR, SpO2, Temp, Movement, HRV, RRV
- Per-window: calculate mean, std_dev, CV, stability flags
- Across 5 windows: calculate weighted confidence score
  * 40% Stability + 35% Consistency + 15% Activity + 10% Variability
  * Range: 0-100%

CRITICAL: After window 5 completes, ALWAYS proceed to monitoring at T=200s:
- IF confidence ≥ 75%: Use 100% Isolation Forest (Full Personalized)
- ELIF 50% ≤ confidence < 75%: Use 60% IF + 40% Global Thresholds (Hybrid)
- ELSE confidence < 50%: Use 30% IF + 70% Global Thresholds (Hybrid Extreme)

NEVER WAIT for more windows. NEVER DISCARD the 5 windows.

PHASE B: MONITORING (Every 40 seconds, indefinitely)
- Calculate z-scores for each vital vs. personalized baseline
- Isolation Forest anomaly detection (using confidence-adjusted weighting)
- Final score: 50% RF + 30% Anomaly + 20% qSOFA
- Status: NORMAL | MILD_STRESS | HIGH_RISK | CRITICAL
- Include confidence_indicator and clinical_recommendation in every output

Full specification attached in 'prompt_with_fallback_strategy.md'
Quick reference attached in 'confidence_fallback_summary.md'
```

### Long Version (Full Context)

See: `prompt_with_fallback_strategy.md`

---

## Implementation Checklist

### Phase A: Baseline Establishment (Windows 1-5)

#### Window Collection
- [ ] Collect exactly 40 seconds of data per window
- [ ] Gather 8 vital parameters: HR, RR, SpO2, Temp, Movement, HRV, RRV
- [ ] Calculate mean for each vital across the 40-second window
- [ ] Calculate std_dev for each vital across the 40-second window
- [ ] Calculate CV (std_dev / mean) × 100 for each vital
- [ ] Generate 7 stability flags (one per vital)

#### Confidence Calculation
- [ ] **Stability Score (40% weight)**
  - [ ] Calculate average CV across all 5 windows for each vital
  - [ ] Formula: 1 - (average_CV / max_acceptable_CV)
  - [ ] Result: 0-100% score

- [ ] **Consistency Score (35% weight)**
  - [ ] Count readings in clinically valid ranges (HR 40-120, RR 8-30, etc.)
  - [ ] Formula: (readings_in_range / total_readings) × 100
  - [ ] Result: 0-100% score

- [ ] **Activity Quality Score (15% weight)**
  - [ ] Calculate average movement across 5 windows
  - [ ] Formula: 1 - (avg_movement / 100)
  - [ ] Penalize if movement > 30 in any window
  - [ ] Result: 0-100% score

- [ ] **Variability Quality Score (10% weight)**
  - [ ] Check HRV within 20-200 ms (quality: 1.0 / 0.5 / 0.0)
  - [ ] Check RRV within 5-30 ms (quality: 1.0 / 0.5 / 0.0)
  - [ ] Average HRV_quality + RRV_quality
  - [ ] Result: 0-100% score

- [ ] **Final Confidence Formula**
  - [ ] confidence = (0.40 × stability) + (0.35 × consistency) + (0.15 × activity) + (0.10 × variability)
  - [ ] Range: 0-100%
  - [ ] Repeat after each window

#### Window 5 Finalization
- [ ] After Window 5 completes, calculate final confidence score
- [ ] Store baseline parameters (mean, std_dev) for all 8 vitals
- [ ] Store confidence breakdown (4 component scores)
- [ ] Store confidence indicator (NORMAL / ACCEPTABLE / UNRELIABLE)
- [ ] Store baseline mode (Full Personalized / Hybrid / Hybrid Extreme)

#### Fallback Decision (CRITICAL)
- [ ] IF confidence ≥ 75%: Set mode = "Full Personalized", IF_weight = 100%, Global_weight = 0%
- [ ] ELIF 50% ≤ confidence < 75%: Set mode = "Hybrid", IF_weight = 60%, Global_weight = 40%
- [ ] ELSE: Set mode = "Hybrid Extreme", IF_weight = 30%, Global_weight = 70%
- [ ] **LOCK baseline immediately at T=200s (no waiting, no re-calculation)**
- [ ] Set monitoring_ready = true

#### Output Format (Windows 1-5)
- [ ] Window number (1-5)
- [ ] Timestamp window start/end
- [ ] Vitals mean and std_dev for all 8 parameters
- [ ] Stability flags (7 boolean values)
- [ ] Confidence scores (4 components + overall)
- [ ] Status: "ESTABLISHING" or "BASELINE_LOCKED"
- [ ] Progress indicator

---

### Phase B: Real-Time Monitoring (Every 40 seconds post-baseline)

#### Data Collection
- [ ] Collect exactly 40 seconds of data per monitoring window
- [ ] Gather same 8 vital parameters: HR, RR, SpO2, Temp, Movement, HRV, RRV
- [ ] Calculate mean for each vital across the 40-second window
- [ ] Calculate std_dev for each vital across the 40-second window

#### Z-Score Calculation
- [ ] For each vital: z_score = (current_mean - baseline_mean) / baseline_std
- [ ] Calculate 8 z-scores (one per vital)
- [ ] Store for output and anomaly detection

#### Isolation Forest Anomaly Detection
- [ ] Prepare 8-dimensional vital vector [HR_mean, RR_mean, ..., RRV_mean]
- [ ] Pass through Isolation Forest trained on baseline windows
- [ ] Get raw anomaly score from Isolation Forest (typically -0.5 to 0.5)
- [ ] Normalize to 0-100 scale

#### Global Threshold Anomaly (if needed)
- [ ] IF baseline_mode is "Full Personalized": skip this step
- [ ] ELSE: Calculate deviation from global thresholds
  - [ ] For each vital out of range: score += (deviation_magnitude / vital_range) × 100
  - [ ] Sum across 8 vitals
  - [ ] global_anomaly = sum / 8

#### Blended Anomaly Score
- [ ] IF baseline_mode = "Full Personalized":
  - [ ] anomaly_score = IF_score × 100
- [ ] ELIF baseline_mode = "Hybrid":
  - [ ] anomaly_score = (0.60 × IF_score) + (0.40 × global_anomaly)
- [ ] ELSE baseline_mode = "Hybrid Extreme":
  - [ ] anomaly_score = (0.30 × IF_score) + (0.70 × global_anomaly)

#### Random Forest (Stage 2) - No Changes
- [ ] Pass engineered features to existing Random Forest model
- [ ] Get probabilities: rf_prob_normal, rf_prob_mild, rf_prob_severe
- [ ] Store rf_prob_severe for final score

#### qSOFA (Stage 3) - No Changes
- [ ] Calculate qSOFA score from RR, SpO2, HR, Temp
- [ ] Range: 0-4
- [ ] Store for final score

#### Final Score Calculation
- [ ] final_score = (0.50 × rf_prob_severe) + (0.30 × anomaly_score/100) + (0.20 × qsofa_score/4)
- [ ] Range: 0-1

#### Status Prediction
- [ ] IF final_score > 0.65: Status = "CRITICAL"
- [ ] ELIF final_score > 0.4 OR rf_prob_severe > 0.40: Status = "HIGH_RISK"
- [ ] ELIF rf_prob_mild > 0.50 OR anomaly_score > 50: Status = "MILD_STRESS"
- [ ] ELSE: Status = "NORMAL"

#### Output Format (Every 40-second window)
- [ ] Window number (6, 7, 8, ...)
- [ ] Timestamp window start/end
- [ ] Vitals mean and std_dev for all 8 parameters
- [ ] Z-scores for all 8 vitals
- [ ] Anomaly score (0-100)
- [ ] Anomaly calculation method (which mode was used)
- [ ] RF components (3 probabilities)
- [ ] qSOFA score
- [ ] Final score (0-1)
- [ ] Status (NORMAL / MILD_STRESS / HIGH_RISK / CRITICAL)
- [ ] Baseline confidence (0-100)
- [ ] Baseline mode (which weighting scheme)
- [ ] Confidence indicator (NORMAL / MODERATE_CAUTION / EXTREME_CAUTION)
- [ ] List of abnormal vitals (which ones deviated)
- [ ] Clinical recommendation (text specific to confidence level)
- [ ] Risk explanation (human-readable description of current state)

---

### Data Storage

#### Baseline Storage
- [ ] Timestamp when baseline established
- [ ] Baseline duration: 200 seconds
- [ ] Total windows: 5
- [ ] Window duration: 40 seconds
- [ ] Confidence score: 0-100
- [ ] Confidence breakdown: 4 components
- [ ] Baseline quality: NORMAL / ACCEPTABLE / UNRELIABLE
- [ ] Baseline mode: Full Personalized / Hybrid / Hybrid Extreme
- [ ] Vital ranges: mean ± 2×std for each of 8 vitals
- [ ] Global thresholds: stored reference ranges
- [ ] Window history: detailed stats for all 5 windows
- [ ] Clinical notes: recommendations based on baseline quality

#### Monitoring History
- [ ] Store every monitoring window's output
- [ ] Include all z-scores, anomaly calculations, final scores
- [ ] Build audit trail for clinical validation
- [ ] Enable trend analysis (deterioration detection)

---

### Quality Assurance

#### Test Case 1: High Confidence Path (≥75%)
- [ ] Create 5 baseline windows with stable vitals (CV < 5%)
- [ ] Verify confidence > 85%
- [ ] Verify baseline_mode = "Full Personalized"
- [ ] Verify IF_weight = 100%, Global_weight = 0%
- [ ] Verify monitoring uses IF-only
- [ ] Verify confidence_indicator = "NORMAL"

#### Test Case 2: Moderate Confidence Path (50-75%)
- [ ] Create 5 baseline windows with some variance (CV 5-10%)
- [ ] Verify 60% < confidence < 75%
- [ ] Verify baseline_mode = "Hybrid"
- [ ] Verify IF_weight = 60%, Global_weight = 40%
- [ ] Verify monitoring_anomaly = (0.60 × IF_score) + (0.40 × global)
- [ ] Verify confidence_indicator = "MODERATE_CAUTION"
- [ ] Verify recommendation includes "suboptimal" warning

#### Test Case 3: Low Confidence Path (<50%)
- [ ] Create 5 baseline windows with high noise (CV > 12%)
- [ ] Verify confidence < 50%
- [ ] Verify baseline_mode = "Hybrid Extreme"
- [ ] Verify IF_weight = 30%, Global_weight = 70%
- [ ] Verify monitoring_anomaly = (0.30 × IF_score) + (0.70 × global)
- [ ] Verify confidence_indicator = "EXTREME_CAUTION"
- [ ] Verify recommendation includes "URGENT" and "manual assessment REQUIRED"

#### Test Case 4: Timeline (Critical)
- [ ] Baseline starts at T=0s
- [ ] Window 1: T=0-40s
- [ ] Window 2: T=40-80s
- [ ] Window 3: T=80-120s
- [ ] Window 4: T=120-160s
- [ ] Window 5: T=160-200s
- [ ] Baseline lock: EXACTLY at T=200s (no delay, no re-calculation)
- [ ] Monitoring starts: T=200-240s (Window 6)
- [ ] Verify NO PAUSE between baseline completion and monitoring start

#### Test Case 5: Output Validation
- [ ] Every monitoring window includes all required fields
- [ ] Confidence_indicator is present and correct (NORMAL / MODERATE / EXTREME)
- [ ] Clinical_recommendation is specific and actionable
- [ ] Z-scores correctly calculated
- [ ] Anomaly blending matches confidence-based weighting
- [ ] Final score correctly weighted (50-30-20)
- [ ] Status matches final_score thresholds

#### Test Case 6: Fallback Integration
- [ ] Verify all 5 baseline windows are used in anomaly calculation
- [ ] Verify 5-window baseline is never discarded
- [ ] Verify monitoring never waits for additional windows
- [ ] Verify hybrid weighting is applied consistently
- [ ] Verify confidence-based adjustments occur in real-time

---

### Edge Cases to Handle

- [ ] Window collection interrupted mid-40s → Handle gracefully (use partial window or retry)
- [ ] Vital sensor malfunction → Handle NaN/null values
- [ ] Extreme outlier in one window → Stability flag catches it; confidence adjusted
- [ ] All vitals out of range → Consistency score reflects it
- [ ] Very high movement (>50) → Activity quality penalized
- [ ] HRV suppressed (<20 ms) → Variability quality penalized
- [ ] Confidence exactly 75.0% → Treat as "≥75%" (benefit of doubt to personalization)
- [ ] Monitoring patient for weeks → Baseline never re-calculated (locked at 200s)

---

### Documentation Requirements

- [ ] Version control: Track this implementation version (v1.0)
- [ ] Comments: Explain the 3 fallback modes in code
- [ ] Logging: Log confidence calculation at Window 5 completion
- [ ] Logging: Log baseline lock decision + mode selected
- [ ] Logging: Log anomaly calculation method for every monitoring window
- [ ] Alerts: Generate clinical alert if baseline quality changes unexpectedly
- [ ] Traceability: Every output references which windows it used

---

### Deployment Checklist

- [ ] Code review: Fallback strategy correctly implemented
- [ ] Code review: All three weighting schemes (100%, 60-40, 30-70) functional
- [ ] Testing: All 6 test cases passing
- [ ] Testing: Edge cases handled gracefully
- [ ] Documentation: Clinical staff briefed on confidence indicators
- [ ] Documentation: Alert system clear about what "MODERATE_CAUTION" means
- [ ] Documentation: Explain why monitoring starts immediately (clinical safety)
- [ ] Monitoring: Track confidence distribution across patient population
- [ ] Monitoring: Alert if no baselines reach >75% (potential system issue)
- [ ] Monitoring: Validate that high-confidence baselines perform better

---

## Files to Give Anthropic

Copy these three files to your prompt:

1. **prompt_with_fallback_strategy.md** (FULL SPECIFICATION)
   - Complete model specification
   - All 3 confidence scenarios detailed
   - Hybrid fallback strategy explained
   - Return formats for all phases
   - Implementation requirements

2. **confidence_fallback_summary.md** (QUICK REFERENCE)
   - One-page summaries
   - Comparison tables
   - Decision tree
   - Test cases

3. **This file** (IMPLEMENTATION CHECKLIST)
   - Step-by-step tasks
   - QA procedures
   - Edge case handling

---

## One-Sentence Summary

**Build a sepsis model that establishes patient baselines in 200 seconds, then uses confidence-weighted hybrid blending of Isolation Forest (trained on baseline) + global thresholds (universal standards) to balance personalization with safety, proceeding to monitoring at T=200s ALWAYS, never waiting, never discarding.**

---

## Success Criteria (Final Validation)

✅ Baseline established in 200 seconds (5 × 40-second windows)
✅ Confidence score calculated correctly (4 weighted components)
✅ Fallback decision made at T=200s (NEVER waits, NEVER discards)
✅ Monitoring begins immediately post-baseline
✅ Anomaly calculation uses confidence-adjusted weighting
✅ Final scores correctly combine RF + Anomaly + qSOFA (50-30-20)
✅ Status predictions align with final scores
✅ Clinical confidence indicator present in every output
✅ Specific recommendations based on baseline confidence
✅ All 8 vital parameters tracked end-to-end
✅ Full audit trail maintained
✅ Patient protected from minute 1 through entire monitoring period
