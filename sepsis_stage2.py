import pandas as pd
import numpy as np
import datetime
import warnings
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import json
import time

warnings.filterwarnings('ignore')

class PatientStreamSimulator:
    def __init__(self, condition=0, baseline_hr=75, baseline_temp=36.8):
        self.condition = condition
        self.baseline_hr = baseline_hr
        self.baseline_temp = baseline_temp
        self.current_time = datetime.datetime.now()
        self.hr = baseline_hr
        self.temp = baseline_temp
        self.spo2 = 98.0
        self.rr = 14.0
        self.hrv = 45.0
        self.rrv = 15.0
        
    def set_condition(self, condition):
        self.condition = condition
        
    def get_next_window(self):
        self.current_time += datetime.timedelta(seconds=40)
        
        if self.condition == 0:
            target_hr, target_temp, target_spo2, target_rr = self.baseline_hr, self.baseline_temp, 98.0, 14.0
            target_hrv, target_rrv, mov_mean = 45.0, 15.0, 10
        elif self.condition == 1:
            target_hr, target_temp, target_spo2, target_rr = self.baseline_hr + 15, self.baseline_temp + 1.0, 95.0, 18.0
            target_hrv, target_rrv, mov_mean = 30.0, 10.0, 20
        else:
            target_hr, target_temp, target_spo2, target_rr = self.baseline_hr + 45, self.baseline_temp + 2.5, 88.0, 26.0
            target_hrv, target_rrv, mov_mean = 12.0, 5.0, 5
            
        alpha = 0.2 if self.condition == 2 else 0.05
        
        self.hr += (target_hr - self.hr) * alpha + np.random.normal(0, 0.5)
        self.temp += (target_temp - self.temp) * alpha + np.random.normal(0, 0.02)
        self.spo2 += (target_spo2 - self.spo2) * alpha + np.random.normal(0, 0.2)
        self.rr += (target_rr - self.rr) * alpha + np.random.normal(0, 0.2)
        self.hrv += (target_hrv - self.hrv) * alpha + np.random.normal(0, 1.0)
        self.rrv += (target_rrv - self.rrv) * alpha + np.random.normal(0, 1.0)
        movement = np.random.normal(mov_mean, 5)
            
        self.hr = np.clip(self.hr, 40, 180)
        self.temp = np.clip(self.temp, 35.5, 41.5)
        self.spo2 = np.clip(self.spo2, 75.0, 100.0)
        self.rr = np.clip(self.rr, 8, 45)
        movement = np.clip(movement, 0, 100)
        self.hrv = np.clip(self.hrv, 5, 200)
        self.rrv = np.clip(self.rrv, 2, 50)
        
        record = {
            'timestamp': self.current_time, 
            'hr': round(self.hr, 2), 'rr': round(self.rr, 2), 'spo2': round(self.spo2, 2), 
            'temp': round(self.temp, 2), 'movement': round(movement, 2), 
            'hrv': round(self.hrv, 2), 'rrv': round(self.rrv, 2),
            'label': int(self.condition)
        }
        return pd.DataFrame([record]).set_index('timestamp')

def get_population_isolation_forest():
    np.random.seed(42)
    pop_data = []
    # simulate 2000 points of diverse normal/abnormal MIMIC-III-like patients across 7 vitals
    for _ in range(2000):
        # 80% normal bounds, 20% abnormal
        if np.random.rand() > 0.2:
            hr, rr, spo2, temp, mv, hrv, rrv = np.random.normal(70, 10), np.random.normal(16, 2), np.random.normal(98, 1), np.random.normal(36.8, 0.3), np.random.normal(10, 5), np.random.normal(45, 10), np.random.normal(15, 3)
        else:
            hr, rr, spo2, temp, mv, hrv, rrv = np.random.normal(110, 20), np.random.normal(24, 4), np.random.normal(92, 3), np.random.normal(38.5, 1.0), np.random.normal(30, 20), np.random.normal(20, 15), np.random.normal(8, 4)
        pop_data.append([hr, rr, spo2, temp, mv, hrv, rrv])
    pop_df = pd.DataFrame(pop_data, columns=['hr', 'rr', 'spo2', 'temp', 'movement', 'hrv', 'rrv'])
    iso = IsolationForest(random_state=42, contamination=0.10)
    iso.fit(pop_df)
    return iso

def establish_baseline_phase_a(baseline_stream):
    windows = baseline_stream.iloc[:5]
    vitals = ['hr', 'rr', 'spo2', 'temp', 'movement', 'hrv', 'rrv']
    
    means = windows[vitals].mean().to_dict()
    stds_raw = windows[vitals].std().replace(0, 1e-5).to_dict()
    cvs = {v: (stds_raw[v] / means[v]) * 100 for v in vitals}
    
    clinical_stds_floor = {'hr': 2.0, 'rr': 1.0, 'spo2': 1.0, 'temp': 0.1, 'movement': 2.0, 'hrv': 5.0, 'rrv': 2.0}
    stds = {v: max(stds_raw[v], clinical_stds_floor[v]) for v in vitals}
    
    max_acc_cv = {'hr': 5, 'rr': 8, 'spo2': 1, 'temp': 0.5, 'movement': 50, 'hrv': 25, 'rrv': 25}
    stability_sum = sum(max(0, 1 - (cvs[v] / max_acc_cv.get(v, 20))) for v in vitals)
    stability_score = (stability_sum / len(vitals)) * 100
    
    global_ranges = {
        'hr': (40, 120), 'rr': (8, 30), 'spo2': (92, 100), 
        'temp': (35.5, 39.5), 'movement': (0, 50), 
        'hrv': (20, 200), 'rrv': (5, 30)
    }
    in_range = sum(1 for _, row in windows.iterrows() for v in vitals if global_ranges[v][0] <= row[v] <= global_ranges[v][1])
    consistency_score = (in_range / 35.0) * 100
    
    avg_movement = means['movement']
    activity_quality = max(0, 1 - (avg_movement / 100)) * 100
    if (windows['movement'] > 30).any(): activity_quality *= 0.5
        
    hrv_q = 1.0 if 20 <= means['hrv'] <= 200 else 0.0
    rrv_q = 1.0 if 5 <= means['rrv'] <= 30 else 0.0
    variability_score = ((hrv_q + rrv_q) / 2) * 100
    
    confidence_score = (0.40 * stability_score) + (0.35 * consistency_score) + (0.15 * activity_quality) + (0.10 * variability_score)
    
    # Section 1 - Fix 4: 4-State Confidence Machine
    if confidence_score >= 75:
        mode = "LOCKED"
    elif confidence_score >= 60:
        mode = "HYBRID"
    else:
        mode = "FALLBACK"
        
    return {
        "confidence": confidence_score,
        "mode": mode,
        "baseline_means": means,
        "baseline_stds": stds
    }

def get_trained_rf():
    # Build RF dataset
    np.random.seed(42)
    rf_data = []
    # Synthetic normal and sick mapped to the 7 base vitals + advanced RF features 
    # to avoid complex dependencies during train-time.
    for _ in range(500):
        # 0 = normal, 1 = mild, 2 = severe 
        cond = np.random.choice([0, 1, 2])
        if cond == 0:
            v_hr, v_rr, v_spo2, v_temp, v_mv, v_hrv, v_rrv = 70, 14, 98, 36.8, 10, 45, 15
            immo, temp_traj, lact, mc = 0, 0, 0, 0  # Normal derived
        elif cond == 1:
            v_hr, v_rr, v_spo2, v_temp, v_mv, v_hrv, v_rrv = 90, 18, 95, 37.8, 20, 30, 10
            immo, temp_traj, lact, mc = 0.2, 0.01, 0.4, 0.1 
        else:
            v_hr, v_rr, v_spo2, v_temp, v_mv, v_hrv, v_rrv = 115, 26, 88, 39.5, 5, 12, 5
            immo, temp_traj, lact, mc = 0.8, 0.05, 0.8, 0.5 
            
        # Add noise
        v_hr += np.random.normal(0, 5)
        rf_data.append([v_hr, v_rr, v_spo2, v_temp, v_mv, v_hrv, v_rrv, immo, temp_traj, lact, mc, cond])
        
    df = pd.DataFrame(rf_data, columns=['hr', 'rr', 'spo2', 'temp', 'movement', 'hrv', 'rrv', 'immobility', 'temp_trajectory', 'lactate_proxy', 'multi_system_corr', 'label'])
    
    X = df.drop(columns=['label'])
    y = df['label']
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)
    return rf

def feature_engine_sepsis_accel(d2):
    cnt = sum([
        d2.get('d2hr', 0) > 0.05,
        d2.get('d2rr', 0) > 0.08,
        d2.get('d2temp', 0) > 0.001,
        d2.get('d2hrv', 0) < -0.10,
        d2.get('d2rrv', 0) < -0.08
    ])
    if cnt >= 3: return 1.0, cnt
    elif cnt == 2: return 0.6, cnt
    elif cnt == 1: return 0.3, cnt
    return 0.0, cnt

def hrv_collapse_severity(hrv_history):
    if len(hrv_history) < 10: return 0.0
    baseline_hrv = np.mean(hrv_history[:6])
    current_hrv = np.mean(hrv_history[-3:])
    if baseline_hrv == 0: return 0.0
    pct_drop = (baseline_hrv - current_hrv) / baseline_hrv
    if pct_drop > 0.50: return 1.0
    elif pct_drop > 0.35: return 0.7
    elif pct_drop > 0.20: return 0.4
    return 0.0

def immobility_score(mov_history):
    if len(mov_history) < 6: return 0.0
    baseline_mv = np.mean(mov_history[:6])
    recent_mv = np.mean(mov_history[-3:])
    drop = (baseline_mv - recent_mv) / max(baseline_mv, 1.0)
    return float(np.clip(drop, 0.0, 1.0))

def temp_trajectory(temp_history):
    if len(temp_history) < 6: return 0.0
    x = list(range(len(temp_history)))
    slope, _ = np.polyfit(x, temp_history, 1)
    return float(slope)

def lactate_proxy(spo2, hr, rr, hrv, mv):
    s = (max(0, 95 - spo2)/10)*0.30 + (max(0, hr - 90)/60)*0.25 + (max(0, rr - 20)/20)*0.20 + (max(0, 60 - hrv)/60)*0.15 + (max(0, 30 - mv)/30)*0.10
    return min(s, 1.0)

def phase_detection(score, sbp_proxy=120):
    if score > 0.65 and sbp_proxy <= 90: return "PHASE_3_SEPTIC_SHOCK"
    elif score > 0.50: return "PHASE_2_INTERMEDIATE"
    elif score > 0.30: return "PHASE_1_EARLY"
    return "PHASE_0_NORMAL"

def multi_system_correlation(history):
    if len(history) < 30: return None
    recent = history[-30:]
    w_3plus = sum(1 for w in recent if sum(1 for z in w['z_scores'].values() if abs(z) > 2.0) >= 3)
    return w_3plus / 30.0

def realtime_monitoring_step(baseline_data, pop_if, rf_model, simulator):
    vitals_keys = ['hr', 'rr', 'spo2', 'temp', 'movement', 'hrv', 'rrv']
    
    score_history = []
    consecutive_normal_window_count = 0
    window_count = 5
    
    try:
        while True:
            window_count += 1
            window = simulator.get_next_window()
            row = window.iloc[0].to_dict()
            ts = str(window.index[0])
            
            # Artifact check
            art_contaminated = row['movement'] > (baseline_data['baseline_means']['movement'] * 2.5)
            
            # Baseline Drift Trigger check (applied at the START of the window if consecutive norms exist from LAST loop)
            if consecutive_normal_window_count >= 10:
                for v in vitals_keys:
                    baseline_data['baseline_means'][v] = 0.9*baseline_data['baseline_means'][v] + 0.1*row[v]
                consecutive_normal_window_count = 0
                
            # Z scores
            z_scores = {v: (row[v] - baseline_data['baseline_means'][v]) / baseline_data['baseline_stds'][v] for v in vitals_keys}
            
            # Derivatives Setup
            first_deriv = {f'd{v}': 0.0 for v in vitals_keys}
            second_deriv = {f'd2{v}': 0.0 for v in vitals_keys}
            smooth_deriv = {f'd{v}': 0.0 for v in vitals_keys}
            deriv_avail = False
            
            if not art_contaminated:
                if len(score_history) >= 1:
                    prev_row = score_history[-1]['vitals_current']
                    for v in vitals_keys:
                        first_deriv[f'd{v}'] = (row[v] - prev_row[v]) / 40.0
                    if len(score_history) >= 2:
                        prev_smooth = score_history[-1].get('smooth_first_deriv', first_deriv) # fallback
                        for v in vitals_keys:
                            smooth_deriv[f'd{v}'] = (0.3 * first_deriv[f'd{v}']) + (0.7 * prev_smooth.get(f'd{v}', 0))
                        
                        prev_1st = score_history[-1]['first_derivatives']
                        for v in vitals_keys:
                            second_deriv[f'd2{v}'] = (first_deriv[f'd{v}'] - prev_1st[f'd{v}']) / 40.0
                        deriv_avail = True
                        
            trajectory_boost, accel_count = feature_engine_sepsis_accel(second_deriv) if deriv_avail else (0.0, 0)
            
            # Anomaly Score (Population IF mixed with Personal Z)
            pop_score = pop_if.decision_function([[row[v] for v in vitals_keys]])[0]
            pop_anomaly = np.clip(-pop_score, 0, 1) * 100
            
            if baseline_data['mode'] == "LOCKED":
                personal_anomaly = min(100, np.mean([abs(z) for z in z_scores.values()]) * 20)
                anomaly_score = 0.5 * pop_anomaly + 0.5 * personal_anomaly
            elif baseline_data['mode'] == "HYBRID":
                personal_anomaly = min(100, np.mean([abs(z) for z in z_scores.values()]) * 20)
                # 60% IF mixed... adapting mapping to global rules 
                anomaly_score = 0.6 * (0.5 * pop_anomaly + 0.5 * personal_anomaly) + 0.4 * pop_anomaly
            else: # FALLBACK
                anomaly_score = pop_anomaly
                
            # Random Forest advanced features wrapper
            hrv_hist = [h['vitals_current']['hrv'] for h in score_history] + [row['hrv']]
            mov_hist = [h['vitals_current']['movement'] for h in score_history] + [row['movement']]
            tmp_hist = [h['vitals_current']['temp'] for h in score_history] + [row['temp']]
            
            immo = immobility_score(mov_hist)
            t_traj = temp_trajectory(tmp_hist)
            lact = lactate_proxy(row['spo2'], row['hr'], row['rr'], row['hrv'], row['movement'])
            msc = multi_system_correlation(score_history)
            msc_val = 0 if msc is None else msc
            
            rf_input = [row['hr'], row['rr'], row['spo2'], row['temp'], row['movement'], row['hrv'], row['rrv'], immo, t_traj, lact, msc_val]
            rf_probs = rf_model.predict_proba([rf_input])[0]
            rf_prob_severe = rf_probs[2] if len(rf_probs) > 2 else 0
            
            qsofa = sum(1 for v, thresh in [('rr', 22), ('hr', 100)] if row[v] >= thresh) + (1 if row['spo2'] < 92 else 0)
            
            # Revised Final Score Formula (45, 28, 17, 10)
            final_raw = (rf_prob_severe * 0.45) + ((anomaly_score/100) * 0.28) + ((qsofa/4) * 0.17) + (trajectory_boost * 0.10)
            
            hrv_c_sev = hrv_collapse_severity(hrv_hist)
            final_score = min(1.0, final_raw * (1 + 0.3 * hrv_c_sev)) if len(score_history) >= 10 else final_raw
            
            if final_score > 0.65: status = "CRITICAL"
            elif final_score > 0.4 or rf_prob_severe > 0.4: status = "HIGH_RISK"
            elif rf_probs[1] > 0.5 or anomaly_score > 50: status = "MILD_STRESS"
            else: status = "NORMAL"
            
            if status == "NORMAL" and all(abs(z) < 1.5 for z in z_scores.values()) and row['movement'] < 25:
                consecutive_normal_window_count += 1
            else:
                consecutive_normal_window_count = 0
                
            s_phase = phase_detection(final_score)
            
            output_json = {
                "phase": "MONITORING",
                "window_number": window_count,
                "timestamp": ts,
                "baseline_state": baseline_data['mode'],
                "baseline_confidence": baseline_data['confidence'],
                "artifact_contaminated": art_contaminated,
                "derivatives_available": deriv_avail,
                "vitals_current": {k: float(v) for k,v in row.items() if k in vitals_keys},
                "z_scores": {k: float(v) for k,v in z_scores.items()},
                "first_derivatives": first_deriv,
                "second_derivatives": second_deriv,
                "anomaly_score": float(anomaly_score),
                "rf_prob_severe": float(rf_prob_severe),
                "qsofa_score": float(qsofa),
                "trajectory_boost": float(trajectory_boost),
                "final_score": float(final_score),
                "status": status,
                "sepsis_phase": s_phase,
                "hrv_collapse_severity": float(hrv_c_sev),
                "lactate_proxy": float(lact),
                "immobility_score": float(immo),
                "multi_system_correlation": float(msc_val) if msc is not None else None,
                "sepsis_acceleration_count": accel_count,
                "consecutive_normal_window_count": consecutive_normal_window_count,
                "provisional_thresholds_active": True,
                "score_history_length": len(score_history) + 1
            }
            
            score_history.append(output_json)
            # Retain max 360 windows (4 hrs)
            if len(score_history) > 360: score_history.pop(0)
            
            # Print execution simulation cleanly mapping to strictly required output exactly
            print("====================================")
            print(json.dumps(output_json, indent=2))
            
            time.sleep(1.0) # simulate pacing dynamically
            
    except KeyboardInterrupt:
        print("Monitoring Terminated.")

if __name__ == "__main__":
    sim = PatientStreamSimulator(condition=0)
    base_frames = [sim.get_next_window() for _ in range(5)]
    base_stream = pd.concat(base_frames)
    b_data = establish_baseline_phase_a(base_stream)
    p_if = get_population_isolation_forest()
    r_mdl = get_trained_rf()
    sim.set_condition(2) # force rapid decline for demonstration purposes during infinite monitoring loop
    realtime_monitoring_step(b_data, p_if, r_mdl, sim)
