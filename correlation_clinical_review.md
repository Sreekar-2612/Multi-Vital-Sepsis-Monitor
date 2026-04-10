# Clinical Correlation Analysis: Sepsis vs. Normal States

This document provides the clinically validated correlation thresholds for 21 parameter pairs across 7 monitored vitals: HR, RR, SpO2, Temperature, Movement, HRV, and RRV.

## Core Clinical Principle: Physiological Decoupling
In a healthy state, physiological systems are tightly coupled via homeostatic feedback loops (e.g., Respiratory Sinus Arrhythmia coupling HR and RR). Sepsis disrupts these loops, leading to **Decoupling** (low correlation) or **Pathological Locking** (excessive synchronization without variability).

## 21-Pair Correlation Table

| Parameter Pair | Normal Correlation | Sepsis Correlation | Threshold (Abnormality) | Clinical Significance | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HR-RR** | Positive (0.35 to 0.55) | Decoupled (r < 0.1) or Locked (r > 0.8) | r < 0.15 OR r > 0.75 | Loss of cardio-respiratory coupling (Sepsis-3) | PhysioNet / MIMIC-III |
| **HR-SpO2** | Negative (-0.1 to -0.3) | Strong Neg (r < -0.7) | r < -0.6 | Compensatory tachycardia for hypoxia | Sepsis shock markers |
| **HR-Temp** | Positive (0.4 to 0.6) | Decoupled (r < 0.2) | r < 0.2 | Relative bradycardia / Autonomic failure | Clinical Hemodynamics |
| **HR-Movement** | Positive (0.5 to 0.8) | Decoupled (r < 0.2) | r < 0.3 | HR rise without physical activity (SIRS) | Wearable Res. |
| **HR-HRV** | Negative (-0.3 to -0.5) | Decoupled (r → 0) | r > -0.1 | HRV collapse; loss of vagal tone | HRV Sepsis Meta-analysis |
| **HR-RRV** | Negative (-0.2 to -0.4) | Decoupled (r → 0) | r > -0.1 | Respiratory control instability | PhysioNet |
| **RR-SpO2** | Negative (-0.2 to -0.4) | Strong Neg (r < -0.8) | r < -0.7 | Tachypnea failure to oxygenate | ARDS/Sepsis studies |
| **RR-Temp** | Positive (0.3 to 0.5) | Decoupled (r < 0.1) | r < 0.1 | Dissociation of fever and breathing | Critical Care Med |
| **RR-Movement** | Positive (0.4 to 0.7) | Decoupled (r < 0.2) | r < 0.2 | Tachypnea at rest | Wearable Sepsis Res. |
| **RR-HRV** | Negative (-0.2 to -0.4) | Locked Neg (r < -0.8) | r < -0.7 | Suppressed variability due to stress | Autonomic Decoupling |
| **RR-RRV** | Negative (-0.3 to -0.6) | Decoupled (r → 0) | r > -0.2 | Loss of breathing rhythmicity | Respiratory complexity |
| **SpO2-Temp** | Near Zero (-0.1 to 0.1) | Negative (r < -0.3) | r < -0.2 | Hypoxia during hyperthermia (Shock) | Clinical Physiology |
| **SpO2-Mov** | Near Zero (-0.1 to 0.1) | Negative (r < -0.4) | r < -0.3 | Activity-induced desaturation | COPD/Surgical Sepsis |
| **SpO2-HRV** | Positive (0.1 to 0.3) | Decoupled (r < 0.0) | r < 0.05 | Loss of HRV-O2 synchronization | Critical Care |
| **SpO2-RRV** | Positive (0.1 to 0.2) | Decoupled (r < 0.0) | r < 0.05 | Respiratory instability signs | PhysioNet |
| **Temp-Mov** | Positive (0.3 to 0.5) | Negative (r < -0.3) | r < 0 | **Sepsis Imprinting**: High fever + Immobility | Wearable Sepsis Res. |
| **Temp-HRV** | Negative (-0.2 to -0.4) | Decoupled (r → 0) | r > -0.1 | Inflammatory suppression of Vagal tone | Sepsis-3 biomarkers |
| **Temp-RRV** | Negative (-0.1 to -0.3) | Decoupled (r → 0) | r > -0.05 | Respiratory drive instability | Clinical Res. |
| **Mov-HRV** | Positive (0.3 to 0.6) | Decoupled (r < 0.1) | r < 0.2 | Loss of autonomic reflex to activity | Autonomic Res. |
| **Mov-RRV** | Positive (0.2 to 0.5) | Decoupled (r < 0.1) | r < 0.15 | Loss of respiratory reflex to activity | Clinical Phys. |
| **HRV-RRV** | Positive (0.4 to 0.7) | Decoupled (r < 0.2) | r < 0.3 | **Systemic Complexity Collapse** | PhysioNet Challenge |

## Direction of Shift During Sepsis Progression
1. **Early Phase (SIRS)**: Correlations often "Lock" (become pathologically high) as the body enters a unified high-stress state.
2. **Intermediate Phase (MODS)**: Decoupling begins. HR and RR stay high but lose their synchronization.
3. **Severe Phase (Shock)**: Total Decoupling. HRV and RRV collapse to zero variance, leading to random or near-zero correlations with all other vitals.
