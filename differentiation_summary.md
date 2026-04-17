# Sepsis Differentiation Architecture

This document outlines how the system distinguishes sepsis from other clinical states using multi-dimensional signals.

## 1. Physiological Decoupling (Primary Differentiator)
The system monitors **21 vital sign pairs** to detect shifts in how physiological systems interact.

| condition | signature logic |
| :--- | :--- |
| **Sepsis** | High HR-RR (Locking), Low HR-TEMP (Decoupling), **Negative TEMP-MOVEMENT** (Immobility with fever). |
| **Simple Infection** | Tight HR-TEMP coupling (Normal fever response), positive TEMP-MOVEMENT (activity-induced heat). |
| **Cardiac Event** | Decoupled HR-RR, but extremely high coupling between HR and HRV. |
| **Normal Stress** | Strong HR-MOVEMENT coupling (Autonomic responsiveness). |

## 2. Second Derivative (Acceleration) Logic
Second derivatives ($d^2/dt^2$) are used to detect **acceleration** in vital signs.

### Role in Differentiation
While first derivatives show the *rate* of change, second derivatives identify an **unstoppable physiological slide**. 
- **Acute Sepsis**: Shows synchronized acceleration (e.g., HR rising faster AND SpO2 falling faster).
- **Chronic/Stable Illness**: May have abnormal first derivatives (e.g., constant high HR), but will have near-zero second derivatives (no acceleration).

### Acceleration Thresholds (`feature_engine.py`)
| Parameter | Acceleration Threshold ($d^2$) |
| :--- | :--- |
| **HR** | $> 0.05$ |
| **RR** | $> 0.08$ |
| **TEMP** | $> 0.001$ |
| **HRV** | $< -0.10$ |

## 3. Sepsis Imprinting
A high-specificity marker for sepsis is the **Inverse Temperature-Movement Correlation**. 
- In most diseases, temperature rises with activity.
- In sepsis, temperature rises as movement drops to zero (Immobility), creating a "Sepsis Fingerprint" in the correlation matrix.
