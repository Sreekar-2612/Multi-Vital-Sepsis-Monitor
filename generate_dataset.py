import pandas as pd
import numpy as np
from simulator import PatientStreamSimulator
from vitals_types import BASELINE_WINDOWS
import os

def generate_patient_data(p_id, condition, num_windows=50):
    sim = PatientStreamSimulator(condition=condition)
    # Give some random variation to baseline
    sim.baseline_hr = float(np.random.normal(75, 10))
    sim.baseline_temp = float(np.random.normal(36.8, 0.5))
    
    rows = []
    for i in range(BASELINE_WINDOWS + num_windows):
        sample = sim.get_next_window()
        d = sample.to_dict()
        d['patient_id'] = p_id
        d['is_baseline'] = i < BASELINE_WINDOWS
        rows.append(d)
    return rows

def main():
    num_patients = 5000
    output_file = "sepsis_dataset_5000.csv"
    
    print(f"Generating data for {num_patients} patients...")
    all_data = []
    
    for p_id in range(num_patients):
        # 40% Normal, 30% Infection, 30% Sepsis
        rand = np.random.random()
        if rand < 0.4: condition = 0
        elif rand < 0.7: condition = 1
        else: condition = 2
        
        patient_rows = generate_patient_data(p_id, condition)
        all_data.extend(patient_rows)
        
        if (p_id + 1) % 100 == 0:
            print(f"Processed {p_id + 1} patients...")
            
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Conditions distribution:\n{df[df['is_baseline']==False]['label'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()
