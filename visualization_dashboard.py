import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# Set plotting style for "Publication Quality"
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SepsisVisualizer:
    """
    Generates the 7 requested publication-quality charts for correlation analysis.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_all_charts(self, score_history: List[Dict]):
        if not score_history:
            print("No history provided for visualization.")
            return

        df_full = self._history_to_df(score_history)
        
        # Split into Normal vs Sepsis for comparison charts
        df_normal = df_full[df_full['status'] == 'NORMAL']
        df_sepsis = df_full[df_full['status'].isin(['HIGH_RISK', 'CRITICAL'])]

        self.chart1_comparison_heatmap(df_normal, df_sepsis)
        self.chart2_diff_heatmap(df_normal, df_sepsis)
        self.chart3_rolling_trajectory(score_history)
        self.chart4_radar_fingerprint(score_history)
        self.chart5_score_timeline(df_full)
        self.chart6_pairplot(df_full)
        self.chart7_abnormality_bars(score_history[-1])

    def _history_to_df(self, history: List[Dict]) -> pd.DataFrame:
        rows = []
        for h in history:
            row = h['vitals_current'].copy()
            row['status'] = h['status']
            row['final_score'] = h['final_score']
            row['sepsis_correlation_score'] = h.get('sepsis_correlation_score', 0.0)
            rows.append(row)
        return pd.DataFrame(rows)

    def chart1_comparison_heatmap(self, df_n, df_s):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Normal
        corr_n = df_n[['hr', 'rr', 'spo2', 'temp', 'movement', 'hrv', 'rrv']].corr()
        sns.heatmap(corr_n, annot=True, cmap="vlag", center=0, ax=axes[0])
        axes[0].set_title("Normal State Correlation")

        # Sepsis
        if not df_s.empty:
            corr_s = df_s[['hr', 'rr', 'spo2', 'temp', 'movement', 'hrv', 'rrv']].corr()
            sns.heatmap(corr_s, annot=True, cmap="vlag", center=0, ax=axes[1])
            axes[1].set_title("Sepsis State Correlation")
        else:
            axes[1].text(0.5, 0.5, "Insufficient Sepsis Data", ha='center')

        plt.suptitle("Inter-Parameter Correlation: Normal vs Sepsis")
        plt.savefig(f"{self.output_dir}/chart1_heatmaps.png", dpi=300)
        plt.close()

    def chart2_diff_heatmap(self, df_n, df_s):
        if df_s.empty: return
        
        cols = ['hr', 'rr', 'spo2', 'temp', 'movement', 'hrv', 'rrv']
        corr_n = df_n[cols].corr()
        corr_s = df_s[cols].corr()
        diff = corr_s - corr_n

        plt.figure(figsize=(10, 8))
        # Mask significant shifts
        sns.heatmap(diff, annot=True, cmap="RdBu_r", center=0)
        plt.title("Correlation Shift: Normal -> Sepsis (Δr)")
        plt.savefig(f"{self.output_dir}/chart2_diff_matrix.png", dpi=300)
        plt.close()

    def chart3_rolling_trajectory(self, history):
        # Extract specific pairs from fingerprint history
        pairs = ["HR_HRV", "HR_RR", "RR_SPO2", "SPO2_HRV", "TEMP_HR", "HRV_RRV"]
        windows = [h['window_number'] for h in history if 'correlation_fingerprint' in h]
        
        plt.figure(figsize=(12, 6))
        for p in pairs:
            values = [h['correlation_fingerprint'].get(p, 0.0) for h in history if 'correlation_fingerprint' in h]
            plt.plot(windows, values, label=p)
        
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        plt.title("Rolling Correlation Trajectory — Key Parameter Pairs")
        plt.ylabel("Pearson r")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/chart3_rolling_trajectory.png", dpi=300)
        plt.close()

    def chart4_radar_fingerprint(self, history):
        from math import pi
        h = history[-1]
        if 'disease_probabilities' not in h: return
        
        probs = h['disease_probabilities']
        categories = list(probs.keys())
        values = list(probs.values())
        values += values[:1] # Repeat first to close circle

        # Angular coordinates
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], categories)
        ax.plot(angles, values, linewidth=1, linestyle='solid', label='Current')
        ax.fill(angles, values, 'b', alpha=0.1)
        plt.title("Disease Discrimination by Correlation Fingerprint")
        plt.savefig(f"{self.output_dir}/chart4_radar.png", dpi=300)
        plt.close()

    def chart5_score_timeline(self, df):
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df['final_score'], label='Final Score', color='red', linewidth=2)
        plt.plot(df.index, df['sepsis_correlation_score'], label='Correlation Score', color='blue', alpha=0.7)
        
        plt.axhline(0.6, color='orange', linestyle='--', label='Warning (0.6)')
        plt.axhline(0.8, color='red', linestyle='--', label='Critical (0.8)')
        
        plt.title("Sepsis Correlation Score vs Final Score — Monitoring Timeline")
        plt.ylabel("Score (0.0 - 1.0)")
        plt.legend()
        plt.savefig(f"{self.output_dir}/chart5_timeline.png", dpi=300)
        plt.close()

    def chart6_pairplot(self, df):
        # Keep manageable size
        subset = df.tail(100)
        cols = ['hr', 'rr', 'spo2', 'temp', 'hrv', 'status']
        g = sns.pairplot(subset, hue='status', palette='husl', diag_kind='kde')
        g.fig.suptitle("Pairwise Parameter Relationships by Disease State", y=1.02)
        plt.savefig(f"{self.output_dir}/chart6_pairplot.png", dpi=300)
        plt.close()

    def chart7_abnormality_bars(self, last_output):
        if 'correlation_fingerprint' not in last_output: return
        
        fp = last_output['correlation_fingerprint']
        pairs = list(fp.keys())
        vals = list(fp.values())
        
        df_bar = pd.DataFrame({'Pair': pairs, 'r': vals})
        df_bar['color'] = ['red' if r > 0.7 or r < -0.6 else 'green' for r in vals]

        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_bar, y='Pair', x='r', palette=df_bar['color'].tolist())
        plt.title("Per-Pair Correlation Deviation from Normal Baseline")
        plt.savefig(f"{self.output_dir}/chart7_bars.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    # Test stub
    print("Visualizer ready. Integration pending data stream.")
