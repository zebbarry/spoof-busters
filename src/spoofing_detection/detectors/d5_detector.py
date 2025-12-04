"""
D5: Combined Detection 
Reference: https://radionavlab.ae.utexas.edu/images/stories/files/papers/gnss_spoofing_detection.pdf
"""

from dataclasses import dataclass
import gnss_lib_py
import numpy as np
import pandas as pd
from typing import Dict, Optional
from ..base import BaseDetector, BaseConfig


@dataclass
class D5Config(BaseConfig):
    min_duration_seconds: int = 0
    max_threshold: float = 9.47


class D5_CombinedDetector(BaseDetector):
    
    def __init__(self, config: Optional[D5Config] = None):
        super().__init__(config or D5Config())
        self.config: D5Config = self.config
    
    def _safe_zscore(self, series: pd.Series) -> pd.Series:
        mu = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std) or std < 1e-10:
            return series - mu
        return (series - mu) / std
    
    def detect(self, navdata: gnss_lib_py.NavData, state_estimates=None, **kwargs) -> Dict:
        df_raw = pd.DataFrame({
            'gps_millis': navdata['gps_millis'],
            'sv_id': navdata['sv_id'],
            'cn0': navdata['cn0_dbhz'],
            'doppler': navdata['doppler_hz'],
            'b_rx_ns': navdata['b_rx_ns'],
            'clock_drift_ns_s': navdata['clock_drift_ns_s']
        })
        
        df = df_raw.groupby('gps_millis').agg(
            n_sats=('sv_id', 'nunique'),
            cn0_mean=('cn0', 'mean'),
            cn0_std=('cn0', 'std'),
            doppler_mean=('doppler', 'mean'),
            doppler_std=('doppler', 'std'),
            b_rx_ns=('b_rx_ns', 'first'),
            clock_drift_ns_s=('clock_drift_ns_s', 'first')
        )

        print(f"  Processing {len(df)} epochs...")

        print("  Computing z-scores...")
        df['z_cn0_mean'] = self._safe_zscore(df['cn0_mean'])
        df['z_cn0_std'] = self._safe_zscore(df['cn0_std'])
        df['z_doppler_mean'] = self._safe_zscore(df['doppler_mean'])
        df['z_doppler_std'] = self._safe_zscore(df['doppler_std'])
        df['z_b_rx_ns'] = self._safe_zscore(df['b_rx_ns'])
        df['z_clock_drift_ns_s'] = self._safe_zscore(df['clock_drift_ns_s'])

        df['D5_score'] = (
            np.abs(df['z_cn0_mean'])
            + np.abs(df['z_cn0_std'])
            + np.abs(df['z_doppler_mean'])
            + np.abs(df['z_doppler_std'])
            + np.abs(df['z_b_rx_ns'])
            + np.abs(df['z_clock_drift_ns_s'])
        )

        mean_score = df['D5_score'].mean()
        std_score = df['D5_score'].std()
        if False:
            threshold = mean_score + 3 * std_score
        else:
            threshold = self.config.max_threshold

        print(f"  Mean D5 score = {mean_score:.3f}, std = {std_score:.3f}, threshold = {threshold:.3f}")

        df['is_anomaly'] = df['D5_score'] > threshold

        if self.config.min_duration_seconds > 0:
            df['anomaly_group'] = (
                (df['is_anomaly'] & ~df['is_anomaly'].shift(fill_value=False))
                .cumsum()
            )
            df.loc[~df['is_anomaly'], 'anomaly_group'] = 0

            if df['anomaly_group'].max() > 0:
                group_durations = df[df['anomaly_group'] > 0].groupby('anomaly_group').size()

                short_groups = group_durations[group_durations < self.config.min_duration_seconds].index
                df.loc[df['anomaly_group'].isin(short_groups), 'is_anomaly'] = False

        self.results = {
            'gps_millis': df.index.tolist(),
            'timestamp': [pd.to_datetime(t, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                         for t in df.index],
            'D5_score': df['D5_score'].tolist(),
            'spoofing_alert': df['is_anomaly'].tolist(),
            'threshold': float(threshold),
            'z_cn0_mean': df['z_cn0_mean'].tolist(),
            'z_cn0_std': df['z_cn0_std'].tolist(),
            'z_doppler_mean': df['z_doppler_mean'].tolist(),
            'z_doppler_std': df['z_doppler_std'].tolist(),
            'z_b_rx_ns': df['z_b_rx_ns'].tolist(),
            'z_clock_drift_ns_s': df['z_clock_drift_ns_s'].tolist(),
            'n_sats': df['n_sats'].tolist(),
            'cn0_mean': df['cn0_mean'].tolist(),
            'cn0_std': df['cn0_std'].tolist(),
            'doppler_mean': df['doppler_mean'].tolist(),
            'doppler_std': df['doppler_std'].tolist()
        }
        n_anomalies = int(df['is_anomaly'].sum())
        print(f"  Detected {n_anomalies} anomalous epochs ({n_anomalies/len(df)*100:.1f}%)")
        print(f"  Score range: {df['D5_score'].min():.2f} - {df['D5_score'].max():.2f}")
        
        return self.results

    
    def get_alerts(self) -> np.ndarray:
        return np.array(self.results['spoofing_alert'])
    
    def get_summary(self) -> Dict:
        return {
            'total_epochs': len(self.results['gps_millis']),
            'anomalous_epochs': int(np.sum(self.results['spoofing_alert'])),
            'detection_rate': float(np.mean(self.results['spoofing_alert'])),
        }
