"""
D1: Pseudorange-based RAIM Detector
"""

import numpy as np
from ..base import BaseDetector, BaseConfig
from dataclasses import dataclass


@dataclass
class D1Config(BaseConfig):
    max_residual_threshold: float = 979.10
    rms_residual_threshold: float = 140.84
    min_satellites: int = 5
    rolling_window_size: int = 5
    alert_on_single_outlier: bool = True


class D1_RAIMDetector(BaseDetector):
    def __init__(self, config: D1Config | None = None):
        super().__init__(config or D1Config())
        self.config: D1Config = self.config
    
    def detect(self, navdata, state_estimates=None, **kwargs):
        all_times = navdata['gps_millis']
        all_residuals = navdata['pr_residual_m']
        all_timestamps = navdata['timestamp_str']

        unique_times, inverse_indices = np.unique(all_times, return_inverse=True)

        n_epochs = len(unique_times)
        mean_residual = np.zeros(n_epochs)
        max_residual = np.zeros(n_epochs)
        rms_residual = np.zeros(n_epochs)
        num_satellites = np.zeros(n_epochs, dtype=int)
        timestamp_strs = []

        for i in range(n_epochs):
            epoch_mask = inverse_indices == i
            epoch_residuals = all_residuals[epoch_mask]

            if len(epoch_residuals) < self.config.min_satellites:
                raise RuntimeWarning(f"Not enough satellites in view for RAIM: ", len(epoch_residuals))
                timestamp_strs.append(all_timestamps[epoch_mask][0])
                continue

            mean_residual[i] = np.mean(np.abs(epoch_residuals))
            max_residual[i] = np.max(np.abs(epoch_residuals))
            rms_residual[i] = np.sqrt(np.mean(epoch_residuals**2))
            num_satellites[i] = len(epoch_residuals)
            timestamp_strs.append(all_timestamps[epoch_mask][0])

        max_alerts = max_residual > self.config.max_residual_threshold
        rms_alerts = rms_residual > self.config.rms_residual_threshold

        weights = np.ones(self.config.rolling_window_size) / self.config.rolling_window_size
        max_alerts = np.convolve(max_alerts, weights, 'same')
        max_alerts = np.convolve(rms_alerts, weights, 'same')
        if self.config.alert_on_single_outlier:
            spoofing_alert = np.logical_or(rms_alerts, max_alerts)
        else:
            spoofing_alert = rms_alerts

        self.results = {
            'gps_millis': unique_times.tolist(),
            'timestamp': timestamp_strs,
            'mean_residual': mean_residual.tolist(),
            'max_residual': max_residual.tolist(),
            'rms_residual': rms_residual.tolist(),
            'num_satellites': num_satellites.tolist(),
            'spoofing_alert': spoofing_alert.tolist()
        }
        
        print(f"  Processed {len(self.results['gps_millis'])} epochs")
        print(f"  Alerts: {sum(self.results['spoofing_alert'])}")
        print(f"  Mean RMS residual: {np.mean(rms_residual):.2f} m")
        
        return self.results
    
    def get_alerts(self) -> np.ndarray:
        if self.results is None:
            return np.array([])
        return np.array(self.results['spoofing_alert'])
