"""
D2: Observable Monitoring Detector
"""

from dataclasses import dataclass
import numpy as np
from ..base import BaseDetector, BaseConfig


@dataclass
class D2Config(BaseConfig):
    cn0_high_threshold: float = 45
    cn0_low_threshold: float = 35.32  # based on mean - 3stddev
    variance_threshold: float = 4.16  # based on mean - 3stddev
    power_step_threshold: float = 2.0
    check_high_cn0: bool = True
    check_low_cn0: bool = True
    check_low_variance: bool = False  # Doesnt seem effective
    check_power_steps: bool = True


class D2_ObservablesDetector(BaseDetector):
    def __init__(self, config: D2Config | None = None):
        super().__init__(config or D2Config())
        self.config: D2Config = self.config
    
    def detect(self, navdata, state_estimates=None, **kwargs):
        all_times = navdata['gps_millis']
        all_cn0 = navdata['cn0_dbhz']
        all_timestamps = navdata['timestamp_str']
        
        unique_times, inverse_indices = np.unique(all_times, return_inverse=True)
        n_epochs = len(unique_times)

        mean_cn0 = np.zeros(n_epochs)
        std_cn0 = np.zeros(n_epochs)
        max_cn0 = np.zeros(n_epochs)
        min_cn0 = np.zeros(n_epochs)
        cn0_range = np.zeros(n_epochs)
        high_cn0_alert = np.zeros(n_epochs, dtype=bool)
        low_cn0_alert = np.zeros(n_epochs, dtype=bool)
        low_cn0_variance = np.zeros(n_epochs, dtype=bool)
        power_step_detected = np.zeros(n_epochs, dtype=bool)
        timestamp_strs = []

        for i in range(n_epochs):
            epoch_mask = inverse_indices == i
            cn0_values = all_cn0[epoch_mask]
            
            if len(cn0_values) == 0:
                continue
            
            mean_cn0[i] = np.mean(cn0_values)
            std_cn0[i] = np.std(cn0_values)
            max_cn0[i] = np.max(cn0_values)
            min_cn0[i] = np.min(cn0_values)
            cn0_range[i] = max_cn0[i] - min_cn0[i]

            if self.config.check_high_cn0:
                high_cn0_alert[i] = mean_cn0[i] > self.config.cn0_high_threshold
            if self.config.check_high_cn0:
                low_cn0_alert[i] = mean_cn0[i] < self.config.cn0_low_threshold

            if self.config.check_low_variance:
                low_cn0_variance[i] = std_cn0[i] < self.config.variance_threshold
            
            timestamp_strs.append(all_timestamps[epoch_mask][0])

        if self.config.check_power_steps:
            mean_diff = np.diff(mean_cn0)
            power_step_detected[1:] = np.abs(mean_diff) > self.config.power_step_threshold

        spoofing_alert = high_cn0_alert | low_cn0_alert | low_cn0_variance | power_step_detected

        self.results = {
            'gps_millis': unique_times.tolist(),
            'timestamp': timestamp_strs,
            'mean_cn0': mean_cn0.tolist(),
            'std_cn0': std_cn0.tolist(),
            'max_cn0': max_cn0.tolist(),
            'min_cn0': min_cn0.tolist(),
            'cn0_range': cn0_range.tolist(),
            'high_cn0_alert': high_cn0_alert.tolist(),
            'low_cn0_alert': low_cn0_alert.tolist(),
            'low_cn0_variance': low_cn0_variance.tolist(),
            'power_step_detected': power_step_detected.tolist(),
            'spoofing_alert': spoofing_alert.tolist()
        }

        print(f"  Processed {len(self.results['gps_millis'])} epochs")
        print(f"  High C/N0 alerts: {sum(self.results['high_cn0_alert'])}")
        print(f"  Low C/N0 alerts: {sum(self.results['low_cn0_alert'])}")
        print(f"  Low variance alerts: {sum(self.results['low_cn0_variance'])}")
        print(f"  Power step detections: {sum(self.results['power_step_detected'])}")
        print(f"  Combined spoofing alerts: {sum(self.results['spoofing_alert'])}")
        
        return self.results
    
    def get_alerts(self) -> np.ndarray:
        if self.results is None:
            return np.array([])
        
        alerts = (
            np.array(self.results['high_cn0_alert']) |
            np.array(self.results['low_cn0_variance']) |
            np.array(self.results['power_step_detected'])
        )
        return alerts
