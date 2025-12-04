"""
D6: Doppler-Pseudorange Consistency Detector
"""

import numpy as np
import scipy
from ..base import BaseDetector, BaseConfig
from dataclasses import dataclass

@dataclass
class D6Config(BaseConfig):
    max_error_threshold: float = 10.51 # See detect_spoofing.ipynb for derivation. Based on mean+3sigma of clean data
    rms_error_threshold: float = 2.29
    max_time_gap: float = 1.5
    rolling_window_size: int = 5
    alert_on_single_outlier: bool = True


class D6_DopplerConsistencyDetector(BaseDetector):
    WAVELENGTH_MAP = {
        ("gps", 0): 1575.42e6,  
        ("gps", 3): 1227.60e6,
 
        ("galileo", 0): 1575.42e6,  
        ("galileo", 6): 1207.14e6,

        ("beidou", 0): 1561.098e6, 
        ("beidou", 2): 1207.14e6,  

        ("qzss", 0): 1575.42e6,  
        ("qzss", 5): 1227.60e6,  
        
        ("glonass", 0): 1602.0e6,
        ("glonass", 2): 1246.0e6, 
    }
    
    def __init__(self, config: D6Config | None = None):
        super().__init__(config or D6Config())
        self.config: D6Config = self.config
    
    def _get_wavelength_array(self, gnss_id_array: np.ndarray, sig_id_array: np.ndarray) -> np.ndarray:
        wavelengths = np.zeros_like(gnss_id_array, dtype=float)
        for i in range(len(gnss_id_array)):
            key = (gnss_id_array[i], sig_id_array[i])
            if key not in self.WAVELENGTH_MAP:
                raise LookupError(f"Unknown GNSS and Signal ID combo: {key}")
            freq = self.WAVELENGTH_MAP[key]
            wavelengths[i] = scipy.constants.c / freq
            
        return wavelengths

    def detect(self, navdata, state_estimates=None, **kwargs):
        times_ms = navdata['gps_millis']
        pr = navdata['raw_pr_m']
        doppler = navdata['doppler_hz']
        gnss_id = navdata['gnss_id']
        sv_id = navdata['sv_id']
        sig_id = navdata['sig_id']
        timestamp_strs = navdata['timestamp_str']
        wavelengths = self._get_wavelength_array(gnss_id, sig_id)
        
        sort_order = np.lexsort((times_ms, sig_id, sv_id, gnss_id))
        
        times_sorted = times_ms[sort_order]
        pr_sorted = pr[sort_order]
        doppler_sorted = doppler[sort_order]
        gnss_sorted = gnss_id[sort_order]
        sv_sorted = sv_id[sort_order]
        sig_sorted = sig_id[sort_order]
        wavelengths_sorted = wavelengths[sort_order]
        
        dt = np.diff(times_sorted) / 1000.0  
        
        d_pr = np.diff(pr_sorted)

        with np.errstate(divide='ignore', invalid='ignore'):
            v_code = d_pr / dt

        v_doppler = -doppler_sorted[1:] * wavelengths_sorted[1:]

        same_sat_mask = (gnss_sorted[1:] == gnss_sorted[:-1]) & \
                        (sv_sorted[1:] == sv_sorted[:-1]) & \
                        (sig_sorted[1:] == sig_sorted[:-1])
        
        valid_time_mask = (dt > 0) & (dt <= self.config.max_time_gap)
        
        valid_mask = same_sat_mask & valid_time_mask

        consistency_error = np.zeros_like(v_code)
        consistency_error[valid_mask] = np.abs(v_code[valid_mask] - v_doppler[valid_mask])
        
        valid_indices = sort_order[1:]
        aligned_errors = np.full(len(times_ms), np.nan)
        aligned_errors[valid_indices[valid_mask]] = consistency_error[valid_mask]

        unique_times_orig, inverse_indices = np.unique(times_ms, return_inverse=True)
        n_epochs = len(unique_times_orig)
        
        mean_consistency_err = np.zeros(n_epochs)
        max_consistency_err = np.zeros(n_epochs)
        rms_consistency_err = np.zeros(n_epochs)
        out_timestamps = []
        
        for i in range(n_epochs):
            epoch_mask = inverse_indices == i

            epoch_errors = aligned_errors[epoch_mask]
            epoch_errors = epoch_errors[~np.isnan(epoch_errors)]
            
            if len(epoch_errors) == 0:
                out_timestamps.append(timestamp_strs[epoch_mask][0])
                continue
                
            mean_consistency_err[i] = np.mean(epoch_errors)
            rms_consistency_err[i] = np.sqrt(np.mean(epoch_errors**2))
            max_consistency_err[i] = np.max(epoch_errors)
                
            out_timestamps.append(timestamp_strs[epoch_mask][0])

        max_alerts = max_consistency_err > self.config.max_error_threshold
        rms_alerts = rms_consistency_err > self.config.rms_error_threshold

        # average
        weights = np.ones(self.config.rolling_window_size) / self.config.rolling_window_size
        max_alerts = np.convolve(max_alerts, weights, 'same')
        max_alerts = np.convolve(rms_alerts, weights, 'same')
        if self.config.alert_on_single_outlier:
            spoofing_alert = np.logical_or(rms_alerts, max_alerts)
        else:
            spoofing_alert = rms_alerts

        self.results = {
            'gps_millis': unique_times_orig.tolist(),
            'timestamp': out_timestamps,
            'mean_consistency_error': mean_consistency_err.tolist(),
            'max_consistency_error': max_consistency_err.tolist(),
            'rms_consistency_error': rms_consistency_err.tolist(),
            'spoofing_alert': spoofing_alert.tolist()
        }
        
        print(f"  Processed {len(self.results['gps_millis'])} epochs")
        print(f"  Alerts: {sum(self.results['spoofing_alert'])}")
        print(f"  Avg Max Error: {np.mean(max_consistency_err):.2f} m/s")
        print(f"  Avg RMS Error: {np.mean(rms_consistency_err):.2f} m/s")
        
        return self.results

    def get_alerts(self) -> np.ndarray:
        if self.results is None:
            return np.array([])
        return np.array(self.results['spoofing_alert'])
