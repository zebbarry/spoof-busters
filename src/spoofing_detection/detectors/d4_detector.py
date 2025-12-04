"""
D4: Drift Monitoring Detector
"""

from dataclasses import dataclass
import numpy as np
from ..base import BaseDetector, BaseConfig


@dataclass
class D4Config(BaseConfig):
    position_threshold: float = 2.0
    clock_threshold: float = 100.0
    min_epochs: int = 2
    max_time_gap: float = 10.0
    check_position: bool = True
    check_clock: bool = True


class D4_DriftDetector(BaseDetector):
    def __init__(self, config: D4Config | None = None):
        super().__init__(config or D4Config())
    
    def detect(self, navdata, state_estimates=None, **kwargs):
        if state_estimates is None:
            raise ValueError(f"No state estimates provided. Cannot run {self.name}.")
        
        if state_estimates.shape[1] < self.config.min_epochs:
            raise ValueError(f"Need at least {self.config.min_epochs} epochs for drift detection")

        sorted_idx = np.argsort(state_estimates['gps_millis'])

        times = state_estimates['gps_millis'][sorted_idx]
        x_pos = state_estimates['x_rx_m'][sorted_idx]
        y_pos = state_estimates['y_rx_m'][sorted_idx]
        z_pos = state_estimates['z_rx_m'][sorted_idx]

        dt = np.diff(times) / 1000.0 

        dx = np.diff(x_pos)
        dy = np.diff(y_pos)
        dz = np.diff(z_pos)
        position_change = np.sqrt(dx**2 + dy**2 + dz**2)

        valid_dt = dt > 0
        velocity = np.zeros_like(dt)
        velocity[valid_dt] = position_change[valid_dt] / dt[valid_dt]

        if 'b_rx_m' in state_estimates.rows:
            clk_bias = state_estimates['b_rx_m'][sorted_idx]
            db = np.diff(clk_bias)

            RESET_STEP_THRESH = 1e5 
            large_steps = np.abs(db) > RESET_STEP_THRESH
            db[large_steps] = np.nan

            clock_drift = np.zeros_like(dt)
            clock_drift[valid_dt] = db[valid_dt] / dt[valid_dt]
        else:
            clock_drift = np.zeros_like(dt)
            
        valid_intervals = (dt > 0) & (dt <= self.config.max_time_gap)

        times_out = times[1:][valid_intervals]
        velocity_out = velocity[valid_intervals]
        clock_drift_out = clock_drift[valid_intervals]

        position_alerts = np.zeros(len(velocity_out), dtype=bool)
        clock_alerts = np.zeros(len(clock_drift_out), dtype=bool)
        
        if self.config.check_position:
            position_alerts = velocity_out > self.config.position_threshold

        if self.config.check_clock:
            clock_alerts = np.abs(clock_drift_out) > self.config.clock_threshold

        spoofing_alert = position_alerts | clock_alerts

        self.results = {
            'gps_millis': times_out.tolist(),
            'timestamp': [str(t) for t in times_out],
            'position_velocity': velocity_out.tolist(),
            'clock_drift_rate': clock_drift_out.tolist(),
            'position_jump_alert': position_alerts.tolist(),
            'clock_jump_alert': clock_alerts.tolist(),
            'spoofing_alert': spoofing_alert.tolist()
        }

        print(f"  Processed {len(self.results['gps_millis'])} epochs")
        print(f"  Position jump alerts: {sum(self.results['position_jump_alert'])}")
        print(f"  Clock jump alerts: {sum(self.results['clock_jump_alert'])}")
        print(f"  Combined spoofing alerts: {sum(self.results['spoofing_alert'])}")
        
        return self.results
    
    def get_alerts(self) -> np.ndarray:
        if self.results is None:
            return np.array([])
        
        alerts = (
            np.array(self.results['position_jump_alert']) |
            np.array(self.results['clock_jump_alert'])
        )
        return alerts
