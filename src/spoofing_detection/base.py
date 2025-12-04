from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


class BaseDetector(ABC):
    def __init__(self, config):
        self.config = config
        self.results = None
        self.name = self.__class__.__name__
    
    @abstractmethod
    def detect(self, navdata, state_estimates=None, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_alerts(self) -> np.ndarray:
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        if self.results is None:
            raise RuntimeError('No results available')
        
        alerts = self.get_alerts()
        return {
            'detector': self.name,
            'total_epochs': len(alerts),
            'alerts': int(np.sum(alerts)),
            'alert_rate': float(np.mean(alerts)),
        }
    
    def __repr__(self):
        return f"{self.name}(config={self.config})"


@dataclass
class BaseConfig:
    """base to inherit"""


class DetectionResults:
    def __init__(self):
        self.results: Dict[str, dict] = {}
        self.performance: Dict[str, dict] = {}
        self.metadata = {}

    def add_result(self, detector_name: str, result: dict):
        self.results[detector_name] = result

    def get_result(self, detector_name: str) -> dict:
        if detector_name not in self.results:
            raise KeyError(f"No results found for detector '{detector_name}'. "
                          f"Available detectors: {list(self.results.keys())}")
        return self.results[detector_name]

    def get_detector_names(self) -> list:
        return list(self.results.keys())
