import os
import numpy as np
import pandas as pd
from typing import Optional

from .plotting import is_spoofing_active
from .csv_to_navdata import CSVToNavData
from .base import DetectionResults, BaseConfig, BaseDetector
from .detectors.d5_detector import D5_CombinedDetector


class SpoofingDetectionPipeline:
    def __init__(self,
                 detectors: list[BaseDetector],
                 csv_path: str,
                 ground_truth_path: Optional[str] = None):
        self.detectors = {d.name.lower(): d for d in detectors}
        self.csv_path = csv_path
        self.ground_truth_path = ground_truth_path

        self.converter = CSVToNavData(csv_path)
        self.navdata, self.state_estimates = self.load_data()

        if ground_truth_path:
            self.load_ground_truth(ground_truth_path)
        else:
            self.ground_truth = None
    
    def load_data(self):
        print("\nLoading data...")
        navdata = self.converter.convert_to_navdata()
        state_estimates = self.converter.get_state_estimates()
        return navdata, state_estimates
    
    def load_ground_truth(self, ground_truth_path: str):
        self.ground_truth_path = ground_truth_path
        print(f"\nLoading ground truth from {self.ground_truth_path}...")
        gt_df = pd.read_csv(
            self.ground_truth_path,
            parse_dates=['start', 'stop'],
            date_format="%Y-%m-%d %H:%M:%S",
        )
        
        gt_df['start_gps_millis'] = gt_df['start'].apply(
            lambda x: (x - self.converter.GPS_EPOCH).total_seconds() * 1000.0
        )
        gt_df['stop_gps_millis'] = gt_df['stop'].apply(
            lambda x: (x - self.converter.GPS_EPOCH).total_seconds() * 1000.0
        )
        
        self.ground_truth = gt_df
        
        print(f"  Loaded {len(gt_df)} spoofing intervals")
        
        return gt_df
    
    def run_detection(self) -> DetectionResults:
        print("\n" + "="*70)
        print("RUNNING SPOOFING DETECTION PIPELINE")
        print("="*70)

        results = DetectionResults()
        results.metadata['csv_path'] = self.csv_path
        results.metadata['detectors'] = list(self.detectors.keys())

        for name, detector in self.detectors.items():
            print(f"\nRunning {detector.name}...")
            result = detector.detect(self.navdata, self.state_estimates)
            results.add_result(name, result)
            print(f"  {detector.name} complete")

        if self.ground_truth is not None:
            print("\n" + "="*70)
            print("EVALUATING DETECTOR PERFORMANCE")
            print("="*70)
            for name in results.get_detector_names():
                result = results.get_result(name)
                performance = self.evaluate_performance(name, result)
                results.performance[name] = performance

        return results
    
    def evaluate_performance(self, detector_name: str, detector_results: dict):
        if self.ground_truth is None:
            raise LookupError("No ground truth available for evaluation")

        print(f"\n{detector_name.upper()} Performance:")

        times = np.array(detector_results['gps_millis'])

        gt_labels = is_spoofing_active(times, self.ground_truth)

        pred_labels = np.array(detector_results['spoofing_alert'])

        true_positives = np.sum(gt_labels & pred_labels)
        false_positives = np.sum(~gt_labels & pred_labels)
        true_negatives = np.sum(~gt_labels & ~pred_labels)
        false_negatives = np.sum(gt_labels & ~pred_labels)

        total = len(times)
        accuracy = (true_positives + true_negatives) / total

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        div = np.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives))
        mcc = (float(true_positives*true_negatives) - float(false_positives*false_negatives)) / float(div) if div != 0 else 0

        latencies = []
        for _, gt_row in self.ground_truth.iterrows():
            start_time = gt_row['start_gps_millis']
            stop_time = gt_row['stop_gps_millis']

            interval_mask = (times >= start_time) & (times <= stop_time)
            interval_detections = pred_labels[interval_mask]
            interval_times = times[interval_mask]

            if np.any(interval_detections):
                first_detection_idx = np.where(interval_detections)[0][0]
                first_detection_time = interval_times[first_detection_idx]
                latency = (first_detection_time - start_time) / 1000.0
                latencies.append(latency)

        results = {
            'detector': detector_name,
            'total_epochs': total,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'detection_rate': recall,
            'false_alarm_rate': false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0,
            'mean_latency_s': np.mean(latencies) if latencies else None,
            'median_latency_s': np.median(latencies) if latencies else None,
            'max_latency_s': np.max(latencies) if latencies else None,
            'intervals_detected': len(latencies),
            'total_intervals': len(self.ground_truth)
        }

        self._print_performance_summary(results)

        return results
    
    def _print_performance_summary(self, results):
        """Print performance metrics"""
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Clean  Spoofed")
        print(f"  Actual Clean   {results['true_negatives']:5d}   {results['false_positives']:5d}")
        print(f"  Actual Spoofed {results['false_negatives']:5d}   {results['true_positives']:5d}")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {results['accuracy']*100:.1f}%")
        print(f"  Precision:          {results['precision']*100:.1f}%")
        print(f"  Recall (Detection): {results['recall']*100:.1f}%")
        print(f"  F1 Score:           {results['f1_score']*100:.1f}%")
        print(f"  MCC:                {results['mcc']:.3g}")
        print(f"  False Alarm Rate:   {results['false_alarm_rate']*100:.1f}%")
        
        print(f"\nInterval Detection:")
        print(f"  Intervals detected: {results['intervals_detected']}/{results['total_intervals']}")
        
        if results['mean_latency_s'] is not None:
            print(f"\nDetection Latency:")
            print(f"  Mean:   {results['mean_latency_s']:.1f}s")
            print(f"  Median: {results['median_latency_s']:.1f}s")
            print(f"  Max:    {results['max_latency_s']:.1f}s")
        
        print("="*70)
    
    def save_results(self, results: DetectionResults, output_path: str):
        os.makedirs(output_path, exist_ok=True)

        for detector_name in results.get_detector_names():
            result = results.get_result(detector_name)
            output_file = f'{output_path}/{detector_name}_results.csv'
            pd.DataFrame(result).to_csv(output_file, index=False)
            print(f"  Saved {detector_name} results to {output_file}")

        if results.performance:
            perf_df = pd.DataFrame(results.performance.values())
            perf_df.to_csv(f'{output_path}/performance_metrics.csv', index=False)
            print(f"  Saved performance metrics to {output_path}/performance_metrics.csv")
    

    
    def print_summary(self, results: DetectionResults):
        print("\n" + "="*70)
        print("SPOOFING DETECTION SUMMARY")
        print("="*70)

        for name in results.get_detector_names():
            detector = self.detectors[name]
            
            summary = detector.get_summary()
            print(f"\n{name.upper()}:")

            if not isinstance(detector, D5_CombinedDetector):
                print(f"  Epochs: {summary.get('total_epochs', 'N/A')}, Alerts: {summary.get('alerts', 'N/A')}")
                continue
            
            print(f"  Total epochs: {summary.get('total_epochs', 'N/A')}")
            print(f"  Anomalous epochs: {summary['anomalous_epochs']} ({summary.get('detection_rate', 0)*100:.1f}%)")
                

        if results.performance:
            print(f"\n{'='*70}")
            print("PERFORMANCE METRICS")
            print(f"{'='*70}")
            for detector_name, perf in results.performance.items():
                print(f"\n{detector_name.upper()}:")
                print(f"  Detection Rate (Recall): {perf['recall']*100:.1f}%")
                print(f"  Precision: {perf['precision']*100:.1f}%")
                print(f"  F1 Score: {perf['f1_score']*100:.1f}%")
                print(f"  MCC: {perf['mcc']:.3g}")
                print(f"  False Alarm Rate: {perf['false_alarm_rate']*100:.1f}%")

        print("="*70)