from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.axes import Axes

from .detectors import (
    D1_RAIMDetector,
    D2_ObservablesDetector,
    D4_DriftDetector,
    D5_CombinedDetector,
    D6_DopplerConsistencyDetector,
)

from .csv_to_navdata import CSVToNavData
from .base import BaseDetector, DetectionResults


def add_ground_truth_shading(
    ax: Axes,
    timestamps: np.ndarray,
    ground_truth_active: np.ndarray,
):
    changes = np.diff(np.concatenate([[False], ground_truth_active, [False]]).astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        if start_idx < len(timestamps) and end_idx <= len(timestamps):
            t_start = timestamps[start_idx]
            t_end = timestamps[end_idx-1] if end_idx > 0 else timestamps[-1]

            ax.axvspan(t_start, t_end, alpha=0.2, color='red', label='True Spoofing' if i == 0 else None, zorder=0)


def format_datetime_axis(ax: Axes, xlabel: str = 'Time'):
    date_formatter = DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(date_formatter)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)


def add_threshold_line(
    ax: Axes,
    threshold: float,
    color: str = 'k',
    label: Optional[str] = None
):
    ax.axhline(
        y=threshold,
        color=color,
        linestyle='--',
        linewidth=1,
        label=label
    )


def setup_subplot(
    ax: Axes,
    title: str,
    ylabel: str,
    legend_loc: str = 'upper right',
):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    if legend_loc:
        ax.legend(loc=legend_loc)


def _plot_generic_detector(
    ax1: Axes,
    ax2: Axes,
    detector_name: str,
    result: dict,
    timestamps: np.ndarray,
    gt_active: Optional[np.ndarray]
):
    # Skip metadata fields
    skip_fields = {'gps_millis', 'spoofing_alert', 'timestamp'}
    plot_fields = [k for k in result.keys() if k not in skip_fields and isinstance(result[k], (list, np.ndarray))]

    if not plot_fields:
        raise LookupError(f"No plottable fields for {detector_name}")

    for field in plot_fields[:3]:
        values = np.array(result[field])
        if len(values) == len(timestamps):
            ax1.plot(timestamps, values, label=field, linewidth=1)

    if gt_active is not None:
        add_ground_truth_shading(ax1, timestamps, gt_active)
        add_ground_truth_shading(ax2, timestamps, gt_active)


    alerts = np.array(result['spoofing_alert']).astype(int)
    ax2.plot(timestamps, alerts, 'r-', linewidth=2, label='Spoofing Alert')
    ax2.fill_between(timestamps, 0, alerts, alpha=0.3, color='red')

    ax2.set_ylim((-0.1, 1.1))
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No Alert', 'Alert'])
    setup_subplot(ax2, 'Detection Status', '', )

    setup_subplot(ax1, f'{detector_name.upper()} Results', 'Value')


def is_spoofing_active(gps_millis: np.ndarray, ground_truth: pd.DataFrame) -> np.ndarray:
    active = np.zeros(len(gps_millis), dtype=bool)
    for _, row in ground_truth.iterrows():
        mask = (gps_millis >= row['start_gps_millis']) & (gps_millis <= row['stop_gps_millis'])
        active |= mask
    return active


def plot_all_detectors(
    results: DetectionResults,
    detectors: dict[str, BaseDetector],
    converter: CSVToNavData,
    ground_truth: Optional[pd.DataFrame],
    output_path: str,
    save: bool = True
):
    detector_names = results.get_detector_names()
    n_detectors = len(detector_names)

    if n_detectors == 0:
        raise ValueError("No detector results to plot")

    for name in detector_names:
        result = results.get_result(name)
        detector = detectors[name]

        gps_millis = np.array(result['gps_millis'])
        timestamps = np.array(converter.gps_millis_to_datetime(gps_millis))

        gt_active = None
        if ground_truth is not None:
            gt_active = is_spoofing_active(gps_millis, ground_truth)


        _fig, axes = plt.subplots(2, 1, figsize=(14, 3), gridspec_kw={'height_ratios': [3, 1]})

        if isinstance(detector, D1_RAIMDetector):
            _plot_raim_detector(axes, result, timestamps, gt_active, detector)
        elif isinstance(detector, D2_ObservablesDetector):
            _plot_observables_detector(axes[1], axes[2], result, timestamps, gt_active, detector)
        elif isinstance(detector, D4_DriftDetector):
            _plot_drift_detector(axes[1], axes[2], result, timestamps, gt_active, detector)
        elif isinstance(detector, D5_CombinedDetector):
            _plot_statistical_detector(axes, result, timestamps, gt_active, detector)
        elif isinstance(detector, D6_DopplerConsistencyDetector):
            _plot_doppler_detector(axes, result, timestamps, gt_active, detector)
        else:
            _plot_generic_detector(axes[1], axes[2], name, result, timestamps, gt_active)

        for ax in axes[:-1]:
            format_datetime_axis(ax, xlabel='')
        format_datetime_axis(axes[-1])

        plt.tight_layout()

        if save:
            filename = 'detector_results.png'
            base_filename = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
            plot_filename = f"{base_filename}_{name}.{ext}"
            plot_path = str(Path(output_path, plot_filename))
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
            plt.close()
        else:
            plt.show()


def _plot_raim_detector(axes: list[Axes], result: dict, timestamps: np.ndarray, gt_active: Optional[np.ndarray], detector: D1_RAIMDetector):
    for idx, (param, threshold, color) in enumerate((('rms', detector.config.rms_residual_threshold, 'b'), ('max', detector.config.max_residual_threshold, 'r'))):

        axes[idx].plot(timestamps, result[param + "_residual"], color+'-', linewidth=1, label=f'{param.upper()} Residual')
        axes[idx].set_yscale('log')

        add_threshold_line(axes[idx], threshold, label=f'Threshold ({threshold:.1f}m)', color=color)

        if gt_active is not None:
            add_ground_truth_shading(axes[idx], timestamps, gt_active)

        setup_subplot(axes[idx], '', 'Residual (m)', legend_loc='upper left')
    setup_subplot(axes[0], f'{detector.name.split("_")[0]}: Pseudorange RAIM', 'Residual (m)', legend_loc='upper left')

    ax2 = axes[-1]
    alerts = np.array(result['spoofing_alert']).astype(int)
    ax2.plot(timestamps, alerts, 'r-', linewidth=2, label='Spoofing Alert')
    ax2.fill_between(timestamps, 0, alerts, alpha=0.3, color='red')

    if gt_active is not None:
        add_ground_truth_shading(ax2, timestamps, gt_active)

    ax2.set_ylim((-0.1, 1.1))
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No Alert', 'Alert'])
    setup_subplot(ax2, 'Detection Status', '', legend_loc='')


def _plot_observables_detector(ax1: Axes, ax2: Axes, result: dict, timestamps: np.ndarray, gt_active: Optional[np.ndarray], detector: D2_ObservablesDetector):
    mean_cn0 = np.array(result['mean_cn0'])
    ax1.plot(timestamps, mean_cn0, 'g-', linewidth=2, label='Mean C/N0')

    std_cn0 = np.array(result['std_cn0'])
    ax1.fill_between(timestamps, mean_cn0 - std_cn0, mean_cn0 + std_cn0,
                    alpha=0.3, color='green', label='±1 Std Dev')

    add_threshold_line(ax1, detector.config.cn0_high_threshold, color='red',
                        linestyle='--', label=f'High Threshold ({detector.config.cn0_high_threshold:.1f} dB-Hz)')
    add_threshold_line(ax1, detector.config.cn0_low_threshold, color='orange',
                        linestyle='--', label=f'Low Threshold ({detector.config.cn0_low_threshold:.1f} dB-Hz)')

    if gt_active is not None:
        add_ground_truth_shading(ax1, timestamps, gt_active)

    setup_subplot(ax1, f'{detector.name.split("_")[0]}: C/N0 Observable Monitoring', 'C/N0 (dB-Hz)')

    alerts = np.array(result['spoofing_alert']).astype(int)
    ax2.plot(timestamps, alerts, 'r-', linewidth=2, label='Spoofing Alert')
    ax2.fill_between(timestamps, 0, alerts, alpha=0.3, color='red')

    if gt_active is not None:
        add_ground_truth_shading(ax2, timestamps, gt_active)

    ax2.set_ylim((-0.1, 1.1))
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No Alert', 'Alert'])
    setup_subplot(ax2, 'Detection Status', '', legend_loc='')


def _plot_drift_detector(ax1: Axes, ax2_bottom: Axes, result: dict, timestamps: np.ndarray, gt_active: Optional[np.ndarray], detector: D4_DriftDetector):
    ax1.plot(timestamps, result['position_velocity'], 'm-', linewidth=2, label='Position Velocity')

    add_threshold_line(ax1, detector.config.position_threshold, color='red', linestyle='--', label=f'Pos Threshold ({detector.config.position_threshold:.1f} m/s)')
    add_threshold_line(ax1, -detector.config.position_threshold, color='red', linestyle='--')

    ax1.set_ylabel('Position Velocity (m/s)', fontsize=11, color='m')
    ax1.tick_params(axis='y', labelcolor='m')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(timestamps, result['clock_drift_rate'], 'c-', linewidth=2,
            label='Clock Drift', alpha=0.7)
    ax1_twin.set_ylabel('Clock Drift Rate (m/s)', fontsize=11, color='c')
    ax1_twin.tick_params(axis='y', labelcolor='c')

    ax1_twin.axhline(y=detector.config.clock_threshold, color='orange', linestyle='--',
                linewidth=1, label=f'Clock Threshold ({detector.config.clock_threshold:.1f} m/s)')
    ax1_twin.axhline(y=-detector.config.clock_threshold, color='orange', linestyle='--',
                linewidth=1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    if gt_active is not None:
        add_ground_truth_shading(ax1, timestamps, gt_active)

    ax1.set_title(f'{detector.name.split("_")[0]}: Position and Clock Drift Monitoring',
                fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    alerts = np.array(result['spoofing_alert']).astype(int)
    ax2_bottom.plot(timestamps, alerts, 'r-', linewidth=2, label='Spoofing Alert')
    ax2_bottom.fill_between(timestamps, 0, alerts, alpha=0.3, color='red')

    if gt_active is not None:
        add_ground_truth_shading(ax2_bottom, timestamps, gt_active)

    ax2_bottom.set_ylim((-0.1, 1.1))
    ax2_bottom.set_yticks([0, 1])
    ax2_bottom.set_yticklabels(['No Alert', 'Alert'])
    setup_subplot(ax2_bottom, 'Detection Status', '', legend_loc='')


def _plot_statistical_detector(axes: list[Axes], result: dict, timestamps: np.ndarray, gt_active: Optional[np.ndarray], detector: D5_CombinedDetector):
    score_field = None
    smooth_field = None
    ax1 = axes[0]
    ax2 = axes[1]

    for key in result.keys():
        if 'score' in key.lower() and 'smooth' not in key.lower():
            score_field = key
        elif 'smooth' in key.lower():
            smooth_field = key

    if smooth_field:
        ax1.plot(timestamps, result[smooth_field], 'b-', linewidth=2,
               label='Score (smoothed)')

    if score_field and score_field != smooth_field:
        ax1.plot(timestamps, result[score_field], 'b-', linewidth=1,
               label='Score (raw)')

    threshold = result['threshold'] if np.isscalar(result['threshold']) else result['threshold'][0]
    threshold_label = f'Threshold ({threshold:.2f})'
    add_threshold_line(ax1, threshold, color='red', label=threshold_label)

    if gt_active is not None:
        add_ground_truth_shading(ax1, timestamps, gt_active)

    setup_subplot(ax1, f'{detector.name.split("_")[0]}: Statistical Anomaly Detection', 'Z-Score Sum', legend_loc='upper left')

    alerts = np.array(result['spoofing_alert']).astype(int)
    ax2.plot(timestamps, alerts, 'r-', linewidth=2, label='Spoofing Alert')
    ax2.fill_between(timestamps, 0, alerts, alpha=0.3, color='red')

    if gt_active is not None:
        add_ground_truth_shading(ax2, timestamps, gt_active)

    ax2.set_ylim((-0.1, 1.1))
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No Alert', 'Alert'])
    setup_subplot(ax2, 'Detection Status', '', legend_loc='')


def _plot_doppler_detector(axes: list[Axes], result: dict, timestamps: np.ndarray, gt_active: Optional[np.ndarray], detector: D6_DopplerConsistencyDetector):
    for idx, (param, threshold, color) in enumerate((('rms', detector.config.rms_error_threshold, 'b'), ('max', detector.config.max_error_threshold, 'r'))):

        axes[idx].plot(timestamps, result[param + "_consistency_error"], color+'-', linewidth=1, label=f'{param.upper()} Error')
        axes[idx].set_yscale('log')

        add_threshold_line(axes[idx], threshold, label=f'Threshold ({threshold:.1f}m)', color=color)

        if gt_active is not None:
            add_ground_truth_shading(axes[idx], timestamps, gt_active)

        setup_subplot(axes[idx], '', 'Velocity Error (m/s)', legend_loc='upper left')
    setup_subplot(axes[0], f'D2: Doppler-Pseudorange Consistency', 'Velocity Error (m/s)', legend_loc='upper left')

    ax2 = axes[-1]
    alerts = np.array(result['spoofing_alert']).astype(int)
    ax2.plot(timestamps, alerts, 'r-', linewidth=2, label='Spoofing Alert')
    ax2.fill_between(timestamps, 0, alerts, alpha=0.3, color='red')

    if gt_active is not None:
        add_ground_truth_shading(ax2, timestamps, gt_active)

    ax2.set_ylim((-0.1, 1.1))
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No Alert', 'Alert'])
    setup_subplot(ax2, 'Detection Status', '', )
