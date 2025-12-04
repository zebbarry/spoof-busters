import sys
import argparse

from spoofing_detection import SpoofingDetectionPipeline
from spoofing_detection.detectors import (
    D1_RAIMDetector,
    D2_ObservablesDetector,
    D4_DriftDetector,
    D5_CombinedDetector,
    D6_DopplerConsistencyDetector,
    D1Config,
    D2Config,
    D4Config,
    D5Config,
    D6Config,
)
from spoofing_detection.plotting import plot_all_detectors


def main():
    parser = argparse.ArgumentParser(
        description='GNSS Spoofing Detection System',
    )
    
    parser.add_argument('csv_file', help='Path to GNSS data CSV file')
    parser.add_argument('output_dir', help='Output directory for results')
    
    parser.add_argument('--ground-truth', '-g', help='Path to ground truth CSV file')
    
    parser.add_argument('--no-d1', action='store_true', help='Disable D1 RAIM detector')
    parser.add_argument('--no-d2', action='store_true', help='Disable D2 Observables detector')
    parser.add_argument('--no-d4', action='store_true', help='Disable D4 Drift detector')
    parser.add_argument('--no-d5', action='store_true', help='Disable D5 Combined detector')
    parser.add_argument('--no-d6', action='store_true', help='Disable D6 Doppler detector')

    args = parser.parse_args()

    d1_config = D1Config()
    d2_config = D2Config()
    d4_config = D4Config()
    d5_config = D5Config()
    d6_config = D6Config()

    detectors = []

    if not args.no_d1:
        detectors.append(D1_RAIMDetector(d1_config))
    if not args.no_d2:
        detectors.append(D2_ObservablesDetector(d2_config))
    if not args.no_d4:
        detectors.append(D4_DriftDetector(d4_config))
    if not args.no_d5:
        detectors.append(D5_CombinedDetector(d5_config))
    if not args.no_d6:
        detectors.append(D6_DopplerConsistencyDetector(d6_config))

    if not detectors:
        print("Error: At least one detector must be enabled")
        sys.exit(1)

    print("="*70)
    print("GNSS SPOOFING DETECTION SYSTEM")
    print("="*70)

    pipeline = SpoofingDetectionPipeline(
        detectors=detectors,
        csv_path=args.csv_file,
        ground_truth_path=args.ground_truth
    )

    results = pipeline.run_detection()

    pipeline.save_results(results, args.output_dir)

    print("\nGenerating plots...")
    plot_all_detectors(
        results=results,
        detectors=pipeline.detectors,
        converter=pipeline.converter,
        ground_truth=pipeline.ground_truth,
        output_path=args.output_dir,
    )

    pipeline.print_summary(results)

    print(f"\nDetection complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
