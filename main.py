#!/usr/bin/env python3
"""
TLCC-Based Anomaly Detection - Main Entry Point
이 파일은 프로젝트의 메인 진입점입니다.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from experiments.train_original import main as train_main
from experiments.comprehensive_experiment_clean import main as experiment_main
from experiments.analyze_comprehensive_results import main as analyze_main


def main():
    """Main entry point for the TLCC anomaly detection system."""
    parser = argparse.ArgumentParser(
        description="TLCC-Based Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a single model
  python main.py train --dataset WADI --epochs 5

  # Run comprehensive experiments
  python main.py experiment --config experiments/config.json

  # Analyze results
  python main.py analyze --results_dir output/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a single model')
    train_parser.add_argument('--dataset', required=True, 
                             choices=['WADI', 'SMAP', 'MSL', 'SMD'],
                             help='Dataset to use')
    train_parser.add_argument('--epochs', type=int, default=5,
                             help='Number of epochs')
    train_parser.add_argument('--tlcc_threshold', type=float, default=0.5,
                             help='TLCC threshold')
    train_parser.add_argument('--batch_size', type=int, default=128,
                             help='Batch size')
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run comprehensive experiments')
    exp_parser.add_argument('--config', help='Configuration file')
    exp_parser.add_argument('--datasets', nargs='+', 
                           choices=['WADI', 'SMAP', 'MSL', 'SMD'],
                           default=['WADI', 'SMAP', 'MSL', 'SMD'],
                           help='Datasets to use')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results')
    analyze_parser.add_argument('--results_dir', required=True,
                               help='Directory containing results')
    analyze_parser.add_argument('--output_dir', default='analysis_output',
                               help='Output directory for analysis')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Set up arguments for training
        train_args = type('Args', (), {
            'dataset': args.dataset,
            'epochs': args.epochs,
            'tlcc_threshold': args.tlcc_threshold,
            'batch_size': args.batch_size,
            'lr': 0.001,
            'gamma': 0.95,
            'comment': f'{args.dataset}_epochs{args.epochs}_tlcc{args.tlcc_threshold}'
        })()
        train_main(train_args)
        
    elif args.command == 'experiment':
        experiment_main()
        
    elif args.command == 'analyze':
        if not os.path.exists(args.results_dir):
            print(f"Error: Results directory '{args.results_dir}' does not exist")
            sys.exit(1)
        analyze_main(args.results_dir, args.output_dir)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
