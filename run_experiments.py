import argparse
import sys
import os

# Add src to path to allow for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import train, evaluate, visualize, config

def main():
    parser = argparse.ArgumentParser(description="Run experiments for the MOTE project.")
    parser.add_argument('--train', action='store_true', help='Run the training pipeline.')
    parser.add_argument('--evaluate', action='store_true', help='Run the evaluation pipeline.')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations.')
    parser.add_argument('--all', action='store_true', help='Run all pipelines: train, evaluate, and visualize.')
    
    parser.add_argument('--architecture', type=str, choices=config.ARCHITECTURES, help='Face recognition architecture to use.')
    parser.add_argument('--dataset', type=str, choices=config.DATASETS, help='Dataset to use for training or evaluation.')

    args = parser.parse_args()

    if args.all:
        for arch in config.ARCHITECTURES:
            for dataset in config.DATASETS:
                train.run_training(arch, dataset)
                evaluate.run_evaluation(arch, dataset)
        visualize.run_visualization()

    else:
        if args.train:
            if not args.architecture or not args.dataset:
                print("Error: --architecture and --dataset are required for training.")
                return
            train.run_training(args.architecture, args.dataset)

        if args.evaluate:
            if not args.architecture or not args.dataset:
                print("Error: --architecture and --dataset are required for evaluation.")
                return
            evaluate.run_evaluation(args.architecture, args.dataset)

        if args.visualize:
            visualize.run_visualization()

    if not (args.train or args.evaluate or args.visualize or args.all):
        print("No action specified. Use --train, --evaluate, --visualize, or --all. Use -h for help.")

if __name__ == '__main__':
    main()
