import os
import json
import matplotlib.pyplot as plt
import numpy as np

from . import config

def plot_roc_curves(results, architecture, dataset):
    plt.figure(figsize=(10, 8))

    # MOTE curve
    mote_fpr = np.array(results['mote']['fpr'])
    mote_tpr = np.array(results['mote']['tpr'])
    mote_auc = results['mote']['roc_auc']
    plt.plot(mote_fpr, mote_tpr, color='blue', lw=2, label=f'MOTE (AUC = {mote_auc:.3f})')

    # Baseline curve
    baseline_fpr = np.array(results['baseline']['fpr'])
    baseline_tpr = np.array(results['baseline']['tpr'])
    baseline_auc = results['baseline']['roc_auc']
    plt.plot(baseline_fpr, baseline_tpr, color='red', lw=2, label=f'Baseline (AUC = {baseline_auc:.3f})')

    plt.xscale('log')
    plt.xlabel('False Match Rate (FMR)', fontsize=14)
    plt.ylabel('1 - False Non-Match Rate (1-FNMR)', fontsize=14)
    plt.title(f'ROC Curve: {architecture.upper()} on {dataset.upper()}', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, which="both", ls="--")
    
    # Save the figure
    figure_path = os.path.join(config.RESULTS_DIR, f"roc_curve_{architecture}_{dataset}.png")
    plt.savefig(figure_path)
    plt.close()
    print(f"ROC curve saved to {figure_path}")

def run_visualization():
    print("Generating visualizations from evaluation results...")
    for arch in config.ARCHITECTURES:
        for dataset in config.DATASETS:
            results_path = os.path.join(config.RESULTS_DIR, f"evaluation_{arch}_{dataset}.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                plot_roc_curves(results, arch, dataset)
            else:
                print(f"Warning: Results file not found for {arch} on {dataset}. Skipping visualization.")
