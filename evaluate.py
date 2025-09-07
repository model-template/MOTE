import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data_loader import load_data_training
from .mote import FaceTemplateClassifier1


def test_model(model, test_loader, device):
    model.eval()
    true_labels, probabilities = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            probabilities.extend(probs.flatten())
            true_labels.extend(labels.numpy())
    return np.array(true_labels), np.array(probabilities)

def calculate_fairness_metrics(fpr, tpr, thresholds, target_fmr=1e-3):
    closest_index = np.argmin(np.abs(fpr - target_fmr))
    fnmr = 1 - tpr[closest_index]
    return fnmr

def run_evaluation(architecture, dataset):
    print(f"Starting evaluation for architecture: {architecture}, dataset: {dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_embeddings, test_identities, _, _, _ = load_data_training(architecture, dataset)
    grouped_embeddings = {identity: test_embeddings[test_identities == identity] for identity in np.unique(test_identities)}

    results = {'mote': {}, 'baseline': {}}

    # Evaluate MOTE models
    model_dir = os.path.join(config.PRETRAINED_MODELS_DIR, f"mote_{architecture}_{dataset}")
    all_true_labels_mote, all_probs_mote = [], []

    for identity, embeddings in grouped_embeddings.items():
        model_path = os.path.join(model_dir, f"model_1_identity_{identity}.pth")
        if not os.path.exists(model_path):
            continue

        model = FaceTemplateClassifier1().to(device)
        model.load_state_dict(torch.load(model_path))

        labels = np.array([1 if i == identity else 0 for i in test_identities])
        test_dataset = TensorDataset(torch.from_numpy(test_embeddings).float(), torch.from_numpy(labels).float())
        test_loader = DataLoader(test_dataset, batch_size=config.TRAINING_PARAMS['batch_size'])

        true_labels, probs = test_model(model, test_loader, device)
        all_true_labels_mote.extend(true_labels)
        all_probs_mote.extend(probs)

    fpr, tpr, thresholds = roc_curve(all_true_labels_mote, all_probs_mote)
    results['mote']['roc_auc'] = auc(fpr, tpr)
    results['mote']['fnmr_at_fmr_1e-3'] = calculate_fairness_metrics(fpr, tpr)
    results['mote']['fpr'] = fpr.tolist()
    results['mote']['tpr'] = tpr.tolist()

    # Evaluate Baseline (cosine similarity)
    all_true_labels_baseline, all_scores_baseline = [], []
    for identity, embeddings in grouped_embeddings.items():
        reference_embedding = embeddings[0].reshape(1, -1)
        scores = cosine_similarity(reference_embedding, test_embeddings).flatten()
        labels = np.array([1 if i == identity else 0 for i in test_identities])
        all_true_labels_baseline.extend(labels)
        all_scores_baseline.extend(scores)

    fpr_b, tpr_b, _ = roc_curve(all_true_labels_baseline, all_scores_baseline)
    results['baseline']['roc_auc'] = auc(fpr_b, tpr_b)
    results['baseline']['fnmr_at_fmr_1e-3'] = calculate_fairness_metrics(fpr_b, tpr_b)
    results['baseline']['fpr'] = fpr_b.tolist()
    results['baseline']['tpr'] = tpr_b.tolist()

    # Save results
    results_path = os.path.join(config.RESULTS_DIR, f"evaluation_{architecture}_{dataset}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {results_path}")
