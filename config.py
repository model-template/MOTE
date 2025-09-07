import os

# --- Project Root ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Data Directories ---
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
PRETRAINED_MODELS_DIR = os.path.join(ROOT_DIR, 'pretrained_models')

# --- Embeddings and Labels Paths ---
# (This section can be expanded to hold all specific file paths,
# making the data_loader more robust to changes in file structure)

# --- Experiment Hyperparameters ---
TRAINING_PARAMS = {
    'num_epochs': 120,
    'patience': 5,
    'batch_size': 128,
}

OPTUNA_PARAMS = {
    'n_trials': 100,
}

# --- Evaluation Parameters ---
EVALUATION_PARAMS = {
    'far_threshold': 1e-3,
    'fdr_alpha': 0.5,
    'garbe_alpha': 0.5,
}

# --- Architectures and Datasets ---
ARCHITECTURES = ['arcface', 'magface']
DATASETS = ['adience', 'colorferet', 'lfw', 'morph']
