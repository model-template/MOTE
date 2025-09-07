import numpy as np
import os

def _load_numpy_file(path):
    """
    Loads a numpy file with error handling.

    Args:
        path (str): The path to the .npy file.

    Returns:
        The loaded numpy array, or None if the file is not found.
    """
    try:
        return np.load(path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Warning: File not found at {path}")
        return None

def load_embeddings(data_dir, architecture):
    """
    Loads the embeddings for a given architecture.

    Args:
        data_dir (str): The path to the data directory.
        architecture (str): The name of the architecture ('arcface', 'facenet', or 'magface').

    Returns:
        A dictionary containing the embeddings for the Adience, ColorFeret, and MORPH datasets.
    """
    embeddings = {}
    arch_path = os.path.join(data_dir, 'stripped_data', architecture)
    
    for dataset in ['adience', 'colorferet', 'morph']:
        emb_path = os.path.join(arch_path, f'matched_{architecture}_{dataset}_emb.npy')
        embeddings[dataset] = _load_numpy_file(emb_path)

    # Special case for LFW
    if architecture == 'facenet':
        lfw_path = os.path.join(data_dir, 'FaceNet', 'ds_lfw_emb.npy')
    elif architecture == 'magface':
        lfw_path = os.path.join(data_dir, 'MagFace', 'ds_lfw_emb.npy')
    else: # arcface
        lfw_path = os.path.join(data_dir, 'ArcFace', 'ArcFace_LFW', 'arcface_lfw_emb.npy')
    embeddings['lfw'] = _load_numpy_file(lfw_path)

    return embeddings

def load_labels(data_dir, architecture):
    """
    Loads the labels for a given architecture.

    Args:
        data_dir (str): The path to the data directory.
        architecture (str): The name of the architecture ('arcface', 'facenet', or 'magface').

    Returns:
        A dictionary containing the labels for the Adience, ColorFeret, and MORPH datasets.
    """
    labels = {}
    arch_path = os.path.join(data_dir, 'stripped_data', architecture)

    for dataset in ['adience', 'colorferet', 'morph']:
        labels[dataset] = {}
        for label_type in ['age', 'gender', 'filenames', 'identities', 'ethnics']:
            if not (dataset == 'adience' and label_type == 'ethnics'): # Adience doesn't have ethnics
                label_path = os.path.join(arch_path, f'matched_{architecture}_{dataset}_{label_type}.npy')
                labels[dataset][label_type] = _load_numpy_file(label_path)

    # Special case for LFW
    labels['lfw'] = {}
    if architecture == 'facenet':
        lfw_path = os.path.join(data_dir, 'FaceNet')
        labels['lfw']['gender'] = _load_numpy_file(os.path.join(lfw_path, 'ds_lfw_gender.npy'))
        labels['lfw']['filenames'] = _load_numpy_file(os.path.join(lfw_path, 'ds_lfw_filenames.npy'))
        labels['lfw']['identities'] = _load_numpy_file(os.path.join(lfw_path, 'ds_lfw_identities.npy'))
    elif architecture == 'magface':
        lfw_path = os.path.join(data_dir, 'MagFace')
        labels['lfw']['gender'] = _load_numpy_file(os.path.join(lfw_path, 'ds_lfw_gender.npy'))
        labels['lfw']['filenames'] = _load_numpy_file(os.path.join(lfw_path, 'ds_filenames_lfw_magface_r100.npy'))
        labels['lfw']['identities'] = _load_numpy_file(os.path.join(lfw_path, 'ds_lfw_identities.npy'))
    else: # arcface
        lfw_path = os.path.join(data_dir, 'ArcFace', 'ArcFace_LFW')
        # Assuming arcface lfw labels are in the same format
        labels['lfw']['gender'] = _load_numpy_file(os.path.join(lfw_path, 'arcface_lfw_gender.npy'))
        labels['lfw']['identities'] = _load_numpy_file(os.path.join(lfw_path, 'arcface_lfw_ids.npy'))


    return labels