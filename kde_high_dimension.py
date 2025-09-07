import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def compute_centroids(embeddings, labels):
    """
    Computes the centroid for each identity in the embedding space.

    Args:
        embeddings (np.ndarray): The face embeddings.
        labels (np.ndarray): The identity labels for the embeddings.

    Returns:
        dict: A dictionary mapping each identity to its centroid.
    """
    unique_labels = np.unique(labels)
    centroids = {label: embeddings[labels == label].mean(axis=0) for label in unique_labels}
    return centroids

def normalize_embeddings(embeddings, labels, centroids):
    """
    Normalizes embeddings by subtracting the corresponding identity centroid.

    Args:
        embeddings (np.ndarray): The face embeddings.
        labels (np.ndarray): The identity labels for the embeddings.
        centroids (dict): A dictionary mapping each identity to its centroid.

    Returns:
        np.ndarray: The normalized embeddings.
    """
    normalized_embeddings = np.empty_like(embeddings)
    for label, centroid in centroids.items():
        mask = (labels == label)
        normalized_embeddings[mask] = embeddings[mask] - centroid
    return normalized_embeddings

def fit_kde(embeddings, kernel='gaussian', cv=5):
    """
    Fits a Kernel Density Estimation (KDE) model to the embeddings.

    Args:
        embeddings (np.ndarray): The embeddings to fit the KDE model to.
        kernel (str): The kernel to use for the KDE.
        cv (int): The number of cross-validation folds for bandwidth selection.

    Returns:
        KernelDensity: The fitted KDE model.
    """
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)
    
    if len(embeddings) < cv:
        cv = len(embeddings)

    bandwidths = 10 ** np.linspace(-2.5, 0.4, 40)
    grid = GridSearchCV(KernelDensity(kernel=kernel), {'bandwidth': bandwidths}, cv=cv, n_jobs=-1)
    grid.fit(embeddings)

    print(f"Best bandwidth for KDE: {grid.best_estimator_.bandwidth:.4f}")

    return grid.best_estimator_

def sample_from_kde(kde_model, n_samples=100, reference_embedding=None):
    """
    Samples new embeddings from a fitted KDE model.

    Args:
        kde_model (KernelDensity): The fitted KDE model.
        n_samples (int): The number of samples to generate.
        reference_embedding (np.ndarray, optional): If provided, the new samples are shifted
            to be centered around this embedding. Defaults to None.

    Returns:
        np.ndarray: The generated samples.
    """
    new_samples = kde_model.sample(n_samples)
    if reference_embedding is not None:
        new_samples += reference_embedding
    return new_samples
