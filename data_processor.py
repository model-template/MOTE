import numpy as np

def create_embeddings_dict(identities, embeddings):
    """
    Groups embeddings by identity.

    Args:
        identities (np.ndarray): An array of identity labels.
        embeddings (np.ndarray): An array of face embeddings.

    Returns:
        dict: A dictionary where keys are unique identities and values are lists of embeddings for each identity.
    """
    embeddings_dict = {}
    unique_identities = np.unique(identities)

    for identity in unique_identities:
        indices = np.where(identities == identity)[0]
        embeddings_dict[identity] = embeddings[indices]

    return embeddings_dict


def find_common_filenames_indices(file_list1, file_list2):
    """
    Finds the indices of common filenames between two lists of file paths.

    Args:
        file_list1 (list): The first list of file paths.
        file_list2 (list): The second list of file paths.

    Returns:
        list: A list of indices of common filenames in the first list.
    """
    file1_stripped = [name.split('\\')[-1].rsplit('.', 1)[0] for name in file_list1]
    file2_stripped = [name.split('\\')[-1].rsplit('.', 1)[0] for name in file_list2]

    common_filenames = set(file1_stripped).intersection(file2_stripped)

    common_filenames_indices = [i for i, filename in enumerate(file1_stripped) if filename in common_filenames]

    return common_filenames_indices


def extract_labels_at_indices(labels, indices):
    """
    Extracts labels at a given list of indices.

    Args:
        labels (np.ndarray): An array of labels.
        indices (list): A list of indices to extract.

    Returns:
        list: A list of labels at the specified indices.
    """
    return [labels[i] for i in indices]

def get_embeddings_for_identity(embeddings, identities, identity):
    """
    Retrieves the embeddings for a specific identity.

    Args:
        embeddings (np.ndarray): An array of face embeddings.
        identities (np.ndarray): An array of identity labels.
        identity: The identity to retrieve embeddings for.

    Returns:
        np.ndarray: An array of embeddings for the specified identity.
        
    Raises:
        ValueError: If the identity is not found in the identities array.
    """
    identity_indices = np.where(identities == identity)[0]
    if len(identity_indices) == 0:
        raise ValueError(f"Identity '{identity}' not found.")

    return embeddings[identity_indices]

def generate_identity_labels(single_sample_label, n_samples):
    """
    Generates an array of identity labels for a set of samples.

    Args:
        single_sample_label: The label of the single sample.
        n_samples (int): The number of new samples generated.

    Returns:
        np.ndarray: An array of labels for all samples.
    """
    new_sample_labels = np.full(n_samples, single_sample_label)
    return np.insert(new_sample_labels, 0, single_sample_label)

def split_embeddings_by_gender(embeddings, gender_labels, identities):
    """
    Splits embeddings into male and female groups based on gender labels.

    Args:
        embeddings (np.ndarray): An array of face embeddings.
        gender_labels (np.ndarray): An array of gender labels.
        identities (np.ndarray): An array of identity labels.

    Returns:
        tuple: A tuple containing:
            - male_embeddings (np.ndarray): Embeddings for male subjects.
            - male_identities (np.ndarray): Identities for male subjects.
            - female_embeddings (np.ndarray): Embeddings for female subjects.
            - female_identities (np.ndarray): Identities for female subjects.
    """
    normalized_labels = np.char.lower(np.asarray(gender_labels, dtype=str))

    male_mask = np.isin(normalized_labels, ['m', 'male'])
    female_mask = np.isin(normalized_labels, ['f', 'female'])

    male_embeddings = embeddings[male_mask]
    male_identities = identities[male_mask]
    female_embeddings = embeddings[female_mask]
    female_identities = identities[female_mask]

    return male_embeddings, male_identities, female_embeddings, female_identities