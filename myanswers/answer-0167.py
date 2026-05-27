import numpy as np
from sklearn.model_selection import KFold

def generar_splits_kfold(X, y, k, random_state):
    """
    Genera k-fold splits para validación cruzada.
    
    Args:
        X (np.ndarray): Features matrix (n_samples, n_features)
        y (np.ndarray): Target vector (n_samples,)
        k (int): Número de folds
        random_state (int): Seed para reproducibilidad
    
    Returns:
        list: Lista de tuplas (train_indices, test_indices) como numpy arrays
    """
    # Crear objeto KFold con shuffle=True
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # Generar splits y guardarlos en una lista
    splits = []
    for train_idx, test_idx in kf.split(X):
        splits.append((train_idx, test_idx))
    
    return splits