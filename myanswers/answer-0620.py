import numpy as np
from sklearn.decomposition import PCA

def reducir_dimensionalidad(X, n_componentes):
    """
    Reduce la dimensionalidad de X usando PCA.
    
    Args:
        X (np.ndarray): Matriz de datos (n_samples, n_features)
        n_componentes (int): Número de componentes principales deseados
    
    Returns:
        np.ndarray: Matriz transformada (n_samples, n_componentes)
    """
    # Crear objeto PCA con número de componentes especificado
    pca = PCA(n_components=n_componentes)
    
    # Aplicar PCA y transformar los datos
    X_reduced = pca.fit_transform(X)
    
    return X_reduced