import numpy as np
import random
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
    # Se utiliza random_state para asegurar reproducibilidad
    pca = PCA(n_components=n_componentes, random_state=42)
    
    # Aplicar PCA y transformar los datos
    X_reduced = pca.fit_transform(X)
    
    return X_reduced


def generar_caso_de_uso_reducir_dimensionalidad():
    """
    Generador de casos de uso CORREGIDO para la función reducir_dimensionalidad.
    
    NOTA: El generador original de Jhon no especificaba random_state,
    causando problemas de reproducibilidad. Este generador usa random_state
    para asegurar consistencia.
    """
    # Usar random_state para reproducibilidad
    seed = random.randint(0, 1000)
    np.random.seed(seed)
    
    n = random.randint(20, 50)
    n_features = random.randint(4, 6)
    n_components = random.randint(2, n_features - 1)
    
    # Generar datos
    X = np.random.rand(n, n_features)
    
    # Aplicar PCA con random_state para reproducibilidad
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    
    input_data = {
        "X": X.copy(),
        "n_componentes": n_components
    }
    
    output_data = X_reduced
    
    return input_data, output_data