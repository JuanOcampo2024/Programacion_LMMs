import pandas as pd
import numpy as np

def seleccionar_top_correlacion(df, target_col, k):
    """
    Selecciona las k variables con mayor correlación (valor absoluto) 
    con la variable objetivo.
    
    Args:
        df (pd.DataFrame): Dataset completo
        target_col (str): Nombre de la columna objetivo
        k (int): Número de variables a seleccionar
    
    Returns:
        np.ndarray: Array con nombres de las k columnas con mayor correlación
    """
    # Calcular la matriz de correlación completa
    correlaciones = df.corr()
    
    # Obtener correlaciones con la columna objetivo
    target_correlaciones = correlaciones[target_col]
    
    # Eliminar la correlación del target consigo mismo
    target_correlaciones = target_correlaciones.drop(labels=[target_col])
    
    # Obtener valores absolutos para considerar correlaciones negativas
    abs_correlaciones = target_correlaciones.abs()
    
    # Ordenar en forma descendente y tomar los top k
    top_k = abs_correlaciones.sort_values(ascending=False).head(k)
    
    # Retornar los nombres como array numpy
    return top_k.index.to_numpy()


def generar_caso_de_uso_seleccionar_top_correlacion():
    """
    Generador de casos de uso CORREGIDO para la función seleccionar_top_correlacion.
    
    NOTA: El generador original de Paola devolvía (args, None) lo cual es un bug.
    Este generador corregido devuelve el output esperado correctamente.
    """
    rows = np.random.randint(40, 60)
    target = np.random.rand(rows)
    df = pd.DataFrame({
        'feat1': target + np.random.normal(0, 0.01, rows),
        'feat2': np.random.rand(rows),
        'target': target
    })
    
    # Argumentos de entrada
    args = {'df': df, 'target_col': 'target', 'k': 1}
    
    # CORRECCIÓN: Calcular el output esperado correctamente
    output = seleccionar_top_correlacion(df, 'target', 1)
    
    return args, output