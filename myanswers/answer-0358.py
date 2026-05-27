import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def escalar_minmax(df, target_col):
    """
    Separa features del target, imputa valores faltantes con la media,
    y escala las features al rango [0, 1] usando MinMaxScaler.
    
    Args:
        df (pd.DataFrame): Dataset con posibles valores NaN
        target_col (str): Nombre de la columna objetivo
    
    Returns:
        tuple: (X_procesada, y) donde:
            - X_procesada (np.ndarray): Features escaladas [0,1]
            - y (np.ndarray): Vector target
    """
    # Separar features del target
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    
    # Imputar valores NaN con la media de cada columna
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Escalar al rango [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, y