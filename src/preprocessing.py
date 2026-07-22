# src/preprocessing.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import load_iris_data

def preprocess_data(test_size=0.15, val_size=0.15, random_state=42):
    """
    Carga los datos, los divide en train/val/test y aplica escalado estándar.
    Retorna:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, class_names
    """
    X, y, feature_names, class_names = load_iris_data()
    
    # Dividir primero en train + temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state, stratify=y
    )
    # Dividir temp en val y test
    val_proportion = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_proportion), random_state=random_state, stratify=y_temp
    )
    
    # Escalado: ajustar scaler solo con entrenamiento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            scaler, class_names)