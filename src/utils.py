# src/utils.py
import joblib
from sklearn.datasets import load_iris


def load_iris_data():
    """Carga el dataset Iris y devuelve X, y, nombres de características y clases."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    return X, y, feature_names, class_names


def save_model(model, filepath):
    """Guarda un modelo de Keras o un scaler de sklearn en un archivo .h5 o .pkl."""
    if hasattr(model, "save"):
        model.save(filepath)
    else:
        joblib.dump(model, filepath)


def load_model(filepath):
    """Carga un modelo de Keras o un scaler desde disco."""
    if filepath.endswith(".h5") or filepath.endswith(".keras"):
        from tensorflow.keras.models import load_model

        return load_model(filepath)
    return joblib.load(filepath)