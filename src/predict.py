# src/predict.py
import numpy as np

def predict_species(model, scaler, features, class_names):
    """
    features: lista o array de 4 valores (largo sépalo, ancho sépalo, largo pétalo, ancho pétalo)
    Retorna: (clase_predicha, probabilidades)
    """
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    proba = model.predict(features_scaled, verbose=0)
    pred_idx = np.argmax(proba, axis=1)[0]
    return class_names[pred_idx], proba[0]