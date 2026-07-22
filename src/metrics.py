# src/metrics.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, class_names):
    """Evalúa el modelo y devuelve precisión, reporte y matriz de confusión."""
    # Obtener predicciones
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Precisión
    accuracy = np.mean(y_pred == y_test)
    
    # Reporte de clasificación
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm, y_pred, y_pred_proba

def plot_confusion_matrix(cm, class_names, title='Matriz de Confusión'):
    """Genera una figura de la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(title)
    return fig