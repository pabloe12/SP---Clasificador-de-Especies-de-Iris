# main.py
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical

from src.metrics import evaluate_model, plot_confusion_matrix
from src.predict import predict_species
from src.preprocessing import preprocess_data
from src.train import train_model
from src.utils import save_model
from src.visualization import plot_decision_boundary, plot_learning_curves

def main():
    # 1. Preprocesar
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, class_names) = preprocess_data()
    
    # Convertir etiquetas a one-hot para la red
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)  # no se usa en train, pero por si acaso
    
    # 2. Entrenar
    print("Entrenando modelo...")
    model, history = train_model(X_train, y_train_cat, X_val, y_val_cat)
    
    # 3. Evaluar en test
    accuracy, report, cm, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, class_names)
    print(f"Precisión en test: {accuracy:.4f}")
    print("\nReporte de clasificación:\n", report)
    
    # 4. Guardar modelo y scaler
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'
    images_dir = base_dir / 'images'
    models_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    save_model(model, models_dir / 'iris_mlp.h5')
    save_model(scaler, models_dir / 'scaler.pkl')
    
    # 5. Generar y guardar gráficos
    fig_curves = plot_learning_curves(history)
    fig_curves.savefig(images_dir / 'learning_curves.png')
    plt.close(fig_curves)
    
    # Matriz de confusión
    fig_cm = plot_confusion_matrix(cm, class_names)
    fig_cm.savefig(images_dir / 'confusion_matrix.png')
    plt.close(fig_cm)
    
    # Frontera de decisión (usando PCA sobre todo el conjunto)
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.hstack([y_train, y_val, y_test])
    fig_db = plot_decision_boundary(model, X_all, y_all, scaler, class_names, already_scaled=False)
    fig_db.savefig(images_dir / 'decision_boundary.png')
    plt.close(fig_db)
    
    print("¡Entrenamiento completado! Modelo y gráficos guardados.")
    
    # (Opcional) prueba de una predicción de ejemplo
    sample = X_test[0]
    pred_class, proba = predict_species(model, scaler, sample, class_names)
    print(f"Ejemplo de predicción: {pred_class} con probabilidades {proba}")

if __name__ == "__main__":
    main()