# app/streamlit_app.py
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent.parent

# Configurar página
st.set_page_config(page_title="Clasificador Iris MLP", layout="wide")
st.title("🌿 Clasificador de Iris con Red Neuronal MLP")

# Cargar modelo y scaler (si no existen, avisar)
model_path = BASE_DIR / "models" / "iris_mlp.h5"
scaler_path = BASE_DIR / "models" / "scaler.pkl"

if not model_path.exists() or not scaler_path.exists():
    st.error("⚠️ No se encontraron el modelo o el scaler. Ejecuta primero `main.py` para entrenar.")
    st.stop()

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Cargar dataset Iris para entrenar KNN (para comparativa)
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Entrenar KNN (con escalado) para comparar
knn_scaler = StandardScaler()
X_scaled = knn_scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Sidebar: entrada de datos
st.sidebar.header("Ingresa las medidas de la flor")
sepal_length = st.sidebar.slider("Largo del sépalo (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Ancho del sépalo (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Largo del pétalo (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Ancho del pétalo (cm)", 0.1, 2.5, 1.2)

features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
features_scaled = scaler.transform(features)

# Predicción MLP
proba_mlp = model.predict(features_scaled, verbose=0)[0]
pred_mlp = np.argmax(proba_mlp)
pred_mlp_name = class_names[pred_mlp]
confidence_mlp = proba_mlp[pred_mlp]

# Predicción KNN
features_knn_scaled = knn_scaler.transform(features)
pred_knn = knn.predict(features_knn_scaled)[0]
pred_knn_name = class_names[pred_knn]
proba_knn = knn.predict_proba(features_knn_scaled)[0]
confidence_knn = proba_knn[pred_knn]

# Mostrar resultados en dos columnas
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 MLP (Red Neuronal)")
    st.metric("Especie predicha", pred_mlp_name, delta=f"Confianza: {confidence_mlp:.2%}")
    st.write("Probabilidades por clase:")
    df_proba = pd.DataFrame({
        'Clase': class_names,
        'Probabilidad': proba_mlp
    })
    st.bar_chart(df_proba.set_index('Clase'))

with col2:
    st.subheader("📏 KNN (Vecinos más cercanos)")
    st.metric("Especie predicha", pred_knn_name, delta=f"Confianza: {confidence_knn:.2%}")
    st.write("Probabilidades por clase:")
    df_proba_knn = pd.DataFrame({
        'Clase': class_names,
        'Probabilidad': proba_knn
    })
    st.bar_chart(df_proba_knn.set_index('Clase'))

# Mostrar comparativa
st.subheader("🔍 Comparativa MLP vs KNN")
if pred_mlp_name == pred_knn_name:
    st.success(f"✅ Ambos modelos coinciden en: **{pred_mlp_name}**")
else:
    st.warning(f"⚠️ Los modelos discrepan: MLP → {pred_mlp_name}, KNN → {pred_knn_name}")

# Mostrar imagen de la frontera de decisión (si existe)
st.subheader("📊 Frontera de Decisión (PCA)")
image_path = BASE_DIR / "images" / "decision_boundary.png"
if image_path.exists():
    st.image(image_path, caption='Frontera de decisión del MLP (proyectada con PCA)', use_container_width=True)
else:
    st.info("La imagen de la frontera de decisión no está disponible. Ejecuta `main.py` para generarla.")