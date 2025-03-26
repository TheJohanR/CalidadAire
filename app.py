import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from PIL import Image

# Cargar modelos
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("labelencoder.joblib")
model = joblib.load("modelRF.joblib")

# Definir valores por defecto
feature_means = {
    "Temperature": 29.977300,
    "Humidity": 70.036240,
    "PM10": 28.215900,
    "NO2": 26.364280,
    "SO2": 9.911630,
    "CO": 1.498345,
    "Proximity_to_Industrial_Areas": 8.419160,
    "Population_Density": 497.406700,
}

# Definir colores y emojis por categoría
category_styles = {
    "Moderate": ("#FFD700", "😐"),
    "Good": ("#32CD32", "😊"),
    "Hazardous": ("#8B0000", "☠️"),
    "Poor": ("#FF4500", "😷"),
}

# UI de la aplicación
st.title("Modelo Predictivo de la Calidad del Aire")
st.subheader("Autores: Johan Rodriguez y Stefania Reyes")

# Imagen
image = Image.open("contaminacion.jpg")
st.image(image, use_container_width=True)

# Introducción
txt = """
Esta aplicación permite predecir la calidad del aire con base en distintos parámetros ambientales. 

**¿Cómo usarla?**
- Ajusta los valores de cada variable usando los deslizadores.
- Presiona el botón de predicción para obtener el resultado.
"""
st.markdown(txt)

# Entrada de variables
st.sidebar.header("Seleccione los valores de las variables")
input_data = []
for feature, mean_value in feature_means.items():
    val = st.sidebar.slider(feature, min_value=float(0), max_value=float(mean_value * 2), value=float(mean_value))
    input_data.append(val)

# Botón de predicción
if st.button("Predecir Calidad del Aire"):
    # Transformar datos
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # Predicción
    prediction = model.predict(input_scaled)
    category = label_encoder.inverse_transform(prediction)[0]
    
    # Mostrar resultado con estilo
    color, emoji = category_styles.get(category, ("#FFFFFF", "❓"))
    st.markdown(f'<div style="background-color:{color}; padding:10px; border-radius:10px; text-align:center;">' \
                f'<h2>{category} {emoji}</h2></div>', unsafe_allow_html=True)

# Línea final
st.markdown("---")
st.markdown("**Ingeniería Industrial**")
st.markdown("**Unab 2025®**")
