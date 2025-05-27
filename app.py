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

# Definir colores, emojis y recomendaciones por categoría
category_styles = {
    "Moderate": ("#FFD700", "😐"),
    "Good": ("#32CD32", "😊"),
    "Hazardous": ("#8B0000", "☠️"),
    "Poor": ("#FF4500", "😷"),
}

category_recommendations = {
    "Moderate": "Evite actividades físicas intensas al aire libre si es sensible. Mantenga ventanas cerradas en las horas pico.",
    "Good": "El aire es saludable. Disfrute actividades al aire libre sin restricciones.",
    "Hazardous": "Permanezca en interiores, cierre puertas y ventanas. Evite todo tipo de exposición al aire exterior.",
    "Poor": "Reduzca el tiempo al aire libre, especialmente niños, adultos mayores y personas con afecciones respiratorias.",
}

category_improvement = {
    "Moderate": "Fomentar el uso de transporte público o bicicleta. Evitar quemas al aire libre. Promover zonas verdes en la ciudad.",
    "Poor": "Reducir emisiones vehiculares, controlar actividades industriales, implementar días sin carro y mejorar la eficiencia energética.",
    "Hazardous": "Cerrar temporalmente fuentes industriales contaminantes, implementar planes de emergencia ambiental y controlar estrictamente el tráfico vehicular.",
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
- Escriba los valores de cada variable usando las cajas de texto.
- Presione el botón de predicción para obtener el resultado.
"""
st.markdown(txt)

# Entrada de variables
st.sidebar.header("Ingrese los valores de las variables")
input_data = []
for feature, mean_value in feature_means.items():
    val = st.sidebar.text_input(f"{feature} (valor numérico con punto decimal)", value=str(mean_value))
    try:
        val_float = float(val)
    except ValueError:
        st.sidebar.error(f"El valor ingresado para {feature} no es válido. Debe ser un número decimal.")
        val_float = mean_value
    input_data.append(val_float)

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
    recommendation = category_recommendations.get(category, "No hay recomendaciones disponibles.")
    improvement = category_improvement.get(category, None)

    st.markdown(f'<div style="background-color:{color}; padding:10px; border-radius:10px; text-align:center;">' \
                f'<h2>{category} {emoji}</h2></div>', unsafe_allow_html=True)

    st.markdown("### Recomendaciones:")
    st.info(recommendation)

    if improvement:
        st.markdown("### ¿Cómo mejorar la calidad del aire?")
        st.warning(improvement)

# Línea final
st.markdown("---")
st.markdown("**Ingeniería Industrial**")
st.markdown("**Unab 2025®**")
