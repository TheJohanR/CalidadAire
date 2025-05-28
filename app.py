import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Cargar modelo y scaler
model = joblib.load("modelGB.joblib")
scaler = joblib.load("scaler.joblib")

# Título y subtítulo
st.title("🧬 Modelo Predictivo Diabetes")
st.subheader("Autores: Laura Sofía Velandia y María Paula Corredor")

# Imagen
image = Image.open("diabetes.jpg")
st.image(image, use_container_width=True)

# Introducción
st.markdown("""
Esta aplicación permite predecir si una persona podría tener riesgo de diabetes, utilizando un modelo entrenado con Gradient Boosting.

Para usar la herramienta, introduce los valores correspondientes en cada campo numérico y presiona el botón **Predecir** para obtener el resultado.
""")

# Valores por defecto (media de la tabla)
default_values = {
    "Pregnancies": 3.84,
    "Glucose": 121.14,
    "BloodPressure": 70.68,
    "SkinThickness": 20.51,
    "Insulin": 73.65,
    "BMI": 32.13,
    "DiabetesPedigreeFunction": 0.46,
    "Age": 33.20
}

# Entradas del usuario
st.markdown("### Ingrese los valores del paciente:")
pregnancies = st.number_input("Número de embarazos", value=default_values["Pregnancies"])
glucose = st.number_input("Nivel de glucosa", value=default_values["Glucose"])
bp = st.number_input("Presión arterial", value=default_values["BloodPressure"])
skin = st.number_input("Espesor de la piel", value=default_values["SkinThickness"])
insulin = st.number_input("Nivel de insulina", value=default_values["Insulin"])
bmi = st.number_input("Índice de masa corporal (BMI)", value=default_values["BMI"])
dpf = st.number_input("Función de herencia de diabetes", value=default_values["DiabetesPedigreeFunction"])
age = st.number_input("Edad", value=default_values["Age"])

# Botón de predicción
if st.button("Predecir"):
    user_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.markdown("""
        <div style='background-color:#ff4c4c; padding: 20px; border-radius: 10px; color: white;'>
            <h3>🔴 Riesgo de Diabetes</h3>
            <p>Se recomienda consultar a un profesional de salud para un diagnóstico más preciso.</p>
            <p><strong>Recomendaciones:</strong></p>
            <ul>
                <li>Mantener una dieta equilibrada baja en azúcares y grasas saturadas.</li>
                <li>Realizar actividad física regular (al menos 30 minutos al día).</li>
                <li>Controlar el peso corporal y evitar el sobrepeso.</li>
                <li>Evitar el consumo de tabaco y alcohol en exceso.</li>
                <li>Monitorear periódicamente los niveles de glucosa en sangre.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color:#4CAF50; padding: 20px; border-radius: 10px; color: white;'>
            <h3>🟢 Sin Riesgo de Diabetes</h3>
            <p>Continúa manteniendo hábitos saludables para conservar tu bienestar.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Ingeniería Industrial  \nUNAB 2025 ®")
