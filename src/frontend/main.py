import streamlit as st
import requests
import os

# --- Configuración de la conexión local ---
# Apunta directamente a tu API corriendo en tu máquina
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/predict")

st.set_page_config(page_title="Predicción de Sueño", page_icon="☕")

st.title("☕ Predicción de Calidad de Sueño")
st.write(f"Conectado a: `{API_URL}`")

# --- Formulario de Datos ---
with st.form("my_form"):
    st.subheader("Ingresa los datos del paciente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Género", ["Male", "Female", "Other"])
        country = st.text_input("País", value="USA")
        occupation = st.text_input("Ocupación", value="Other")
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0)
        heart_rate = st.number_input("Ritmo Cardíaco", min_value=40, max_value=200, value=70)
        stress_level = st.selectbox("Nivel de Estrés", ["Low", "Medium", "High"])

    with col2:
        coffee_intake = st.number_input("Tazas de Café", min_value=0.0, value=2.0, step=0.5)
        # Calculamos la cafeína estimada (aprox 95mg por taza)
        caffeine_mg = st.number_input("Cafeína (mg)", min_value=0.0, value=coffee_intake * 95.0)
        sleep_hours = st.number_input("Horas de Sueño", min_value=0.0, value=7.0)
        phys_activity = st.number_input("Horas Actividad Física", min_value=0.0, value=5.0)
        health_issues = st.selectbox("Problemas de Salud", ["None", "Diabetes", "Hypertension", "Insomnia", "Asthma"])
        
        # Inputs binarios (0 o 1) como requiere tu modelo
        smoking = st.selectbox("¿Fuma?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        alcohol = st.selectbox("¿Bebe Alcohol?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

    submitted = st.form_submit_button("🔮 Predecir")

if submitted:
    # 1. Crear el diccionario de datos (Payload)
    # Incluimos el ID dummy porque tu modelo lo requiere
    payload = {
        "ID": 0,
        "Age": age,
        "Gender": gender,
        "Country": country,
        "Coffee_Intake": coffee_intake,
        "Caffeine_mg": caffeine_mg,
        "Sleep_Hours": sleep_hours,
        "BMI": bmi,
        "Heart_Rate": heart_rate,
        "Stress_Level": stress_level,
        "Physical_Activity_Hours": phys_activity,
        "Health_Issues": health_issues,
        "Occupation": occupation,
        "Smoking": smoking,
        "Alcohol_Consumption": alcohol
    }

    # 2. Enviar a la API
    try:
        with st.spinner("Consultando a la API..."):
            response = requests.post(API_URL, json=payload)
        
        # 3. Mostrar Resultados
        if response.status_code == 200:
            resultado = response.json()
            prediccion = resultado.get("prediction", "Error")
            
            st.success("¡Predicción Exitosa!")
            st.metric(label="Calidad de Sueño Predicha", value=prediccion)
            
            # Mostrar el JSON que enviamos y recibimos (para debug)
            with st.expander("Ver detalles técnicos"):
                st.json(payload)
                st.write("Respuesta:", resultado)
        else:
            st.error(f"Error en la API: {response.status_code}")
            st.error(response.text)
            
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar con la API.")
        st.info("Asegúrate de que 'uvicorn api:app --reload' esté corriendo en otra terminal.")