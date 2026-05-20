import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración inicial de la página
st.set_page_config(page_title="Predicción de Enfermedades Cardíacas", page_icon="🫀", layout="centered")

# 2. Branding y Encabezado (Requisito 10: Datos ISIL)
st.title("🫀 Predicción de Enfermedades Cardíacas")
st.markdown("""
Esta aplicación evalúa el riesgo de enfermedad cardíaca basándose en variables clínicas.
Utiliza modelos de Machine Learning entrenados previamente para generar un diagnóstico.
""")
st.markdown("---")
st.markdown("**Desarrollado por:** [Ricardo Gian Jesús Navarro Aponte]")
st.markdown("**Código ISIL:** [70884066]")
st.markdown("[🔗 Ver Cuaderno de Google Colab (Modo Lector)](https://colab.research.google.com/drive/1-9YkW9Bhm_ML0utwG-aQyCGkfuabXxVu?usp=sharing)")
st.markdown("---")

# 3. Función con caché para cargar los modelos y el escalador
@st.cache_resource
def cargar_recursos():
    # Ruta absoluta basada en la ubicación de este archivo (app.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelos_dir = os.path.join(base_dir, 'modelos')
    
    # Verificar que la carpeta exista
    if not os.path.exists(modelos_dir):
        st.error(f"❌ Error: No se encontró la carpeta 'modelos/' en: {modelos_dir}")
        st.write("Contenido del directorio actual:", os.listdir(base_dir))
        return None, None, None
    
    try:
        modelo_lr = joblib.load(os.path.join(modelos_dir, 'modelo_regresion_logistica.pkl'))
        modelo_rf = joblib.load(os.path.join(modelos_dir, 'modelo_random_forest.pkl'))
        scaler = joblib.load(os.path.join(modelos_dir, 'escalador_heart_disease.pkl'))
        return modelo_lr, modelo_rf, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Error al cargar archivos .pkl: {e}")
        st.write("Archivos disponibles en modelos/:", os.listdir(modelos_dir))
        return None, None, None

modelo_lr, modelo_rf, scaler = cargar_recursos()

# Si los recursos cargan correctamente, mostramos la interfaz
if modelo_lr is not None and modelo_rf is not None and scaler is not None:
    # 4. Selector de modelo en la pantalla principal
    st.subheader("🤖 Configuración del Modelo")
    modelo_seleccionado = st.selectbox("Seleccione el modelo predictivo:", ["Regresión Logística", "Random Forest"])

    # 5. Entrada de datos del usuario (Sidebar)
    st.sidebar.header("📝 Ingrese los datos clínicos")

    def obtener_datos_usuario():
        age = st.sidebar.slider("Edad", 20, 100, 50)
        sex = st.sidebar.selectbox("Sexo", options=[0, 1], format_func=lambda x: "0 - Femenino" if x == 0 else "1 - Masculino")
        cp = st.sidebar.selectbox("Tipo de dolor de pecho (cp)", options=[0, 1, 2, 3])
        trestbps = st.sidebar.slider("Presión arterial en reposo (trestbps)", 90, 200, 120)
        chol = st.sidebar.slider("Colesterol sérico (chol)", 100, 600, 200)
        fbs = st.sidebar.selectbox("Azúcar en sangre > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "0 - Falso" if x == 0 else "1 - Verdadero")
        restecg = st.sidebar.selectbox("Resultados ECG en reposo (restecg)", options=[0, 1, 2])
        thalach = st.sidebar.slider("Frecuencia cardíaca máxima (thalach)", 70, 220, 150)
        exang = st.sidebar.selectbox("Angina inducida por ejercicio (exang)", options=[0, 1], format_func=lambda x: "0 - No" if x == 0 else "1 - Sí")
        oldpeak = st.sidebar.slider("Depresión del ST (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.sidebar.selectbox("Pendiente del segmento ST (slope)", options=[0, 1, 2])
        ca = st.sidebar.slider("Número de vasos principales (ca)", 0, 4, 0)
        thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

        data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        return pd.DataFrame(data, index=[0])

    input_df = obtener_datos_usuario()

    st.subheader("📋 Resumen del Paciente")
    st.dataframe(input_df, hide_index=True)

    # 6. Botón de predicción y lógica
    if st.button("Generar Diagnóstico", type="primary"):
        input_scaled = scaler.transform(input_df)
        modelo_activo = modelo_lr if modelo_seleccionado == "Regresión Logística" else modelo_rf
        prediccion = modelo_activo.predict(input_scaled)[0]
        probabilidades = modelo_activo.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.subheader("🩺 Resultados del Diagnóstico")

        if prediccion == 1:
            st.error("⚠️ **Paciente con Enfermedad Cardíaca** (Riesgo Detectado)")
            st.metric(label="Probabilidad de Riesgo", value=f"{probabilidades[1]*100:.2f} %")
            st.warning("Se recomienda derivar al paciente para exámenes cardiovasculares adicionales.")
        else:
            st.success("✅ **Paciente Sano** (Bajo Riesgo)")
            st.metric(label="Probabilidad de estar sano", value=f"{probabilidades[0]*100:.2f} %")
