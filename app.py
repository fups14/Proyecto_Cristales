import streamlit as st
from keras.models import load_model
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import streamlit as st
import base64

# Configuración de la página
st.set_page_config(page_title="Proyecto-Cristales", page_icon="🔬", layout="wide")

# Función para convertir la imagen a base64
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        img_bytes = img_file.read()
        return base64.b64encode(img_bytes).decode()

# Convertir la imagen de fondo a base64
fondo_path = "CRISTAL_FONDO.jpeg"
img_base64 = img_to_base64(fondo_path)

# Estilo CSS para fondo con la imagen
st.markdown(
    f"""
    <style>
    body {{
        background-image: url('data:image/jpeg;base64,{img_base64}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        height: 100vh;
        margin: 0;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Cargar datos cristalográficos desde el CSV
datos_cristales_df = pd.read_csv("datos_cristales.csv")

# Encabezados y estructura de la página
st.header("🔬 Cristalografía")

# Subheader: Sistema Cristalino (Centrado automáticamente por CSS)
st.subheader("Sistema Cristalino")

# Lista de sistemas cristalinos
sistemas_cristalinos_info = {
    "Cúbico": "Los parámetros de red son a = b = c y los ángulos α = β = γ = 90°.",
    "Tetragonal": "Los parámetros de red son a = b ≠ c y los ángulos α = β = γ = 90°.",
    "Ortorrómbico": "Los parámetros de red son a ≠ b ≠ c y los ángulos α = β = γ = 90°.",
    "Trigonal": "Los parámetros de red son a = b = c y los ángulos α = β = γ ≠ 90°.",
    "Hexagonal": "Los parámetros de red son a = b ≠ c y los ángulos α = β = 90°, γ = 120°.",
    "Monoclínico": "Los parámetros de red son a ≠ b ≠ c y los ángulos α = γ = 90°, β ≠ 90°.",
    "Triclínico": "Los parámetros de red son a ≠ b ≠ c y los ángulos α ≠ β ≠ γ ≠ 90°."
}

# Mostrar lista de sistemas cristalinos en un selector
sistema_seleccionado = st.selectbox("Selecciona un sistema cristalino para obtener más información:", sistemas_cristalinos_info.keys())

# Mostrar información del sistema cristalino seleccionado
if sistema_seleccionado:
    st.write(f"### {sistema_seleccionado}")
    st.write(sistemas_cristalinos_info[sistema_seleccionado])

# Subheader: Cristales de Chihuahua
st.subheader("Cristales de Chihuahua")

# Crear dos columnas para los cristales
col1, col2 = st.columns(2)
cristales_chihuahua = {
    "Selenita": "CaSO₄ · 2H₂O",
    "Fluorita": "CaF₂",
    "Cuarzo": "SiO₂",
    "Calcopirita": "CaFeS₂",
    "Galena": "PbS",
    "Pirita": "FeS₂",
    "Baritina": "BaSO₄",
    "Calcita": "CaCO₃",
    "Hematita": "Fe₂O₃"
}

# Distribuir los cristales en las dos columnas
mitad = len(cristales_chihuahua) // 2
cristales_items = list(cristales_chihuahua.items())

for i, (cristal, formula) in enumerate(cristales_items):
    if i < mitad:
        col1.write(f"- **{cristal}**: ({formula})")
    else:
        col2.write(f"- **{cristal}**: ({formula})")

# Entrada de datos
st.subheader("Parámetros cristalográficos", divider='red')
col1, col23, col4 = st.columns([1, 6, 1])

# Campos de entrada individuales para cada parámetro
a = col23.number_input("Valor de a (Å)", min_value=0.0, step=0.01, format="%.2f")
b = col23.number_input("Valor de b (Å)", min_value=0.0, step=0.01, format="%.2f")
c = col23.number_input("Valor de c (Å)", min_value=0.0, step=0.01, format="%.2f")
alpha = col23.number_input("Valor de α (grados)", min_value=0.0, step=0.01, format="%.2f")
beta = col23.number_input("Valor de β (grados)", min_value=0.0, step=0.01, format="%.2f")
gamma = col23.number_input("Valor de γ (grados)", min_value=0.0, step=0.01, format="%.2f")

# Botón para realizar la predicción
boton = col23.button("Clasificar el cristal")

# Cargar el modelo
modelo_path = r"C:\Users\jazmy\Documents\UACH\9no Semestre\Reconocimiento de Patrones\PROYECTO_CRISTALES\modelo_cristales.keras"
try:
    modelo = load_model(modelo_path)
except Exception as e:
    col23.error(f"Error al cargar el modelo desde {modelo_path}: {e}")
    st.stop()

# Función para encontrar el cristal más similar
def encontrar_cristal_mas_similar(parametros, datos_cristales_df):
    valores_cristales = datos_cristales_df[["a", "b", "c", "alpha", "beta", "gamma"]].values
    distancias = euclidean_distances([parametros], valores_cristales).flatten()
    indice_mas_cercano = np.argmin(distancias)
    formula_mas_cercana = datos_cristales_df.iloc[indice_mas_cercano]["formula"]
    distancia_minima = distancias[indice_mas_cercano]
    return formula_mas_cercana, distancia_minima

# Predicción
if boton:
    cristal = np.array([a, b, c, alpha, beta, gamma])
    prediccion = modelo.predict(np.array([cristal]))
    cris_pred = np.argmax(prediccion)
    sistemas_cristalinos = {
        0: "Cúbico",
        1: "Tetragonal",
        2: "Ortorrómbico",
        3: "Trigonal",
        4: "Hexagonal",
        5: "Monoclínico",
        6: "Triclínico",
    }
    resultado = sistemas_cristalinos.get(cris_pred, "Sistema desconocido")
    col23.markdown(f"<h3 style='color: #FF6347;'>El sistema cristalino predicho es: {resultado}</h3>", unsafe_allow_html=True)
    formula_mas_cercana, distancia = encontrar_cristal_mas_similar(cristal, datos_cristales_df)
    col23.markdown(f"""
        <div style='display: flex; justify-content: center; align-items: center; height: 100px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f0f8ff;'>
            <h2 style='font-size: 36px; font-weight: bold; color: #000000; text-align: center;'>
                {formula_mas_cercana}
            </h2>
        </div>
    """, unsafe_allow_html=True)

