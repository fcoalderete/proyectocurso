# Version 1.6: Display second logo via st.image centered using columns
import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
from openai import OpenAI
import unicodedata

# 0. Configuración inicial
st.set_page_config(page_title="Horarios y docentes de Asesorías Académicas de la FCA UACH", layout="wide")

# Sidebar con logos e información de contacto
# Primer logo + Universidad, Segundo logo + Facultad

def setup_sidebar():
    with st.sidebar:
        # Primer logo: escala al ancho del contenedor
        if os.path.exists("escudo-texto-color.png"):
            st.image("escudo-texto-color.png", use_container_width=True)
        else:
            st.write("**[Logo FCA no disponible]**")
        # Información de la Universidad debajo del primer logo
        st.markdown("---")
        st.header("Universidad Autónoma de Chihuahua")
        st.write("C. Escorza 900, Col. Centro 31000")
        st.write("Tel. +52 (614) 439 1500")
        st.write("Chihuahua, Chih. México")
        # Segundo logo: centrado en una columna intermedia y con ancho fijo
        if os.path.exists("fca-escudo.png"):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("fca-escudo.png", width=100)
        else:
            st.write("**[Logo UACH no disponible]**")
        # Información de la Facultad debajo del segundo logo
        st.markdown("---")
        st.header("Facultad de Contaduría y Administración")
        st.write("Circuito Universitario Campus II")
        st.write("Tel. +52 (614) 442 0000")
        st.write("Chihuahua, Chih. México")
        st.markdown("---")
        st.write("**Realizado por Francisco Aldrete**")

setup_sidebar()

# Animación de bienvenida
st.balloons()


# Título y subtítulo de la aplicación
st.title("Horarios y docentes de Asesorías Académicas de la FCA UACH")
st.subheader("Consulta tutorías por materia y recibe recomendaciones personalizadas de profesores y horarios.")

# Version history:
# 1.0 - Initial implementation.
# 1.1 - Removed semantic fallback.
# 1.2 - Added sidebar.
# 1.3 - Fixed decorator placement for cargar_tutores caching.
# 1.4 - Handled missing logo files in sidebar.
# 1.5 - Replaced use_column_width; centered and resized logos with HTML.
# 1.6 - Use st.image within columns to center second logo and avoid HTML.

# 1. Validación y cliente de OpenAI
api_key = st.secrets.get("api_key")
if not api_key:
    st.error("Clave de OpenAI no encontrada. Define 'api_key' en .streamlit/secrets.toml o en Secrets de Streamlit Cloud.")
    st.stop()
client = OpenAI(api_key=api_key)

# Función de normalización de texto
def normalize_text(s):
    nkfd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nkfd if not unicodedata.combining(c)).lower().strip()

# 2. Carga de datos de tutores
@st.cache_data(ttl=3600)
def cargar_tutores(path="tutores.csv"):
    df_local = pd.read_csv(path)
    df_local = df_local.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df_local.columns = [c.strip().lower() for c in df_local.columns]
    return df_local.to_dict(orient="records")

# Carga efectiva
tutores = cargar_tutores()

# 3. Preparación del índice semántico
@st.cache_resource
def preparar_indice(data):
    embs = []
    for t in data:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=t["materia"])
        embs.append(resp.data[0].embedding)
    arr = np.array(embs, dtype="float32")
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index

index = preparar_indice(tutores)

# 4. Historial conversacional
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "Eres un asistente experto en tutorías de la FCA-UACH."}]
for msg in st.session_state.history[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 5. Función de búsqueda (solo substring)
def buscar_tutores(consulta, k=3):
    norm = normalize_text(consulta)
    sub_matches = [t for t in tutores if norm in normalize_text(t['materia'])]
    return sub_matches[:k]

# 6. Input y salida en chat
consulta = st.chat_input("¿En qué materia necesitas asesoría?")
if consulta:
    st.session_state.history.append({"role": "user", "content": consulta})
    with st.chat_message("user"):
        st.write(consulta)

    recomendados = buscar_tutores(consulta)
    if recomendados:
        st.subheader("Profesores recomendados:")
        for t in recomendados:
            line = f"**{t['maestro']}** | _{t['materia']}_ | 📅 {t['días']} | ⏰ {t['hora']} | 📍 {t['lugar']}"
            st.markdown(line)
    else:
        st.warning("No hay maestro asesor disponible para esa materia.")
        st.info("Sin embargo, puedo ayudarte con otras dudas o brindarte más información.")

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.history,
        max_tokens=800,
        temperature=0
    )
    ia_resp = stream.choices[0].message.content
    st.session_state.history.append({"role": "assistant", "content": ia_resp})
    with st.chat_message("assistant"):
        st.write(ia_resp)
