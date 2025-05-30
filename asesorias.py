# Version 1.9: Mejorar búsqueda tokenizada y por prefijo para materias
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

# Función de normalización de texto
def normalize_text(s):
    nkfd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nkfd if not unicodedata.combining(c)).lower().strip()

# 1. Sidebar con logos e información de contacto
# Primer logo + Universidad, Segundo logo + Facultad
def setup_sidebar():
    with st.sidebar:
        st.markdown("---")
        # Primer logo: centrado y con borde
        if os.path.exists("escudo-texto-color.png"):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown(
                    "<div style='border:1px solid #ccc; border-radius:8px; padding:4px;'>",
                    unsafe_allow_html=True
                )
                st.image("escudo-texto-color.png", width=120)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("**[Logo FCA no disponible]**")
        st.markdown("---")
        # Información de la Universidad
        st.header("Universidad Autónoma de Chihuahua")
        st.write("C. Escorza 900, Col. Centro 31000")
        st.write("Tel. +52 (614) 439 1500")
        st.write("Chihuahua, Chih. México")
        st.markdown("---")
        # Segundo logo: centrado y con borde
        if os.path.exists("fca-escudo.png"):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown(
                    "<div style='border:1px solid #ccc; border-radius:8px; padding:4px;'>",
                    unsafe_allow_html=True
                )
                st.image("fca-escudo.png", width=100)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("**[Logo UACH no disponible]**")
        st.markdown("---")
        # Información de la Facultad
        st.header("Facultad de Contaduría y Administración")
        st.write("Circuito Universitario Campus II")
        st.write("Tel. +52 (614) 442 0000")
        st.write("Chihuahua, Chih. México")
        st.markdown("---")
        st.write("**Realizado por Francisco Aldrete**")

setup_sidebar()

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
# 1.7 - Removed balloons; added dividers and frames.
# 1.8 - Refined centering and borders for both logos with st.image in columns.
# 1.9 - Mejorada búsqueda tokenizada y por prefijo para materias.

# 2. Validación y cliente de OpenAI
api_key = st.secrets.get("api_key")
if not api_key:
    st.error("Clave de OpenAI no encontrada. Define 'api_key' en .streamlit/secrets.toml o en Secrets de Streamlit Cloud.")
    st.stop()
client = OpenAI(api_key=api_key)

# 3. Carga de datos de tutores
@st.cache_data(ttl=3600)
def cargar_tutores(path="tutores.csv"):
    df_local = pd.read_csv(path)
    df_local = df_local.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df_local.columns = [c.strip().lower() for c in df_local.columns]
    return df_local.to_dict(orient="records")

tutores = cargar_tutores()

# 4. Preparación del índice semántico
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

# 5. Historial conversacional
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "Eres un asistente experto en tutorías de la FCA-UACH."}]
for msg in st.session_state.history[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 6. Función de búsqueda mejorada

def buscar_tutores(consulta):
    norm = normalize_text(consulta)
    matches = []
    for t in tutores:
        mat = normalize_text(t['materia'])
        tokens = mat.replace('-', ' ').split()
        # Coincidencia exacta de palabra o prefijo
        if norm in tokens or any(token.startswith(norm) for token in tokens):
            matches.append(t)
    # Si no hay matches tokenizados, fallback a substring en toda la materia
    if not matches:
        matches = [t for t in tutores if norm in normalize_text(t['materia'])]
    return matches

# 7. Input y salida en chat
consulta = st.chat_input("¿En qué materia necesitas asesoría?")
if consulta:
    st.session_state.history.append({"role": "user", "content": consulta})
    with st.chat_message("user"):
        st.write(consulta)

    with st.spinner("🔍 Buscando profesores..."):
        recomendados = buscar_tutores(consulta)

    st.markdown("---")
    if recomendados:
        st.subheader("Profesores recomendados:")
        for t in recomendados:
            line = f"**{t['maestro']}** | _{t['materia']}_ | 📅 {t['días']} | ⏰ {t['hora']} | 📍 {t['lugar']}"
            st.markdown(line)
        st.markdown("---")
        st.info("¿Necesitas más ayuda o detalles específicos? Escribe tu pregunta para recibir asistencia de IA.")
    else:
        st.warning("No hay maestro asesor disponible para esa materia.")
        with st.spinner("⌛ Generando respuesta de IA..."):
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
