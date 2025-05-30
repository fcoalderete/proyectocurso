# Version 1.2: Added sidebar with contact info and logos
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
def setup_sidebar():
    with st.sidebar:
        # Logos
        st.image("escudo-texto-color.png", use_column_width=True)
        st.image("fca-escudo.png", use_column_width=True)
        st.markdown("---")
        # FCA
        st.header("Facultad de Contaduría y Administración")
        st.write("Circuito Universitario Campus II")
        st.write("Tel. +52 (614) 442 0000")
        st.write("Chihuahua, Chih. México")
        st.markdown("---")
        # UACH
        st.header("Universidad Autónoma de Chihuahua")
        st.write("C. Escorza 900, Col. Centro 31000")
        st.write("Tel. +52 (614) 439 1500")
        st.write("Chihuahua, Chih. México")
        st.markdown("---")
        st.write("**Realizado por Francisco Aldrete**")

setup_sidebar()

st.title("Horarios y docentes de Asesorías Académicas de la FCA UACH")
st.subheader("Consulta tutorías por materia y recibe recomendaciones personalizadas de profesores y horarios.")

# Version history:
# 1.0 - Initial implementation with substring and semantic fallback.
# 1.1 - Removed semantic fallback; only substring search.
# 1.2 - Added sidebar with contact info and logos.

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
def cargar_tutores(path="tutores.csv"):
    df = pd.read_csv(path)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.to_dict(orient="records")

@st.cache_data(ttl=3600)
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

    # Conversación adicional con IA
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
