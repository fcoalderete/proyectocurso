# Version 2.1: Simplified substring search and header updated
import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
from openai import OpenAI
import unicodedata

# 0. Configuración inicial
titulo = "Horarios y docentes de Asesorías Académicas de la FCA UACH"
st.set_page_config(page_title=titulo, layout="wide")

# Función de normalización de texto
def normalize_text(s):
    nkfd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nkfd if not unicodedata.combining(c)).lower().strip()

# 1. Sidebar con logos e información de contacto
def setup_sidebar():
    with st.sidebar:
        st.markdown("---")
        st.write("**Versión 2.2**")
        # Logo Universidad
        if os.path.exists("escudo-texto-color.png"):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image("escudo-texto-color.png", width=120)
        else:
            st.write("**[Logo FCA no disponible]**")
        st.markdown("---")
        st.header("Universidad Autónoma de Chihuahua")
        st.write("C. Escorza 900, Col. Centro 31000")
        st.write("Tel. +52 (614) 439 1500")
        st.write("Chihuahua, Chih. México")
        st.markdown("---")
        # Logo Facultad
        if os.path.exists("fca-escudo.png"):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image("fca-escudo.png", width=100)
        else:
            st.write("**[Logo UACH no disponible]**")
        st.markdown("---")
        st.header("Facultad de Contaduría y Administración")
        st.write("Circuito Universitario Campus II")
        st.write("Tel. +52 (614) 442 0000")
        st.write("Chihuahua, Chih. México")
        st.markdown("---")
        st.write("**Realizado por Francisco Aldrete**")

setup_sidebar()

# Título y subtítulo
st.title(titulo)
st.subheader("Consulta tutorías por materia y recibe recomendaciones personalizadas de profesores y horarios.")

# 2. Validación y cliente de OpenAI
api_key = st.secrets.get("api_key")
if not api_key:
    st.error("Clave de OpenAI no encontrada. Define 'api_key' en Secrets.")
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
    embs = [client.embeddings.create(model="text-embedding-ada-002", input=t["materia"]).data[0].embedding for t in data]
    arr = np.array(embs, dtype="float32")
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index

index = preparar_indice(tutores)

# 5. Historial conversacional
if "history" not in st.session_state:
    st.session_state.history = [{"role":"system","content":"Eres un asistente experto en tutorías de la FCA-UACH."}]
for msg in st.session_state.history[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 6. Función de búsqueda simplificada (substring normalize)

def buscar_tutores(consulta):
    norm = normalize_text(consulta)
    return [t for t in tutores if norm in normalize_text(t['materia'])]

# 7. Interacción en chat
consulta = st.chat_input("¿En qué materia necesitas asesoría?")
if consulta:
    st.session_state.history.append({"role":"user","content":consulta})
    with st.chat_message("user"):
        st.write(consulta)

    with st.spinner("🔍 Buscando profesores..."):
        recomendados = buscar_tutores(consulta)
    st.markdown("---")
    if recomendados:
        st.subheader("Profesores recomendados:")
        for t in recomendados:
            st.markdown(f"**{t['maestro']}** | _{t['materia']}_ | 📅 {t['días']} | ⏰ {t['hora']} | 📍 {t['lugar']}")
        st.markdown("---")
        st.info("¿Necesitas más ayuda? Haz otra consulta para IA.")
    else:
        st.warning("No hay maestro asesor disponible para esa materia.")
        with st.spinner("⌛ Generando respuesta de IA..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.history,
                max_tokens=800,
                temperature=0
            )
            ia_resp = response.choices[0].message.content
        st.session_state.history.append({"role":"assistant","content":ia_resp})
        with st.chat_message("assistant"):
            st.write(ia_resp)
