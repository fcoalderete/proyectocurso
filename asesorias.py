# Version 1.1: Removed semantic fallback; using only substring matching for tutor lookup.
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
from openai import OpenAI
import unicodedata

# Version history:
# 1.0 - Initial implementation with substring and semantic fallback.
# 1.1 - Removed semantic fallback to prevent irrelevant matches; now only substring search.

# 0. Configuración inicial
st.set_page_config(page_title="Horarios y docentes de Asesorías Académicas de la FCA UACH", layout="wide")
st.title("Horarios y docentes de Asesorías Académicas de la FCA UACH")
st.subheader("Consulta tutorías por materia y recibe recomendaciones personalizadas de profesores y horarios.")

# 1. Validación y cliente de OpenAI
api_key = st.secrets.get("api_key")
if not api_key:
    st.error("Clave de OpenAI no encontrada. Define 'api_key' en .streamlit/secrets.toml o en Secrets de Streamlit Cloud.")
    st.stop()
client = OpenAI(api_key=api_key)

# 2. Carga de datos de tutores
def normalize_text(s):
    nkfd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nkfd if not unicodedata.combining(c)).lower().strip()

@st.cache_data(ttl=3600)
def cargar_tutores(path="tutores.csv"):
    df = pd.read_csv(path)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.to_dict(orient="records")

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
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.history,
            max_tokens=800,
            temperature=0
        )
        ia_resp = stream.choices[0].message.content
    except Exception as e:
        st.error(f"Error generando respuesta de chat: {e}")
        st.stop()

    st.session_state.history.append({"role": "assistant", "content": ia_resp})
    with st.chat_message("assistant"):
        st.write(ia_resp)
