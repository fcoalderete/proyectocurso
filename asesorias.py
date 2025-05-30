import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
from openai import OpenAI

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
@st.cache_data(ttl=3600)
def cargar_tutores(path="tutores.csv"):
    df = pd.read_csv(path)
    # Limpiar espacios en blanco en todas las celdas de texto
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.to_dict(orient="records")

# Carga efectiva
tutores = cargar_tutores()

# 3. Preparación del índice semántico
@st.cache_resource
def preparar_indice(data):
    embs = []
    for t in data:
        try:
            resp = client.embeddings.create(model="text-embedding-ada-002", input=t["materia"])
            embs.append(resp.data[0].embedding)
        except Exception as e:
            st.error(f"Error generando embeddings: {e}")
            st.stop()
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

# 5. Función de búsqueda con tolerancia a errores tipográficos y coincidencias
from difflib import get_close_matches

def buscar_tutores(consulta, k=3):
    norm = consulta.lower().strip()
    # 5.1 Búsqueda difusa (fuzzy) sobre nombres de materia
    materias_set = list({t["materia"].lower() for t in tutores})
    close = get_close_matches(norm, materias_set, n=k, cutoff=0.6)
    if close:
        matches = [t for t in tutores if t["materia"].lower() in close]
        return matches[:k]
    # 5.2 Coincidencia exacta por palabra completa
    word_matches = [t for t in tutores if norm in t["materia"].lower().split()]
    if word_matches:
        return word_matches[:k]
    # 5.3 Coincidencia por substring
    sub_matches = [t for t in tutores if norm in t["materia"].lower()]
    if sub_matches:
        return sub_matches[:k]
    # 5.4 Búsqueda semántica como fallback
    try:
        q_resp = client.embeddings.create(model="text-embedding-ada-002", input=consulta)
        q_emb = q_resp.data[0].embedding
    except Exception as e:
        st.error(f"Error en búsqueda semántica: {e}")
        st.stop()
    D, I = index.search(np.array([q_emb], dtype="float32"), k=k)
    return [tutores[i] for i in I[0]]

# 6. Input y salida en chat Input y salida en chat
consulta = st.chat_input("¿En qué materia necesitas asesoría?")
if consulta:
    st.session_state.history.append({"role": "user", "content": consulta})
    with st.chat_message("user"):
        st.write(consulta)

    recomendados = buscar_tutores(consulta)
    st.subheader("Profesores recomendados:")
    for t in recomendados:
        line = f"**{t['maestro']}** | _{t['materia']}_ | 📅 {t['días']} | ⏰ {t['hora']} | 📍 {t['lugar']}"
        st.markdown(line)

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
