import streamlit as st
import pandas as pd
from openai import OpenAI
import faiss
import numpy as np

# Título y configuración inicial
st.set_page_config(page_title="Horarios y docentes de Asesorías Académicas de la FCA UACH", layout="wide")
st.title("Horarios y docentes de Asesorías Académicas de la FCA UACH")
st.subheader("Consulta tutorías por materia y recibe recomendaciones personalizadas de profesores y horarios.")

# Cliente de OpenAI
client = OpenAI(api_key=st.secrets["api_key"])

# 1. Carga de datos de tutores
def cargar_tutores(path="tutores.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.to_dict(orient="records")

tutores = st.cache_data(ttl=3600)(cargar_tutores)()

# 2. Preparación del índice semántico (embedding de materias)
@st.cache_resource
def preparar_indice(tutores):
    embs = []
    for t in tutores:
        emb = client.embeddings.create(
            model="text-embedding-ada-002",
            input=t["materia"]
        ).data[0].embedding
        embs.append(emb)
    arr = np.array(embs, dtype="float32")
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index

index = preparar_indice(tutores)

# 3. Historial conversacional
def init_history():
    return [{"role": "system", "content": "Eres un asistente experto en tutorías de la FCA-UACH."}]

if "history" not in st.session_state:
    st.session_state.history = init_history()

# Mostrar historial previo
for msg in st.session_state.history[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 4. Input del alumno
def buscar_tutores(consulta, k=3):
    # 4.1. Búsqueda exacta por nombre de materia
    exact = [t for t in tutores if consulta.lower() in t["materia"].lower()]
    if exact:
        return exact[:k]
    # 4.2. Búsqueda semántica fallback
    q_emb = client.embeddings.create(
        model="text-embedding-ada-002",
        input=consulta
    ).data[0].embedding
    D, I = index.search(np.array([q_emb], dtype="float32"), k=k)
    return [tutores[i] for i in I[0]]

consulta = st.chat_input("¿En qué materia necesitas asesoría?")
if consulta:
    # Guardar mensaje de usuario
    st.session_state.history.append({"role": "user", "content": consulta})
    with st.chat_message("user"):
        st.write(consulta)

    # Buscar y formatear recomendaciones
    recomendados = buscar_tutores(consulta)
    st.subheader("Profesores recomendados:")
    for t in recomendados:
        st.markdown(
            f"**{t['maestro']}** – _{t['materia']}_  
             📅 {t['días']}  |  ⏰ {t['hora']}  |  📍 {t['lugar']}")

    # Propuesta de interacción adicional
    st.session_state.history.append({"role": "assistant", "content": "¿En qué más te puedo ayudar?"})
    with st.chat_message("assistant"):
        st.write("¿En qué más te puedo ayudar?")
