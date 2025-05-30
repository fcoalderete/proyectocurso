import streamlit as st
import pandas as pd
from openai import OpenAI
import faiss
import numpy as np

# 0. Configuración inicial
st.set_page_config("🤖 Asesor IA de Tutorías", layout="wide")
client = OpenAI(api_key=st.secrets["api_key"])

# 1. Carga de datos
@st.cache_data(ttl=3600)
def cargar_tutores(path="tutores.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.to_dict(orient="records")

# 2. Preparación del índice semántico
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

# Inicialización
tutores = cargar_tutores()
index = preparar_indice(tutores)

# 3. Historial conversacional
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "system", "content": "Eres un asistente experto en tutorías de la FCA-UACH."}
    ]

# Mostrar mensajes previos
for msg in st.session_state.history[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 4. Entrada del alumno
consulta = st.chat_input("¿En qué tema necesitas asesoría?")
if consulta:
    # Añadir mensaje del usuario
    st.session_state.history.append({"role": "user", "content": consulta})
    with st.chat_message("user"):
        st.write(consulta)

    # 5. Búsqueda semántica
    q_emb = client.embeddings.create(
        model="text-embedding-ada-002",
        input=consulta
    ).data[0].embedding
    D, I = index.search(np.array([q_emb], dtype="float32"), k=3)

    # Formatear recomendaciones
    recomendaciones = []
    for i in I[0]:
        t = tutores[i]
        recomendaciones.append(
            f"- **{t['maestro']}** ({t['materia']}): {t['días']} · {t['hora']} · {t['lugar']}"
        )

    respuesta = (
        "Según tu consulta semántica, te recomiendo estos profesores:\n\n"
        + "\n".join(recomendaciones)
        + "\n\n¿En qué más te puedo ayudar?"
    )

    # Mostrar respuesta del asistente
    st.session_state.history.append({"role": "assistant", "content": respuesta})
    with st.chat_message("assistant"):
        st.markdown(respuesta)
