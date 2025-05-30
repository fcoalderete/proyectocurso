import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
from openai import OpenAI
from openai.error import AuthenticationError

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
def cargar_tutores(path="tutores.csv"):
    df = pd.read_csv(path)
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
        try:
            resp = client.embeddings.create(model="text-embedding-ada-002", input=t["materia"])
            embs.append(resp.data[0].embedding)
        except AuthenticationError:
            st.error("Error de autenticación con OpenAI. Verifica tu API key.")
            st.stop()
        except Exception as e:
            st.error(f"Error generando embeddings: {e}")
            st.stop()
    arr = np.array(embs, dtype="float32")
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index

# Índice instanciado
index = preparar_indice(tutores)

# 4. Historial conversacional
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "Eres un asistente experto en tutorías de la FCA-UACH."}]

# Mostrar historial previo
for msg in st.session_state.history[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 5. Función de búsqueda (exacta y semántica)
def buscar_tutores(consulta, k=3):
    # Búsqueda exacta
    exact = [t for t in tutores if consulta.lower() in t["materia"].lower()]
    if exact:
        return exact[:k]
    # Si no hay coincidencias exactas, búsqueda semántica
    try:
        q_resp = client.embeddings.create(model="text-embedding-ada-002", input=consulta)
        q_emb = q_resp.data[0].embedding
    except AuthenticationError:
        st.error("Error de autenticación al buscar embeddings.")
        st.stop()
    except Exception as e:
        st.error(f"Error en búsqueda semántica: {e}")
        st.stop()
    D, I = index.search(np.array([q_emb], dtype="float32"), k=k)
    return [tutores[i] for i in I[0]]

# 6. Input y salida en chat
consulta = st.chat_input("¿En qué materia necesitas asesoría?")
if consulta:
    # Añadir mensaje del usuario al historial
    st.session_state.history.append({"role": "user", "content": consulta})
    with st.chat_message("user"):
        st.write(consulta)

    # Obtener recomendaciones
    recomendados = buscar_tutores(consulta)
    st.subheader("Profesores recomendados:")
    for t in recomendados:
        line = f"**{t['maestro']}** | _{t['materia']}_ | 📅 {t['días']} | ⏰ {t['hora']} | 📍 {t['lugar']}"
        st.markdown(line)

    # Llamada al chat de OpenAI para conversación adicional
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.history,
            max_tokens=800,
            temperature=0
        )
        ia_resp = stream.choices[0].message.content
    except AuthenticationError:
        st.error("Error de autenticación al generar respuesta de chat.")
        st.stop()
    except Exception as e:
        st.error(f"Error en llamada de chat completions: {e}")
        st.stop()

    # Mostrar respuesta de IA
    st.session_state.history.append({"role": "assistant", "content": ia_resp})
    with st.chat_message("assistant"):
        st.write(ia_resp)
