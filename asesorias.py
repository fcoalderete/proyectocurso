# Versión 2.2: Búsqueda mejorada con múltiples estrategias
import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
from openai import OpenAI
import unicodedata
import re
from difflib import SequenceMatcher

# 0. Configuración inicial
titulo = "Horarios y docentes de Asesorías Académicas de la FCA UACH"
st.set_page_config(page_title=titulo, layout="wide")

# Función de normalización de texto mejorada
def normalize_text(s):
    """Normaliza texto removiendo acentos, convirtiendo a minúsculas y limpiando espacios"""
    if not isinstance(s, str):
        return str(s).lower().strip()
    nkfd = unicodedata.normalize('NFKD', s)
    normalized = ''.join(c for c in nkfd if not unicodedata.combining(c))
    # Remover caracteres especiales excepto espacios y números
    cleaned = re.sub(r'[^\w\s]', ' ', normalized)
    return ' '.join(cleaned.lower().split())

# Nueva función para extraer palabras clave
def extraer_palabras_clave(texto):
    """Extrae palabras clave relevantes del texto"""
    texto_norm = normalize_text(texto)
    # Dividir en palabras y filtrar palabras comunes irrelevantes
    stop_words = {'de', 'del', 'la', 'el', 'en', 'y', 'o', 'para', 'con', 'por', 'a', 'al', 'las', 'los', 'una', 'un'}
    palabras = [p for p in texto_norm.split() if len(p) > 2 and p not in stop_words]
    return palabras

# Función de similitud entre textos
def calcular_similitud(texto1, texto2):
    """Calcula similitud entre dos textos usando SequenceMatcher"""
    return SequenceMatcher(None, normalize_text(texto1), normalize_text(texto2)).ratio()

# 1. Sidebar con logos e información de contacto (sin cambios)
def setup_sidebar():
    with st.sidebar:
        st.markdown("---")
        st.write("**Versión 2.2 - Búsqueda Mejorada**")
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
    try:
        df_local = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df_local = pd.read_csv(path, encoding='latin-1')
    
    # Limpiar datos
    df_local = df_local.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df_local.columns = [c.strip().lower() for c in df_local.columns]
    
    # Agregar campo de búsqueda normalizado
    df_local['materia_normalizada'] = df_local['materia'].apply(normalize_text)
    
    return df_local.to_dict(orient="records")

tutores = cargar_tutores()

# 4. Preparación del índice semántico (mantenido para casos complejos)
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
    st.session_state.history = [{"role":"system","content":"Eres un asistente experto en tutorías de la FCA-UACH. Ayuda a encontrar profesores y horarios de asesorías académicas."}]

for msg in st.session_state.history[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 6. Función de búsqueda mejorada con múltiples estrategias
def buscar_tutores(consulta):
    """
    Búsqueda mejorada usando múltiples estrategias:
    1. Coincidencia exacta de substring
    2. Búsqueda por palabras clave
    3. Similitud textual
    4. Búsqueda semántica con embeddings (fallback)
    """
    consulta_norm = normalize_text(consulta)
    palabras_consulta = extraer_palabras_clave(consulta)
    
    resultados = []
    puntuaciones = []
    
    for tutor in tutores:
        materia_norm = tutor['materia_normalizada']
        puntuacion = 0
        
        # Estrategia 1: Coincidencia exacta de substring (peso alto)
        if consulta_norm in materia_norm:
            puntuacion += 100
        
        # Estrategia 2: Coincidencia de palabras clave (peso medio-alto)
        palabras_materia = extraer_palabras_clave(tutor['materia'])
        palabras_comunes = set(palabras_consulta) & set(palabras_materia)
        if palabras_comunes:
            puntuacion += len(palabras_comunes) * 30
        
        # Estrategia 3: Similitud textual (peso medio)
        similitud = calcular_similitud(consulta, tutor['materia'])
        if similitud > 0.3:  # Umbral de similitud
            puntuacion += similitud * 50
        
        # Estrategia 4: Búsqueda parcial en cada palabra
        for palabra in palabras_consulta:
            if len(palabra) > 2:  # Solo palabras significativas
                for palabra_materia in palabras_materia:
                    if palabra in palabra_materia or palabra_materia in palabra:
                        puntuacion += 20
        
        if puntuacion > 0:
            resultados.append(tutor)
            puntuaciones.append(puntuacion)
    
    # Ordenar por puntuación descendente
    if resultados:
        resultados_ordenados = [x for _, x in sorted(zip(puntuaciones, resultados), reverse=True)]
        return resultados_ordenados[:10]  # Limitar a 10 mejores resultados
    
    return []

# Función de búsqueda semántica como respaldo
def buscar_semantica(consulta, k=5):
    """Búsqueda semántica usando embeddings como último recurso"""
    try:
        emb_consulta = client.embeddings.create(model="text-embedding-ada-002", input=consulta).data[0].embedding
        arr_consulta = np.array([emb_consulta], dtype="float32")
        distancias, indices = index.search(arr_consulta, k)
        
        # Filtrar resultados con distancia razonable
        resultados_semanticos = []
        for i, dist in zip(indices[0], distancias[0]):
            if dist < 1.0:  # Umbral de distancia semántica
                resultados_semanticos.append(tutores[i])
        
        return resultados_semanticos
    except Exception as e:
        st.error(f"Error en búsqueda semántica: {e}")
        return []

# 7. Interacción en chat mejorada
consulta = st.chat_input("¿En qué materia necesitas asesoría? (ej: redes, mercadotecnia, contabilidad)")

if consulta:
    st.session_state.history.append({"role":"user","content":consulta})
    with st.chat_message("user"):
        st.write(consulta)

    with st.spinner("🔍 Buscando profesores..."):
        # Búsqueda principal mejorada
        recomendados = buscar_tutores(consulta)
        
        # Si no hay resultados, intentar búsqueda semántica
        if not recomendados:
            st.info("Probando búsqueda semántica...")
            recomendados = buscar_semantica(consulta)
    
    st.markdown("---")
    
    if recomendados:
        st.subheader(f"✅ {len(recomendados)} profesor(es) encontrado(s):")
        
        # Mostrar resultados en formato mejorado
        for i, t in enumerate(recomendados, 1):
            with st.expander(f"{i}. **{t['maestro']}** - {t['materia']}", expanded=(i <= 3)):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"📅 **Días:** {t['días']}")
                with col2:
                    st.write(f"⏰ **Horario:** {t['hora']}")
                with col3:
                    st.write(f"📍 **Lugar:** {t['lugar']}")
        
        st.markdown("---")
        st.success("¿Necesitas más información? Haz otra consulta.")
        
    else:
        st.warning("❌ No se encontraron profesores para esa materia.")
        st.info("💡 **Sugerencias:**\n- Intenta con palabras clave más específicas\n- Usa términos como 'contabilidad', 'administración', 'mercadotecnia'\n- Consulta la lista completa de materias disponibles")
        
        # Generar respuesta de IA con contexto mejorado
        with st.spinner("🤖 Generando sugerencias de IA..."):
            prompt_mejorado = f"""Usuario busca: "{consulta}"
            
            Contexto: Sistema de asesorías FCA-UACH. No se encontraron tutores.
            
            Proporciona:
            1. Sugerencias de términos alternativos de búsqueda
            2. Materias relacionadas que podrían existir
            3. Consejos para refinar la búsqueda
            
            Responde de manera académica y útil."""
            
            st.session_state.history.append({"role":"user","content":prompt_mejorado})
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.history,
                max_tokens=500,
                temperature=0.3
            )
            ia_resp = response.choices[0].message.content
            
        st.session_state.history.append({"role":"assistant","content":ia_resp})
        with st.chat_message("assistant"):
            st.write(ia_resp)

# 8. Sección adicional: Mostrar estadísticas
with st.expander("📊 Estadísticas del sistema"):
    st.write(f"**Total de tutores registrados:** {len(tutores)}")
    materias_unicas = len(set(t['materia'] for t in tutores))
    st.write(f"**Materias disponibles:** {materias_unicas}")
    
    # Mostrar algunas materias de ejemplo
    st.write("**Ejemplos de materias disponibles:**")
    ejemplos = [t['materia'] for t in tutores[:10]]
    for ejemplo in ejemplos:
        st.write(f"• {ejemplo}")
