# Versión 2.3: Búsqueda optimizada basada en análisis del CSV real
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

# Diccionario de sinónimos y términos relacionados (basado en el CSV real)
SINONIMOS_MATERIAS = {
    'redes': ['redes y comunicaciones', 'comunicaciones', 'sistemas operativos', 'internet', 'tecnologias'],
    'red': ['redes y comunicaciones', 'comunicaciones', 'sistemas operativos', 'internet', 'tecnologias'],
    'mercado': ['mercadotecnia', 'investigacion de mercados', 'mercado de capitales', 'mercado de dinero', 'mercado de valores'],
    'marketing': ['mercadotecnia', 'comunicacion integral de mercadotecnia', 'publicidad', 'promocion'],
    'mercadotecnia': ['mercadotecnia', 'marketing', 'comunicacion integral', 'publicidad', 'promocion'],
    'sistemas': ['sistemas operativos', 'analisis y diseno de sistemas', 'fundamentos de sistemas', 'administracion de sistemas'],
    'computacion': ['sistemas operativos', 'arquitectura de computadoras', 'fundamentos de sistemas'],
    'informatica': ['sistemas operativos', 'tecnologias y manejo de la informacion', 'administracion de las operaciones'],
    'finanzas': ['mercado de capitales', 'mercado de dinero', 'mercado de valores', 'administracion de inversiones'],
    'contabilidad': ['aspectos basicos de contabilidad', 'analisis e interpretacion de estados financieros', 'auditoria'],
    'recursos humanos': ['administracion de recursos humanos', 'administracion de la compensacion'],
    'administracion': ['administracion i', 'administracion ii', 'administracion de la pyme', 'administracion bancaria'],
    'produccion': ['administracion de la produccion', 'administracion de proyectos'],
    'comercio': ['entorno comercial internacional', 'entorno comercial latinoamericano', 'negocios por internet'],
    'internacional': ['entorno comercial internacional', 'mercadotecnia internacional', 'negocios por internet'],
    'publicidad': ['publicidad y promocion', 'comunicacion integral de mercadotecnia'],
    'ventas': ['comunicacion integral de mercadotecnia y ventas', 'mercadotecnia'],
    'investigacion': ['investigacion de mercados', 'investigacion de operaciones']
}

# Función para expandir consulta con sinónimos
def expandir_consulta(consulta):
    """Expande la consulta original con sinónimos y términos relacionados"""
    consulta_norm = normalize_text(consulta)
    terminos_expandidos = [consulta_norm]
    
    # Buscar sinónimos para cada palabra en la consulta
    palabras = consulta_norm.split()
    for palabra in palabras:
        if palabra in SINONIMOS_MATERIAS:
            terminos_expandidos.extend(SINONIMOS_MATERIAS[palabra])
    
    return list(set(terminos_expandidos))  # Eliminar duplicados

# Nueva función para extraer palabras clave
def extraer_palabras_clave(texto):
    """Extrae palabras clave relevantes del texto"""
    texto_norm = normalize_text(texto)
    # Dividir en palabras y filtrar palabras comunes irrelevantes
    stop_words = {'de', 'del', 'la', 'el', 'en', 'y', 'o', 'para', 'con', 'por', 'a', 'al', 'las', 'los', 'una', 'un', 'ii', 'i'}
    palabras = [p for p in texto_norm.split() if len(p) > 1 and p not in stop_words]
    return palabras

# Función de similitud entre textos
def calcular_similitud(texto1, texto2):
    """Calcula similitud entre dos textos usando SequenceMatcher"""
    return SequenceMatcher(None, normalize_text(texto1), normalize_text(texto2)).ratio()

# 1. Sidebar con logos e información de contacto
def setup_sidebar():
    with st.sidebar:
        st.markdown("---")
        st.write("**Versión 2.3 - Búsqueda Inteligente**")
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
        try:
            df_local = pd.read_csv(path, encoding='latin-1')
        except:
            df_local = pd.read_csv(path, encoding='cp1252')
    
    # Limpiar datos
    df_local = df_local.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df_local.columns = [c.strip().lower() for c in df_local.columns]
    
    # Agregar campo de búsqueda normalizado
    df_local['materia_normalizada'] = df_local['materia'].apply(normalize_text)
    df_local['palabras_clave'] = df_local['materia'].apply(extraer_palabras_clave)
    
    return df_local.to_dict(orient="records")

tutores = cargar_tutores()

# 4. Preparación del índice semántico (mantenido para casos complejos)
@st.cache_resource
def preparar_indice(data):
    try:
        embs = [client.embeddings.create(model="text-embedding-ada-002", input=t["materia"]).data[0].embedding for t in data]
        arr = np.array(embs, dtype="float32")
        index = faiss.IndexFlatL2(arr.shape[1])
        index.add(arr)
        return index
    except Exception as e:
        st.warning(f"No se pudo crear el índice semántico: {e}")
        return None

index = preparar_indice(tutores)

# 5. Historial conversacional
if "history" not in st.session_state:
    st.session_state.history = [{"role":"system","content":"Eres un asistente experto en tutorías de la FCA-UACH. Ayuda a encontrar profesores y horarios de asesorías académicas usando el catálogo real de materias disponibles."}]

# Mostrar historial de conversación
for msg in st.session_state.history[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 6. Función de búsqueda optimizada para el CSV real
def buscar_tutores(consulta):
    """
    Búsqueda optimizada usando múltiples estrategias y sinónimos:
    1. Expansión de consulta con sinónimos
    2. Coincidencia exacta de substring
    3. Búsqueda por palabras clave individual
    4. Similitud textual con umbral bajo
    5. Búsqueda parcial flexible
    """
    try:
        # Expandir consulta con sinónimos
        terminos_busqueda = expandir_consulta(consulta)
        
        resultados_con_puntuacion = []
        
        for tutor in tutores:
            materia_norm = tutor['materia_normalizada']
            palabras_materia = tutor['palabras_clave']
            puntuacion_total = 0
            
            # Probar cada término de búsqueda expandido
            for termino in terminos_busqueda:
                puntuacion = 0
                
                # Estrategia 1: Coincidencia exacta de substring (peso muy alto)
                if termino in materia_norm:
                    puntuacion += 150
                
                # Estrategia 2: Coincidencia de palabras individuales (peso alto)
                palabras_termino = termino.split()
                for palabra_busq in palabras_termino:
                    if len(palabra_busq) > 2:  # Solo palabras significativas
                        for palabra_mat in palabras_materia:
                            if palabra_busq == palabra_mat:
                                puntuacion += 100
                            elif palabra_busq in palabra_mat or palabra_mat in palabra_busq:
                                puntuacion += 60
                
                # Estrategia 3: Similitud textual (peso medio)
                try:
                    similitud = calcular_similitud(termino, tutor['materia'])
                    if similitud > 0.2:  # Umbral más bajo para mayor flexibilidad
                        puntuacion += similitud * 80
                except:
                    pass
                
                # Estrategia 4: Búsqueda parcial muy flexible
                try:
                    if any(palabra_busq in palabra_mat for palabra_busq in palabras_termino if len(palabra_busq) > 2 
                           for palabra_mat in palabras_materia):
                        puntuacion += 40
                except:
                    pass
                
                puntuacion_total = max(puntuacion_total, puntuacion)
            
            if puntuacion_total > 20:  # Umbral mínimo más bajo
                resultados_con_puntuacion.append((puntuacion_total, tutor))
        # … después de llenar “resultados_con_puntuacion” …
        st.write("DEBUG – resultados_con_puntuacion:", resultados_con_puntuacion)
        st.write("DEBUG – tipo(resultados_con_puntuacion):", type(resultados_con_puntuacion))
        # Eliminar duplicados por materia manteniendo el de mayor puntuación
        materias_vistas = {}
        for puntuacion, tutor in resultados_con_puntuacion:
            materia = tutor['materia']
            if materia not in materias_vistas or materias_vistas[materia][0] < puntuacion:
                materias_vistas[materia] = (puntuacion, tutor)
        
        # Ordenar por puntuación descendente
        resultados_finales = sorted(materias_vistas.values(), key=lambda x: x[0], reverse=True)
        return [tutor for puntuacion, tutor in resultados_finales[:15]]
    
    except Exception as e:
        st.error(f"Error en la búsqueda: {e}")
        return []

# Función de búsqueda semántica como respaldo
def buscar_semantica(consulta, k=8):
    """Búsqueda semántica usando embeddings como último recurso"""
    if index is None:
        return []
    
    try:
        emb_consulta = client.embeddings.create(model="text-embedding-ada-002", input=consulta).data[0].embedding
        arr_consulta = np.array([emb_consulta], dtype="float32")
        distancias, indices = index.search(arr_consulta, k)
        
        resultados_semanticos = []
        materias_agregadas = set()
        
        for i, dist in zip(indices[0], distancias[0]):
            if dist < 1.2:  # Umbral más flexible
                tutor = tutores[i]
                if tutor['materia'] not in materias_agregadas:
                    resultados_semanticos.append(tutor)
                    materias_agregadas.add(tutor['materia'])
        
        return resultados_semanticos
    except Exception as e:
        st.error(f"Error en búsqueda semántica: {e}")
        return []

# 7. Función para mostrar sugerencias de búsqueda
def mostrar_sugerencias_busqueda():
    """Muestra sugerencias de términos de búsqueda populares"""
    st.markdown("### 💡 **Sugerencias de búsqueda:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Administración & Negocios:**")
        st.markdown("• administración\n• mercadotecnia\n• recursos humanos\n• proyectos\n• pyme")
    
    with col2:
        st.markdown("**💰 Finanzas & Contabilidad:**")
        st.markdown("• contabilidad\n• finanzas\n• mercado\n• inversiones\n• auditoría")
    
    with col3:
        st.markdown("**💻 Tecnología & Sistemas:**")
        st.markdown("• redes\n• sistemas\n• computación\n• internet\n• tecnologías")

# 8. Interacción en chat optimizada
consulta = st.chat_input("🔍 Escribe el nombre de la materia (ej: redes, mercadotecnia, contabilidad, administración)")

if consulta:
    st.session_state.history.append({"role":"user","content":consulta})
    with st.chat_message("user"):
        st.write(consulta)

    with st.spinner("🔍 Analizando consulta y buscando profesores..."):
        # Mostrar términos de búsqueda expandidos
        terminos_expandidos = expandir_consulta(consulta)
        if len(terminos_expandidos) > 1:
            st.info(f"🎯 **Búsqueda expandida:** {', '.join(terminos_expandidos[:5])}")
        
        # Búsqueda principal optimizada
        recomendados = buscar_tutores(consulta)
        
        # Si pocos resultados, intentar búsqueda semántica
        if len(recomendados) < 3 and index is not None:
            st.info("🤖 Complementando con búsqueda semántica...")
            semanticos = buscar_semantica(consulta)
            # Agregar resultados semánticos que no estén ya incluidos
            materias_existentes = {t['materia'] for t in recomendados}
            recomendados.extend([t for t in semanticos if t['materia'] not in materias_existentes])
    
    st.markdown("---")
    
    if recomendados:
        st.success(f"✅ **{len(recomendados)} tutor(es) encontrado(s) para tu consulta**")
        
        # Agrupar por materia para mejor presentación
        from itertools import groupby
        
        # Agrupar resultados por materia
        recomendados_ordenados = sorted(recomendados, key=lambda x: x['materia'])
        grupos_materias = {k: list(v) for k, v in groupby(recomendados_ordenados, key=lambda x: x['materia'])}
        
        for materia, tutores_materia in grupos_materias.items():
            with st.expander(f"📚 **{materia}** ({len(tutores_materia)} tutor{'es' if len(tutores_materia) > 1 else ''})", 
                           expanded=len(grupos_materias) <= 3):
                
                for i, tutor in enumerate(tutores_materia, 1):
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
                    with col1:
                        st.write(f"**👨‍🏫 {tutor['maestro']}**")
                    with col2:
                        st.write(f"📅 {tutor['días']}")
                    with col3:
                        st.write(f"⏰ {tutor['hora']}")
                    with col4:
                        st.write(f"📍 {tutor['lugar']}")
                    
                    if i < len(tutores_materia):
                        st.markdown("---")
        
        st.markdown("---")
        st.info("💬 **¿Necesitas información de otra materia?** Escribe una nueva consulta.")
        
    else:
        st.warning("❌ **No se encontraron tutores para esa consulta**")
        
        # Mostrar sugerencias de búsqueda
        mostrar_sugerencias_busqueda()
        
        # Generar respuesta de IA con mejor contexto
        with st.spinner("🤖 Generando sugerencias personalizadas..."):
            # Obtener muestra de materias disponibles para contexto
            materias_disponibles = list(set(t['materia'] for t in tutores))[:20]
            
            prompt_contextual = f"""La consulta del usuario fue: "{consulta}"

No se encontraron tutores para esta búsqueda en el sistema de asesorías de la FCA-UACH.

Algunas materias disponibles incluyen: {', '.join(materias_disponibles[:10])}...

Como experto en asesorías académicas, proporciona:
1. Términos de búsqueda alternativos más específicos
2. Materias similares que podrían interesar al usuario
3. Sugerencias para refinar la búsqueda

Responde de manera académica, útil y alentadora."""
            
            st.session_state.history.append({"role":"user","content":prompt_contextual})
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.history,
                    max_tokens=400,
                    temperature=0.3
                )
                ia_resp = response.choices[0].message.content
                
                st.session_state.history.append({"role":"assistant","content":ia_resp})
                with st.chat_message("assistant"):
                    st.write(ia_resp)
            except Exception as e:
                st.error(f"Error al generar respuesta de IA: {e}")

# 9. Sección informativa mejorada
with st.expander("📊 **Información del sistema**"):
    materias_unicas = len(set(t['materia'] for t in tutores))
    maestros_unicos = len(set(t['maestro'] for t in tutores))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de registros", len(tutores))
    with col2:
        st.metric("Materias disponibles", materias_unicas)
    with col3:
        st.metric("Profesores registrados", maestros_unicos)
    
    # Mostrar muestra de materias más populares
    st.markdown("**📚 Algunas materias disponibles:**")
    materias_muestra = sorted(set(t['materia'] for t in tutores))[:15]
    for i, materia in enumerate(materias_muestra):
        if i % 3 == 0:
            cols = st.columns(3)
        with cols[i % 3]:
            st.write(f"• {materia}")
