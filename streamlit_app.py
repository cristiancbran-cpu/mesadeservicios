import streamlit as st
import os
# from dotenv import load_dotenv # Ya no es necesario

# Importaciones de LangChain, ahora modulares
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# NUEVAS IMPORTACIONES para resolver el error de 'ConversationalRetrievalChain'
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import tempfile

# ----------------------------------------------------
# PASO 1: Vincular la Clave de API (Opción 2: Solicitud al Usuario)
# ----------------------------------------------------

# Intentar leer la clave de una variable de entorno
api_key = os.getenv("GOOGLE_API_KEY") 

if not api_key:
    with st.sidebar:
        st.warning("⚠️ Introduce tu clave de API de Gemini para continuar.")
        api_key_input = st.text_input("Clave de API de Google Gemini", type="password")
    
    if api_key_input:
        api_key = api_key_input
    else:
        st.info("Introduce la clave de API en la barra lateral.")
        st.stop()

# Configurar la clave para el resto del script
os.environ["GOOGLE_API_KEY"] = api_key

# --- Configuración de Streamlit ---
st.set_page_config(page_title="Chat con Documentos (RAG + Gemini)", layout="wide")
st.title("📄 Chat Asistente para Documentos")
st.caption("Sube un PDF y haz preguntas basadas en su contenido.")

# --- Funciones de RAG ---

def process_documents(uploaded_file):
    """
    Carga el archivo, lo divide en fragmentos y crea un vector store (ChromaDB).
    """
    if uploaded_file is None:
        return None

    # Guardar el archivo subido temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    try:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        else:
            st.error(f"Tipo de archivo no soportado: {uploaded_file.type}. Solo se aceptan PDF.")
            os.remove(temp_file_path)
            return None
        
        documents = loader.load()

        # División de documentos (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        texts = text_splitter.split_documents(documents)

        # Inicialización de Embeddings 
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004") 
        
        # Creación de Vector Store
        vectorstore = Chroma.from_documents(texts, embeddings)
        
        # Limpiar el archivo temporal
        os.remove(temp_file_path)
        
        return vectorstore.as_retriever()
        
    except Exception as e:
        st.error(f"Error al procesar el documento: {e}")
        if os.path.exists(temp_file_path):
             os.remove(temp_file_path)
        return None


def get_conversation_chain(retriever):
    """
    Crea la cadena de conversación RAG modular (LLM + Retriever con historial).
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # 1. Prompt para entender el historial del chat
    contextualize_q_system_prompt = (
        "Dado el historial de chat y la última pregunta del usuario, "
        "genera una pregunta independiente que pueda ser utilizada para la búsqueda en la base de datos."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Prompt para generar la respuesta final usando el contexto recuperado
    qa_system_prompt = (
        "Eres un asistente experto para la solución de fallas. Usa SOLO el contexto proporcionado "
        "para responder la pregunta. Si no sabes la respuesta basada en el contexto, simplemente di que "
        "no puedes encontrar la información específica en el documento. "
        "Tu objetivo es dar sugerencias de pasos a seguir. \n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Combinar las dos cadenas
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


# --- Lógica de la Aplicación Streamlit ---

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False


# Sidebar para subir el archivo
with st.sidebar:
    st.header("1. Sube tu Documento")
    uploaded_file = st.file_uploader(
        "Sube un archivo PDF",
        type=["pdf"], 
        accept_multiple_files=False
    )
    
    if st.button("Procesar Documento"):
        if uploaded_file:
            with st.spinner("Procesando y creando base de conocimiento..."):
                retriever = process_documents(uploaded_file)
                if retriever:
                    st.session_state.conversation = get_conversation_chain(retriever)
                    st.session_state.chat_history = []
                    st.session_state.processing_done = True
                    st.success("¡Documento procesado! Ahora puedes preguntar en el chat principal.")
                else:
                    st.session_state.processing_done = False
        else:
            st.warning("Por favor, sube un documento primero.")
    
    st.markdown("---")
    st.write("Impulsado por Gemini 2.5 Flash y RAG modular")


# Panel principal de chat
if st.session_state.processing_done:
    st.header(f"🤖 Asistente de Chat: {uploaded_file.name}")
    st.markdown("Ahora puedes hacer preguntas como: *'¿Cuáles son los pasos recomendados para la falla de conectividad?'*")
    
    # Mostrar historial de chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de chat
    user_question = st.chat_input("Escribe tu pregunta aquí...")
    if user_question:
        if st.session_state.conversation:
            # Añadir la pregunta del usuario al historial para LangChain Core
            st.session_state.chat_history.append(HumanMessage(content=user_question))

            with st.chat_message("user"):
                st.markdown(user_question)
                
            # Llamar a la cadena de conversación
            with st.spinner("Gemini está pensando..."):
                # Llamada usando el historial de chat acumulado
                response = st.session_state.conversation.invoke({
                    "input": user_question, 
                    # El historial se pasa ahora como objetos HumanMessage/AIMessage
                    "chat_history": st.session_state.chat_history
                })
            
            # Procesar y mostrar la respuesta del asistente
            assistant_response = response["answer"]
            
            # Añadir la respuesta del asistente al historial
            st.session_state.chat_history.append(AIMessage(content=assistant_response))
            
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

        else:
            st.warning("Por favor, procesa un documento en la barra lateral primero.")

else:
    st.info("Sube un documento PDF en la barra lateral para empezar a chatear.")
