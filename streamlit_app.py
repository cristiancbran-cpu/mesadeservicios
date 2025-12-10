import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import tempfile

# Cargar variables de entorno (para GOOGLE_API_KEY)
load_dotenv()

# --- Configuración de Streamlit ---
st.set_page_config(page_title="Chat con Documentos (RAG + Gemini)", layout="wide")
st.title("📄 Chat Asistente para Documentos")
st.caption("Sube un PDF/Documento y haz preguntas basadas en su contenido.")

# --- Funciones de RAG ---

def process_documents(uploaded_file):
    """
    Carga el archivo, lo divide en fragmentos y crea un vector store (ChromaDB).
    """
    if uploaded_file is None:
        return None

    # Streamlit file_uploader no tiene un path, así que guardamos temporalmente.
    # Esto es necesario para que LangChain pueda cargar el documento.
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    # Nota: Este ejemplo solo maneja PDF (usa PyPDFLoader de LangChain).
    # Para Word (DOCX), necesitarías una librería adicional como Docx2txtLoader.
    # Por simplicidad, nos enfocaremos en PDF.
    try:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        # elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        #     from langchain.document_loaders import Docx2txtLoader # Requeriría instalar docx2txt
        #     loader = Docx2txtLoader(temp_file_path)
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

        # Creación de Embeddings y Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
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
    Crea la cadena de conversación RAG (LLM + Retriever).
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # La cadena de recuperación de conversación combina el historial de chat con la recuperación de documentos
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        # 'memory' se manejará manualmente en st.session_state
    )
    return conversation_chain


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
        type=["pdf"], # Puedes añadir "docx" si implementas Docx2txtLoader
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
    st.write("Impulsado por Gemini 2.5 Flash y RAG (LangChain + Streamlit)")


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
            # Añadir la pregunta del usuario al historial
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
                
            # Llamar a la cadena de conversación
            with st.spinner("Gemini está pensando..."):
                # Llamada usando el historial de chat acumulado
                response = st.session_state.conversation.invoke({
                    "question": user_question, 
                    "chat_history": [(msg["content"], st.session_state.chat_history[i+1]["content"]) 
                                     for i, msg in enumerate(st.session_state.chat_history) 
                                     if msg["role"] == "user" and i+1 < len(st.session_state.chat_history) and st.session_state.chat_history[i+1]["role"] == "assistant"]
                })
            
            # Procesar y mostrar la respuesta del asistente
            assistant_response = response["answer"]
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

        else:
            st.warning("Por favor, procesa un documento en la barra lateral primero.")

else:
    st.info("Sube un documento PDF en la barra lateral para empezar a chatear.")

# Limpiar el path temporal si existe, por si acaso (aunque se limpia en process_documents)
# No es necesario aquí, ya que el manejo de archivos temporales se hace dentro de la función.
