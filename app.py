import os
import gc
import tempfile
import uuid
from dotenv import load_dotenv

import streamlit as st
from llama_index.core import SimpleDirectoryReader
import rag_engine
import rag_engine_sentence_window


# Load env vars
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Session Management (Auto-Load)
SESSION_FILE = ".latest_session"
if "id" not in st.session_state:
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            st.session_state.id = f.read().strip()
    else:
        st.session_state.id = str(uuid.uuid4())
    st.session_state.file_cache = {}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_engine_choice" not in st.session_state:
    st.session_state.rag_engine_choice = "Standard"

session_id = st.session_state.id

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    if 'chat_engine' in st.session_state and hasattr(st.session_state.chat_engine, 'reset'):
        st.session_state.chat_engine.reset()
    gc.collect()

# --- Sidebar ---
with st.sidebar:
    st.header("🤖 Configuration")
    
    # Engine Selector
    engine_choice = st.radio(
        "Select RAG Strategy:",
        ["Standard", "Sentence Window"],
        index=0 if st.session_state.rag_engine_choice == "Standard" else 1
    )
    st.session_state.rag_engine_choice = engine_choice

    # Determine active module
    if engine_choice == "Standard":
        active_engine = rag_engine

    else:
        active_engine = rag_engine_sentence_window

    st.divider()
    st.subheader("Add Documents")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        file_path = f"./docs/{uploaded_file.name}"
        os.makedirs("./docs", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Process & Index"):
            with st.spinner(f"Indexing with {engine_choice} Strategy..."):
                # Create a NEW session ID for this new index
                new_session_id = str(uuid.uuid4())
                st.session_state.id = new_session_id
                session_id = new_session_id
                
                reader = SimpleDirectoryReader(input_files=[file_path])
                documents = reader.load_data()
                
                # Ingest
                index = active_engine.create_index_from_docs(documents, session_id)
                st.session_state.chat_engine = active_engine.create_chat_engine(index)
                
                # Save session ID for reuse
                with open(SESSION_FILE, "w") as f:
                    f.write(session_id)
                
                st.success(f"Indexed! Session ID: {session_id}")

    st.divider()
    
    # Resume Session
    st.subheader("Resume Session")
    resume_id = st.text_input("Enter Session ID:", value=session_id)
    if st.button("Load Session"):
        if resume_id:
            try:
                index = active_engine.load_index_from_store(resume_id)
                st.session_state.chat_engine = active_engine.create_chat_engine(index)
                st.session_state.id = resume_id
                
                with open(SESSION_FILE, "w") as f:
                    f.write(resume_id)
                    
                st.success(f"Resumed Session: {resume_id}")
            except Exception as e:
                st.error(f"Could not load session: {e}")

# --- Main Layout ---
col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Docs ({engine_choice} RAG)")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is Agentic AI?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if 'chat_engine' in st.session_state:
            try:
                streaming_response = st.session_state.chat_engine.stream_chat(prompt)
                for chunk in streaming_response.response_gen:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("⚠️ No index loaded. Please upload a document or resume a session from the sidebar.")