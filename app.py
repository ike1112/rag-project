import os

import gc
import tempfile
import uuid
from dotenv import load_dotenv

import streamlit as st
import streamlit as st
from llama_index.core import SimpleDirectoryReader
import rag_engine  # Import our new engine

# Load env vars
load_dotenv()

# --- Page Config ---
st.set_page_config(layout="wide") 


# Session Management (Auto-Load)
SESSION_FILE = ".latest_session"
if "id" not in st.session_state:
    # Try to load last session
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            st.session_state.id = f.read().strip()
    else:
        st.session_state.id = str(uuid.uuid4())
    
    st.session_state.file_cache = {}

session_id = st.session_state.id

# --- Load Models ---

# Note: Models are now loaded inside rag_engine.py, but we initialization is handled when needed.


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    # Reset chat engine memory if it exists
    if 'chat_engine' in st.session_state:
        st.session_state.chat_engine.reset()
    gc.collect()



# --- Sidebar ---
with st.sidebar:
    st.header(f"Add your documents!")
    
    # Session Management
    # Check if we have a saved session to default to "Resume"
    default_resume = os.path.exists(SESSION_FILE)
    resume_session = st.checkbox("Resume previous session?", value=default_resume)
    
    uploaded_file = None
    if resume_session:
        existing_id = st.text_input("Enter Session ID to resume:", value=session_id)
        if existing_id:
            st.session_state.id = existing_id
            session_id = existing_id
            # Auto-save manually entered ID
            with open(SESSION_FILE, "w") as f:
                f.write(session_id)
            st.success(f"Resuming session: {session_id}")
    else:
        uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
        st.info(f"Current Session ID: `{session_id}`")

    if uploaded_file or resume_session:
        try:
            # Logic for NEW Upload
            if uploaded_file:
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    start_indexing = True
                    # Write to temp file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # SAVE SESSION ID AUTOMATICALLY
                    with open(SESSION_FILE, "w") as f:
                        f.write(session_id)
                    
                    file_key = f"{session_id}-{uploaded_file.name}"
                    
                    if file_key not in st.session_state.get('file_cache', {}):
                        st.write("Indexing your document...")
                        
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            required_exts=[".pdf"],
                            recursive=True
                        )
                        docs = loader.load_data()

                        # 1. Create Index (Ingest Data)
                        index = rag_engine.create_index_from_docs(docs, session_id)
                        
                        # 2. Create Chat Engine
                        chat_engine = rag_engine.create_chat_engine(index)
                        
                        st.session_state.file_cache[file_key] = chat_engine
                    
                    # PERSISTENCE:
                    st.session_state.chat_engine = st.session_state.file_cache[file_key]
                    st.success("Ready to Chat!")

            # Logic for RESUMING Session (No Upload)
            elif resume_session and session_id:
                # 1. Load Existing Index
                index = rag_engine.load_index_from_store(session_id)
                
                # 2. Create Chat Engine
                st.session_state.chat_engine = rag_engine.create_chat_engine(index)
                
                st.success(f"Connected to session: {session_id}")

                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# --- Main Layout ---
col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Docs using Gemini 2.0")

with col2:
    st.button("Clear ↺", on_click=reset_chat)


if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # VALIDATION:
        # Ensure the chat engine is actually loaded before trying to query it.
        # This prevents errors if a user attempts to chat without uploading a file first.
        if 'chat_engine' in st.session_state:
            try:
                # GENERATION:
                # The Augmented Prompt (User Query + Retrieved Context) is sent to Gemini.
                # Gemini generates the answer using the facts provided in the context.
                streaming_response = st.session_state.chat_engine.stream_chat(prompt)
                for chunk in streaming_response.response_gen:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.error("Please upload a PDF document first!")