import os

import gc
import tempfile
import uuid
from dotenv import load_dotenv

import streamlit as st
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load env vars
load_dotenv()

# --- Page Config ---
st.set_page_config(layout="wide") 

if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())
    st.session_state.file_cache = {}

session_id = st.session_state.id

# --- Load Models ---
@st.cache_resource
def load_llm():
    return GoogleGenAI(model="models/gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_resource
def load_embedding_model():
    # EMBEDDING MODEL: Converts text into 1536-dimensional vectors.
    # We use OpenAI's 'text-embedding-3-small' for high performance and standard compatibility.
    return OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_reranker():
    # RERANKING: A second-pass filter.
    # After Pinecone finds the top 10 broadly similar chunks, this model (Cross-Encoder)
    # reads them carefully to select the top 3 *truly* relevant ones.
    return SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=3 
    )

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
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Write to temp file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                
                if file_key not in st.session_state.get('file_cache', {}):
                    st.write("Indexing your document...")
                    
                    loader = SimpleDirectoryReader(
                        input_dir=temp_dir,
                        required_exts=[".pdf"],
                        recursive=True
                    )
                    docs = loader.load_data()

                    # Setup LLM & Embeddings
                    Settings.llm = load_llm()
                    Settings.embed_model = load_embedding_model()
                    reranker = load_reranker()

                    # Pinecone Init
                    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                    index_name = "rag-project-index"

                    # Create index if needed
                    existing_indexes = [i.name for i in pc.list_indexes()]
                    if index_name not in existing_indexes:
                        pc.create_index(
                            name=index_name,
                            dimension=1536,
                            # QUERY ALGORITHM: Cosine Similarity.
                            # Why? It measures the angle between vectors (semantic meaning) rather than magnitude.
                            # It is the industry standard for semantic search tasks.
                            metric="cosine", 
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                    
                    pinecone_index = pc.Index(index_name)
                    
                    # VECTOR STORE SETUP:
                    # We use a specific namespace for this session to ensure data isolation.
                    # This prevents one user's document from mixing with another's if they share the index.
                    vector_store = PineconeVectorStore(
                        pinecone_index=pinecone_index, 
                        namespace=session_id 
                    )
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)

                    # EMBEDDING & STORAGE:
                    # 1. Documents are chunked into smaller text blocks.
                    # 2. Each chunk is passed to OpenAIEmbedding to get a vector.
                    # 3. The vector + text is uploaded to Pinecone (Vector Store).
                    index = VectorStoreIndex.from_documents(
                        docs, 
                        storage_context=storage_context
                    )

                    # RETRIEVAL, AUGMENTATION & GENERATION SETUP:
                    # 1. Query Embedding: Your chat question is embedded into a vector.
                    # 2. Retrieval: Pinecone finds top 10 chunks using Cosine Similarity.
                    # 3. Reranking: The Cross-Encoder selects the best 3 chunks.
                    # 4. Augmentation: 'condense_plus_context' injects these 3 chunks into the system prompt.
                    
                    # Custom Prompt Template
                    # This instructs Gemini to behave specifically (e.g., say "I don't know").
                    custom_prompt = PromptTemplate(
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )

                    chat_engine = index.as_chat_engine(
                        chat_mode="condense_plus_context",
                        streaming=True,
                        similarity_top_k=10, 
                        node_postprocessors=[reranker],
                        context_prompt=custom_prompt
                    )
                    
                    st.session_state.file_cache[file_key] = chat_engine
                
                # PERSISTENCE:
                # Store the ready-to-use engine in session_state.
                # This ensures we can access it later in the main app loop without re-running initialization.
                st.session_state.chat_engine = st.session_state.file_cache[file_key]
                st.success("Ready to Chat!")
                
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