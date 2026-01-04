
import os
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load env vars
load_dotenv()

# --- Configuration Constants ---
PINECONE_INDEX_NAME = "rag-project-index" # We can reuse the same index, just different namespaces
EMBEDDING_DIMENSION = 1536
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "models/gemini-2.0-flash-exp"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"

# --- Initialization Functions ---

def get_llm():
    return GoogleGenAI(model=LLM_MODEL, api_key=os.getenv("GOOGLE_API_KEY"))

def get_embedding_model():
    return OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=os.getenv("OPENAI_API_KEY"))

def get_reranker():
    return SentenceTransformerRerank(model=RERANKER_MODEL, top_n=3)

def initialize_settings():
    """Initializes global LlamaIndex settings."""
    Settings.llm = get_llm()
    Settings.embed_model = get_embedding_model()

def get_pinecone_index():
    """Initializes Pinecone connection and ensures index exists."""
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # Create index if needed
    existing_indexes = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(PINECONE_INDEX_NAME)

def get_vector_store(session_id):
    """Returns a PineconeVectorStore isolated by session_id namespace."""
    pinecone_index = get_pinecone_index()
    return PineconeVectorStore(pinecone_index=pinecone_index, namespace=session_id)

def create_index_from_docs(documents, session_id):
    """Ingests documents into Pinecone using Sentence Window Retrieval."""
    print(f"✨ [Sentence Window Engine] Indexing {len(documents)} docs for session {session_id}")
    initialize_settings()
    vector_store = get_vector_store(session_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the Sentence Window Parser
    # This splits text into sentences and adds the surrounding context to metadata
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    
    # Extract nodes manually (or let VectorStoreIndex do it if we pass transformations)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    return VectorStoreIndex(
        nodes, 
        storage_context=storage_context
    )

def load_index_from_store(session_id):
    """Loads an existing index from Pinecone based on session_id."""
    initialize_settings()
    vector_store = get_vector_store(session_id)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

def create_chat_engine(index):
    """Creates the chat engine with metadata replacement + reranking."""
    reranker = get_reranker()
    
    # The PostProcessor that swaps the single sentence for the full window
    postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )
    
    custom_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above I want you to answer the query.\n"
        "Rules:\n"
        "1. Use markdown formatting (e.g. **bolding** for key terms).\n"
        "2. Keep the tone professional but easy to understand.\n"
        "3. If you don't know the answer, say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        streaming=True,
        similarity_top_k=10, 
        # Order matters! Replace metadata FIRST, then Rerank.
        node_postprocessors=[postprocessor, reranker],
        context_prompt=custom_prompt
    )
