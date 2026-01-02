# Gemini RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) application that allows you to upload a PDF document and chat with it using Google's advanced **Gemini 2.0** model.

## 🌟 Features

*   **PDF Upload**: Upload any PDF document to chat with its content.
*   **Intelligent Indexing**: Uses **OpenAI Embeddings** to create a searchable vector index of your document's content.
*   **Advanced AI Chat**: Powered by **Google Gemini 2.0 Flash Experimental**, enabling high-quality, context-aware answers.
*   **Streaming Responses**: Answers are streamed in real-time for a responsive user experience.
*   **Context-Aware Reranking**: Uses a **Cross-Encoder model** to re-evaluate retrieval results, ensuring the AI only sees the most relevant information.
*   **Chat History**: Maintains the context of your conversation ("memory") so you can ask follow-up questions naturally.


## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Orchestration**: [LlamaIndex](https://www.llamaindex.ai/)
*   **LLM**: Google Gemini (`gemini-2.0-flash-exp`)

*   **Embeddings**: OpenAI (`text-embedding-3-small`)
*   **Vector Database**: [Pinecone](https://www.pinecone.io/)
*   **Reranking**: Sentence Transformers (`cross-encoder/ms-marco-MiniLM-L-2-v2`)

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Clone/Download
Navigate to the project directory where `app.py` is located.

### 3. Create Virtual Environment (Recommended)
Isolate your dependencies to avoid conflicts.

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
You need to install the required Python libraries. Run the following command:

```bash
pip install -r requirements.txt
```

### 5. Configure API Keys
This application requires API keys from Google, OpenAI, and Pinecone.

**Option A (Environment Variables - Recommended):**
Set these variables in your terminal or a `.env` file before running the app.

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your_google_api_key_here"
$env:OPENAI_API_KEY="your_openai_api_key_here"
$env:PINECONE_API_KEY="your_pinecone_api_key_here"
```

**Mac/Linux:**
```bash
export GOOGLE_API_KEY="your_google_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"
export PINECONE_API_KEY="your_pinecone_api_key_here"
```

### 6. Run the Application
Launch the Streamlit app with the following command:

```bash
streamlit run app.py
```

## 📖 How to Use

1.  **Launch the App**: The app should open automatically in your browser (usually at `http://localhost:8501`).
2.  **Upload a PDF**: Use the sidebar on the left to browse and upload a PDF file.
3.  **Wait for Indexing**: The app will process the file (this gathers the knowledge). You will see a "Ready to Chat!" message when done.
4.  **Ask Questions**: Type your query in the chat input box at the bottom. The AI will answer based strictly on the content of your uploaded PDF.
5.  **Clear Chat**: Use the "Clear ↺" button to restart the conversation.
