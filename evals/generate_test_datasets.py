import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

load_dotenv()

def main():
    print("🚀 Starting Ragas Testset Generation...")
    
    # 1. Load Documents
    # Resolve path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(current_dir, "..", "docs", "agentic-ai-patterns.pdf")
    
    loader = PyPDFLoader(docs_path)
    documents = loader.load()
    
    # 2. Setup LLM & Embeddings (Wrapped for Ragas)
    # Ragas recommends GPT-4 for generation/criticism to ensure quality
    # But using gpt-3.5-turbo to avoid Rate Limit (429) errors on lower tiers
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # 3. Initialize the Ragas Generator
    # This replaces your manual loop and prompt template
    generator = TestsetGenerator(
        llm=generator_llm, 
        embedding_model=generator_embeddings
    )
    
    # 4. Generate the Testset
    # Ragas handles the chunking, node creation, and query evolution automatically
    dataset = generator.generate_with_langchain_docs(documents, testset_size=10)
    
    # 5. Save to CSV
    df = dataset.to_pandas()
    df.to_csv("golden_dataset.csv", index=False)
    print(f"✅ Success! Generated {len(df)} questions with Ragas evolutions.")

if __name__ == "__main__":
    main()