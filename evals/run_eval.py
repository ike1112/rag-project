
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Disable OTEL
os.environ["TRULENS_OTEL_TRACING"] = "0"

# Add parent directory to path so we can import rag_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trulens.core import Tru, Feedback
from trulens.apps.llamaindex import TruLlama
from trulens.providers.openai import OpenAI as OpenAIProvider

load_dotenv()

def run_evaluation():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation")
    parser.add_argument("--mode", choices=["standard", "sentence-window"], default="standard", help="Select RAG Engine")
    args = parser.parse_args()

    print(f"🚀 Starting RAG Evaluation: {args.mode.upper()} Mode")

    # Dynamic Import based on mode
    if args.mode == "sentence-window":
        import rag_engine_sentence_window as active_engine
        app_id = "RAG_Bot_Sentence_Window"
        print("✨ Using Sentence Window Retrieval Engine (baseline)")
    else:
        import rag_engine as active_engine
        app_id = "RAG_Bot_v1"
        print("🔹 Using Standard Engine")

    # 1. Initialize TruLens
    # Using a new DB file approach (appending)
    tru = Tru(database_url="sqlite:///eval_db.sqlite") 
    
    # 2. Define Feedback Functions (RAG Triad)
    provider = OpenAIProvider()
    context_selection = TruLlama.select_source_nodes().node.text

    f_groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(context_selection)
        .on_output()
    )
    f_answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )
    f_context_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    )
    
    feedbacks_list = [f_answer_relevance, f_context_relevance, f_groundedness]

    # 3. Load Session & Engine
    session_file_path = os.path.join(os.path.dirname(__file__), '..', '.latest_session')
    
    if os.path.exists(session_file_path):
        with open(session_file_path, "r") as f:
            session_id = f.read().strip()
    else:
        print("❌ No session ID found. Please run the app and index a document first.")
        return

    print(f"🔗 Connecting to Session ID: {session_id}")
    
    try:
        index = active_engine.load_index_from_store(session_id)
        chat_engine = active_engine.create_chat_engine(index)
    except Exception as e:
        print(f"❌ Error loading engine: {e}")
        return

    # 4. Wrap with TruLens
    tru_recorder = TruLlama(
        chat_engine,
        app_id=app_id,
        feedbacks=feedbacks_list
    )

    # 5. Load Data
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.csv")
    if not os.path.exists(dataset_path):
        print("❌ golden_dataset.csv not found.")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"📊 Loaded {len(df)} questions.")
    
    # 6. Evaluation Loop
    print("\n⏳ Running Evaluation Loop...")
    for i, row in df.iterrows():
        question = row['user_input']
        print(f"   [{i+1}/{len(df)}] Q: {question}")
        
        try:
            # Reset memory
            if hasattr(chat_engine, 'reset'):
                chat_engine.reset()
                
            with tru_recorder as recording:
                chat_engine.chat(question)
                
            print("   💤 Sleeping 10s...")
            time.sleep(10)
            
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

    print("\n✅ Evaluation Complete! Launching Dashboard...")
    tru.run_dashboard()
    
    print("\n   [Dashboard is running in the background]")
    input("   >> Press ENTER to stop the dashboard and exit script...")

if __name__ == "__main__":
    run_evaluation()
