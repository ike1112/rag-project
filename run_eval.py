
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Disable OTEL to allow Legacy Lens selectors
os.environ["TRULENS_OTEL_TRACING"] = "0"

# Import our RAG engine
import rag_engine
from llama_index.core import VectorStoreIndex

# TruLens Imports (Modern v2.x)
from trulens.core import Tru, Feedback
from trulens.apps.llamaindex import TruLlama
from trulens.providers.openai import OpenAI as OpenAIProvider

try:
    from trulens.feedback import Groundedness
except ImportError:
    try:
        from trulens_eval.feedback import Groundedness
    except ImportError:
        Groundedness = None

load_dotenv()

def run_evaluation():
    print("🚀 Starting RAG Evaluation with TruLens...")
    
    # 1. Initialize TruLens
    # Manually clean up old DB to ensure fresh start
    if os.path.exists("eval_db.sqlite"):
        try:
            os.remove("eval_db.sqlite")
        except:
            pass # If locked, we might fail, but let's try
            
    # Using a new DB file
    tru = Tru(database_url="sqlite:///eval_db.sqlite") 
    # tru.reset_database()  <-- Removing this as it might be causing schema issues
    
    # 2. Setup Feedback Functions (The "Triad")
    provider = OpenAIProvider()
    
    # Try to find Groundedness method
    f_groundedness = None
    if hasattr(provider, "groundedness_measure_with_cot_reasons"):
        f_groundedness = (
            Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(TruLlama.select_source_nodes().node.text)
            .on_output()
        )
    else:
        # Fallback for some versions where it is in a separate class
        print("⚠️ 'groundedness_measure_with_cot_reasons' not found on provider. Skipping Groundedness.")

    # A. Answer Relevance
    f_answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )
    
    # B. Context Relevance
    f_context_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )
    
    feedbacks_list = [f_answer_relevance, f_context_relevance]

    # C. Groundedness
    if f_groundedness:
        feedbacks_list.append(f_groundedness)
    
    # 3. Load RAG Engine
    session_id = None
    if os.path.exists(".latest_session"):
        with open(".latest_session", "r") as f:
            session_id = f.read().strip()
    
    if not session_id:
        print("❌ No existing session found via .latest_session.")
        return

    print(f"🔗 Connecting to Session ID: {session_id}")
    index = rag_engine.load_index_from_store(session_id)
    chat_engine = rag_engine.create_chat_engine(index)
    
    # 4. Wrap with TruLens Recorder
    tru_recorder = TruLlama(
        chat_engine,
        app_id="RAG_Bot_v1",
        feedbacks=feedbacks_list
    )
    
    # 5. Load Test Dataset
    if not os.path.exists("golden_dataset.csv"):
        print("❌ golden_dataset.csv not found.")
        return
        
    df = pd.read_csv("golden_dataset.csv")
    print(f"📊 Loaded {len(df)} test questions.")
    
    import time

    # 6. Run Evaluation Loop
    print("\n⏳ Running Evaluation Loop...")
    with tru_recorder as recording:
        for i, row in df.iterrows():
            question = row['user_input']
            reference = row.get('reference', 'N/A')
            
            print(f"   [{i+1}/{len(df)}] Q: {question}")
            print(f"         Make sure to compare with Ground Truth: {reference[:100]}...")
            
            try:
                # Use .chat() for ChatEngine
                chat_engine.chat(question) 
                
                # Attempt to tag the record with ground truth for the dashboard
                record = recording.get()
                record.meta = {"ground_truth": reference}
                
                # Sleep to avoid Rate Limits (429)
                time.sleep(2) 
                
            except Exception as e:
                print(f"   ⚠️ Error: {e}")

    # 7. Show Results
    print("\n✅ Evaluation Complete!")
    print("Launching Dashboard...")
    tru.run_dashboard()

if __name__ == "__main__":
    run_evaluation()
