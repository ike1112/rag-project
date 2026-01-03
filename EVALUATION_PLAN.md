# RAG Evaluation Implementation Plan

This plan outlines the steps to integrate **RAGAS** (for synthetic data generation) and **TruLens** (for tracking and evaluation) into your project.

## 📋 Phase 1: Preparation & Dependencies
Before writing code, we need to install the heavy-lifting libraries.
*   **Action**: Update `requirements.txt` and install.
*   **New Tools**: `ragas`, `trulens-eval`, `spacy` (possibly needed for RAGAS chunking).

## 🛠️ Phase 2: Refactoring Core Logic
Currently, your generic RAG logic is locked inside `app.py` and mixed with Streamlit UI code. To run evaluations automatically, we need to extract the "Brain" from the "UI".
*   **Action**: Create `rag_engine.py`.
*   **Task**: Move `load_embedding_model`, `load_reranker`, and the index creation logic into this new file.
*   **Result**: 
    *   `app.py` becomes smaller and just handles the UI.
    *   Our new eval scripts can import the RAG engine directly without launching a website.

## 🧪 Phase 3: Synthetic Data Generation (RAGAS)
We will use RAGAS to read your `docs/dspy.pdf` and automatically create a "Gold Standard" test sheet.
*   **Script**: Create `1_generate_dataset.py`.
*   **Logic**: 
    *   Load PDF.
    *   Use `TestsetGenerator` to create 10-20 QA pairs (Simple and Reasoning types).
    *   Save to `eval_dataset.csv`.

## 📊 Phase 4: Running Evaluation (TruLens)
We will feed the generated questions into your engine and have TruLens record the quality.
*   **Script**: Create `2_run_eval.py`.
*   **Logic**:
    *   Load `eval_dataset.csv`.
    *   Wrap your engine with `TruLlama`.
    *   Define "Feedback Functions" (Triad of Truth):
        1.  **Context Relevance**: Did Pinecone find good stuff?
        2.  **Groundedness**: Did the answer come from the context (or hallucinated)?
        3.  **Answer Relevance**: Did it actually answer the user?
    *   Run the loop.
    *   Launch the Dashboard.

---

**Ready to start? We will execute Phase 1 first.**
