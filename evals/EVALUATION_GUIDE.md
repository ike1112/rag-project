
# 🧪 RAG Evaluation Guide

This guide explains how to generate synthetic test data and run automated evaluations for the RAG Chatbot.

## 📋 Prerequisites

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

*Key libraries used: `ragas`, `trulens-core`, `trulens-apps-llamaindex`, `trulens-providers-openai`.*

---

## 💡 Methodology: Why Ragas + TruLens?

We use a hybrid approach to ensure our evaluation is both rigorous and easy to manage:

1.  **Synthetic Data (Ragas)**: Writing test questions for large documents is tedious. Ragas uses an LLM (Guard Model) to scan your PDF and generate complex questions (Reasoning, Multi-hop) automatically. This ensures your test set covers the entire document and isn't biased by human assumptions.
2.  **Evaluation Engine (TruLens)**: While Ragas can also evaluate, **TruLens** excels at *observability*. It logs every step of the chain (Retrieval -> Generation) and provides a visual dashboard to verify **intermediate results** (retrieved chunks) alongside the final answer.

## 📐 The "RAG Triad" Metrics

We evaluate the system using the industry-standard **RAG Triad**. These three metrics check the health of every component in your pipeline:

![RAG Triad Diagram](rag%20triad.png)


1.  **Context Relevance** (Retrieval Quality):
    *   *Question:* "Did I find the right information?"
    *   *Check:* Measures if the retrieved document chunks are actually related to the user's query.
    *   *Failure Mode:* If low, your embedding model or chunking strategy might need improvement.

2.  **Groundedness** (Hallucination Check):
    *   *Question:* "Is the answer supported by the data?"
    *   *Check:* Verifies that every claim in the AI's answer can be found in the retrieved chunks.
    *   *Failure Mode:* If low, the LLM is "hallucinating" or making things up not present in the source text.

3.  **Answer Relevance** (Response Quality):
    *   *Question:* "Did I answer the user's question?"
    *   *Check:* Measures if the final response actually addresses the user's intent.
    *   *Failure Mode:* If low, the LLM might be rambling or being too defensive ("I don't know") even when it has context.

---

## 🚀 Step 1: Generate Test Dataset (Golden Data)

We use **Ragas** to automatically generate complex test questions (Reasoning, Multi-Hop) from your PDF documents.

1.  Place your source PDF in `./docs/agentic-ai-patterns.pdf` (or update path in script).
2.  Run the generation script:
    ```bash
    python generate_test_datasets.py
    ```
3.  **Output**: A file named `golden_dataset.csv` will be created containing the questions and ground truth.

---

## ⚙️ Step 2: Prepare the RAG Application

Before evaluating, the RAG engine needs to index the data so it can answer the questions.

1.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2.  **Upload your PDF** in the browser UI.
3.  Wait for the "Ready to Chat!" message.
4.  This creates a **Session ID** and saves it to a hidden file `.latest_session`.
5.  You can close the Streamlit app now (Ctrl+C).

---

## 📊 Step 3: Run Evaluation (TruLens)

We use **TruLens** to grade the RAG system's answers against the generated questions.

1.  Run the evaluation script:
    ```bash
    python run_eval.py
    ```

2.  **What happens?**
    *   The script reads the Session ID from `.latest_session`.
    *   It loads your indexed documents (Pinecone).
    *   It loops through `golden_dataset.csv` and asks the Chatbot every question.
    *   It uses GPT-4/3.5 to grade the response on:
        *   **Answer Relevance**: Did it answer the user's question?
        *   **Context Relevance**: Did it find the right chunk in the PDF?
        *   *(Optional)* **Groundedness**: Is the answer hallucinated?

3.  **View Results**:
    *   A local dashboard will launch automatically or at `http://localhost:8501`.
    *   You will see Leaderboards and detailed trace logs for every question.

---

### 📉 Baseline Benchmark (Appendix)

Here is an example result from our standard RAG pipeline (v1):

![Baseline Metrics](metrics%20with%20no%20optimization.png)

*Notice the high Relevance (0.8+) but lower Groundedness (0.57). This indicates room for improvement via advanced retrieval techniques.*


