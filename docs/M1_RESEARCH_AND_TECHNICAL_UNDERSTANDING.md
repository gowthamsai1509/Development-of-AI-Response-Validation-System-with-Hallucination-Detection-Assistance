# Milestone 1.1: Research & Technical Understanding Report

**Project Title:** Development of AI Response Validation System with Hallucination Detection Assistance  
**Program:** Infosys Springboard  
**Author:** Project Developer  
**Date:** November 2026  

---

## 1. Executive Summary

As Large Language Models (LLMs) are increasingly integrated into production decision-making workflows, their susceptibility to **hallucinations** (generating plausible-sounding but factually ungrounded or contradictory outputs) poses significant reliability risks. 

This research report documents the foundational study conducted under **Milestone 1.1**, covering:
1. Automated AI response evaluation techniques & LLM-as-a-Judge paradigms.
2. Hallucination detection taxonomies and key evaluation dimensions (*Relevance, Accuracy, Faithfulness/Hallucination, Completeness*).
3. Retrieval-Augmented Generation (RAG) architecture, semantic embedding models, and vector database indexing.
4. Comparative analysis of existing evaluation frameworks (**RAGAS** and **TruLens**).
5. Public benchmark QA datasets (**TruthfulQA** and **SQuAD**) from Hugging Face.
6. Technology choices and the implementation roadmap for subsequent milestones.

---

## 2. LLM Evaluation Workflows & Automated Response Validation

### 2.1 Automated Evaluation Paradigms
Traditional NLP metrics (BLEU, ROUGE, METEOR) rely strictly on n-gram surface matching against fixed reference texts. While useful for machine translation, they suffer from critical limitations when evaluating generative LLM responses:
- **Paraphrase Blindness:** An accurate response written with synonymous phrasing receives low n-gram overlap.
- **Factuality Insensitivity:** A response can have a 95% ROUGE score while flipping a critical negation (e.g., *"Knuckle cracking does cause arthritis"* vs *"does not"*), leading to catastrophic misjudgments.
- **Reference Dependency:** Traditional metrics cannot evaluate generative responses when ground-truth answers are unavailable.

### 2.2 LLM-as-a-Judge Methodology
The LLM-as-a-Judge approach utilizes specialized prompt topologies and fine-tuned models to act as impartial adjudicators. In our system architecture, we implement a **multi-agent committee of specialized judges**:
- Instead of relying on a single monolithic prompt, distinct agents independently evaluate individual dimensions (*Relevance Judge, Accuracy Judge, Hallucination Detection Agent, Completeness Judge*).
- An **Orchestrator** coordinates parallel execution, preventing evaluation bias and cross-contamination.
- A **Verdict Agent** aggregates individual dimension scores into a unified confidence metric and categorical verdict.

---

## 3. Evaluation Dimensions & Hallucination Taxonomy

### 3.1 Taxonomy of AI Hallucinations
Hallucinations in generative AI are formally categorized into:
1. **Intrinsic Hallucinations:** The generated response directly contradicts the provided source context or established reference answer (e.g., claiming a study concluded the opposite of its actual findings).
2. **Extrinsic Hallucinations:** The generated response introduces claims or named entities that cannot be verified from or are completely absent in the source context or trusted knowledge base.

### 3.2 Core Evaluation Dimensions
Our validation framework assesses four complementary dimensions on a normalized continuous scale of `[0.0, 1.0]`:

| Evaluation Dimension | Focus Question | Description & Objective |
| :--- | :--- | :--- |
| **Relevance** | *Does the AI response answer what was asked?* | Evaluates semantic alignment, topical adherence, and avoidance of off-topic digression. |
| **Accuracy / Factuality** | *Is the information factually correct?* | Cross-examines assertions against ground truth, reference materials, or retrieved benchmark context. |
| **Hallucination Risk** | *Does the response contain fabricated claims?* | Detects ungrounded assertions, contradiction markers, and popular misconception perpetuation (`0.0` = grounded, `1.0` = hallucinated). |
| **Completeness** | *Did the AI address all parts of the query?* | Assesses whether multi-part prompts, constraints, and necessary explanatory nuances are adequately covered. |

---

## 4. RAG Architecture & Vector Indexing

### 4.1 RAG Retrieval Pipeline
Retrieval-Augmented Generation (RAG) grounds both generation and evaluation in an authoritative, external knowledge corpus. The architecture comprises:
1. **Document Chunking:** Splitting raw text into bounded chunks ($C_{size} \approx 400$ characters) with sliding window overlap ($C_{overlap} \approx 50$ characters) to preserve sentence coherence.
2. **Dense Semantic Embeddings:** Transforming textual chunks into normalized vector embeddings $\mathbf{v} \in \mathbb{R}^d$.
3. **Vector Store & Indexing:** Storing vectors in an optimized index (FAISS Flat Inner Product index) where vector inner product corresponds to cosine similarity:
   $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}}$$
4. **Top-$K$ Retrieval:** Querying the index with the user's prompt to retrieve the top $K=3$ most semantically relevant evidence chunks with similarity confidence scores.

---

## 5. Comparative Study: RAGAS vs. TruLens

| Feature / Dimension | RAGAS (Retrieval Augmented Generation Assessment) | TruLens (TruEra) | Our Implemented Architecture |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | Component-level metric framework for RAG evaluation. | Feedback function observability framework. | Multi-Agent Committee Orchestration with RAG Grounding. |
| **Key Metrics** | - Faithfulness<br>- Answer Relevance<br>- Context Precision<br>- Context Recall | - **RAG Triad:**<br>  1. Context Relevance<br>  2. Groundedness<br>  3. Answer Relevance | - Relevance Score (0-1)<br>- Accuracy Score (0-1)<br>- Hallucination Risk (0-1)<br>- Completeness Score (0-1) |
| **Agent Topology** | Programmatic pipeline over LangChain/LlamaIndex. | Recorder wrapper tracking instrumentation traces. | Modular Specialized Judge Agents + Orchestrator + Verdict Agent. |
| **Benchmark Grounding** | Synthetic test data generator based on source docs. | User-supplied queries with golden dataset logging. | Dual benchmark integration: TruthfulQA (misconceptions) + SQuAD (factual extraction). |
| **Execution Mode** | Batch offline evaluation. | Real-time tracing and offline dashboard. | Dual mode: Real-time single submission web UI + REST API + Batch pipeline. |

---

## 6. Public QA Benchmark Datasets Analysis

### 6.1 TruthfulQA (Hugging Face / Lin et al.)
- **Purpose:** Specifically designed to test whether language models mimic human falsehoods, popular myths, and superstitions.
- **Dataset Structure:** 817 questions spanning 38 categories including Health, Nutrition, Neuroscience, Physics, Geography, and Law.
- **Application in Our System:** Seeded in our Reference Knowledge Base as factual ground-truth evidence, with known incorrect/hallucinatory patterns indexed to detect model mimicry of popular misconceptions (e.g., knuckle cracking causing arthritis, 10% brain myth, naked-eye visibility of the Great Wall from the Moon).

### 6.2 SQuAD — Stanford Question Answering Dataset (Hugging Face / Rajpurkar et al.)
- **Purpose:** Reading comprehension benchmark testing extractive and factual QA against reference passage contexts.
- **Dataset Structure:** Passages extracted from Wikipedia accompanied by question-answer pairs with exact character span annotations.
- **Application in Our System:** Seeded in our Reference Knowledge Base to validate chunking, passage boundary retrieval, and factual extraction fidelity.

---

## 7. Technology Stack Selection Justification

1. **FastAPI & Uvicorn:** High-performance asynchronous REST API framework with native Pydantic type validation and automatic OpenAPI Swagger documentation.
2. **Pydantic v2:** Robust schema validation and sanitization for evaluation submissions and structured judge verdicts.
3. **SQLAlchemy & SQLite:** Lightweight, zero-configuration embedded relational database for tracking evaluation submissions, agent scores, and audit logs.
4. **FAISS (Facebook AI Similarity Search) & NumPy:** Fast vector similarity search with dual-backend fallback ensuring 100% operational reliability.
5. **Scikit-Learn (TF-IDF + TruncatedSVD / LSA):** High-speed, reproducible semantic embedding engine providing dense vector representations with zero external network dependency.
6. **Modern HTML5/CSS3/Vanilla JS UI:** Clean, responsive, dependency-free dashboard interface enabling instant submission and real-time visualization of agent verdicts and RAG citations.
