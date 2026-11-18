# AI Response Validation System with Hallucination Detection Assistance

[![Infosys Springboard](https://img.shields.io/badge/Infosys%20Springboard-Milestone%201%20%26%202%20Complete-green.svg)](https://springboard.infosys.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal.svg)](https://fastapi.tiangolo.com/)
[![FAISS Vector DB](https://img.shields.io/badge/FAISS-Indexed-orange.svg)](https://github.com/facebookresearch/faiss)
[![Consistency Validation](https://img.shields.io/badge/Consistency%20Test-12%2F12%20(100%25%20Verdict)-success.svg)](./scripts/validate_agent_consistency.py)

An automated AI Response Validation and Hallucination Detection framework built for **Infosys Springboard**. The system evaluates AI-generated responses using a multi-agent **LLM-as-a-Judge** architecture grounded in a **Reference Knowledge Base** seeded with public QA benchmarks (**TruthfulQA** and **SQuAD** from Hugging Face).

---

## 📋 Milestone Completion Overview

### Milestone 1: Foundation & Evaluation Understanding
- **M1.1 — Research & Technical Understanding:** Studied LLM evaluation techniques, hallucination taxonomies, RAG architectures, RAGAS, and TruLens. Documented in [`docs/M1_RESEARCH_AND_TECHNICAL_UNDERSTANDING.md`](./docs/M1_RESEARCH_AND_TECHNICAL_UNDERSTANDING.md).
- **M1.2 — System Architecture:** Defined multi-agent committee, scoring scales, weights, and decision matrix in [`docs/M1_SYSTEM_ARCHITECTURE.md`](./docs/M1_SYSTEM_ARCHITECTURE.md).
- **M1.3 — Evaluation Input Module:** Built single submission interface (Web UI & REST API) with Pydantic v2 validation and SQLite persistence.
- **M1.4 — Reference Knowledge Base:** Ingested TruthfulQA and SQuAD, implemented sliding-window chunking, dense semantic embeddings, and FAISS vector index with 100% retrieval accuracy.

### Milestone 2: Evaluation Judge Agents & Consistency Validation
- **M2.1 — Relevance Judge Agent:** Evaluates prompt alignment with clear rubrics (`FULLY_RELEVANT`, `PARTIALLY_RELEVANT`, `PERIPHERAL`, `OFF_TOPIC`) and topic extraction.
- **M2.2 — Accuracy Judge Agent:** Cross-references factual correctness against reference answers or RAG-retrieved chunks, outputting supporting and contradicting evidence.
- **M2.3 — Hallucination Detection Agent:** Implements **claim-level decomposition**, breaking responses into individual factual propositions, cross-referencing each against RAG evidence, and flagging specific unsupported or contradictory statements with granular reasoning.
- **M2.4 — Agent Evaluation & Consistency Validation:** Built 12-case benchmark consistency dataset across 8 response categories, achieving **100% recall on hallucination detection** and **100% verdict accuracy** via [`scripts/validate_agent_consistency.py`](./scripts/validate_agent_consistency.py). Documented in [`docs/M2_AGENT_SPECIFICATIONS_AND_VALIDATION.md`](./docs/M2_AGENT_SPECIFICATIONS_AND_VALIDATION.md).


---

## 🏛️ System Architecture

```mermaid
graph TD
    %% User and UI Layer
    subgraph UI_Layer ["1. Evaluation Input / User Interface Layer"]
        User(["User / Evaluator"])
        WebUI["Web Evaluation Dashboard (HTML5 / CSS3 / JS)"]
        PresetSelector["1-Click Test Presets (TruthfulQA / SQuAD / Hallucination)"]
    end

    %% Backend API Layer
    subgraph API_Layer ["2. Backend / API Layer (FastAPI)"]
        API_Gateway["FastAPI Gateway / Router"]
        InputValidator["Evaluation Input Processing & Pydantic Validation"]
        SubEndpoints["Submission Endpoints (/api/submissions)"]
        EvalEndpoints["Evaluation Endpoints (/api/evaluations)"]
        KBEndpoints["Knowledge Base Endpoints (/api/knowledge-base)"]
    end

    %% Storage & Submission Database
    subgraph Storage_Layer ["3. Evaluation Submission & Storage Layer"]
        SubDB[("Evaluation Submission Database (SQLite / SQLAlchemy)")]
        SubHistory["Evaluation Records & Historical Audit Logs"]
    end

    %% Reference Knowledge Base (RAG)
    subgraph RAG_Layer ["4. Reference Knowledge Base & RAG Pipeline"]
        RawData["Public QA Benchmarks (TruthfulQA + SQuAD)"]
        IngestionMod["Benchmark Dataset Ingestion Module"]
        CleaningPipeline["Data Cleaning & Standardization"]
        ChunkerPipeline["Recursive Text Chunking Pipeline (400 chars, 50 overlap)"]
        EmbeddingGen["Semantic Embedding Generation Module (Dense LSA / TF-IDF)"]
        VectorDB[("FAISS Vector Database / Flat IP Index")]
        MetadataStore[("Chunk Metadata Repository (JSON Store)")]
        RAGRetriever["RAG Retrieval Pipeline (Top-K Semantic Search)"]
        KBUpdateMod["Knowledge Base Update & Re-indexing Module"]
    end

    %% Multi-Agent Evaluation Committee Layer
    subgraph Agent_Layer ["5. AI Evaluation Agent Layer (LLM-as-a-Judge Committee)"]
        Orchestrator["Agent Orchestrator"]
        RelevanceAgent["Relevance Judge Agent (0.0 - 1.0)"]
        AccuracyAgent["Accuracy Judge Agent (0.0 - 1.0)"]
        HallucinationAgent["Hallucination Detection Agent (0.0 - 1.0)"]
        CompletenessAgent["Completeness Judge Agent (0.0 - 1.0)"]
        VerdictAgent["Verdict Agent (Aggregate Confidence & Decision Matrix)"]
    end

    %% Results and Output Layer
    subgraph Output_Layer ["6. Structured Evaluation Results & Dashboard"]
        ScoringVerdictMod["Evaluation Scoring & Verdict Module"]
        StructuredResultsMod["Structured Evaluation Results Module (JSON / Pydantic)"]
        ReportGenMod["Evaluation Report Generation Module"]
        BatchEvalMod["Batch Evaluation Module"]
        DashboardView["Results & Evaluation Dashboard Interface"]
    end

    %% Data Flow Connections
    User -->|1. Enters Question, AI Response, Optional Ref/Source| WebUI
    PresetSelector -.-> WebUI
    WebUI -->|HTTP POST Request| API_Gateway
    API_Gateway --> InputValidator
    InputValidator --> SubEndpoints
    InputValidator --> EvalEndpoints
    
    SubEndpoints -->|Store Input Record| SubDB
    SubDB --> SubHistory

    %% Knowledge Base Ingestion Flow
    RawData --> IngestionMod
    IngestionMod --> CleaningPipeline
    CleaningPipeline --> ChunkerPipeline
    ChunkerPipeline --> EmbeddingGen
    EmbeddingGen --> VectorDB
    ChunkerPipeline -.-> MetadataStore
    KBUpdateMod -.-> IngestionMod
    KBEndpoints --> KBUpdateMod

    %% Retrieval & Orchestration Flow
    EvalEndpoints --> Orchestrator
    Orchestrator -->|Query Semantic Match| RAGRetriever
    RAGRetriever -->|Search Query Vector| VectorDB
    VectorDB -->|Top-K Chunks| MetadataStore
    MetadataStore -->|Return Grounded Evidence| RAGRetriever
    RAGRetriever -->|Provide Source Context| Orchestrator

    %% Agent Dispatch Flow
    Orchestrator -->|Dispatch Query + Response| RelevanceAgent
    Orchestrator -->|Dispatch Ground Truth + Response| AccuracyAgent
    Orchestrator -->|Dispatch Evidence + Response| HallucinationAgent
    Orchestrator -->|Dispatch Prompt + Response| CompletenessAgent

    %% Verdict Flow
    RelevanceAgent -->|Relevance Score & Flags| VerdictAgent
    AccuracyAgent -->|Accuracy Score & Flags| VerdictAgent
    HallucinationAgent -->|Hallucination Risk & Flags| VerdictAgent
    CompletenessAgent -->|Completeness Score & Flags| VerdictAgent

    VerdictAgent --> ScoringVerdictMod
    ScoringVerdictMod --> StructuredResultsMod
    StructuredResultsMod -->|Update Evaluation Record| SubDB
    StructuredResultsMod --> ReportGenMod
    StructuredResultsMod --> DashboardView
    BatchEvalMod -.-> Orchestrator
    DashboardView -->|Render Results, Meters, Citations| WebUI
```

---

## 🛠️ Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Backend & Web Framework** | **FastAPI**, **Uvicorn**, **Pydantic v2** |
| **Database & ORM** | **SQLAlchemy**, **SQLite** |
| **Vector Store & Indexing** | **FAISS (IndexFlatIP)** with seamless **NumPy** fallback |
| **Semantic Embeddings** | **Scikit-Learn (TF-IDF + TruncatedSVD / LSA)** dense vector space (100% offline & reproducible) |
| **Reference Benchmarks** | **TruthfulQA** (misconceptions) & **SQuAD** (factual reading comprehension) from Hugging Face |
| **Frontend UI** | **HTML5**, **CSS3**, **Vanilla JavaScript** (Responsive Dashboard with glassmorphism styling) |
| **Testing & Quality** | **Pytest** and automated regression test runner |

---

## 📊 Evaluation Scoring Dimensions

Our multi-agent committee evaluates responses across 4 core dimensions normalized to `[0.0, 1.0]`:

| Dimension | Weight | Description | Threshold / Target |
| :--- | :---: | :--- | :--- |
| **Relevance** | `0.25` | Does response directly address the question intent? | $\ge 0.50$ (Relevance) |
| **Accuracy** | `0.35` | Factual correctness against reference / knowledge base. | $\ge 0.65$ (Factual Match) |
| **Hallucination Risk** | `0.25` | Extent of fabricated or ungrounded claims. | $\le 0.25$ (Grounded) |
| **Completeness** | `0.15` | Coverage of essential entities and question facets. | $\ge 0.50$ (Complete) |

$$\text{Overall Confidence Score} = 0.25 \times \text{Relevance} + 0.35 \times \text{Accuracy} + 0.25 \times (1.0 - \text{Hallucination}) + 0.15 \times \text{Completeness}$$

### Final Verdict Classification:
- **`PASSED`**: Response is accurate, relevant, complete, and grounded in evidence.
- **`SUSPECTED_HALLUCINATION`**: High risk ($\ge 0.50$) of ungrounded or contradictory assertions.
- **`FACTUALLY_INCORRECT`**: Response contradicts reference answer or benchmark knowledge.
- **`OFF_TOPIC`**: Response does not address prompt's question.
- **`INCOMPLETE`**: Response fails to provide sufficient explanation.

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/gowthamsai1509/Development-of-AI-Response-Validation-System-with-Hallucination-Detection-Assistance.git
cd Development-of-AI-Response-Validation-System-with-Hallucination-Detection-Assistance


# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Seed the Reference Knowledge Base
Ingests TruthfulQA and SQuAD benchmark datasets, applies sliding window chunking, generates embeddings, and creates the FAISS vector index:
```bash
python scripts/seed_knowledge_base.py
```

### 3. Validate Semantic Retrieval Quality
Runs benchmark queries to verify top-$K$ semantic retrieval accuracy:
```bash
python scripts/validate_retrieval.py
```

### 4. Run the Automated Test Suite & Consistency Validation
```bash
# Run unit & integration test suite (16/16 passed)
python scripts/run_tests.py

# Run Milestone 2 agent consistency validation across 12 benchmark cases
python scripts/validate_agent_consistency.py
```


### 5. Launch the Web Application
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Web UI:** Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **Interactive Swagger API Docs:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 📂 Project Repository Structure

```
Development-of-AI-Response-Validation-System-with-Hallucination-Detection-Assistance/
├── README.md                                # Project overview, setup, and submission report
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore configuration
│
├── docs/                                    # Detailed project documentation
│   ├── M1_RESEARCH_AND_TECHNICAL_UNDERSTANDING.md # Milestone 1.1 Research Report
│   ├── M1_SYSTEM_ARCHITECTURE.md            # Milestone 1.2 System Architecture & Flow
│   ├── M1_TECH_STACK_AND_SPECIFICATION.md   # Complete Tech Stack Specifications
│   └── diagrams/
│       └── architecture_diagram.mermaid     # Complete Mermaid architecture source
│
├── data/
│   ├── raw/                                 # Raw public QA benchmark datasets
│   │   ├── truthfulqa_sample.json           # TruthfulQA benchmark subset
│   │   └── squad_sample.json                # SQuAD reading comprehension subset
│   └── processed/
│       └── faiss_index/                     # FAISS vector index & metadata repository
│           ├── faiss.index                  # Binary vector index
│           ├── vectors.npy                  # NumPy vector array backup
│           └── metadata.json                # Chunk metadata and text store
│
├── app/                                     # Core Application Package
│   ├── config.py                            # Central configuration & hyperparameters
│   ├── main.py                              # FastAPI application & route mounting
│   ├── database/                            # SQLite persistence layer
│   │   ├── connection.py                    # SQLAlchemy session & engine
│   │   └── models.py                        # EvaluationRecord ORM model
│   ├── schemas/                             # Pydantic data models & contracts
│   │   ├── submission.py                    # Input validation & submission schemas
│   │   ├── evaluation.py                    # Scores, AgentVerdict, and EvaluationResult
│   │   └── knowledge_base.py                # Chunk and retrieval schemas
│   ├── knowledge_base/                      # RAG Knowledge Base & Indexing
│   │   ├── chunker.py                       # Recursive sliding window text chunker
│   │   ├── embeddings.py                    # Dense semantic vector generator (LSA/TF-IDF)
│   │   ├── vector_store.py                  # FAISS & NumPy dual-backend vector store
│   │   ├── ingestion.py                     # Benchmark dataset ingestion pipeline
│   │   └── retriever.py                     # Top-K semantic retrieval engine
│   ├── agents/                              # Multi-Agent Judge Committee
│   │   ├── base_agent.py                    # Abstract BaseJudgeAgent
│   │   ├── relevance_agent.py               # Relevance Judge Agent
│   │   ├── accuracy_agent.py                # Accuracy Judge Agent
│   │   ├── hallucination_agent.py           # Hallucination Detection Agent
│   │   ├── completeness_agent.py            # Completeness Judge Agent
│   │   ├── verdict_agent.py                 # Verdict Agent & Decision Matrix
│   │   └── orchestrator.py                  # Evaluation Orchestrator
│   ├── api/                                 # REST API Routers
│   │   ├── routes_submission.py             # Single submission endpoints
│   │   ├── routes_evaluation.py             # Multi-agent evaluation endpoints
│   │   └── routes_knowledge_base.py         # Knowledge base & retrieval endpoints
│   └── static/                              # Evaluation Input Module UI
│       ├── index.html                       # Modern evaluation dashboard
│       ├── style.css                        # Glassmorphism styling & status badges
│       └── app.js                           # Frontend API bindings & presets
│
├── scripts/                                 # Executable CLI Scripts
│   ├── seed_knowledge_base.py               # Ingests & indexes benchmark datasets
│   ├── validate_retrieval.py                # Semantic retrieval quality validation
│   └── run_tests.py                         # Automated 16-point test suite
│
└── tests/                                   # Pytest Unit & Integration Tests
    ├── test_input_validation.py             # Input sanitization & edge cases
    ├── test_knowledge_base.py               # Chunking & vector index unit tests
    └── test_evaluation_flow.py              # End-to-end evaluation flow tests
```

---

## 📌 Submission Summary for Milestone 1

- **GitHub Repository Link:** [https://github.com/gowthamsai1509/Development-of-AI-Response-Validation-System-with-Hallucination-Detection-Assistance](https://github.com/gowthamsai1509/Development-of-AI-Response-Validation-System-with-Hallucination-Detection-Assistance)
- **Interactive Web Interface:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **System Architecture:** Detailed architecture diagram and agent specifications available at [`docs/M1_SYSTEM_ARCHITECTURE.md`](./docs/M1_SYSTEM_ARCHITECTURE.md).
- **Tech Stack:** Comprehensive tech stack breakdown and specification available at [`docs/M1_TECH_STACK_AND_SPECIFICATION.md`](./docs/M1_TECH_STACK_AND_SPECIFICATION.md).
- **Automated Testing & Validation:** 100% test pass rate across all modules verified by [`scripts/run_tests.py`](./scripts/run_tests.py).

