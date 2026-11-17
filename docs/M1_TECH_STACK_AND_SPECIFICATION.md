# Milestone 1: Tech Stack & System Specifications

**Project Title:** Development of AI Response Validation System with Hallucination Detection Assistance  
**Program:** Infosys Springboard  
**Date:** November 2026  

---

## 1. Complete Technology Stack

| Layer / Component | Technology Selected | Version | Purpose / Technical Justification |
| :--- | :--- | :--- | :--- |
| **Backend & Web Framework** | **FastAPI** | `>=0.115.0` | Asynchronous high-performance REST API with automated OpenAPI / Swagger documentation and high throughput. |
| **ASGI Server** | **Uvicorn** | `>=0.30.6` | Lightning-fast ASGI production-grade web server. |
| **Data Validation** | **Pydantic v2** | `>=2.9.0` | Type-safe schema definition, input sanitization, error messages, and response validation. |
| **Database & ORM** | **SQLAlchemy + SQLite** | `>=2.0.30` | Embedded relational database with zero-setup maintenance for storing submissions, evaluations, and audit logs. |
| **Vector Store** | **FAISS (Facebook AI Similarity Search)** | `>=1.9.0` | Dense vector indexing with Inner Product (cosine similarity on unit-normalized vectors). Includes seamless NumPy fallback. |
| **Embedding Generation** | **Scikit-Learn (TF-IDF + TruncatedSVD / LSA)** | `>=1.5.0` | High-fidelity dense semantic vector engine; runs 100% offline with zero external network or paid API dependencies. |
| **Array & Data Processing** | **NumPy & Pandas** | `>=1.26.0` / `>=2.2.0` | High-performance vector transformations, matrix operations, and dataset preprocessing. |
| **Evaluation Benchmarks** | **TruthfulQA & SQuAD** | Hugging Face datasets | TruthfulQA (misconceptions and hallucination patterns) and SQuAD (factual reading comprehension extraction). |
| **Frontend / UI Layer** | **HTML5, CSS3, Vanilla JavaScript** | Modern Standards | Clean, responsive Single Submission Web Interface with real-time scoring meters and test presets. |
| **Test Suite** | **Pytest & Custom Test Runner** | `>=8.3.0` | 100% automated test coverage across input validation, chunking, indexing, retrieval, and multi-agent pipeline. |

---

## 2. API Endpoints Specification

### 2.1 Evaluation Submission Endpoints
- **`POST /api/submissions`**: Single submission interface accepting question, AI response, optional reference answer, optional source document. Persists record and returns submission receipt.
- **`GET /api/submissions`**: Retrieves paginated history of all submitted evaluations.
- **`GET /api/submissions/{id}`**: Retrieves complete details, scores, and retrieved contexts for a specific submission.

### 2.2 Multi-Agent Evaluation Endpoints
- **`POST /api/evaluations/evaluate`**: Submits and immediately triggers end-to-end evaluation pipeline with RAG retrieval, agent scoring, and verdict generation.
- **`POST /api/evaluations/run/{id}`**: Runs or re-runs evaluation for an existing stored submission ID.

### 2.3 Reference Knowledge Base Endpoints
- **`POST /api/knowledge-base/retrieve`**: Performs semantic search across indexed chunks for a query string and returns top-$K$ chunks with similarity scores.
- **`GET /api/knowledge-base/stats`**: Returns vector database index status, indexed document count, and dataset breakdown.
- **`POST /api/knowledge-base/reindex`**: Re-ingests raw benchmark datasets, applies chunking, generates embeddings, and rebuilds the vector index.

---

## 3. Data Schema & Model Specifications

### 3.1 Input Submission Schema (`SubmissionRequest`)
```python
class SubmissionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    ai_response: str = Field(min_length=2, max_length=10000)
    reference_answer: Optional[str] = Field(default=None, max_length=10000)
    source_document: Optional[str] = Field(default=None, max_length=50000)
```

### 3.2 Evaluation Result Schema (`EvaluationResult`)
```python
class EvaluationResult(BaseModel):
    submission_id: str
    status: str
    verdict: VerdictType  # PASSED, SUSPECTED_HALLUCINATION, FACTUALLY_INCORRECT, OFF_TOPIC, INCOMPLETE
    scores: ScoringDimensions  # relevance, accuracy, hallucination, completeness, overall
    agent_analyses: List[AgentVerdict]
    retrieved_evidences: List[RetrievalEvidence]
    timestamp: datetime
```

### 3.3 Database Storage Model (`EvaluationRecord`)
- Table: `evaluation_records`
- Fields: `id`, `question`, `ai_response`, `reference_answer`, `source_document`, `status`, `created_at`, `retrieved_contexts` (JSON), `relevance_score`, `accuracy_score`, `hallucination_score`, `completeness_score`, `overall_confidence_score`, `verdict`, `agent_feedback` (JSON).
