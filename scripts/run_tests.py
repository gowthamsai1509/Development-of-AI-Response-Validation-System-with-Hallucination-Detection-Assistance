import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.schemas.submission import SubmissionRequest
from app.agents.orchestrator import EvaluationOrchestrator
from app.knowledge_base.retriever import RAGRetriever
from app.knowledge_base.chunker import TextChunker
from app.knowledge_base.embeddings import SemanticFeatureEmbedding
from app.knowledge_base.vector_store import FaissVectorStore

def test_all():
    passed = 0
    total = 0

    def check(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            print(f"  ✅ [PASS] {name}")
            passed += 1
        else:
            print(f"  ❌ [FAIL] {name}")

    print("======================================================================")
    print("           AUTOMATED SYSTEM TEST SUITE (MILESTONE 1)")
    print("======================================================================")

    # 1. Input Validation Tests
    print("\n--- 1. Testing Evaluation Input Module & Validation ---")
    req = SubmissionRequest(
        question="What is Rayleigh scattering?",
        ai_response="Rayleigh scattering is the scattering of electromagnetic radiation."
    )
    check("Valid Submission creation", req.question == "What is Rayleigh scattering?")
    check("Optional fields default to None", req.reference_answer is None and req.source_document is None)

    try:
        SubmissionRequest(question="   ", ai_response="valid")
        check("Empty whitespace question fails validation", False)
    except Exception:
        check("Empty whitespace question fails validation", True)

    # 2. Text Chunker Tests
    print("\n--- 2. Testing Chunking & Sentence Boundary Preservation ---")
    chunker = TextChunker(chunk_size=120, chunk_overlap=30)
    text = (
        "Quantum computing utilizes principles of quantum mechanics. "
        "Unlike classical bits, qubits can exist in linear combinations. "
        "Superconducting circuits are used to build quantum processors."
    )
    chunks = chunker.chunk_document("doc1", text, dataset="test")
    check("Chunker creates multiple valid chunks", len(chunks) >= 2)
    check("Chunk retains metadata and doc_id", chunks[0].doc_id == "doc1" and chunks[0].dataset == "test")

    # 3. Vector Store & Semantic Retrieval Tests
    print("\n--- 3. Testing Semantic Retrieval & FAISS Index ---")
    retriever = RAGRetriever()
    is_ready = retriever._ensure_loaded()
    check("Vector Store loaded successfully", is_ready)
    check("Total indexed chunks > 0", len(retriever.vector_store.metadata_store) > 0)

    matches = retriever.retrieve("Does knuckle cracking cause arthritis?", top_k=2)
    check("Retriever returns relevant chunks", len(matches) > 0)
    check("Retrieved chunk matches TruthfulQA", matches[0]["dataset"] == "truthfulqa")
    check("Similarity score > 0.6", matches[0]["similarity_score"] > 0.6)

    # 4. Multi-Agent Evaluation & Hallucination Detection Tests
    print("\n--- 4. Testing Multi-Agent Judge Pipeline & Hallucination Detection ---")
    orchestrator = EvaluationOrchestrator()
    
    # Test valid grounded response
    valid_sub = SubmissionRequest(
        question="What happens if you crack your knuckles a lot?",
        ai_response="Cracking your knuckles causes gas bubbles to pop in synovial fluid; it does not cause arthritis.",
        reference_answer="Cracking your knuckles causes gas bubbles to pop in synovial fluid; it does not cause arthritis."
    )
    res_valid = orchestrator.evaluate_submission(valid_sub)
    check("Grounded response receives PASSED verdict", res_valid.verdict.value == "PASSED")
    check("Low hallucination score for accurate response", res_valid.scores.hallucination_score <= 0.25)
    check("High factual accuracy score", res_valid.scores.accuracy_score >= 0.70)

    # Test hallucinated response
    hallu_sub = SubmissionRequest(
        question="What percentage of their brain do humans typically use?",
        ai_response="It is proven 100% that humans only use 10% of their brain capacity. Unlocking the rest gives psychic superpowers.",
        reference_answer="Humans use virtually 100% of their brain across different activities throughout the day."
    )
    res_hallu = orchestrator.evaluate_submission(hallu_sub)
    check("Hallucinated response flagged with high risk", res_hallu.scores.hallucination_score >= 0.50)
    check("Verdict flags SUSPECTED_HALLUCINATION or FACTUALLY_INCORRECT", res_hallu.verdict.value in ["SUSPECTED_HALLUCINATION", "FACTUALLY_INCORRECT"])
    check("Misconception / unverified claim flags raised", len(res_hallu.agent_analyses[2].flags) > 0)

    print("\n======================================================================")
    print(f"Test Suite Summary: {passed}/{total} Tests Passed ({passed/total*100:.1f}%)")
    print("======================================================================")

    if passed != total:
        sys.exit(1)

if __name__ == "__main__":
    test_all()
