import pytest
import numpy as np
from app.knowledge_base.chunker import TextChunker
from app.knowledge_base.embeddings import SemanticFeatureEmbedding
from app.knowledge_base.vector_store import FaissVectorStore

def test_chunker_splits_correctly():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    long_text = (
        "Quantum computing is a rapidly-emerging technology. "
        "It harnesses the laws of quantum mechanics to solve problems too complex for classical computers. "
        "Qubits can exist in superposition of states. "
        "Entanglement allows qubits to be correlated."
    )
    chunks = chunker.chunk_document(doc_id="test_doc", text=long_text, dataset="test")
    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_id == "test_doc"
        assert len(c.text) > 0

def test_embedding_and_faiss(tmp_path):
    temp_model_path = tmp_path / "test_model.joblib"
    embedder = SemanticFeatureEmbedding(dimension=16, model_path=temp_model_path)
    corpus = [
        "Quantum superposition allows qubits to be in multiple states simultaneously.",
        "Solar photovoltaic panels convert sunlight into electrical power using silicon.",
        "Rayleigh scattering makes the daylight sky appear blue."
    ]
    vectors = embedder.fit(corpus)
    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0

    vector_store = FaissVectorStore(dimension=vectors.shape[1])
    vector_store.index_file = tmp_path / "faiss.index"
    vector_store.meta_file = tmp_path / "metadata.json"

    # Create chunks
    chunker = TextChunker()
    chunks = []
    for i, t in enumerate(corpus):
        chunks.extend(chunker.chunk_document(f"doc_{i}", t, dataset="test"))

    vector_store.initialize_index(vectors.shape[1])
    vector_store.add_chunks(chunks, vectors)

    # Search for solar
    query_vec = embedder.embed_query("photovoltaic solar energy panels")
    matches = vector_store.search(query_vec, top_k=1)
    assert len(matches) == 1
    assert "solar" in matches[0]["text"].lower() or "photovoltaic" in matches[0]["text"].lower()
