import os
import json
import logging
from typing import List, Dict, Any
from app.config import settings
from app.schemas.knowledge_base import DocumentChunk
from app.knowledge_base.chunker import TextChunker
from app.knowledge_base.embeddings import SemanticFeatureEmbedding
from app.knowledge_base.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)

class BenchmarkIngestionPipeline:
    """
    Ingests public QA benchmark datasets (TruthfulQA, SQuAD),
    cleans and normalizes contexts, applies chunking strategy, generates embeddings,
    and indexes into FAISS vector database.
    """
    def __init__(self):
        self.chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        self.embedder = SemanticFeatureEmbedding(dimension=settings.EMBEDDING_DIM)
        self.vector_store = FaissVectorStore(dimension=settings.EMBEDDING_DIM)

    def load_truthfulqa(self, file_path: str = None) -> List[Dict[str, Any]]:
        path = file_path or (settings.RAW_DATA_DIR / "truthfulqa_sample.json")
        if not os.path.exists(path):
            logger.warning(f"TruthfulQA dataset not found at {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        standardized = []
        for r in records:
            # Construct unified ground-truth knowledge document
            content = (
                f"Question: {r.get('question')}\n"
                f"Verified Answer: {r.get('best_answer')}\n"
                f"Factual Evidence: {r.get('source_context', '')}\n"
                f"Alternative Factual Formulations: {'; '.join(r.get('correct_answers', []))}"
            )
            standardized.append({
                "doc_id": r.get("id", f"tqa_{len(standardized)}"),
                "dataset": "truthfulqa",
                "title": f"TruthfulQA: {r.get('category', 'General')}",
                "text": content,
                "metadata": {
                    "category": r.get("category"),
                    "question": r.get("question"),
                    "best_answer": r.get("best_answer"),
                    "known_hallucinations": r.get("incorrect_answers", [])
                }
            })
        return standardized

    def load_squad(self, file_path: str = None) -> List[Dict[str, Any]]:
        path = file_path or (settings.RAW_DATA_DIR / "squad_sample.json")
        if not os.path.exists(path):
            logger.warning(f"SQuAD dataset not found at {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        standardized = []
        for r in records:
            answers_text = "; ".join([ans.get("text", "") for ans in r.get("answers", []) if ans.get("text")])
            content = (
                f"Topic: {r.get('title')}\n"
                f"Context: {r.get('context')}\n"
                f"Question: {r.get('question')}\n"
                f"Extracted Answer: {answers_text}"
            )
            standardized.append({
                "doc_id": r.get("id", f"squad_{len(standardized)}"),
                "dataset": "squad",
                "title": r.get("title", "SQuAD Comprehension"),
                "text": content,
                "metadata": {
                    "question": r.get("question"),
                    "extracted_answers": answers_text,
                    "title": r.get("title")
                }
            })
        return standardized

    def run_ingestion(self) -> Dict[str, Any]:
        """
        Executes complete ingestion pipeline:
        Dataset extraction -> Standardization -> Chunking -> Embedding -> FAISS Indexing
        """
        tqa_docs = self.load_truthfulqa()
        squad_docs = self.load_squad()
        all_docs = tqa_docs + squad_docs

        if not all_docs:
            raise ValueError("No benchmark documents found to ingest.")

        all_chunks: List[DocumentChunk] = []
        for doc in all_docs:
            chunks = self.chunker.chunk_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                dataset=doc["dataset"],
                title=doc["title"],
                metadata=doc.get("metadata", {})
            )
            all_chunks.extend(chunks)

        # Generate embeddings
        chunk_texts = [c.text for c in all_chunks]
        embeddings = self.embedder.fit(chunk_texts)

        # Store in FAISS
        self.vector_store.initialize_index(embeddings.shape[1])
        self.vector_store.add_chunks(all_chunks, embeddings)
        self.vector_store.save()

        summary = {
            "total_documents": len(all_docs),
            "datasets": {
                "truthfulqa": len(tqa_docs),
                "squad": len(squad_docs)
            },
            "total_chunks": len(all_chunks),
            "embedding_dimension": int(embeddings.shape[1]),
            "index_path": str(self.vector_store.index_file)
        }
        return summary
