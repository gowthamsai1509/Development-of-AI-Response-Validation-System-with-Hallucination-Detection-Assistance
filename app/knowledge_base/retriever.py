import os
from typing import List, Dict, Any, Optional
from app.config import settings
from app.knowledge_base.embeddings import SemanticFeatureEmbedding
from app.knowledge_base.vector_store import FaissVectorStore

class RAGRetriever:
    """
    RAG Semantic Retrieval Engine for grounding LLM responses against benchmark knowledge base.
    """
    def __init__(self):
        self.embedder = SemanticFeatureEmbedding(dimension=settings.EMBEDDING_DIM)
        self.vector_store = FaissVectorStore(dimension=settings.EMBEDDING_DIM)
        self.is_ready = False
        self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if not self.is_ready:
            if os.path.exists(self.embedder.model_path):
                self.embedder.load()
            if self.vector_store.load():
                self.is_ready = True
        return self.is_ready

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        dataset_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self._ensure_loaded():
            return []

        query_vector = self.embedder.embed_query(query)
        matches = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            dataset_filter=dataset_filter
        )
        return matches

    def retrieve_grounding_context(self, question: str, top_k: int = 3) -> str:
        matches = self.retrieve(question, top_k=top_k)
        if not matches:
            return ""
        
        context_parts = []
        for i, m in enumerate(matches, 1):
            context_parts.append(
                f"[Source Evidence {i} - {m['dataset'].upper()}: {m['title']}] (Similarity: {m['similarity_score']:.2f})\n"
                f"{m['text']}"
            )
        return "\n\n".join(context_parts)
