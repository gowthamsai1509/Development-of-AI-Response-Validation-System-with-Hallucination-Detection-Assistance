import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from app.schemas.knowledge_base import DocumentChunk
from app.config import settings

logger = logging.getLogger(__name__)

# Try loading FAISS; if DLL fails on Windows/Python 3.13, fallback gracefully to NumPy vector search
try:
    import faiss
    HAS_FAISS = True
except Exception as e:
    logger.warning(f"FAISS native library not available ({e}). Using high-performance NumPy vector backend.")
    HAS_FAISS = False

class FaissVectorStore:
    """
    Dual-backend Vector Store:
    Uses FAISS IndexFlatIP when native library is loaded, and falls back to
    exact NumPy normalized matrix inner product (cosine similarity) seamlessly.
    """
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.index = None
        self.vectors: Optional[np.ndarray] = None
        self.metadata_store: List[Dict[str, Any]] = []
        self.index_file = settings.VECTOR_INDEX_DIR / "faiss.index"
        self.numpy_vector_file = settings.VECTOR_INDEX_DIR / "vectors.npy"
        self.meta_file = settings.VECTOR_INDEX_DIR / "metadata.json"

    def initialize_index(self, dimension: int):
        self.dimension = dimension
        if HAS_FAISS:
            try:
                self.index = faiss.IndexFlatIP(dimension)
            except Exception:
                self.index = None
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.metadata_store = []

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match.")

        dim = embeddings.shape[1]
        if self.dimension != dim or self.vectors is None:
            self.initialize_index(dim)

        float_embeddings = embeddings.astype(np.float32)
        
        if HAS_FAISS and self.index is not None:
            try:
                self.index.add(float_embeddings)
            except Exception as e:
                logger.warning(f"FAISS add failed: {e}. Switching to NumPy storage.")
                self.index = None

        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = float_embeddings
        else:
            self.vectors = np.vstack([self.vectors, float_embeddings])

        for chunk in chunks:
            self.metadata_store.append(chunk.model_dump())

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        dataset_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        n_total = len(self.metadata_store)
        if n_total == 0:
            return []

        query_arr = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        # 1. Use FAISS if available and loaded
        if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
            try:
                k_search = min(self.index.ntotal, top_k * 3 if dataset_filter else top_k)
                scores, indices = self.index.search(query_arr, k_search)
                matched_scores = scores[0]
                matched_indices = indices[0]
            except Exception:
                matched_scores, matched_indices = self._numpy_search(query_arr[0], top_k, dataset_filter)
        else:
            matched_scores, matched_indices = self._numpy_search(query_arr[0], top_k, dataset_filter)

        results = []
        for score, idx in zip(matched_scores, matched_indices):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            meta = self.metadata_store[idx]
            if dataset_filter and meta.get("dataset") != dataset_filter:
                continue

            results.append({
                "chunk_id": meta.get("chunk_id"),
                "doc_id": meta.get("doc_id"),
                "dataset": meta.get("dataset"),
                "title": meta.get("title"),
                "text": meta.get("text"),
                "similarity_score": float(np.clip(score, 0.0, 1.0)),
                "metadata": meta.get("metadata", {})
            })

            if len(results) >= top_k:
                break

        return results

    def _numpy_search(self, query_vec: np.ndarray, top_k: int, dataset_filter: Optional[str]):
        if self.vectors is None or len(self.vectors) == 0:
            return [], []
        # Normalized vectors: inner product is cosine similarity
        sims = np.dot(self.vectors, query_vec)
        k_search = min(len(sims), top_k * 3 if dataset_filter else top_k)
        top_indices = np.argsort(-sims)[:k_search]
        return sims[top_indices], top_indices

    def save(self):
        os.makedirs(settings.VECTOR_INDEX_DIR, exist_ok=True)
        if HAS_FAISS and self.index is not None:
            try:
                faiss.write_index(self.index, str(self.index_file))
            except Exception as e:
                logger.warning(f"Could not persist FAISS index: {e}")
        
        if self.vectors is not None:
            np.save(str(self.numpy_vector_file), self.vectors)

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, indent=2)

    def load(self) -> bool:
        if not os.path.exists(self.meta_file):
            return False

        with open(self.meta_file, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)

        # Try loading FAISS first
        if HAS_FAISS and os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(str(self.index_file))
                self.dimension = self.index.d
            except Exception:
                self.index = None

        # Load NumPy vectors as well
        if os.path.exists(self.numpy_vector_file):
            self.vectors = np.load(str(self.numpy_vector_file))
            self.dimension = self.vectors.shape[1]

        return len(self.metadata_store) > 0
