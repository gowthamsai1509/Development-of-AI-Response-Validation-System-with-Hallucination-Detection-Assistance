import os
from pathlib import Path
import joblib
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from app.config import settings

class BaseEmbeddingEngine(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        pass

class SemanticFeatureEmbedding(BaseEmbeddingEngine):
    """
    High-performance semantic dense embedding generator using TF-IDF sublinear n-gram
    representation and Truncated SVD (LSA) projected to unit hypersphere.
    Fully offline, fast, reproducible, and zero external network dependency.
    """
    def __init__(self, dimension: int = 128, model_path: Optional[Path] = None):
        self.dimension = dimension
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            stop_words="english"
        )
        self.svd = None
        self.is_fitted = False
        self.model_path = model_path or (settings.PROCESSED_DATA_DIR / "embedding_model.joblib")


    def fit(self, texts: List[str]):
        if not texts or len(texts) == 0:
            return
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_samples, n_features = tfidf_matrix.shape
        actual_dim = min(self.dimension, n_features, n_samples - 1 if n_samples > 1 else 1)
        actual_dim = max(actual_dim, 2)
        
        self.svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        dense_vectors = self.svd.fit_transform(tfidf_matrix)
        self.is_fitted = True
        self.save()
        return normalize(dense_vectors, norm="l2")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            if os.path.exists(self.model_path):
                self.load()
            else:
                return self.fit(texts)

        tfidf_matrix = self.vectorizer.transform(texts)
        if self.svd is not None:
            dense_vectors = self.svd.transform(tfidf_matrix)
        else:
            dense_vectors = tfidf_matrix.toarray()
            
        normed = normalize(dense_vectors, norm="l2")
        return normed.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vectors = self.embed_documents([query])
        return vectors[0]

    def save(self):
        joblib.dump({
            "vectorizer": self.vectorizer,
            "svd": self.svd,
            "dimension": self.dimension,
            "is_fitted": self.is_fitted
        }, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.vectorizer = data["vectorizer"]
            self.svd = data["svd"]
            self.dimension = data.get("dimension", self.dimension)
            self.is_fitted = data.get("is_fitted", True)
