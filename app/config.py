import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

class AppConfig(BaseModel):
    PROJECT_NAME: str = "AI Response Validation & Hallucination Detection System"
    VERSION: str = "1.0.0-milestone1"
    DEBUG: bool = True
    
    # Storage paths
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    VECTOR_INDEX_DIR: Path = BASE_DIR / "data" / "processed" / "faiss_index"
    SQLITE_DB_PATH: Path = BASE_DIR / "evaluations.db"
    
    # RAG & Chunking
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    EMBEDDING_DIM: int = 128
    RETRIEVAL_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.25
    
    # Scoring Weights
    WEIGHT_RELEVANCE: float = 0.25
    WEIGHT_ACCURACY: float = 0.35
    WEIGHT_HALLUCINATION: float = 0.25  # Inverted (1 - hallucination score)
    WEIGHT_COMPLETENESS: float = 0.15

settings = AppConfig()
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_INDEX_DIR, exist_ok=True)
