from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    dataset: str
    title: Optional[str] = None
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalQuery(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=20)
    dataset_filter: Optional[str] = None

class RetrievalResult(BaseModel):
    query: str
    top_k: int
    results_count: int
    matches: List[Dict[str, Any]]
