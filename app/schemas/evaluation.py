from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class VerdictType(str, Enum):
    PASSED = "PASSED"
    SUSPECTED_HALLUCINATION = "SUSPECTED_HALLUCINATION"
    FACTUALLY_INCORRECT = "FACTUALLY_INCORRECT"
    OFF_TOPIC = "OFF_TOPIC"
    INCOMPLETE = "INCOMPLETE"

class ScoringDimensions(BaseModel):
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Measures if response addresses user query (0.0 - 1.0)")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Measures factual correctness against reference/knowledge base (0.0 - 1.0)")
    hallucination_score: float = Field(..., ge=0.0, le=1.0, description="Probability/extent of fabricated or unfaithful content (0.0 = grounded, 1.0 = completely hallucinated)")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Measures coverage of critical answer components (0.0 - 1.0)")
    overall_confidence_score: float = Field(..., ge=0.0, le=1.0, description="Aggregated confidence metric combining dimensions")

class AgentVerdict(BaseModel):
    agent_name: str
    dimension: str
    score: float
    reasoning: str
    flags: List[str] = Field(default_factory=list)

class RetrievalEvidence(BaseModel):
    chunk_id: str
    dataset: str
    score: float
    snippet: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvaluationResult(BaseModel):
    submission_id: str
    status: str
    verdict: VerdictType
    scores: ScoringDimensions
    agent_analyses: List[AgentVerdict]
    retrieved_evidences: List[RetrievalEvidence]
    timestamp: datetime
