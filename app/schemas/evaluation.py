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

class RelevanceLevel(str, Enum):
    FULLY_RELEVANT = "FULLY_RELEVANT"
    PARTIALLY_RELEVANT = "PARTIALLY_RELEVANT"
    PERIPHERAL = "PERIPHERAL"
    OFF_TOPIC = "OFF_TOPIC"

class AccuracyLevel(str, Enum):
    CORRECT = "CORRECT"
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
    INCORRECT = "INCORRECT"
    CONTRADICTORY = "CONTRADICTORY"

class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"

class ClaimAnalysis(BaseModel):
    claim_id: int
    statement: str
    status: ClaimStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    evidence_quote: Optional[str] = None

class RelevanceDetail(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    level: RelevanceLevel
    reasoning: str
    matched_topics: List[str] = Field(default_factory=list)
    missing_aspects: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)

class AccuracyDetail(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    level: AccuracyLevel
    reasoning: str
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    source_used: str = "None"
    flags: List[str] = Field(default_factory=list)

class HallucinationDetail(BaseModel):
    hallucination_score: float = Field(..., ge=0.0, le=1.0)
    is_hallucinated: bool
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    claim_breakdown: List[ClaimAnalysis] = Field(default_factory=list)
    flagged_statements: List[str] = Field(default_factory=list)
    reasoning: str
    flags: List[str] = Field(default_factory=list)

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
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    relevance_detail: Optional[RelevanceDetail] = None
    accuracy_detail: Optional[AccuracyDetail] = None
    hallucination_detail: Optional[HallucinationDetail] = None
    timestamp: datetime
