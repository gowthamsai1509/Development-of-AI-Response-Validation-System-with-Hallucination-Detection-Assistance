import datetime
from sqlalchemy import Column, String, Text, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class EvaluationRecord(Base):
    __tablename__ = "evaluation_records"
    
    id = Column(String(36), primary_key=True, index=True)
    question = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    reference_answer = Column(Text, nullable=True)
    source_document = Column(Text, nullable=True)
    
    # Processing & Status
    status = Column(String(50), default="SUBMITTED", nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    # RAG Retrieval Context
    retrieved_contexts = Column(JSON, nullable=True)
    
    # Agent Scores (0.0 to 1.0)
    relevance_score = Column(Float, nullable=True)
    accuracy_score = Column(Float, nullable=True)
    hallucination_score = Column(Float, nullable=True)  # Higher means more hallucination
    completeness_score = Column(Float, nullable=True)
    overall_confidence_score = Column(Float, nullable=True)
    
    # Final Verdict & Explanation
    verdict = Column(String(50), nullable=True)  # e.g., "PASSED", "SUSPECTED_HALLUCINATION", "FAILED"
    agent_feedback = Column(JSON, nullable=True)
