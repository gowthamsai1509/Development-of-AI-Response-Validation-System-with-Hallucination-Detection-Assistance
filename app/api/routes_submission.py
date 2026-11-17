import uuid
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import EvaluationRecord
from app.schemas.submission import SubmissionRequest, SubmissionResponse

router = APIRouter(prefix="/api/submissions", tags=["Evaluation Submission"])

@router.post("", response_model=SubmissionResponse, status_code=status.HTTP_201_CREATED)
def submit_for_evaluation(
    submission: SubmissionRequest,
    db: Session = Depends(get_db)
):
    """
    Evaluation Input Module: Single submission interface accepting question, AI response,
    and optional reference answer or source document.
    """
    sub_id = str(uuid.uuid4())
    record = EvaluationRecord(
        id=sub_id,
        question=submission.question,
        ai_response=submission.ai_response,
        reference_answer=submission.reference_answer,
        source_document=submission.source_document,
        status="SUBMITTED",
        created_at=datetime.utcnow()
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return SubmissionResponse(
        submission_id=sub_id,
        status="SUBMITTED",
        message="Evaluation input validated and queued successfully.",
        created_at=record.created_at,
        question=record.question,
        has_reference_answer=bool(record.reference_answer),
        has_source_document=bool(record.source_document)
    )

@router.get("", response_model=List[dict])
def list_submissions(limit: int = 20, db: Session = Depends(get_db)):
    records = db.query(EvaluationRecord).order_by(EvaluationRecord.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "question": r.question,
            "status": r.status,
            "verdict": r.verdict,
            "created_at": r.created_at,
            "overall_score": r.overall_confidence_score
        }
        for r in records
    ]

@router.get("/{submission_id}")
def get_submission(submission_id: str, db: Session = Depends(get_db)):
    record = db.query(EvaluationRecord).filter(EvaluationRecord.id == submission_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Submission ID not found.")
    return {
        "id": record.id,
        "question": record.question,
        "ai_response": record.ai_response,
        "reference_answer": record.reference_answer,
        "source_document": record.source_document,
        "status": record.status,
        "verdict": record.verdict,
        "relevance_score": record.relevance_score,
        "accuracy_score": record.accuracy_score,
        "hallucination_score": record.hallucination_score,
        "completeness_score": record.completeness_score,
        "overall_score": record.overall_confidence_score,
        "created_at": record.created_at,
        "agent_feedback": record.agent_feedback,
        "retrieved_contexts": record.retrieved_contexts
    }
