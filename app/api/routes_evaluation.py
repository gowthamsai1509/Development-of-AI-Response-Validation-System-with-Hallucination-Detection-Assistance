from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.schemas.submission import SubmissionRequest
from app.schemas.evaluation import EvaluationResult
from app.agents.orchestrator import EvaluationOrchestrator

router = APIRouter(prefix="/api/evaluations", tags=["Evaluation Pipeline"])
orchestrator = EvaluationOrchestrator()

@router.post("/evaluate", response_model=EvaluationResult)
def evaluate_submission(
    submission: SubmissionRequest,
    db: Session = Depends(get_db)
):
    """
    Submits and immediately executes full multi-agent evaluation pipeline with RAG grounding.
    """
    try:
        result = orchestrator.evaluate_submission(submission=submission, db=db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation pipeline error: {str(e)}")

@router.post("/run/{submission_id}", response_model=EvaluationResult)
def run_existing_submission(
    submission_id: str,
    db: Session = Depends(get_db)
):
    """
    Executes evaluation pipeline for a previously submitted input record.
    """
    from app.database.models import EvaluationRecord
    record = db.query(EvaluationRecord).filter(EvaluationRecord.id == submission_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Submission record not found.")

    sub_req = SubmissionRequest(
        question=record.question,
        ai_response=record.ai_response,
        reference_answer=record.reference_answer,
        source_document=record.source_document
    )
    result = orchestrator.evaluate_submission(submission=sub_req, db=db, submission_id=submission_id)
    return result
