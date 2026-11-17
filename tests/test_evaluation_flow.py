import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base
from app.schemas.submission import SubmissionRequest
from app.api.routes_submission import submit_for_evaluation, list_submissions, get_submission
from app.api.routes_evaluation import evaluate_submission
from app.api.routes_knowledge_base import get_knowledge_base_stats, retrieve_knowledge
from app.schemas.knowledge_base import RetrievalQuery
from app.agents.orchestrator import EvaluationOrchestrator

# Setup in-memory test database
TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

def test_knowledge_base_stats():
    stats = get_knowledge_base_stats()
    assert stats["status"] == "READY"
    assert stats["total_chunks_indexed"] > 0
    assert "truthfulqa" in stats["datasets"]

def test_knowledge_base_retrieval():
    query_obj = RetrievalQuery(query="quantum processors trapped ions", top_k=2)
    res = retrieve_knowledge(query_obj)
    assert res.results_count > 0
    assert len(res.matches) > 0

def test_submission_endpoint(db_session):
    payload = SubmissionRequest(
        question="Does cracking your knuckles cause arthritis?",
        ai_response="Cracking knuckles does not cause arthritis; it pops gas bubbles.",
        reference_answer="No arthritis is caused by knuckle cracking."
    )
    res = submit_for_evaluation(submission=payload, db=db_session)
    assert res.submission_id is not None
    assert res.status == "SUBMITTED"
    assert res.has_reference_answer is True

    # Check listing
    submissions = list_submissions(limit=10, db=db_session)
    assert len(submissions) == 1
    assert submissions[0]["id"] == res.submission_id

    # Check detail
    detail = get_submission(res.submission_id, db=db_session)
    assert detail["question"] == payload.question

def test_multi_agent_evaluation(db_session):
    orchestrator = EvaluationOrchestrator()
    payload = SubmissionRequest(
        question="What percentage of the brain do humans use?",
        ai_response="It is proven 100% that humans only use 10% of their brain.",
        reference_answer="Humans use virtually 100% of their brain."
    )
    result = orchestrator.evaluate_submission(submission=payload, db=db_session)
    assert result.status == "SUCCESS"
    assert result.verdict.value in ["SUSPECTED_HALLUCINATION", "FACTUALLY_INCORRECT"]
    assert result.scores.hallucination_score >= 0.3
    assert len(result.agent_analyses) == 4
    assert len(result.retrieved_evidences) > 0
