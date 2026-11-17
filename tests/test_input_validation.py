import pytest
from pydantic import ValidationError
from app.schemas.submission import SubmissionRequest

def test_valid_submission():
    payload = {
        "question": "What is Rayleigh scattering?",
        "ai_response": "Rayleigh scattering is the scattering of light by particles in the atmosphere.",
        "reference_answer": "Rayleigh scattering causes shorter blue wavelengths to scatter.",
        "source_document": "Physics textbook chapter 4."
    }
    req = SubmissionRequest(**payload)
    assert req.question == "What is Rayleigh scattering?"
    assert req.reference_answer == "Rayleigh scattering causes shorter blue wavelengths to scatter."

def test_optional_fields_can_be_omitted():
    req = SubmissionRequest(
        question="What is Rayleigh scattering?",
        ai_response="It scatters blue light."
    )
    assert req.reference_answer is None
    assert req.source_document is None

def test_empty_question_fails():
    with pytest.raises(ValidationError):
        SubmissionRequest(
            question="   ",
            ai_response="Some response."
        )

def test_too_short_question_fails():
    with pytest.raises(ValidationError):
        SubmissionRequest(
            question="Hi",
            ai_response="Valid response."
        )
