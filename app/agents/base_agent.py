from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.schemas.evaluation import AgentVerdict

class BaseJudgeAgent(ABC):
    """
    Abstract base class for all LLM-as-a-Judge evaluation agents.
    """
    def __init__(self, name: str, dimension: str):
        self.name = name
        self.dimension = dimension

    @abstractmethod
    def evaluate(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        source_document: Optional[str] = None
    ) -> AgentVerdict:
        """
        Executes evaluation for the given dimension and returns structured verdict.
        """
        pass
