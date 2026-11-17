import re
from typing import Optional, List
from app.agents.base_agent import BaseJudgeAgent
from app.schemas.evaluation import AgentVerdict

class CompletenessJudgeAgent(BaseJudgeAgent):
    """
    Evaluates whether the response thoroughly addresses all required aspects of the prompt.
    """
    def __init__(self):
        super().__init__(name="CompletenessJudgeAgent", dimension="completeness")

    def evaluate(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        source_document: Optional[str] = None
    ) -> AgentVerdict:
        flags: List[str] = []
        q_len = len(question.split())
        resp_len = len(ai_response.split())

        # Check for brevity failure
        if resp_len < 3:
            return AgentVerdict(
                agent_name=self.name,
                dimension=self.dimension,
                score=0.1,
                reasoning="Response is severely truncated or one word.",
                flags=["EXTREME_BREVITY"]
            )

        # Detect multiple clauses or questions (e.g., and, what, why, how)
        question_marks = question.count("?")
        has_multi_part = question_marks > 1 or (" and " in question.lower() and "?" in question)

        # Sentence structure check
        sentences = [s for s in re.split(r'[.!?]', ai_response) if len(s.strip()) > 3]

        # Entity coverage from question
        q_entities = set(re.findall(r'\b[a-zA-Z]{4,}\b', question.lower()))
        resp_entities = set(re.findall(r'\b[a-zA-Z]{4,}\b', ai_response.lower()))
        entity_coverage = len(q_entities.intersection(resp_entities)) / max(len(q_entities), 1)

        length_factor = min(1.0, resp_len / max(q_len * 1.2, 10))
        
        completeness_score = 0.5 * entity_coverage + 0.3 * length_factor + 0.2 * min(1.0, len(sentences) / 2.0)
        completeness_score = round(float(min(max(completeness_score, 0.0), 1.0)), 3)

        if has_multi_part and completeness_score < 0.6:
            flags.append("POTENTIAL_PARTIAL_ANSWER")
            reasoning = f"Prompt contains multiple parts or detailed inquiry; response provides only partial coverage ({completeness_score:.2f})."
        elif completeness_score < 0.4:
            flags.append("INCOMPLETE_ANSWER")
            reasoning = f"Response is too terse to provide a comprehensive explanation ({completeness_score:.2f})."
        else:
            reasoning = f"Response covers key aspects and entities required by the question ({completeness_score:.2f})."

        return AgentVerdict(
            agent_name=self.name,
            dimension=self.dimension,
            score=completeness_score,
            reasoning=reasoning,
            flags=flags
        )
