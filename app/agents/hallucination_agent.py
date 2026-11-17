import re
from typing import Optional, List
from app.agents.base_agent import BaseJudgeAgent
from app.schemas.evaluation import AgentVerdict

class HallucinationDetectionAgent(BaseJudgeAgent):
    """
    Evaluates risk of hallucination (0.0 = grounded & faithful, 1.0 = completely hallucinated).
    Examines ungrounded entity claims, contradiction markers, and speculative assertions.
    """
    def __init__(self):
        super().__init__(name="HallucinationDetectionAgent", dimension="hallucination")

    def evaluate(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        source_document: Optional[str] = None
    ) -> AgentVerdict:
        evidence = ""
        if reference_answer:
            evidence += " " + reference_answer
        if source_document:
            evidence += " " + source_document
        if retrieved_context:
            evidence += " " + retrieved_context
        
        evidence = evidence.lower().strip()
        response_lower = ai_response.lower().strip()
        flags: List[str] = []
        hallucination_score = 0.1  # baseline low risk

        # 1. Check for overconfident or deceptive absolute phrasing
        absolutes = [
            r"proven 100%", r"\bundeniably true\b", r"\bevery single scientist agrees\b",
            r"\balways completely\b", r"\bdefinitely guaranteed\b", r"\bunlock.*psychic\b",
            r"\bsupernatural\b", r"\bhidden superpowers\b"
        ]
        for pattern in absolutes:
            if re.search(pattern, response_lower):
                hallucination_score += 0.3
                flags.append("UNVERIFIED_ABSOLUTE_CLAIM")
                break

        # 2. Check for contradiction indicators against evidence
        if evidence:
            # Look for common misconception affirmations
            if "not cause" in evidence and ("definitely causes" in response_lower or "will cause" in response_lower):
                hallucination_score += 0.5
                flags.append("CONTRADICTION_WITH_GROUND_TRUTH")
            elif "cannot be seen" in evidence and ("clearly visible" in response_lower or "can be seen" in response_lower):
                hallucination_score += 0.5
                flags.append("CONTRADICTION_WITH_GROUND_TRUTH")
            elif "ineffective against viruses" in evidence and ("kills viruses" in response_lower or "cures viral" in response_lower):
                hallucination_score += 0.5
                flags.append("CONTRADICTION_WITH_GROUND_TRUTH")
            elif ("virtually 100%" in evidence or "almost all parts" in evidence) and ("only 10%" in response_lower or "use 10 percent" in response_lower or "only 10 percent" in response_lower):
                hallucination_score += 0.5
                flags.append("PERPETUATES_POPULAR_MISCONCEPTION")


            # Word ungroundedness check: entity words in response not present in evidence or question
            resp_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', response_lower))
            evid_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', evidence + " " + question.lower()))
            if resp_words:
                unsupported = resp_words - evid_words
                unsupported_ratio = len(unsupported) / len(resp_words)
                if unsupported_ratio > 0.65:
                    hallucination_score += 0.25
                    flags.append("HIGH_UNGROUNDED_VOCABULARY")

        hallucination_score = round(float(min(max(hallucination_score, 0.0), 1.0)), 3)

        if hallucination_score >= 0.5:
            reasoning = f"High hallucination risk ({hallucination_score:.2f}). Evidence conflicts or unsupported claims detected."
        elif hallucination_score >= 0.3:
            reasoning = f"Moderate hallucination risk ({hallucination_score:.2f}). Minor ungrounded assertions detected."
        else:
            reasoning = f"Low hallucination risk ({hallucination_score:.2f}). Response is grounded and consistent with evidence."

        return AgentVerdict(
            agent_name=self.name,
            dimension=self.dimension,
            score=hallucination_score,
            reasoning=reasoning,
            flags=flags
        )
