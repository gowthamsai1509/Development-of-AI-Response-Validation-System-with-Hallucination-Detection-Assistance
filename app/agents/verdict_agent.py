from typing import List, Dict, Tuple
from app.config import settings
from app.schemas.evaluation import AgentVerdict, VerdictType, ScoringDimensions

class VerdictAgent:
    """
    Synthesizes individual agent dimensions into a final holistic verdict,
    weighted confidence score, and actionable summary.
    """
    def __init__(self):
        self.w_rel = settings.WEIGHT_RELEVANCE
        self.w_acc = settings.WEIGHT_ACCURACY
        self.w_hal = settings.WEIGHT_HALLUCINATION
        self.w_com = settings.WEIGHT_COMPLETENESS

    def compute_verdict(self, agent_verdicts: List[AgentVerdict]) -> Tuple[VerdictType, ScoringDimensions, List[str]]:
        scores: Dict[str, float] = {}
        all_flags: List[str] = []

        for v in agent_verdicts:
            scores[v.dimension] = v.score
            all_flags.extend(v.flags)

        rel = scores.get("relevance", 0.0)
        acc = scores.get("accuracy", 0.0)
        hal = scores.get("hallucination", 0.0)
        com = scores.get("completeness", 0.0)

        groundedness = 1.0 - hal
        overall = (
            self.w_rel * rel +
            self.w_acc * acc +
            self.w_hal * groundedness +
            self.w_com * com
        )
        overall = round(float(min(max(overall, 0.0), 1.0)), 3)

        scoring_dims = ScoringDimensions(
            relevance_score=rel,
            accuracy_score=acc,
            hallucination_score=hal,
            completeness_score=com,
            overall_confidence_score=overall
        )

        # Decision threshold matrix
        if rel < 0.25:
            verdict = VerdictType.OFF_TOPIC
        elif hal >= 0.50:
            verdict = VerdictType.SUSPECTED_HALLUCINATION
        elif acc < 0.40 and hal >= 0.35:
            verdict = VerdictType.SUSPECTED_HALLUCINATION
        elif acc < 0.35:
            verdict = VerdictType.FACTUALLY_INCORRECT
        elif rel < 0.35:
            verdict = VerdictType.OFF_TOPIC
        elif com < 0.30:
            verdict = VerdictType.INCOMPLETE
        elif overall >= 0.60:
            verdict = VerdictType.PASSED
        else:
            verdict = VerdictType.FACTUALLY_INCORRECT


        return verdict, scoring_dims, list(set(all_flags))
