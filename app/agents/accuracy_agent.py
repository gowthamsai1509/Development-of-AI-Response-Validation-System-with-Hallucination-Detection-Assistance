import re
from typing import Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.agents.base_agent import BaseJudgeAgent
from app.schemas.evaluation import AgentVerdict

class AccuracyJudgeAgent(BaseJudgeAgent):
    """
    Evaluates factual correctness against reference ground truth or retrieved RAG context.
    """
    def __init__(self):
        super().__init__(name="AccuracyJudgeAgent", dimension="accuracy")

    def evaluate(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        source_document: Optional[str] = None
    ) -> AgentVerdict:
        ground_truth = ""
        source_label = "None"

        if reference_answer and reference_answer.strip():
            ground_truth = reference_answer.strip()
            source_label = "User Provided Reference Answer"
        elif source_document and source_document.strip():
            ground_truth = source_document.strip()
            source_label = "User Provided Source Document"
        elif retrieved_context and retrieved_context.strip():
            ground_truth = retrieved_context.strip()
            source_label = "RAG Retrieved Benchmark Knowledge"

        flags: List[str] = []

        if not ground_truth:
            return AgentVerdict(
                agent_name=self.name,
                dimension=self.dimension,
                score=0.5,
                reasoning="No ground truth or retrieved context available to verify factual accuracy.",
                flags=["NO_GROUND_TRUTH"]
            )

        # Semantic similarity against ground truth
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            mat = vec.fit_transform([ground_truth, ai_response])
            sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        except Exception:
            sim = 0.5

        # Check entity and key phrase preservation
        gt_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', ground_truth.lower()))
        resp_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', ai_response.lower()))
        common = gt_words.intersection(resp_words)
        coverage = len(common) / max(len(resp_words), 1)

        accuracy_score = round(float(min(max(0.6 * sim + 0.4 * coverage, 0.0), 1.0)), 3)

        if accuracy_score < 0.35:
            flags.append("FACTUAL_DIVERGENCE")
            reasoning = f"Low factual alignment ({accuracy_score:.2f}) verified against {source_label}."
        elif accuracy_score < 0.65:
            flags.append("PARTIAL_FACTUAL_MATCH")
            reasoning = f"Moderate factual consistency ({accuracy_score:.2f}) observed with {source_label}."
        else:
            reasoning = f"High factual accuracy ({accuracy_score:.2f}) verified against {source_label}."

        return AgentVerdict(
            agent_name=self.name,
            dimension=self.dimension,
            score=accuracy_score,
            reasoning=reasoning,
            flags=flags
        )
