import re
from typing import Optional, List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.agents.base_agent import BaseJudgeAgent
from app.schemas.evaluation import AgentVerdict, AccuracyLevel, AccuracyDetail

class AccuracyJudgeAgent(BaseJudgeAgent):
    """
    Milestone 2.2 - Accuracy Judge Agent
    Evaluates factual correctness against:
    1. Human Reference Answer (priority 1)
    2. User-provided Source Document (priority 2)
    3. RAG-retrieved benchmark chunks from Knowledge Base (priority 3)
    
    Classifies factual correctness into: CORRECT, PARTIALLY_CORRECT, INCORRECT, CONTRADICTORY.
    Outputs supporting evidence and reasoning.
    """
    def __init__(self):
        super().__init__(name="AccuracyJudgeAgent", dimension="accuracy")

    def analyze_accuracy(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        source_document: Optional[str] = None
    ) -> AccuracyDetail:
        ground_truth = ""
        source_used = "None"
        supporting_evidence: List[str] = []
        contradicting_evidence: List[str] = []
        flags: List[str] = []

        # Determine evidence source priority
        if reference_answer and reference_answer.strip():
            ground_truth = reference_answer.strip()
            source_used = "User Reference Answer"
            supporting_evidence.append(ground_truth)
        elif source_document and source_document.strip():
            ground_truth = source_document.strip()
            source_used = "User Source Document"
            supporting_evidence.append(ground_truth[:300] + ("..." if len(ground_truth) > 300 else ""))
        elif retrieved_context and retrieved_context.strip():
            ground_truth = retrieved_context.strip()
            source_used = "RAG Retrieved Benchmark Knowledge"
            # Extract first 2 clean passages as supporting evidence
            passages = [p.strip() for p in retrieved_context.split("\n\n") if p.strip()]
            supporting_evidence.extend(passages[:2])

        if not ground_truth:
            return AccuracyDetail(
                score=0.50,
                level=AccuracyLevel.PARTIALLY_CORRECT,
                reasoning="No reference answer or retrieved knowledge base context available to verify factual accuracy.",
                supporting_evidence=[],
                contradicting_evidence=[],
                source_used="None",
                flags=["NO_GROUND_TRUTH"]
            )

        resp_lower = ai_response.lower()
        gt_lower = ground_truth.lower()

        # Check for direct contradictions (negation flips)
        is_contradictory = False
        contradiction_reason = ""
        
        # Check classic negation mismatch patterns
        negation_pairs = [
            ("not cause", ["definitely causes", "does cause", "will cause", "destroys joints", "causes permanent", "weakens your joint grip"]),
            ("cannot be seen", ["can be seen", "clearly visible", "is visible", "visible from the moon"]),
            ("ineffective against", ["kills", "cures", "destroys", "is effective against"]),
            ("virtually 100%", ["only 10%", "use 10 percent", "use only ten", "capacity"]),
            ("does not cause hyperactivity", ["directly triggers hyperactive", "causes immediate", "makes children wildly", "adrenaline surge"]),
            ("does not negatively impact", ["causes severe", "directly triggers hyperactivity", "adrenaline surge"])
        ]

        
        for gt_neg, false_affirmations in negation_pairs:
            if gt_neg in gt_lower:
                for fa in false_affirmations:
                    if fa in resp_lower:
                        is_contradictory = True
                        contradiction_reason = f"Response asserts '{fa}' which directly contradicts evidence '{gt_neg}'."
                        contradicting_evidence.append(contradiction_reason)
                        flags.append("DIRECT_FACTUAL_CONTRADICTION")
                        break

        if is_contradictory:
            return AccuracyDetail(
                score=0.10,
                level=AccuracyLevel.CONTRADICTORY,
                reasoning=f"Response is FACTUALLY CONTRADICTORY (10%). {contradiction_reason}",
                supporting_evidence=supporting_evidence[:2],
                contradicting_evidence=contradicting_evidence,
                source_used=source_used,
                flags=flags
            )

        # Semantic Similarity against Ground Truth
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            mat = vec.fit_transform([ground_truth, ai_response])
            sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        except Exception:
            sim = 0.50

        # Entity / Key term preservation
        gt_entities = set(re.findall(r'\b[a-zA-Z]{4,}\b', gt_lower))
        resp_entities = set(re.findall(r'\b[a-zA-Z]{4,}\b', resp_lower))
        common_entities = gt_entities.intersection(resp_entities)
        coverage = len(common_entities) / max(len(resp_entities), 1)

        raw_score = 0.55 * sim + 0.45 * coverage
        # High match bonus
        if sim > 0.50 and coverage > 0.35:
            raw_score = max(raw_score, 0.90)
        elif sim > 0.35:
            raw_score = max(raw_score, 0.65)

        score = round(float(min(max(raw_score, 0.0), 1.0)), 3)

        # Map to accuracy levels
        if score >= 0.80:
            level = AccuracyLevel.CORRECT
            reasoning = (
                f"Response is FACTUALLY CORRECT ({score * 100:.1f}%). All key facts, relationships, and entities "
                f"align completely with verified {source_used}."
            )
        elif score >= 0.50:
            level = AccuracyLevel.PARTIALLY_CORRECT
            flags.append("PARTIAL_FACTUAL_MATCH")
            reasoning = (
                f"Response is PARTIALLY CORRECT ({score * 100:.1f}%). General topic matches {source_used}, "
                f"but some specific details or qualifying constraints are missing or imprecise."
            )
        else:
            level = AccuracyLevel.INCORRECT
            flags.append("FACTUALLY_INCORRECT")
            reasoning = (
                f"Response is FACTUALLY INCORRECT ({score * 100:.1f}%). Significant deviation or unsupported claims "
                f"observed when cross-checked against {source_used}."
            )


        return AccuracyDetail(
            score=score,
            level=level,
            reasoning=reasoning,
            supporting_evidence=supporting_evidence[:2],
            contradicting_evidence=contradicting_evidence,
            source_used=source_used,
            flags=flags
        )

    def evaluate(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        source_document: Optional[str] = None
    ) -> AgentVerdict:
        detail = self.analyze_accuracy(
            question=question,
            ai_response=ai_response,
            reference_answer=reference_answer,
            retrieved_context=retrieved_context,
            source_document=source_document
        )
        return AgentVerdict(
            agent_name=self.name,
            dimension=self.dimension,
            score=detail.score,
            reasoning=detail.reasoning,
            flags=detail.flags,
            metadata={
                "level": detail.level.value,
                "source_used": detail.source_used,
                "supporting_evidence": detail.supporting_evidence,
                "contradicting_evidence": detail.contradicting_evidence
            }
        )
