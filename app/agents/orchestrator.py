import uuid
import datetime
from typing import Optional, List
from sqlalchemy.orm import Session

from app.schemas.submission import SubmissionRequest
from app.schemas.evaluation import EvaluationResult, AgentVerdict, RetrievalEvidence
from app.agents.relevance_agent import RelevanceJudgeAgent
from app.agents.accuracy_agent import AccuracyJudgeAgent
from app.agents.hallucination_agent import HallucinationDetectionAgent
from app.agents.completeness_agent import CompletenessJudgeAgent
from app.agents.verdict_agent import VerdictAgent
from app.knowledge_base.retriever import RAGRetriever
from app.database.models import EvaluationRecord

class EvaluationOrchestrator:
    """
    Coordinates end-to-end evaluation lifecycle:
    Retrieval -> Agent Dispatch -> Verdict Synthesis -> Database Persistence
    Integrates Milestone 2 granular agent evaluations (claim-level hallucination breakdown,
    structured evidence, and relevance levels).
    """
    def __init__(self):
        self.retriever = RAGRetriever()
        self.relevance_agent = RelevanceJudgeAgent()
        self.accuracy_agent = AccuracyJudgeAgent()
        self.hallucination_agent = HallucinationDetectionAgent()
        self.completeness_agent = CompletenessJudgeAgent()
        self.verdict_agent = VerdictAgent()

    def evaluate_submission(
        self,
        submission: SubmissionRequest,
        db: Optional[Session] = None,
        submission_id: Optional[str] = None
    ) -> EvaluationResult:
        sub_id = submission_id or str(uuid.uuid4())
        
        # Step 1: Grounding Retrieval via Knowledge Base
        retrieved_raw = self.retriever.retrieve(
            query=f"{submission.question} {submission.ai_response}",
            top_k=3
        )

        retrieval_evidences: List[RetrievalEvidence] = []
        retrieved_context_str = ""
        for item in retrieved_raw:
            evidence = RetrievalEvidence(
                chunk_id=item["chunk_id"],
                dataset=item["dataset"],
                score=item["similarity_score"],
                snippet=item["text"],
                metadata=item.get("metadata", {})
            )
            retrieval_evidences.append(evidence)
            retrieved_context_str += f"\n[{item['dataset']}: {item['title']}] {item['text']}\n"

        # Step 2: Milestone 2 Granular Analysis
        rel_detail = self.relevance_agent.analyze_relevance(
            question=submission.question,
            ai_response=submission.ai_response
        )
        acc_detail = self.accuracy_agent.analyze_accuracy(
            question=submission.question,
            ai_response=submission.ai_response,
            reference_answer=submission.reference_answer,
            retrieved_context=retrieved_context_str,
            source_document=submission.source_document
        )
        hal_detail = self.hallucination_agent.analyze_hallucination(
            question=submission.question,
            ai_response=submission.ai_response,
            reference_answer=submission.reference_answer,
            retrieved_context=retrieved_context_str,
            source_document=submission.source_document
        )
        com_verdict = self.completeness_agent.evaluate(
            question=submission.question,
            ai_response=submission.ai_response,
            reference_answer=submission.reference_answer,
            retrieved_context=retrieved_context_str,
            source_document=submission.source_document
        )

        # Build standard AgentVerdicts for synthesis
        agent_verdicts: List[AgentVerdict] = [
            AgentVerdict(
                agent_name=self.relevance_agent.name,
                dimension=self.relevance_agent.dimension,
                score=rel_detail.score,
                reasoning=rel_detail.reasoning,
                flags=rel_detail.flags,
                metadata={"level": rel_detail.level.value, "matched_topics": rel_detail.matched_topics}
            ),
            AgentVerdict(
                agent_name=self.accuracy_agent.name,
                dimension=self.accuracy_agent.dimension,
                score=acc_detail.score,
                reasoning=acc_detail.reasoning,
                flags=acc_detail.flags,
                metadata={"level": acc_detail.level.value, "source_used": acc_detail.source_used}
            ),
            AgentVerdict(
                agent_name=self.hallucination_agent.name,
                dimension=self.hallucination_agent.dimension,
                score=hal_detail.hallucination_score,
                reasoning=hal_detail.reasoning,
                flags=hal_detail.flags,
                metadata={
                    "is_hallucinated": hal_detail.is_hallucinated,
                    "flagged_statements": hal_detail.flagged_statements,
                    "total_claims": hal_detail.total_claims
                }
            ),
            com_verdict
        ]

        # Step 3: Synthesis & Verdict
        verdict, scores, flags = self.verdict_agent.compute_verdict(agent_verdicts)

        # Step 4: Database Storage
        if db is not None:
            db_record = EvaluationRecord(
                id=sub_id,
                question=submission.question,
                ai_response=submission.ai_response,
                reference_answer=submission.reference_answer,
                source_document=submission.source_document,
                status="EVALUATED",
                created_at=datetime.datetime.utcnow(),
                retrieved_contexts=[e.model_dump() for e in retrieval_evidences],
                relevance_score=scores.relevance_score,
                accuracy_score=scores.accuracy_score,
                hallucination_score=scores.hallucination_score,
                completeness_score=scores.completeness_score,
                overall_confidence_score=scores.overall_confidence_score,
                verdict=verdict.value,
                agent_feedback={
                    "agent_analyses": [a.model_dump() for a in agent_verdicts],
                    "relevance_detail": rel_detail.model_dump(),
                    "accuracy_detail": acc_detail.model_dump(),
                    "hallucination_detail": hal_detail.model_dump(),
                    "flags": flags
                }
            )
            db.merge(db_record)
            db.commit()

        # Step 5: Structured Evaluation Result
        return EvaluationResult(
            submission_id=sub_id,
            status="SUCCESS",
            verdict=verdict,
            scores=scores,
            agent_analyses=agent_verdicts,
            retrieved_evidences=retrieval_evidences,
            relevance_detail=rel_detail,
            accuracy_detail=acc_detail,
            hallucination_detail=hal_detail,
            timestamp=datetime.datetime.utcnow()
        )
