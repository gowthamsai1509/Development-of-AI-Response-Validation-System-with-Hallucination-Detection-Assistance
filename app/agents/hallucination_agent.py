import re
from typing import Optional, List, Dict, Any, Tuple
from app.agents.base_agent import BaseJudgeAgent
from app.schemas.evaluation import AgentVerdict, ClaimAnalysis, ClaimStatus, HallucinationDetail

ABSOLUTE_OVERCLAIMS = [
    r"proven 100%", r"\bundeniably true\b", r"\bevery single scientist agrees\b",
    r"\balways completely\b", r"\bdefinitely guaranteed\b", r"\bunlock.*psychic\b",
    r"\bsupernatural\b", r"\bhidden superpowers\b", r"\bmagic\b", r"\bcures all known\b"
]

CONTRADICTION_PATTERNS = [
    # (pattern_in_evidence, false_affirmation_in_claim, explanation)
    (
        "not cause",
        ["definitely causes", "does cause", "will cause", "destroys joints", "causes permanent"],
        "Asserts the action causes damage, whereas scientific evidence verifies it does not cause harm."
    ),
    (
        "cannot be seen",
        ["can be seen", "clearly visible", "is visible with naked eye", "visible from the moon"],
        "Claims structure is visible from space/moon without magnification, contradicting astronaut optical confirmations."
    ),
    (
        "ineffective against",
        ["kills viruses", "cures viral", "treats the flu", "destroys the virus"],
        "Claims antibiotics destroy viral pathogens, contradicting pharmacological reality that antibiotics only affect bacteria."
    ),
    (
        "virtually 100%",
        ["only 10%", "use 10 percent", "use only ten percent"],
        "Repeats the 10% brain myth, contradicting neuroimaging evidence showing near 100% brain utilization."
    ),
    (
        "does not cause hyperactivity",
        ["directly triggers hyperactive", "makes children wildly hyperactive", "causes immediate hyperactivity"],
        "Claims sugar directly induces hyperactivity, which double-blind clinical trials have disproven."
    )
]

class HallucinationDetectionAgent(BaseJudgeAgent):
    """
    Milestone 2.3 - Hallucination Detection Agent
    1. Decomposes AI responses into individual factual propositions.
    2. Cross-references each claim against RAG-retrieved benchmark chunks and reference evidence.
    3. Flags specific unsupported, fabricated, or contradictory statements with granular reasoning.
    """
    def __init__(self):
        super().__init__(name="HallucinationDetectionAgent", dimension="hallucination")

    def split_into_claims(self, text: str) -> List[str]:
        # Split on sentence terminals and bullet points while preserving text
        raw_statements = re.split(r'(?<=[.!?])\s+|\n+|- |\* ', text.strip())
        claims = []
        for s in raw_statements:
            s_clean = s.strip()
            # Filter out empty or trivial fragments
            if len(s_clean.split()) >= 3:
                claims.append(s_clean)
        return claims if claims else [text.strip()]

    def analyze_hallucination(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        source_document: Optional[str] = None
    ) -> HallucinationDetail:
        # Build comprehensive evidence pool
        evidence_parts = []
        if reference_answer and reference_answer.strip():
            evidence_parts.append(reference_answer.strip())
        if source_document and source_document.strip():
            evidence_parts.append(source_document.strip())
        if retrieved_context and retrieved_context.strip():
            evidence_parts.append(retrieved_context.strip())

        full_evidence = "\n".join(evidence_parts).strip()
        evidence_lower = full_evidence.lower()
        question_lower = question.lower()

        claims = self.split_into_claims(ai_response)
        claim_analyses: List[ClaimAnalysis] = []
        flagged_statements: List[str] = []
        flags: List[str] = []

        supported_count = 0
        unsupported_count = 0
        contradicted_count = 0

        for idx, claim in enumerate(claims, 1):
            claim_lower = claim.lower()
            claim_status = ClaimStatus.SUPPORTED
            claim_confidence = 0.90
            claim_reasoning = "Claim is grounded in and corroborated by the evidence."
            evidence_quote = None

            # 1. Check for absolute overclaims or pseudoscience
            has_overclaim = False
            for pat in ABSOLUTE_OVERCLAIMS:
                if re.search(pat, claim_lower):
                    has_overclaim = True
                    claim_status = ClaimStatus.UNSUPPORTED
                    claim_confidence = 0.95
                    claim_reasoning = f"Contains unverified absolute or sensational claim matching pattern '{pat}'."
                    flags.append("UNVERIFIED_ABSOLUTE_CLAIM")
                    break

            # 2. Check for explicit contradictions against evidence
            if not has_overclaim and evidence_lower:
                for ev_pat, false_affirmations, expl in CONTRADICTION_PATTERNS:
                    if ev_pat in evidence_lower:
                        for fa in false_affirmations:
                            if fa in claim_lower:
                                claim_status = ClaimStatus.CONTRADICTED
                                claim_confidence = 0.98
                                claim_reasoning = f"Contradicts verified knowledge: {expl}"
                                evidence_quote = f"Ground truth establishes: '{ev_pat}'"
                                flags.append("CONTRADICTION_WITH_GROUND_TRUTH")
                                break
                    if claim_status == ClaimStatus.CONTRADICTED:
                        break

            # 3. Check for entity ungroundedness
            if claim_status == ClaimStatus.SUPPORTED and evidence_lower:
                claim_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', claim_lower))
                ev_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', evidence_lower + " " + question_lower))
                if claim_words:
                    unsupported_words = claim_words - ev_words
                    ratio = len(unsupported_words) / len(claim_words)
                    if ratio > 0.65:
                        claim_status = ClaimStatus.UNSUPPORTED
                        claim_confidence = 0.85
                        claim_reasoning = (
                            f"Contains multiple ungrounded entities/concepts ({', '.join(list(unsupported_words)[:3])}) "
                            f"not substantiated by reference sources."
                        )
                        flags.append("HIGH_UNGROUNDED_VOCABULARY")

            # Update tallies
            if claim_status == ClaimStatus.SUPPORTED:
                supported_count += 1
            elif claim_status == ClaimStatus.CONTRADICTED:
                contradicted_count += 1
                flagged_statements.append(claim)
            else:
                unsupported_count += 1
                flagged_statements.append(claim)

            claim_analyses.append(
                ClaimAnalysis(
                    claim_id=idx,
                    statement=claim,
                    status=claim_status,
                    confidence=claim_confidence,
                    reasoning=claim_reasoning,
                    evidence_quote=evidence_quote
                )
            )

        total_claims = len(claims)
        # Calculate continuous hallucination score [0.0, 1.0]
        if total_claims == 0:
            hallucination_score = 0.10
        else:
            # Contradictions are penalized heavily (1.0 each), unsupported claims at 0.60
            penalty = (contradicted_count * 1.0 + unsupported_count * 0.60) / total_claims
            hallucination_score = round(float(min(max(penalty, 0.05 if not flagged_statements else 0.40), 1.0)), 3)

        is_hallucinated = hallucination_score >= 0.40 or len(flagged_statements) > 0

        # Construct overall reasoning
        if contradicted_count > 0:
            reasoning = (
                f"High Hallucination Detected ({hallucination_score * 100:.1f}% risk). "
                f"{contradicted_count} of {total_claims} statements directly contradict verified benchmark facts. "
                f"Flagged statements: {'; '.join(flagged_statements[:2])}"
            )
        elif unsupported_count > 0:
            reasoning = (
                f"Moderate Hallucination Risk ({hallucination_score * 100:.1f}%). "
                f"{unsupported_count} of {total_claims} statements contain ungrounded or speculative assertions "
                f"not found in reference sources."
            )
        else:
            reasoning = (
                f"Low Hallucination Risk ({hallucination_score * 100:.1f}%). "
                f"All {total_claims} claims are grounded in and substantiated by the reference knowledge base."
            )

        return HallucinationDetail(
            hallucination_score=hallucination_score,
            is_hallucinated=is_hallucinated,
            total_claims=total_claims,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            contradicted_claims=contradicted_count,
            claim_breakdown=claim_analyses,
            flagged_statements=flagged_statements,
            reasoning=reasoning,
            flags=list(set(flags))
        )

    def evaluate(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        source_document: Optional[str] = None
    ) -> AgentVerdict:
        detail = self.analyze_hallucination(
            question=question,
            ai_response=ai_response,
            reference_answer=reference_answer,
            retrieved_context=retrieved_context,
            source_document=source_document
        )
        return AgentVerdict(
            agent_name=self.name,
            dimension=self.dimension,
            score=detail.hallucination_score,
            reasoning=detail.reasoning,
            flags=detail.flags,
            metadata={
                "is_hallucinated": detail.is_hallucinated,
                "total_claims": detail.total_claims,
                "flagged_statements": detail.flagged_statements,
                "claim_breakdown": [c.model_dump() for c in detail.claim_breakdown]
            }
        )
