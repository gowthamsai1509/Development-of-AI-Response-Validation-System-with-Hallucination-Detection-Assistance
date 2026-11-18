import re
from typing import Optional, List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.agents.base_agent import BaseJudgeAgent
from app.schemas.evaluation import AgentVerdict, RelevanceLevel, RelevanceDetail

STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "do", "did", "have", "has", "had", "the", "a", "an", "and", "or",
    "but", "if", "then", "else", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "once", "here", "there", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now", "lot",
    "you", "your", "tell", "me", "explain", "describe", "please", "would", "could"
}

def simple_stem(word: str) -> str:
    w = word.lower().strip()
    for suffix in ["ingly", "ing", "tion", "able", "ment", "ness", "ies", "ed", "es", "s"]:
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w

class RelevanceJudgeAgent(BaseJudgeAgent):
    """
    Milestone 2.1 - Relevance Judge Agent
    Evaluates whether an AI response directly and appropriately addresses the submitted question.
    Classifies relevance into: FULLY_RELEVANT, PARTIALLY_RELEVANT, PERIPHERAL, and OFF_TOPIC.
    """
    def __init__(self):
        super().__init__(name="RelevanceJudgeAgent", dimension="relevance")

    def analyze_relevance(
        self,
        question: str,
        ai_response: str
    ) -> RelevanceDetail:
        q_clean = question.strip()
        resp_clean = ai_response.strip()

        q_raw_words = re.findall(r'\b[a-zA-Z]{3,}\b', q_clean.lower())
        q_keywords = [w for w in q_raw_words if w not in STOPWORDS]
        q_stems = {simple_stem(w): w for w in q_keywords}

        resp_raw_words = re.findall(r'\b[a-zA-Z]{3,}\b', resp_clean.lower())
        resp_keywords = [w for w in resp_raw_words if w not in STOPWORDS]
        resp_stems = {simple_stem(w): w for w in resp_keywords}

        flags: List[str] = []
        if not resp_keywords:
            return RelevanceDetail(
                score=0.0,
                level=RelevanceLevel.OFF_TOPIC,
                reasoning="Response contains no meaningful content or topical keywords.",
                matched_topics=[],
                missing_aspects=list(q_stems.values()),
                flags=["EMPTY_RESPONSE", "OFF_TOPIC"]
            )

        # Keyword alignment
        matched_stem_keys = set(q_stems.keys()).intersection(set(resp_stems.keys()))
        matched_topics = [q_stems[k] for k in matched_stem_keys]
        missing_aspects = [q_stems[k] for k in set(q_stems.keys()) - set(resp_stems.keys())]

        overlap_ratio = len(matched_stem_keys) / max(len(q_stems), 1)

        # TF-IDF Cosine Similarity
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            tfidf_mat = vectorizer.fit_transform([q_clean, resp_clean])
            sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        except Exception:
            sim = overlap_ratio

        # Composite Relevance Score
        raw_score = 0.40 * sim + 0.60 * min(1.0, overlap_ratio * 1.4)
        if overlap_ratio >= 0.50:
            raw_score = max(raw_score, 0.85)
        if overlap_ratio >= 0.80:
            raw_score = max(raw_score, 0.95)

        score = round(float(min(max(raw_score, 0.0), 1.0)), 3)

        # Categorization based on tuned rubric
        if score >= 0.70:
            level = RelevanceLevel.FULLY_RELEVANT
            reasoning = (
                f"Response is FULLY RELEVANT ({score * 100:.1f}%). It directly addresses the core prompt question, "
                f"matching key concepts: {', '.join(matched_topics[:4]) if matched_topics else 'all topics'}."
            )
        elif score >= 0.40:
            level = RelevanceLevel.PARTIALLY_RELEVANT
            flags.append("PARTIALLY_RELEVANT")
            reasoning = (
                f"Response is PARTIALLY RELEVANT ({score * 100:.1f}%). It touches on question topics ({', '.join(matched_topics[:3])}), "
                f"but omits key aspects: {', '.join(missing_aspects[:3])}."
            )
        elif score >= 0.20:
            level = RelevanceLevel.PERIPHERAL
            flags.append("PERIPHERAL_TOPIC")
            reasoning = (
                f"Response has PERIPHERAL relevance ({score * 100:.1f}%). Mentions marginal keywords but does not "
                f"directly answer the intended query."
            )
        else:
            level = RelevanceLevel.OFF_TOPIC
            flags.append("OFF_TOPIC")
            reasoning = (
                f"Response is OFF-TOPIC ({score * 100:.1f}%). It fails to address the submitted question "
                f"and discusses unrelated concepts."
            )


        return RelevanceDetail(
            score=score,
            level=level,
            reasoning=reasoning,
            matched_topics=matched_topics,
            missing_aspects=missing_aspects,
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
        detail = self.analyze_relevance(question=question, ai_response=ai_response)
        return AgentVerdict(
            agent_name=self.name,
            dimension=self.dimension,
            score=detail.score,
            reasoning=detail.reasoning,
            flags=detail.flags,
            metadata={
                "level": detail.level.value,
                "matched_topics": detail.matched_topics,
                "missing_aspects": detail.missing_aspects
            }
        )
