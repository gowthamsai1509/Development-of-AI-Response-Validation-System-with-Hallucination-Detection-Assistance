import re
from typing import Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.agents.base_agent import BaseJudgeAgent
from app.schemas.evaluation import AgentVerdict

def simple_stem(word: str) -> str:
    w = word.lower().strip()
    for suffix in ["ingly", "ing", "tion", "able", "ment", "ness", "ies", "ed", "es", "s"]:
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w

class RelevanceJudgeAgent(BaseJudgeAgent):
    """
    Evaluates whether the AI response is on-topic and directly answers the user prompt.
    """
    def __init__(self):
        super().__init__(name="RelevanceJudgeAgent", dimension="relevance")

    def evaluate(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        source_document: Optional[str] = None
    ) -> AgentVerdict:
        q_clean = question.strip()
        resp_clean = ai_response.strip()

        # Extract stems of content words
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "does", "do", "did", "have", "has", "had", "the", "a", "an", "and", "or",
            "but", "if", "then", "else", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "can", "will", "just", "should", "now", "lot", "you", "your"
        }

        q_raw_words = re.findall(r'\b[a-zA-Z]{3,}\b', q_clean.lower())
        q_words = {simple_stem(w) for w in q_raw_words if w not in stopwords}
        
        resp_raw_words = re.findall(r'\b[a-zA-Z]{3,}\b', resp_clean.lower())
        resp_words = {simple_stem(w) for w in resp_raw_words if w not in stopwords}

        flags: List[str] = []
        if not resp_raw_words:
            return AgentVerdict(
                agent_name=self.name,
                dimension=self.dimension,
                score=0.0,
                reasoning="Response contains no meaningful words.",
                flags=["EMPTY_RESPONSE"]
            )

        # Keyword intersection
        overlap = q_words.intersection(resp_words)
        overlap_ratio = len(overlap) / max(len(q_words), 1)

        # Cosine similarity using TF-IDF
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            tfidf_mat = vectorizer.fit_transform([q_clean, resp_clean])
            sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        except Exception:
            sim = overlap_ratio

        # Composite score
        relevance_score = 0.4 * sim + 0.6 * min(1.0, overlap_ratio * 1.5)
        # If overlap is high, ensure score reflects topical match
        if overlap_ratio >= 0.5:
            relevance_score = max(relevance_score, 0.75)
        relevance_score = round(float(min(max(relevance_score, 0.0), 1.0)), 3)

        if relevance_score < 0.30:
            flags.append("POTENTIALLY_OFF_TOPIC")
            reasoning = f"Low semantic relevance ({relevance_score:.2f}) between question keywords and AI response."
        else:
            reasoning = f"Response demonstrates strong thematic and semantic alignment ({relevance_score:.2f}) with user question."

        return AgentVerdict(
            agent_name=self.name,
            dimension=self.dimension,
            score=relevance_score,
            reasoning=reasoning,
            flags=flags
        )

