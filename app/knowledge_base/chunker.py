import re
from typing import List, Dict, Any
from app.schemas.knowledge_base import DocumentChunk

class TextChunker:
    """
    Sliding window chunker preserving sentence and paragraph boundaries.
    """
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_into_sentences(self, text: str) -> List[str]:
        # Split on sentence terminals while keeping punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        dataset: str = "general",
        title: str = "",
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        metadata = metadata or {}
        text = text.strip()
        if not text:
            return []

        # If document is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [
                DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_0",
                    doc_id=doc_id,
                    dataset=dataset,
                    title=title,
                    text=text,
                    chunk_index=0,
                    metadata=metadata
                )
            ]

        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > self.chunk_size and current_chunk_sentences:
                chunk_str = " ".join(current_chunk_sentences)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        dataset=dataset,
                        title=title,
                        text=chunk_str,
                        chunk_index=chunk_idx,
                        metadata=metadata
                    )
                )
                chunk_idx += 1
                
                # Overlap logic: keep last few sentences that fit in overlap
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk_sentences = overlap_sentences
                current_length = sum(len(s) for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_length += sentence_len

        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    dataset=dataset,
                    title=title,
                    text=chunk_str,
                    chunk_index=chunk_idx,
                    metadata=metadata
                )
            )

        return chunks
