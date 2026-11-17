from fastapi import APIRouter, HTTPException
from app.schemas.knowledge_base import RetrievalQuery, RetrievalResult
from app.knowledge_base.retriever import RAGRetriever
from app.knowledge_base.ingestion import BenchmarkIngestionPipeline

router = APIRouter(prefix="/api/knowledge-base", tags=["Reference Knowledge Base"])
retriever = RAGRetriever()

@router.post("/retrieve", response_model=RetrievalResult)
def retrieve_knowledge(query_data: RetrievalQuery):
    """
    Performs semantic vector search across TruthfulQA and SQuAD reference knowledge base chunks.
    """
    matches = retriever.retrieve(
        query=query_data.query,
        top_k=query_data.top_k,
        dataset_filter=query_data.dataset_filter
    )
    return RetrievalResult(
        query=query_data.query,
        top_k=query_data.top_k,
        results_count=len(matches),
        matches=matches
    )

@router.get("/stats")
def get_knowledge_base_stats():
    """
    Returns statistics about indexed benchmark datasets and FAISS vector index.
    """
    ready = retriever._ensure_loaded()
    total_vectors = retriever.vector_store.index.ntotal if (ready and retriever.vector_store.index) else 0
    return {
        "status": "READY" if ready else "NOT_INITIALIZED",
        "total_chunks_indexed": total_vectors,
        "embedding_dimension": retriever.vector_store.dimension if ready else 0,
        "datasets": ["truthfulqa", "squad"]
    }

@router.post("/reindex")
def trigger_reindex():
    """
    Re-ingests benchmark datasets and rebuilds vector store index.
    """
    pipeline = BenchmarkIngestionPipeline()
    stats = pipeline.run_ingestion()
    # Refresh retriever instance
    retriever.is_ready = False
    retriever._ensure_loaded()
    return {"message": "Knowledge base re-indexed successfully", "stats": stats}
