from app.api.routes_submission import router as submission_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_knowledge_base import router as knowledge_base_router

__all__ = ["submission_router", "evaluation_router", "knowledge_base_router"]
