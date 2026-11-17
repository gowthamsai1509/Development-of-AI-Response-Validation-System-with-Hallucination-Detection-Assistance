from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.config import settings
from app.database.connection import init_db
from app.api import submission_router, evaluation_router, knowledge_base_router

# Ensure DB initialized
init_db()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Automated AI Response Validation System with Multi-Agent Hallucination Detection & Public QA Benchmark Knowledge Base (Infosys Springboard Milestone 1)"
)

# Mount static directory for frontend
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include modular API routers
app.include_router(submission_router)
app.include_router(evaluation_router)
app.include_router(knowledge_base_router)

@app.get("/", include_in_schema=False)
def serve_home():
    return FileResponse(str(static_dir / "index.html"))

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

