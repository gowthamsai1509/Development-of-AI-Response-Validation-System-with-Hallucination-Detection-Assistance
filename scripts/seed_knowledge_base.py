import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.knowledge_base.ingestion import BenchmarkIngestionPipeline

def main():
    print("=== Seeding Reference Knowledge Base (TruthfulQA + SQuAD) ===")
    pipeline = BenchmarkIngestionPipeline()
    stats = pipeline.run_ingestion()
    print("Ingestion Successful!")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"TruthfulQA Docs: {stats['datasets']['truthfulqa']}")
    print(f"SQuAD Docs:      {stats['datasets']['squad']}")
    print(f"Total Chunks:    {stats['total_chunks']}")
    print(f"Vector Dim:      {stats['embedding_dimension']}")
    print(f"FAISS Index:     {stats['index_path']}")

if __name__ == "__main__":
    main()
