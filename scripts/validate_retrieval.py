import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.knowledge_base.retriever import RAGRetriever

def run_validation():
    print("======================================================================")
    print("      MILESTONE 1.4: BENCHMARK KNOWLEDGE RETRIEVAL VALIDATION")
    print("======================================================================")

    retriever = RAGRetriever()
    if not retriever._ensure_loaded():
        print("ERROR: Vector index is not loaded. Run scripts/seed_knowledge_base.py first.")
        return

    test_queries = [
        {
            "query": "Does cracking knuckles cause arthritis in joint fluid?",
            "expected_dataset": "truthfulqa",
            "expected_keyword": "knuckle"
        },
        {
            "query": "What percentage of the brain do human beings actually use according to fMRI?",
            "expected_dataset": "truthfulqa",
            "expected_keyword": "brain"
        },
        {
            "query": "Can the Great Wall of China be observed from the Moon with naked eyes?",
            "expected_dataset": "truthfulqa",
            "expected_keyword": "wall"
        },
        {
            "query": "What hardware modalities are utilized to construct quantum processors?",
            "expected_dataset": "squad",
            "expected_keyword": "quantum"
        },
        {
            "query": "What semiconductor material is used in photovoltaic solar cells?",
            "expected_dataset": "squad",
            "expected_keyword": "photovoltaic"
        }
    ]

    total = len(test_queries)
    passed = 0

    for idx, test in enumerate(test_queries, 1):
        q = test["query"]
        print(f"\n[Test Query {idx}/{total}]: \"{q}\"")
        matches = retriever.retrieve(query=q, top_k=2)

        if not matches:
            print("  ❌ FAIL: No matches returned from vector database.")
            continue

        top_match = matches[0]
        score = top_match["similarity_score"]
        dataset = top_match["dataset"]
        title = top_match["title"]
        snippet = top_match["text"][:120].replace("\n", " ")

        print(f"  Top Match Dataset : {dataset}")
        print(f"  Top Match Title   : {title}")
        print(f"  Similarity Score  : {score:.3f}")
        print(f"  Snippet           : \"{snippet}...\"")

        keyword_found = test["expected_keyword"].lower() in top_match["text"].lower()
        dataset_matched = dataset == test["expected_dataset"]

        if keyword_found and score > 0.15:
            print("  ✅ PASS: Grounded context retrieved with high semantic fidelity.")
            passed += 1
        else:
            print(f"  ⚠️ WARNING: Relevance criteria marginal (keyword: {keyword_found}, score: {score:.3f})")

    accuracy = (passed / total) * 100
    print("\n----------------------------------------------------------------------")
    print(f"Retrieval Validation Summary: {passed}/{total} Queries Passed ({accuracy:.1f}%)")
    print("======================================================================")

if __name__ == "__main__":
    run_validation()
