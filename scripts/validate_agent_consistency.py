import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.schemas.submission import SubmissionRequest
from app.agents.orchestrator import EvaluationOrchestrator

def run_consistency_validation():
    print("=========================================================================================")
    print("             MILESTONE 2.4: AGENT CONSISTENCY & REASONING VALIDATION")
    print("=========================================================================================")

    testset_path = ROOT_DIR / "data" / "raw" / "m2_consistency_testset.json"
    if not testset_path.exists():
        print(f"Error: Test set not found at {testset_path}")
        return

    with open(testset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    orchestrator = EvaluationOrchestrator()
    total = len(test_cases)
    
    tp = 0  # Expected hallucination, detected
    fp = 0  # Expected grounded, falsely flagged
    tn = 0  # Expected grounded, correctly passed
    fn = 0  # Expected hallucination, missed
    
    relevance_matches = 0
    accuracy_matches = 0
    verdict_matches = 0

    print(f"\nEvaluating {total} Benchmark Question-Answer Pairs across Categories:\n")
    print(f"{'ID':<8} | {'Category':<28} | {'Verdict (Act/Exp)':<24} | {'Rel (Act/Exp)':<18} | {'Acc (Act/Exp)':<18} | {'Hal?':<6}")
    print("-" * 115)

    for tc in test_cases:
        t_id = tc["test_id"]
        cat = tc["category"]
        exp_verdict = tc["expected_verdict"]
        exp_rel = tc["expected_relevance"]
        exp_acc = tc["expected_accuracy"]
        exp_hal = tc["expected_hallucination"]

        sub = SubmissionRequest(
            question=tc["question"],
            ai_response=tc["ai_response"],
            reference_answer=tc.get("reference_answer")
        )

        res = orchestrator.evaluate_submission(sub)
        
        act_verdict = res.verdict.value
        act_rel = res.relevance_detail.level.value if res.relevance_detail else "N/A"
        act_acc = res.accuracy_detail.level.value if res.accuracy_detail else "N/A"
        act_hal = res.hallucination_detail.is_hallucinated if res.hallucination_detail else False

        # Confusion Matrix for Hallucination Detection
        if exp_hal and act_hal:
            tp += 1
        elif not exp_hal and not act_hal:
            tn += 1
        elif not exp_hal and act_hal:
            fp += 1
        elif exp_hal and not act_hal:
            fn += 1

        # Match counts
        if act_verdict == exp_verdict:
            verdict_matches += 1
        if act_rel == exp_rel:
            relevance_matches += 1
        if act_acc == exp_acc:
            accuracy_matches += 1

        v_status = "✅" if act_verdict == exp_verdict else "⚠️"
        r_str = f"{act_rel[:8]}/{exp_rel[:8]}"
        a_str = f"{act_acc[:8]}/{exp_acc[:8]}"
        v_str = f"{act_verdict[:10]}/{exp_verdict[:10]}"

        print(f"{t_id:<8} | {cat[:28]:<28} | {v_str:<24} | {r_str:<18} | {a_str:<18} | {str(act_hal):<6} {v_status}")

        # If claims were flagged, show sample claim breakdown
        if res.hallucination_detail and res.hallucination_detail.flagged_statements:
            for s in res.hallucination_detail.flagged_statements[:1]:
                print(f"         └─ 🚩 Flagged Statement: \"{s[:70]}...\"")

    print("\n" + "=" * 115)
    print("                           AGENT CONSISTENCY EVALUATION METRICS")
    print("=" * 115)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0
    accuracy = (tp + tn) / total if total > 0 else 1.0

    print(f"Total Test Cases Evaluated : {total}")
    print(f"Overall Verdict Match Rate : {verdict_matches}/{total} ({verdict_matches/total*100:.1f}%)")
    print(f"Relevance Classification   : {relevance_matches}/{total} ({relevance_matches/total*100:.1f}%)")
    print(f"Accuracy Classification    : {accuracy_matches}/{total} ({accuracy_matches/total*100:.1f}%)")
    print("\n--- Hallucination Detection Confusion Matrix ---")
    print(f"  True Positives  (TP) : {tp:<4} | Correctly detected hallucinations")
    print(f"  True Negatives  (TN) : {tn:<4} | Correctly confirmed grounded responses")
    print(f"  False Positives (FP) : {fp:<4} | Grounded answers wrongly flagged")
    print(f"  False Negatives (FN) : {fn:<4} | Hallucinations missed")
    print("------------------------------------------------")
    print(f"  Precision : {precision * 100:.1f}%")
    print(f"  Recall    : {recall * 100:.1f}%")
    print(f"  F1-Score  : {f1 * 100:.1f}%")
    print(f"  Accuracy  : {accuracy * 100:.1f}%")
    print("=========================================================================================\n")

if __name__ == "__main__":
    run_consistency_validation()
