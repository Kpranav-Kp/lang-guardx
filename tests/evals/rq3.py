#!/usr/bin/env python
"""
evaluate_rq3.py

RQ3: Output Validation (Layer 3) evaluation.
Evaluates the IndirectScanner on a realistic dataset of poisoned and clean database results.
"""

import json
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lang_guardx.detection.core import Detector

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "indirect_injection_eval_dataset.json"
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_layer3(detector: Detector, dataset: list) -> dict:
    """Run the evaluation and compute metrics."""
    tp = fn = fp = tn = 0

    for entry in dataset:
        text = entry["text"]
        expected_label = entry["label"]  # 1 = poisoned (attack), 0 = clean

        # Simulate a database row containing this text (e.g., a product review)
        db_rows = [{"id": 1, "review": text}]
        _, flags = detector.scan_db_results(db_rows)

        predicted_attack = bool(flags)  # True if any flag detected

        # Update confusion matrix
        if expected_label == 1 and predicted_attack:
            tp += 1
        elif expected_label == 1 and not predicted_attack:
            fn += 1
        elif expected_label == 0 and predicted_attack:
            fp += 1
        elif expected_label == 0 and not predicted_attack:
            tn += 1

    total_poisoned = tp + fn
    total_clean = fp + tn
    detection_rate = (tp / total_poisoned * 100) if total_poisoned > 0 else 0
    false_positive_rate = (fp / total_clean * 100) if total_clean > 0 else 0
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100 if (tp + tn + fp + fn) > 0 else 0
    total_requests = len(dataset)
    critic_invocation_rate = (tp + fp) / total_requests * 100  # proportion of requests that triggered a scan flag

    return {
        "total_poisoned": total_poisoned,
        "total_clean": total_clean,
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "metrics": {
            "ri1_detection_rate_%": round(detection_rate, 2),
            "false_positive_rate_%": round(false_positive_rate, 2),
            "accuracy_%": round(accuracy, 2),
            "critic_invocation_rate_%": round(critic_invocation_rate, 2),
        },
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} records.")

    print("\nInitializing LangGuardX Detector...")
    detector = Detector(distilbert_threshold=0.975)

    print("\nRunning RQ3 evaluation...")
    results = evaluate_layer3(detector, dataset)

    # Print summary
    print("\n" + "=" * 60)
    print("RQ3 EVALUATION RESULTS (Output Validation)")
    print("=" * 60)
    print(f"Poisoned Records: {results['total_poisoned']}")
    print(f"Clean Records: {results['total_clean']}")
    print("\nConfusion Matrix:")
    print(f"  TP (correctly flagged): {results['confusion_matrix']['TP']}")
    print(f"  FN (missed attacks):    {results['confusion_matrix']['FN']}")
    print(f"  FP (false alarms):      {results['confusion_matrix']['FP']}")
    print(f"  TN (correct passes):    {results['confusion_matrix']['TN']}")
    print("\nMetrics:")
    print(f"  RI.1 Detection Rate: {results['metrics']['ri1_detection_rate_%']:.2f}%")
    print(f"  False Positive Rate: {results['metrics']['false_positive_rate_%']:.2f}%")
    print(f"  Accuracy:            {results['metrics']['accuracy_%']:.2f}%")
    print(f"  Critic Invocation Rate: {results['metrics']['critic_invocation_rate_%']:.2f}%")
    print("=" * 60)

    # Save results
    output_path = Path("tests/evals/output/rq3_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
