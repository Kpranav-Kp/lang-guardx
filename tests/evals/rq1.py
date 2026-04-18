#!/usr/bin/env python
"""
rq1_cascade.py

RQ1: Cascade evaluation of Layer 1 (Bloom → Regex → DistilBERT) using
Bayesian risk minimization for the final decision (BLOCK / PASS / UNCERTAIN).

Results are saved to rq1_cascade_results.json.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from lang_guardx.detection.bloom import BloomDetector
from lang_guardx.detection.regex import RegexDetector
from lang_guardx.detection.sql_intent import SQLIntentClassifier

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
DATASET_PATH = Path(__file__).parent / "data/prompt_injection.csv"

# Costs for Bayesian risk minimization
COST_FP = 1.0
COST_FN = 2.0
UNCERTAIN_RATIO = 0.5


# ----------------------------------------------------------------------
# Load dataset
# ----------------------------------------------------------------------
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    if "text" in df.columns:
        df.rename(columns={"text": "prompt"}, inplace=True)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain 'label' column (1=attack, 0=safe)")
    df["label"] = df["label"].astype(int)
    return df


# ----------------------------------------------------------------------
# Cascade evaluation with BRM
# ----------------------------------------------------------------------
def evaluate_cascade(df, bloom, regex, classifier):
    results = []
    stage_counts = {
        "bloom_blocked": 0,
        "bloom_passed": 0,
        "regex_blocked": 0,
        "regex_passed": 0,
        "distilbert_blocked": 0,
        "distilbert_uncertain": 0,
        "distilbert_passed": 0,
    }
    latency_bloom = []
    latency_regex = []
    latency_distilbert = []

    for _, row in df.iterrows():
        prompt = row["prompt"]
        true_label = row["label"]
        blocked = False
        stage = None

        # Bloom
        t0 = time.perf_counter()
        bloom_hit = bloom.might_be_attack(prompt)
        latency_bloom.append((time.perf_counter() - t0) * 1000)

        if bloom_hit:
            blocked = True
            stage = "bloom"
            stage_counts["bloom_blocked"] += 1
        else:
            stage_counts["bloom_passed"] += 1
            # Regex
            t0 = time.perf_counter()
            regex_match = regex.check(prompt)
            latency_regex.append((time.perf_counter() - t0) * 1000)

            if regex_match.matched:
                blocked = True
                stage = "regex"
                stage_counts["regex_blocked"] += 1
            else:
                stage_counts["regex_passed"] += 1
                # DistilBERT with BRM
                t0 = time.perf_counter()
                decision, _ = classifier.decide(prompt, COST_FP, COST_FN, UNCERTAIN_RATIO)
                latency_distilbert.append((time.perf_counter() - t0) * 1000)

                if decision == "BLOCK":
                    blocked = True
                    stage = "distilbert_block"
                    stage_counts["distilbert_blocked"] += 1
                elif decision == "UNCERTAIN":
                    blocked = False
                    stage = "distilbert_uncertain"
                    stage_counts["distilbert_uncertain"] += 1
                else:  # PASS
                    blocked = False
                    stage = "distilbert_pass"
                    stage_counts["distilbert_passed"] += 1

        results.append(
            {
                "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                "true_label": true_label,
                "blocked": blocked,
                "blocking_stage": stage if blocked else "none",
            }
        )

    y_true = df["label"].to_numpy()
    y_pred = [1 if r["blocked"] else 0 for r in results]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    latency_summary = {
        "bloom": {"mean": statistics.mean(latency_bloom), "p95": sorted(latency_bloom)[int(0.95 * len(latency_bloom))]},
        "regex": {"mean": statistics.mean(latency_regex) if latency_regex else 0, "p95": sorted(latency_regex)[int(0.95 * len(latency_regex))] if latency_regex else 0},
        "distilbert": {"mean": statistics.mean(latency_distilbert) if latency_distilbert else 0, "p95": sorted(latency_distilbert)[int(0.95 * len(latency_distilbert))] if latency_distilbert else 0},
    }

    return stage_counts, latency_summary, tp, fn, fp, tn


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = load_dataset()
    bloom = BloomDetector.with_defaults()
    regex = RegexDetector()
    classifier = SQLIntentClassifier(threshold=0.95)

    stage_counts, lat, tp, fn, fp, tn = evaluate_cascade(df, bloom, regex, classifier)

    # Convert numpy ints to Python ints for JSON serialization
    tp = int(tp)
    fn = int(fn)
    fp = int(fp)
    tn = int(tn)
    total_samples = int(len(df))
    attacks = tp + fn
    safe = fp + tn

    block_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100

    # Convert stage counts to ints
    stage_counts_int = {k: int(v) for k, v in stage_counts.items()}

    output_data = {
        "total_samples": total_samples,
        "attacks": attacks,
        "safe": safe,
        "pipeline_flow": {
            "bloom_blocked": stage_counts_int["bloom_blocked"],
            "bloom_passed": stage_counts_int["bloom_passed"],
            "regex_blocked": stage_counts_int["regex_blocked"],
            "regex_passed": stage_counts_int["regex_passed"],
            "distilbert_blocked": stage_counts_int["distilbert_blocked"],
            "distilbert_uncertain": stage_counts_int["distilbert_uncertain"],
            "distilbert_passed": stage_counts_int["distilbert_passed"],
            "final_passed": stage_counts_int["distilbert_passed"] + stage_counts_int["distilbert_uncertain"],
        },
        "latency_ms": {
            "bloom": {"mean": lat["bloom"]["mean"], "p95": lat["bloom"]["p95"], "inputs": total_samples},
            "regex": {"mean": lat["regex"]["mean"], "p95": lat["regex"]["p95"], "inputs": stage_counts_int["bloom_passed"]},
            "distilbert": {"mean": lat["distilbert"]["mean"], "p95": lat["distilbert"]["p95"], "inputs": stage_counts_int["regex_passed"]},
        },
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "metrics": {"block_rate_%": round(block_rate, 2), "false_positive_rate_%": round(fpr, 2), "accuracy_%": round(accuracy, 2)},
    }

    print("\n========== RQ1 CASCADE RESULTS ==========")
    print(f"Total samples: {total_samples}")
    print(f"Attacks: {attacks}, Safe: {safe}\n")
    print("--- Pipeline flow ---")
    print(f"Bloom blocked:      {stage_counts_int['bloom_blocked']}")
    print(f"  → passed to Regex: {stage_counts_int['bloom_passed']}")
    print(f"Regex blocked:      {stage_counts_int['regex_blocked']}")
    print(f"  → passed to DistilBERT: {stage_counts_int['regex_passed']}")
    print(f"DistilBERT BLOCK:   {stage_counts_int['distilbert_blocked']}")
    print(f"DistilBERT UNCERTAIN: {stage_counts_int['distilbert_uncertain']} (passed)")
    print(f"DistilBERT PASS:    {stage_counts_int['distilbert_passed']}")
    print(f"  → final passed: {stage_counts_int['distilbert_passed'] + stage_counts_int['distilbert_uncertain']}\n")
    print("--- Final confusion matrix ---")
    print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
    print(f"Block Rate (TPR): {block_rate:.2f}%")
    print(f"False Positive Rate: {fpr:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")
    print("========================================\n")

    output_path = Path("tests/evals/output/rq1_results.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
