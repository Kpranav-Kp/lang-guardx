#!/usr/bin/env python
"""
evaluate_rq4_fixed.py

Corrected RQ4 evaluation with separate adaptive Bloom filter.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lang_guardx.adaptive.adaptive_engine import AdaptiveEngine
from lang_guardx.adaptive.threat_ontology import ThreatOntology
from lang_guardx.detection.bloom import BloomDetector
from lang_guardx.detection.core import Detector

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "adaptive_dataset.json"
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

malicious = [item["text"] for item in dataset if item["label"] == 1]
benign = [item["text"] for item in dataset if item["label"] == 0]

print(f"Loaded {len(dataset)} entries: {len(malicious)} malicious, {len(benign)} benign")

ATTACK_ID = "RI.1"
LOG_PATH = "rq4_adapt_log.jsonl"


# ----------------------------------------------------------------------
# Helper: detection rates
# ----------------------------------------------------------------------
def bloom_detection_rate(bloom: BloomDetector, patterns: list[str]) -> float:
    detected = sum(1 for p in patterns if bloom.might_be_attack(p))
    return (detected / len(patterns)) * 100 if patterns else 0.0


def adaptive_bloom_detection_rate(adaptive_bloom: BloomDetector, patterns: list[str]) -> float:
    if adaptive_bloom is None:
        return 0.0
    detected = sum(1 for p in patterns if adaptive_bloom.might_be_attack(p))
    return (detected / len(patterns)) * 100 if patterns else 0.0


# ----------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("RQ4 EVALUATION (Static + Adaptive Bloom)")
    print("=" * 70)

    # Create static Bloom and adaptive engine
    static_bloom = BloomDetector.with_defaults()
    ontology = ThreatOntology()
    engine = AdaptiveEngine(static_bloom, ontology, log_path=LOG_PATH)

    # Create detector and inject adaptive Bloom
    detector = Detector(distilbert_threshold=0.975)
    detector.bloom = static_bloom  # share static Bloom
    adaptive_bloom = engine.get_adaptive_bloom()
    detector.set_adaptive_bloom(adaptive_bloom)

    # Also ensure scanner uses static bloom
    if hasattr(detector, "scanner") and hasattr(detector.scanner, "bloom"):
        detector.scanner.bloom = static_bloom

    # --- Baseline detection BEFORE adaptation ---
    print("\n[1] Baseline detection (before adaptation)")
    static_before = bloom_detection_rate(static_bloom, malicious)
    print(f"    Static Bloom detection rate: {static_before:.1f}%")
    adaptive_before = adaptive_bloom_detection_rate(adaptive_bloom, malicious)
    print(f"    Adaptive Bloom detection rate: {adaptive_before:.1f}% (should be 0)")
    full_before = sum(1 for p in malicious if detector.check(p).blocked)
    full_before_rate = (full_before / len(malicious)) * 100
    print(f"    Full detector detection rate: {full_before_rate:.1f}%")

    # --- Identify truly novel patterns (not already in static Bloom) ---
    new_patterns = [p for p in malicious if not static_bloom.might_be_attack(p)]
    already_detected = len(malicious) - len(new_patterns)
    print(f"\n[2] Patterns already in static Bloom: {already_detected}")
    print(f"    New patterns to add: {len(new_patterns)}")

    if not new_patterns:
        print("    No new patterns – nothing to adapt. Exiting.")
        return

    # --- Add novel patterns via AdaptiveEngine (adds to adaptive Bloom only) ---
    print(f"\n[3] Adding {len(new_patterns)} novel patterns to adaptive Bloom (min_window=3)")
    start_time = time.perf_counter()
    for i, pattern in enumerate(new_patterns, 1):
        engine.add_pattern(ATTACK_ID, pattern)  # this uses adaptive_bloom with min_window=3
        if i % 50 == 0 or i == len(new_patterns):
            print(f"    Added {i}/{len(new_patterns)}")
    adapt_time_ms = (time.perf_counter() - start_time) * 1000
    print(f"    Adaptation time: {adapt_time_ms:.2f} ms")

    # --- Detection AFTER adaptation ---
    print("\n[4] Detection after adaptation")
    static_after = bloom_detection_rate(static_bloom, malicious)
    print(f"    Static Bloom detection rate: {static_after:.1f}% (unchanged)")
    adaptive_after = adaptive_bloom_detection_rate(adaptive_bloom, malicious)
    print(f"    Adaptive Bloom detection rate: {adaptive_after:.1f}%")
    print(f"    Improvement (adaptive): +{adaptive_after - adaptive_before:.1f} pp")

    full_after = sum(1 for p in malicious if detector.check(p).blocked)
    full_after_rate = (full_after / len(malicious)) * 100
    print(f"    Full detector detection rate: {full_after_rate:.1f}%")
    print(f"    Improvement (full): +{full_after_rate - full_before_rate:.1f} pp")

    # --- False positive rate on benign texts ---
    print("\n[5] False positive rate on benign texts (full detector)")
    fp_before = sum(1 for t in benign if detector.check(t).blocked)
    fpr_before = (fp_before / len(benign)) * 100
    fp_after = sum(1 for t in benign if detector.check(t).blocked)
    fpr_after = (fp_after / len(benign)) * 100
    print(f"    FPR before: {fpr_before:.1f}%")
    print(f"    FPR after:  {fpr_after:.1f}%")
    print(f"    Change: {fpr_after - fpr_before:+.1f} pp")

    # Show first few false positives (optional)
    if fpr_before > 0:
        print("\n    Sample false positives (first 5):")
        fp_count = 0
        for t in benign:
            if detector.check(t).blocked:
                print(f"      - {t[:80]}...")
                fp_count += 1
                if fp_count >= 5:
                    break

    # --- Adaptation log verification ---
    print("\n[6] Adaptation log")
    count = engine.get_adaptation_count()
    print(f"    Total logged adaptations: {count}")
    if count >= len(new_patterns):
        print("    ✓ All patterns correctly logged.")
    else:
        print("    ⚠️ Log count mismatch – check log file.")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RQ4 SUMMARY")
    print("=" * 70)
    print(f"Novel attack patterns added: {len(new_patterns)}")
    print(f"Static Bloom detection: {static_before:.1f}% → {static_after:.1f}% (unchanged by design)")
    print(f"Adaptive Bloom detection: {adaptive_before:.1f}% → {adaptive_after:.1f}%")
    print(f"Full detector detection: {full_before_rate:.1f}% → {full_after_rate:.1f}%")
    print(f"False positive rate: {fpr_before:.1f}% → {fpr_after:.1f}%")
    print(f"Adaptation time: {adapt_time_ms:.2f} ms")
    print(f"Adaptation count: {count}")
    print("=" * 70)

    # Save results to JSON
    results = {
        "malicious_total": len(malicious),
        "benign_total": len(benign),
        "patterns_new": len(new_patterns),
        "patterns_already_detected": already_detected,
        "static_bloom_before_%": round(static_before, 2),
        "static_bloom_after_%": round(static_after, 2),
        "adaptive_bloom_before_%": round(adaptive_before, 2),
        "adaptive_bloom_after_%": round(adaptive_after, 2),
        "full_detector_before_%": round(full_before_rate, 2),
        "full_detector_after_%": round(full_after_rate, 2),
        "fpr_before_%": round(fpr_before, 2),
        "fpr_after_%": round(fpr_after, 2),
        "adaptation_time_ms": round(adapt_time_ms, 2),
        "adaptation_log_count": count,
    }
    with open("rq4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to rq4_results.json")


if __name__ == "__main__":
    main()
