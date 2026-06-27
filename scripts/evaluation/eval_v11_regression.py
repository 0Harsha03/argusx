"""
ArgusX v11 — Regression Validation Benchmark
==============================================
Runs the DeepSet benchmark against the unified AdaptiveDetectionPipeline.
"""

import io, json, logging, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset as hf_load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("v11_regression")

from app.services.model_registry import ModelRegistry
from app.services.adaptive_detection_pipeline import AdaptiveDetectionPipeline

def main():
    log.info("STEP 1 — Initializing ModelRegistry and AdaptiveDetectionPipeline ...")
    registry = ModelRegistry()
    registry.load_all()
    pipeline = AdaptiveDetectionPipeline(registry)

    log.info("STEP 2 — Loading DeepSet test split ...")
    ds = hf_load_dataset("deepset/prompt-injections")
    test_texts = [r["text"] for r in ds["test"]]
    test_labels = [int(r["label"]) for r in ds["test"]]
    log.info("  %d test samples loaded", len(test_labels))

    log.info("STEP 3 — Running inference via AdaptiveDetectionPipeline ...")
    preds = []
    
    POSITIVE_DECISIONS = {"BLOCK", "FLAG", "SANITIZE"}

    for i, text in enumerate(test_texts):
        res = pipeline.analyze(text)
        decision = res["decision"]
        
        if decision in POSITIVE_DECISIONS:
            preds.append(1)
        else:
            preds.append(0)

    log.info("STEP 4 — Computing metrics ...")
    preds_arr = np.array(preds)
    labels = np.array(test_labels)

    acc = accuracy_score(labels, preds_arr)
    prec = precision_score(labels, preds_arr, zero_division=0)
    rec = recall_score(labels, preds_arr, zero_division=0)
    f1 = f1_score(labels, preds_arr, zero_division=0)

    cm = confusion_matrix(labels, preds_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 70)
    print("V11 REGRESSION VALIDATION RESULTS")
    print("=" * 70)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("\n  Confusion Matrix:")
    print(f"    [[{tn}, {fp}],\n     [{fn}, {tp}]]")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print("=" * 70)

if __name__ == "__main__":
    main()
