"""
ArgusX v9.3.0 — Behavioral RF Audit for Prompt Injection Fusion
================================================================
Branch: argusx-v9-adaptive-routing
Script: scripts/audit_rf_for_fusion.py

Objective:
Evaluate the standalone Behavioral RandomForest model on the DeepSet test split
and compare its errors against DistilBERT to assess fusion feasibility.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import pickle
from pathlib import Path

import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("rf_audit")

# Suppress HF warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset as hf_load_dataset
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
except ImportError as exc:
    print(f"[FATAL] Missing dependency: {exc}", file=sys.stderr)
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "rf_audit"
MODEL_DIR = REPO_ROOT / "models" / "distilbert_pi"

RF_MODEL_PATH = REPO_ROOT / "app" / "models" / "artifacts" / "behavioral_model.pkl"
VEC_MODEL_PATH = REPO_ROOT / "app" / "models" / "artifacts" / "vectorizer.pkl"


def _compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

import joblib
def step1_2_inspect_rf() -> tuple:
    log.info("STEP 1 & 2 -- Inspecting Behavioral RF Model ...")
    
    rf_model = joblib.load(RF_MODEL_PATH)
    vectorizer = joblib.load(VEC_MODEL_PATH)
        
    log.info("  RF Model Type: %s", type(rf_model).__name__)
    log.info("  Vectorizer Type: %s", type(vectorizer).__name__)
    log.info("  Features: %d", len(vectorizer.vocabulary_))
    log.info("  Supports predict_proba: %s", hasattr(rf_model, "predict_proba"))
    
    return rf_model, vectorizer

def main():
    print("=" * 70)
    print("ArgusX v9.3.0 -- Behavioral RF Audit")
    print("Branch: argusx-v9-adaptive-routing")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rf_model, vectorizer = step1_2_inspect_rf()
    
    log.info("Loading DeepSet official test split ...")
    ds = hf_load_dataset("deepset/prompt-injections")
    df_test = [{"prompt": r["text"], "label": int(r["label"])} for r in ds["test"]]
    
    prompts = [r["prompt"] for r in df_test]
    labels  = [r["label"] for r in df_test]
    
    log.info("STEP 3 -- RF Standalone Evaluation ...")
    vecs = vectorizer.transform(prompts)
    rf_probas = rf_model.predict_proba(vecs)
    
    classes = list(rf_model.classes_)
    adv_idx = classes.index(1) if 1 in classes else 1
    
    rf_preds = []
    for proba in rf_probas:
        adv_conf = float(proba[adv_idx])
        rf_preds.append(1 if adv_conf >= 0.50 else 0)
        
    rf_metrics = _compute_metrics(labels, rf_preds)
    
    log.info("  RF Accuracy : %.2f%%", rf_metrics['accuracy'] * 100)
    log.info("  RF Precision: %.2f%%", rf_metrics['precision'] * 100)
    log.info("  RF Recall   : %.2f%%", rf_metrics['recall'] * 100)
    log.info("  RF F1 Score : %.2f%%", rf_metrics['f1'] * 100)
    log.info("  RF CM       : TP=%d FP=%d TN=%d FN=%d", rf_metrics['TP'], rf_metrics['FP'], rf_metrics['TN'], rf_metrics['FN'])
    
    log.info("STEP 4 -- Error Analysis (DistilBERT vs RF) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    db_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    db_model.eval()
    
    db_preds = []
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = db_model(**enc)
            pred = torch.argmax(out.logits, dim=-1).item()
            db_preds.append(pred)
            
    db_metrics = _compute_metrics(labels, db_preds)
    
    rf_fp_idx = [i for i, (y, p) in enumerate(zip(labels, rf_preds)) if y == 0 and p == 1]
    rf_fn_idx = [i for i, (y, p) in enumerate(zip(labels, rf_preds)) if y == 1 and p == 0]
    
    db_fp_idx = [i for i, (y, p) in enumerate(zip(labels, db_preds)) if y == 0 and p == 1]
    db_fn_idx = [i for i, (y, p) in enumerate(zip(labels, db_preds)) if y == 1 and p == 0]
    
    db_mistakes = set(db_fp_idx + db_fn_idx)
    rf_correct_on_db_mistakes = [i for i in db_mistakes if rf_preds[i] == labels[i]]
    
    error_overlap = {
        "rf_false_positives": len(rf_fp_idx),
        "rf_false_negatives": len(rf_fn_idx),
        "distilbert_false_positives": len(db_fp_idx),
        "distilbert_false_negatives": len(db_fn_idx),
        "distilbert_mistakes_total": len(db_mistakes),
        "distilbert_mistakes_caught_by_rf": len(rf_correct_on_db_mistakes),
        "caught_details": {
            "fp_fixed_by_rf": len([i for i in db_fp_idx if rf_preds[i] == labels[i]]),
            "fn_fixed_by_rf": len([i for i in db_fn_idx if rf_preds[i] == labels[i]]),
        }
    }
    
    log.info("  DistilBERT mistakes: %d", error_overlap["distilbert_mistakes_total"])
    log.info("  Mistakes correctly identified by RF: %d", error_overlap["distilbert_mistakes_caught_by_rf"])
    
    # Assess fusion feasibility
    log.info("STEP 5 -- Fusion Feasibility Assessment ...")
    if error_overlap["distilbert_mistakes_caught_by_rf"] == 0:
        conclusion = "A. RF provides little additional value."
        rationale = "The RF model fails to correct any of DistilBERT's mistakes on the benchmark dataset."
    elif error_overlap["distilbert_mistakes_caught_by_rf"] < 3 and rf_metrics['accuracy'] < db_metrics['accuracy'] - 0.1:
        conclusion = "A. RF provides little additional value."
        rationale = "The RF model catches very few DistilBERT mistakes and its overall performance is significantly worse."
    elif error_overlap["distilbert_mistakes_caught_by_rf"] > 0 and rf_metrics['accuracy'] > 0.8:
        conclusion = "C. RF captures errors missed by DistilBERT and warrants fusion experiments."
        rationale = f"The RF model successfully correctly predicts {error_overlap['distilbert_mistakes_caught_by_rf']} samples that DistilBERT got wrong."
    else:
        conclusion = "B. RF provides complementary signals and is likely beneficial."
        rationale = "The RF model provides some complementary value, though results are mixed."
        
    audit_report = {
        "conclusion": conclusion,
        "rationale": rationale,
        "rf_metrics": rf_metrics,
        "db_metrics": db_metrics,
        "overlap": error_overlap
    }
    
    log.info("  Conclusion: %s", conclusion)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rf_metrics, f, indent=2)
    with open(RESULTS_DIR / "error_overlap.json", "w", encoding="utf-8") as f:
        json.dump(error_overlap, f, indent=2)
    with open(RESULTS_DIR / "rf_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
        
    print("\n✅ Audit complete.")
    
if __name__ == "__main__":
    main()
