"""
ArgusX v7.1 — DeepSet Evaluation Protocol
========================================
Rigorous evaluation of the deepset/prompt-injections dataset against the 
Strategy A (Baseline) and Strategy D (v7.1 Dynamic Threshold) policies.

This script evaluates the train, test, and combined splits separately to
identify data contamination issues and establish the most publication-defensible
performance metrics.

Metrics Defined:
  TP (True Positive)  = Malicious prompt (Label 1) correctly caught (BLOCK, SANITIZE, FLAG)
  FP (False Positive) = Benign prompt (Label 0) incorrectly caught (BLOCK, SANITIZE, FLAG)
  TN (True Negative)  = Benign prompt (Label 0) correctly permitted (ALLOW)
  FN (False Negative) = Malicious prompt (Label 1) incorrectly permitted (ALLOW)
"""

import json
import logging
from typing import List, Dict
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.services.model_registry import ModelRegistry
from app.services.sbert_detection_pipeline import SBERTDetectionPipeline
from app.core.config import settings

# Temporarily disable logs from the pipeline so we can format our own output
logging.getLogger("app.detection").setLevel(logging.WARNING)
logging.getLogger("app.services").setLevel(logging.WARNING)

POSITIVE_DECISIONS = {"BLOCK", "SANITIZE", "FLAG"}


def evaluate_split(pipeline: SBERTDetectionPipeline, data: pd.DataFrame, split_name: str, strategy_d: bool) -> dict:
    """Run pipeline and compute rigorous metrics for a given dataset split."""
    # Enforce strategy directly on the ThreatScorer instance
    pipeline._threat_scorer._strategy_d = strategy_d
    
    y_true = []
    y_pred = []
    
    for _, row in data.iterrows():
        label = int(row['label'])
        text = row['text']
        
        # Analyze using full ArgusX pipeline
        res = pipeline.analyze(text)
        
        # Label 1 = Malicious, Label 0 = Benign
        decision = res['decision']
        pred = 1 if decision in POSITIVE_DECISIONS else 0
        
        y_true.append(label)
        y_pred.append(pred)
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "split": split_name,
        "strategy": "Strategy D" if strategy_d else "Strategy A (Baseline)",
        "N": len(data),
        "N_pos": sum(y_true),
        "N_neg": len(data) - sum(y_true),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "acc": acc * 100,
        "prec": prec * 100,
        "rec": rec * 100,
        "f1": f1 * 100
    }

def print_results(results: List[dict]):
    print(f"\n{'='*95}")
    print(f"{'Split':<12} | {'Strategy':<22} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'TP':>4}/{'FN':>4} | {'FP':>4}/{'TN':>4}")
    print("-" * 95)
    for r in results:
        print(f"{r['split']:<12} | {r['strategy']:<22} | {r['acc']:>6.2f}% | {r['prec']:>6.2f}% | {r['rec']:>6.2f}% | {r['f1']:>6.2f}% | {r['TP']:>4}/{r['FN']:<4} | {r['FP']:>4}/{r['TN']:<4}")


def main():
    print("=" * 70)
    print("ArgusX v7.1 — DeepSet Prompt Injection Evaluation")
    print("=" * 70)
    
    print("\n[1/3] Loading deepset/prompt-injections from HuggingFace...")
    ds = load_dataset("deepset/prompt-injections")
    
    df_train = pd.DataFrame([{"text": r["text"], "label": r["label"]} for r in ds["train"]])
    df_test = pd.DataFrame([{"text": r["text"], "label": r["label"]} for r in ds["test"]])
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    print(f"      Train split: {len(df_train)} samples")
    print(f"      Test split:  {len(df_test)} samples")
    print(f"      Total:       {len(df_all)} samples")
    
    print("\n[2/3] Initializing ArgusX Pipeline (v7 SBERT + Random Forest)...")
    registry = ModelRegistry()
    registry.load_all()
    pipeline = SBERTDetectionPipeline(registry)
    
    print("\n[3/3] Executing Evaluations...")
    all_results = []
    
    # Evaluate Baseline (Strategy A)
    print("      Evaluating Baseline (Strategy A, threshold=35)...")
    all_results.append(evaluate_split(pipeline, df_train, "Train", False))
    all_results.append(evaluate_split(pipeline, df_test, "Test", False))
    all_results.append(evaluate_split(pipeline, df_all, "Combined", False))
    
    # Evaluate Strategy D
    print("      Evaluating Strategy D (Dynamic threshold)...")
    all_results.append(evaluate_split(pipeline, df_train, "Train", True))
    all_results.append(evaluate_split(pipeline, df_test, "Test", True))
    all_results.append(evaluate_split(pipeline, df_all, "Combined", True))
    
    print_results(all_results)
    
    # Generate Publication Recommendations
    print("\n" + "=" * 70)
    print("METHODOLOGICAL ANALYSIS & PUBLICATION RECOMMENDATIONS")
    print("=" * 70)
    
    # Calculate deltas for test set
    base_test = next(r for r in all_results if r["split"] == "Test" and "Strategy A" in r["strategy"])
    strat_d_test = next(r for r in all_results if r["split"] == "Test" and "Strategy D" in r["strategy"])
    
    print(f"\n1. Data Contamination & Which Split to Report")
    print("The ArgusX Behavioral Engine (Random Forest) was previously trained on a corpus")
    print("that incorporated the DeepSet 'train' split. Therefore, reporting performance")
    print("on the 'Combined' or 'Train' split constitutes data leakage and is mathematically")
    print("invalid for a zero-shot or generalization claim.")
    print("\n=> PUBLICATION PROTOCOL: Only the 'Test' split (N=116) may be reported in the paper.")
    
    print(f"\n2. Baseline Performance Discrepancy")
    print("If previous reports cited DeepSet Recall=60.0%, this metric was likely derived")
    print("from the 'Combined' split or the contaminated 'Train' split.")
    print(f"The actual held-out Baseline Recall on the Test split is {base_test['rec']:.2f}%.")
    
    print(f"\n3. Strategy D Impact (Test Split Only)")
    print(f"   Recall:    {base_test['rec']:>6.2f}%  ->  {strat_d_test['rec']:>6.2f}%  (Delta: {strat_d_test['rec'] - base_test['rec']:>+6.2f}%)")
    print(f"   Precision: {base_test['prec']:>6.2f}%  ->  {strat_d_test['prec']:>6.2f}%  (Delta: {strat_d_test['prec'] - base_test['prec']:>+6.2f}%)")
    print(f"   F1 Score:  {base_test['f1']:>6.2f}%  ->  {strat_d_test['f1']:>6.2f}%  (Delta: {strat_d_test['f1'] - base_test['f1']:>+6.2f}%)")

if __name__ == "__main__":
    main()
