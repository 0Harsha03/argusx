"""
ArgusX Blind Benchmark Evaluation
====================================
Evaluates ArgusX v6.1 (TF-IDF) and v7 (SBERT) against a 100-prompt
human-authored blind dataset with NO overlap with any training corpus.

Dataset: evaluation/blind_dataset.json
  50 Benign prompts  (label=0)  — realistic edge-cases
  50 Malicious prompts (label=1) — 10 attack categories

Label mapping:
  label=0 → ALLOW expected         → TN if correct, FP if flagged
  label=1 → BLOCK|SANITIZE|FLAG    → TP if caught, FN if missed

No models retrained. No code modified. Evaluation only.

Run:
    $env:PYTHONPATH = "."
    python scripts/eval_blind_benchmark.py
"""

import json
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from app.services.model_registry import ModelRegistry
from app.services.detection_pipeline import DetectionPipeline
from app.services.sbert_detection_pipeline import SBERTDetectionPipeline


DATASET_PATH = Path("evaluation/blind_dataset.json")
POSITIVE_DECISIONS = {"FLAG", "SANITIZE", "BLOCK"}


def load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    benign   = [d for d in data if d["label"] == 0]
    malicious= [d for d in data if d["label"] == 1]
    print(f"  Loaded {len(data)} prompts — benign={len(benign)}, malicious={len(malicious)}")
    return data


def run_evaluation(pipeline, dataset: list, pipeline_name: str) -> dict:
    y_true, y_pred, raw = [], [], []

    for item in dataset:
        res  = pipeline.analyze(item["prompt"])
        pred = 1 if res["decision"] in POSITIVE_DECISIONS else 0

        y_true.append(item["label"])
        y_pred.append(pred)
        raw.append({
            "id":       item["id"],
            "prompt":   item["prompt"],
            "label":    item["label"],
            "category": item["category"],
            "ps":       res["pattern_score"],
            "ss":       res["semantic_score"],
            "bs":       res["behavioral_score"],
            "ans":      res["anomaly_score"],
            "fs":       res["final_score"],
            "dec":      res["decision"],
            "pred":     pred,
        })

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    fns = sorted([r for r in raw if r["label"] == 1 and r["pred"] == 0],
                 key=lambda x: x["fs"], reverse=True)[:20]
    fps = sorted([r for r in raw if r["label"] == 0 and r["pred"] == 1],
                 key=lambda x: x["fs"], reverse=True)[:20]

    return {
        "pipeline": pipeline_name,
        "acc":  accuracy_score(y_true, y_pred)                     * 100,
        "prec": precision_score(y_true, y_pred, zero_division=0)    * 100,
        "rec":  recall_score(y_true, y_pred, zero_division=0)        * 100,
        "f1":   f1_score(y_true, y_pred, zero_division=0)            * 100,
        "cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fns": fns, "fps": fps, "raw": raw,
    }


def print_failure_table(title: str, items: list):
    n = len(items)
    print(f"\n{title}  ({n} items)")
    print(f"{'#':<4} {'ID':<6} {'Category':<26} {'Pat':>5} {'Sem':>5} {'Beh':>5} {'FS':>6}  Prompt")
    print("-" * 115)
    for i, r in enumerate(items):
        snippet = r["prompt"][:65].replace("\n", " ")
        print(f"{i+1:<4} {r['id']:<6} {r['category']:<26} "
              f"{r['ps']:>5.1f} {r['ss']:>5.1f} {r['bs']:>5.1f} "
              f"{r['fs']:>6.2f}  {snippet}")


def category_recall(results: dict, cat_type: str) -> dict:
    """For malicious prompts: detection rate per category."""
    cats = {}
    for r in results["raw"]:
        if r["label"] == 1:
            cat = r["category"]
            cats.setdefault(cat, {"total": 0, "detected": 0})
            cats[cat]["total"] += 1
            if r["pred"] == 1:
                cats[cat]["detected"] += 1
    return {k: v["detected"] / v["total"] * 100 for k, v in cats.items()}


def main():
    print("\n" + "=" * 65)
    print("ArgusX Blind Benchmark  —  v6.1 TF-IDF  vs  v7 SBERT")
    print("=" * 65)

    # 1. Dataset
    print("\n[1/4] Loading blind dataset…")
    dataset = load_dataset()

    # 2. Pipelines
    print("\n[2/4] Initialising pipelines (using frozen v6.0 LODO artifacts)…")
    registry = ModelRegistry()
    registry.load_all()
    pipeline_v6 = DetectionPipeline(registry)
    pipeline_v7 = SBERTDetectionPipeline(registry)

    # 3. Evaluate
    print("\n[3/4] Running evaluations…")
    print("      v6.1 TF-IDF…")
    res_v6 = run_evaluation(pipeline_v6, dataset, "v6.1 TF-IDF")
    print("      v7 SBERT…")
    res_v7 = run_evaluation(pipeline_v7, dataset, "v7 SBERT")

    # 4. Comparison table
    print("\n" + "=" * 65)
    print("COMPARISON TABLE  —  Blind Benchmark (N=100)")
    print("=" * 65)
    print(f"{'Metric':<12} | {'v6.1 TF-IDF':>12} | {'v7 SBERT':>12} | {'Delta':>10}")
    print("-" * 55)
    for metric in ("acc", "prec", "rec", "f1"):
        v6 = res_v6[metric]
        v7 = res_v7[metric]
        print(f"{metric.upper():<12} | {v6:>11.2f}% | {v7:>11.2f}% | {v7-v6:>+9.2f}%")

    # 5. Confusion matrices
    print("\n" + "=" * 65)
    print("CONFUSION MATRICES")
    print("=" * 65)
    for res in (res_v6, res_v7):
        print(f"\n{res['pipeline']}:")
        print(f"  TN={res['tn']}  FP={res['fp']}")
        print(f"  FN={res['fn']}  TP={res['tp']}")
        print(f"  Matrix:\n{res['cm']}")

    # 6. Category-level detection
    print("\n" + "=" * 65)
    print("ATTACK CATEGORY DETECTION RATES  (malicious prompts, N=50)")
    print("=" * 65)
    cr_v6 = category_recall(res_v6, "v6")
    cr_v7 = category_recall(res_v7, "v7")
    all_cats = sorted(set(list(cr_v6.keys()) + list(cr_v7.keys())))
    print(f"\n{'Category':<28} | {'v6.1 Recall':>12} | {'v7 Recall':>10} | {'Delta':>8}")
    print("-" * 68)
    for cat in all_cats:
        v6r = cr_v6.get(cat, 0)
        v7r = cr_v7.get(cat, 0)
        print(f"{cat:<28} | {v6r:>11.1f}% | {v7r:>9.1f}% | {v7r-v6r:>+7.1f}%")

    # 7. Failure analysis
    print("\n" + "=" * 65)
    print("FAILURE ANALYSIS  — v6.1 TF-IDF")
    print("=" * 65)
    print_failure_table("TOP 20 FALSE NEGATIVES  (missed malicious)", res_v6["fns"])
    print(f"\nFN Cluster counts: {dict(Counter(r['category'] for r in res_v6['fns']))}")
    print_failure_table("TOP 20 FALSE POSITIVES  (benign incorrectly flagged)", res_v6["fps"])
    print(f"\nFP Cluster counts: {dict(Counter(r['category'] for r in res_v6['fps']))}")

    print("\n" + "=" * 65)
    print("FAILURE ANALYSIS  — v7 SBERT")
    print("=" * 65)
    print_failure_table("TOP 20 FALSE NEGATIVES  (missed malicious)", res_v7["fns"])
    print(f"\nFN Cluster counts: {dict(Counter(r['category'] for r in res_v7['fns']))}")
    print_failure_table("TOP 20 FALSE POSITIVES  (benign incorrectly flagged)", res_v7["fps"])
    print(f"\nFP Cluster counts: {dict(Counter(r['category'] for r in res_v7['fps']))}")

    # 8. Per-prompt scorecard (full)
    print("\n" + "=" * 65)
    print("FULL PER-PROMPT SCORECARD  — v7 SBERT")
    print("=" * 65)
    print(f"{'ID':<6} {'Lbl':>4} {'Cat':<26} {'Pat':>5} {'Sem':>5} {'Beh':>5} {'FS':>6} {'Dec':<10} {'OK?':>4}")
    print("-" * 85)
    for r in res_v7["raw"]:
        ok = "✓" if r["pred"] == r["label"] else "✗"
        print(f"{r['id']:<6} {r['label']:>4} {r['category']:<26} "
              f"{r['ps']:>5.1f} {r['ss']:>5.1f} {r['bs']:>5.1f} "
              f"{r['fs']:>6.2f} {r['dec']:<10} {ok:>4}")

    # 9. Export
    export = {
        "v6.1 TF-IDF": {
            "acc": res_v6["acc"], "prec": res_v6["prec"],
            "rec": res_v6["rec"], "f1":   res_v6["f1"],
            "TP": int(res_v6["tp"]), "FP": int(res_v6["fp"]),
            "TN": int(res_v6["tn"]), "FN": int(res_v6["fn"]),
            "category_recall": cr_v6,
        },
        "v7 SBERT": {
            "acc": res_v7["acc"], "prec": res_v7["prec"],
            "rec": res_v7["rec"], "f1":   res_v7["f1"],
            "TP": int(res_v7["tp"]), "FP": int(res_v7["fp"]),
            "TN": int(res_v7["tn"]), "FN": int(res_v7["fn"]),
            "category_recall": cr_v7,
        },
    }
    with open("blind_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print("\nResults saved to blind_eval_results.json")

    print("\n" + "=" * 65)
    print("Blind benchmark evaluation complete.")


if __name__ == "__main__":
    main()
