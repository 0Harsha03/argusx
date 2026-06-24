"""
ArgusX v9.4.0 -- DistilBERT + Behavioral RF Fusion Study
=========================================================
Branch  : argusx-v9-adaptive-routing
Script  : scripts/eval_distilbert_rf_fusion.py

Objective:
Ablation study to determine whether fusing DistilBERT semantic probabilities
with Behavioral RF probabilities improves prompt injection detection over
the validated DistilBERT baseline.

Pipeline
--------
  STEP 1  -- Load DistilBERT + Behavioral RF (no retraining)
  STEP 2  -- Load DeepSet official TEST split ONLY (N=116, frozen)
  STEP 3  -- Generate per-sample probabilities for both models
  STEP 4  -- Evaluate fusion strategies (mean, max, weighted grid)
  STEP 5  -- Compute metrics per strategy
  STEP 6  -- Error recovery analysis vs DistilBERT baseline
  STEP 7  -- Rank strategies and recommend best
  STEP 8  -- Export results

Constraints
-----------
  * No retraining of either model.
  * No modifications to routing, ThreatScorer, LOF, or production code.
  * DeepSet test split is frozen -- inference only.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("fusion_study")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

try:
    import joblib
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset as hf_load_dataset
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )
except ImportError as exc:
    print(f"[FATAL] Missing dependency: {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[1]
MODEL_DIR    = REPO_ROOT / "models" / "distilbert_pi"
RF_PATH      = REPO_ROOT / "app" / "models" / "artifacts" / "behavioral_model.pkl"
VEC_PATH     = REPO_ROOT / "app" / "models" / "artifacts" / "vectorizer.pkl"
RESULTS_DIR  = REPO_ROOT / "results" / "distilbert_rf_fusion"

MAX_LENGTH   = 256
BATCH_SIZE   = 16
THRESHOLD    = 0.5

# DistilBERT validated baseline (frozen)
BASELINE = {
    "accuracy":  0.9483,
    "precision": 0.9655,
    "recall":    0.9333,
    "f1":        0.9492,
    "TP": 56, "FP": 2, "TN": 54, "FN": 4,
}

# Weighted fusion grid to evaluate (db_weight, rf_weight)
WEIGHT_GRID = [
    (0.90, 0.10),
    (0.85, 0.15),
    (0.80, 0.20),
    (0.75, 0.25),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
]


# ===========================================================================
# Dataset wrapper
# ===========================================================================

class PIDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ===========================================================================
# Metrics helper
# ===========================================================================

def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


# ===========================================================================
# STEP 1 -- Load models
# ===========================================================================

def step1_load_models(device: torch.device) -> tuple:
    log.info("STEP 1 -- Loading models ...")

    # DistilBERT
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"DistilBERT checkpoint not found: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    db_model  = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    db_model  = db_model.to(device)
    db_model.eval()
    log.info("  DistilBERT loaded from %s", MODEL_DIR)

    # Behavioral RF
    rf_model   = joblib.load(RF_PATH)
    vectorizer = joblib.load(VEC_PATH)
    log.info("  RF (%s) loaded | Vectorizer: %d features",
             type(rf_model).__name__, len(vectorizer.vocabulary_))

    return tokenizer, db_model, rf_model, vectorizer


# ===========================================================================
# STEP 2 -- Load DeepSet frozen test split
# ===========================================================================

def step2_load_data() -> tuple[list[str], list[int]]:
    log.info("STEP 2 -- Loading deepset/prompt-injections [test] ...")
    ds      = hf_load_dataset("deepset/prompt-injections")
    prompts = [r["text"] for r in ds["test"]]
    labels  = [int(r["label"]) for r in ds["test"]]
    log.info("  Loaded: %d samples", len(labels))
    return prompts, labels


# ===========================================================================
# STEP 3 -- Generate probabilities
# ===========================================================================

def step3_get_probabilities(
    tokenizer,
    db_model,
    rf_model,
    vectorizer,
    prompts: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    log.info("STEP 3 -- Generating probabilities ...")

    # ── DistilBERT probs ────────────────────────────────────────────────────
    enc     = tokenizer(prompts, max_length=MAX_LENGTH, padding="max_length",
                        truncation=True)
    ds_obj  = PIDataset(enc, [0] * len(prompts))   # labels irrelevant here
    loader  = DataLoader(ds_obj, batch_size=BATCH_SIZE, shuffle=False)

    db_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = db_model(input_ids=input_ids, attention_mask=attention_mask)
            # softmax -> probability for class 1 (malicious)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            db_probs.extend(probs.tolist())

    db_probs = np.array(db_probs)
    log.info("  DistilBERT probs: min=%.4f  max=%.4f  mean=%.4f",
             db_probs.min(), db_probs.max(), db_probs.mean())

    # ── RF probs ────────────────────────────────────────────────────────────
    vecs    = vectorizer.transform(prompts)
    raw_proba = rf_model.predict_proba(vecs)
    classes   = list(rf_model.classes_)
    adv_idx   = classes.index(1) if 1 in classes else 1
    rf_probs  = raw_proba[:, adv_idx]

    log.info("  RF probs       : min=%.4f  max=%.4f  mean=%.4f",
             rf_probs.min(), rf_probs.max(), rf_probs.mean())

    return db_probs, rf_probs


# ===========================================================================
# STEP 4 + 5 -- Fusion experiments + metrics
# ===========================================================================

def _threshold_predict(scores: np.ndarray, t: float = THRESHOLD) -> list[int]:
    return [1 if s >= t else 0 for s in scores]


def step4_5_fuse_and_evaluate(
    db_probs: np.ndarray,
    rf_probs: np.ndarray,
    labels:   list[int],
) -> dict[str, dict]:
    log.info("STEP 4 & 5 -- Running fusion experiments ...")
    results: dict[str, dict] = {}

    # Baseline (DistilBERT only)
    baseline_preds = _threshold_predict(db_probs)
    results["distilbert_baseline"] = {
        "description": "DistilBERT only (baseline)",
        "db_weight": 1.0, "rf_weight": 0.0,
        **_metrics(labels, baseline_preds),
    }

    # A. Mean fusion
    mean_scores = (db_probs + rf_probs) / 2.0
    results["mean_fusion"] = {
        "description": "Mean fusion: (DB + RF) / 2",
        "db_weight": 0.5, "rf_weight": 0.5,
        **_metrics(labels, _threshold_predict(mean_scores)),
    }

    # B. Max fusion
    max_scores = np.maximum(db_probs, rf_probs)
    results["max_fusion"] = {
        "description": "Max fusion: max(DB, RF)",
        "db_weight": "max", "rf_weight": "max",
        **_metrics(labels, _threshold_predict(max_scores)),
    }

    # C. Weighted grid
    for w_db, w_rf in WEIGHT_GRID:
        key    = f"w{int(w_db*100)}db_{int(w_rf*100)}rf"
        scores = w_db * db_probs + w_rf * rf_probs
        results[key] = {
            "description": f"Weighted: {w_db:.2f}*DB + {w_rf:.2f}*RF",
            "db_weight": w_db, "rf_weight": w_rf,
            **_metrics(labels, _threshold_predict(scores)),
        }

    log.info("  Evaluated %d strategies.", len(results))
    return results


# ===========================================================================
# STEP 6 -- Error recovery analysis
# ===========================================================================

def step6_error_analysis(
    db_probs:  np.ndarray,
    rf_probs:  np.ndarray,
    labels:    list[int],
    all_results: dict[str, dict],
) -> dict:
    log.info("STEP 6 -- Error recovery analysis ...")

    db_preds    = _threshold_predict(db_probs)
    db_fp_idx   = {i for i, (y, p) in enumerate(zip(labels, db_preds)) if y == 0 and p == 1}
    db_fn_idx   = {i for i, (y, p) in enumerate(zip(labels, db_preds)) if y == 1 and p == 0}

    recovery: dict[str, dict] = {}

    for name, res in all_results.items():
        if name == "distilbert_baseline":
            continue

        # Reconstruct fused preds from stored confusion numbers
        # We need the actual fusion probabilities -- easier to recompute
        w_db = res["db_weight"]
        w_rf = res["rf_weight"]

        if isinstance(w_db, float) and isinstance(w_rf, float):
            fused_scores = w_db * db_probs + w_rf * rf_probs
        elif res["description"].startswith("Mean"):
            fused_scores = (db_probs + rf_probs) / 2.0
        else:  # Max
            fused_scores = np.maximum(db_probs, rf_probs)

        fused_preds = _threshold_predict(fused_scores)

        fn_recovered  = len([i for i in db_fn_idx  if fused_preds[i] == 1])
        fp_corrected  = len([i for i in db_fp_idx  if fused_preds[i] == 0])
        new_fn        = len([i for i, (y, p) in enumerate(zip(labels, fused_preds))
                             if y == 1 and p == 0 and i not in db_fn_idx])
        new_fp        = len([i for i, (y, p) in enumerate(zip(labels, fused_preds))
                             if y == 0 and p == 1 and i not in db_fp_idx])

        bl = BASELINE
        recovery[name] = {
            "fn_recovered":       fn_recovered,
            "fp_corrected":       fp_corrected,
            "new_fn_introduced":  new_fn,
            "new_fp_introduced":  new_fp,
            "delta_recall":    round(res["recall"]    - bl["recall"],    4),
            "delta_precision": round(res["precision"] - bl["precision"], 4),
            "delta_f1":        round(res["f1"]        - bl["f1"],        4),
            "delta_accuracy":  round(res["accuracy"]  - bl["accuracy"],  4),
        }

    return recovery


# ===========================================================================
# STEP 7 -- Rank and recommend
# ===========================================================================

def step7_rank(all_results: dict[str, dict]) -> dict:
    log.info("STEP 7 -- Ranking fusion strategies ...")

    rows = [(name, r) for name, r in all_results.items()]
    by_acc = sorted(rows, key=lambda x: x[1]["accuracy"],  reverse=True)
    by_rec = sorted(rows, key=lambda x: x[1]["recall"],    reverse=True)
    by_f1  = sorted(rows, key=lambda x: x[1]["f1"],        reverse=True)

    best_acc = by_acc[0]
    best_rec = by_rec[0]
    best_f1  = by_f1[0]

    log.info("  Best Accuracy  : %-30s  %.4f", best_acc[0], best_acc[1]["accuracy"])
    log.info("  Best Recall    : %-30s  %.4f", best_rec[0], best_rec[1]["recall"])
    log.info("  Best F1        : %-30s  %.4f", best_f1[0],  best_f1[1]["f1"])

    # Recommendation: prefer the strategy that improves both F1 and recall
    # without sacrificing precision by more than 2 pp vs baseline
    candidates = [
        (n, r) for n, r in rows
        if r["f1"] >= BASELINE["f1"] - 0.005      # no significant F1 drop
        and r["precision"] >= BASELINE["precision"] - 0.02  # allow 2pp slack
        and r["recall"] >= BASELINE["recall"]      # must improve recall
        and n != "distilbert_baseline"
    ]

    if candidates:
        recommended = max(candidates, key=lambda x: (x[1]["recall"], x[1]["f1"]))
    else:
        # Fallback: pick best F1 among non-baseline strategies
        non_baseline = [(n, r) for n, r in rows if n != "distilbert_baseline"]
        recommended  = max(non_baseline, key=lambda x: x[1]["f1"])

    log.info("  Recommended    : %s  (recall=%.4f  f1=%.4f  precision=%.4f)",
             recommended[0], recommended[1]["recall"],
             recommended[1]["f1"], recommended[1]["precision"])

    return {
        "best_accuracy":  {"strategy": best_acc[0], **best_acc[1]},
        "best_recall":    {"strategy": best_rec[0], **best_rec[1]},
        "best_f1":        {"strategy": best_f1[0],  **best_f1[1]},
        "recommended":    {"strategy": recommended[0], **recommended[1]},
        "full_ranking_by_f1": [
            {"rank": i+1, "strategy": n, **r}
            for i, (n, r) in enumerate(by_f1)
        ],
    }


# ===========================================================================
# STEP 8 -- Export
# ===========================================================================

def step8_export(
    all_results:  dict[str, dict],
    recovery:     dict[str, dict],
    ranking:      dict,
) -> None:
    log.info("STEP 8 -- Exporting results to %s ...", RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "fusion_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=float)

    comparison = {
        "baseline": BASELINE,
        "error_recovery_per_strategy": recovery,
    }
    with open(RESULTS_DIR / "fusion_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=float)

    with open(RESULTS_DIR / "best_strategy.json", "w", encoding="utf-8") as f:
        json.dump(ranking, f, indent=2, default=float)

    log.info("  3 files written.")


# ===========================================================================
# Print tables
# ===========================================================================

def _print_table(all_results: dict, recovery: dict, ranking: dict) -> None:
    cols = ["Strategy", "Acc", "Prec", "Rec", "F1", "TP", "FP", "TN", "FN"]
    w    = [32, 7, 7, 7, 7, 4, 4, 4, 4]
    sep  = "  ".join("-" * wi for wi in w)

    header = "  ".join(c.ljust(wi) if i == 0 else c.rjust(wi)
                       for i, (c, wi) in enumerate(zip(cols, w)))

    print("\n" + "=" * 80)
    print("COMPLETE FUSION METRIC TABLE (DeepSet Test, N=116)")
    print("=" * 80)
    print(header)
    print(sep)

    for name, r in all_results.items():
        tag = " <-- baseline" if name == "distilbert_baseline" else ""
        row = [
            (name + tag).ljust(w[0]),
            f"{r['accuracy']*100:.2f}%".rjust(w[1]),
            f"{r['precision']*100:.2f}%".rjust(w[2]),
            f"{r['recall']*100:.2f}%".rjust(w[3]),
            f"{r['f1']*100:.2f}%".rjust(w[4]),
            str(r["TP"]).rjust(w[5]),
            str(r["FP"]).rjust(w[6]),
            str(r["TN"]).rjust(w[7]),
            str(r["FN"]).rjust(w[8]),
        ]
        print("  ".join(row))

    print("\n" + "=" * 80)
    print("ERROR RECOVERY vs DistilBERT Baseline")
    print("=" * 80)
    rcols = ["Strategy", "FN Rec.", "FP Fix.", "New FN", "New FP", "dRecall", "dPrec", "dF1"]
    rw    = [32, 7, 7, 7, 7, 8, 8, 8]
    rheader = "  ".join(c.ljust(rw[0]) if i == 0 else c.rjust(rw[i])
                        for i, c in enumerate(rcols))
    rsep = "  ".join("-" * wi for wi in rw)
    print(rheader)
    print(rsep)
    for name, rec in recovery.items():
        row = [
            name.ljust(rw[0]),
            str(rec["fn_recovered"]).rjust(rw[1]),
            str(rec["fp_corrected"]).rjust(rw[2]),
            str(rec["new_fn_introduced"]).rjust(rw[3]),
            str(rec["new_fp_introduced"]).rjust(rw[4]),
            f"{rec['delta_recall']:+.4f}".rjust(rw[5]),
            f"{rec['delta_precision']:+.4f}".rjust(rw[6]),
            f"{rec['delta_f1']:+.4f}".rjust(rw[7]),
        ]
        print("  ".join(row))

    rec  = ranking["recommended"]
    bf1  = ranking["best_f1"]
    br   = ranking["best_recall"]
    bacc = ranking["best_accuracy"]

    print("\n" + "=" * 80)
    print("RECOMMENDATION SUMMARY")
    print("=" * 80)
    print(f"  Best Accuracy  : {bacc['strategy']:35s}  acc={bacc['accuracy']:.4f}")
    print(f"  Best Recall    : {br['strategy']:35s}  rec={br['recall']:.4f}")
    print(f"  Best F1        : {bf1['strategy']:35s}  f1={bf1['f1']:.4f}")
    print(f"\n  Recommended for ArgusX v9:")
    print(f"  Strategy  : {rec['strategy']}")
    print(f"  Accuracy  : {rec['accuracy']:.4f}  ({rec['accuracy']*100:.2f}%)")
    print(f"  Precision : {rec['precision']:.4f}  ({rec['precision']*100:.2f}%)")
    print(f"  Recall    : {rec['recall']:.4f}  ({rec['recall']*100:.2f}%)")
    print(f"  F1        : {rec['f1']:.4f}  ({rec['f1']*100:.2f}%)")
    print(f"  TP={rec['TP']}  FP={rec['FP']}  TN={rec['TN']}  FN={rec['FN']}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("ArgusX v9.4.0 -- DistilBERT + Behavioral RF Fusion Study")
    print("Branch: argusx-v9-adaptive-routing")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    tokenizer, db_model, rf_model, vectorizer = step1_load_models(device)
    prompts, labels = step2_load_data()
    db_probs, rf_probs = step3_get_probabilities(
        tokenizer, db_model, rf_model, vectorizer, prompts, device
    )
    all_results = step4_5_fuse_and_evaluate(db_probs, rf_probs, labels)
    recovery    = step6_error_analysis(db_probs, rf_probs, labels, all_results)
    ranking     = step7_rank(all_results)
    step8_export(all_results, recovery, ranking)

    _print_table(all_results, recovery, ranking)

    print("\n--- Output Files ---")
    for fname in ("fusion_metrics.json", "fusion_comparison.json", "best_strategy.json"):
        p  = RESULTS_DIR / fname
        kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {fname:<30}  {kb:>7.1f} KB  ->  {p}")

    print("\nFusion study complete.")
    print("Next step: implement recommended strategy in adaptive routing.")


if __name__ == "__main__":
    main()
