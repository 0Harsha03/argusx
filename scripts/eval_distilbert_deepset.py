"""
ArgusX v9.2.1 -- DistilBERT Benchmark Reproduction
===================================================
Branch  : argusx-v9-adaptive-routing
Script  : scripts/eval_distilbert_deepset.py

Objective
---------
Verify reproducibility of the trained DistilBERT prompt injection classifier
by re-running inference on the frozen DeepSet test split using the saved checkpoint.

Pipeline
--------
  STEP 1  -- Load model & tokenizer from models/distilbert_pi/
  STEP 2  -- Load DeepSet official TEST split ONLY
  STEP 3  -- Inference only (no training/fine-tuning)
  STEP 4  -- Compute metrics
  STEP 5  -- Export results to results/distilbert_reproduction/

Expected target metrics:
  Accuracy  ≈ 94.83%
  Precision ≈ 96.55%
  Recall    ≈ 93.33%
  F1        ≈ 94.92%
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Force UTF-8 stdout on Windows
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("eval_distilbert")

# Suppress noisy HuggingFace warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset as hf_load_dataset
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
except ImportError as exc:
    print(f"\n[FATAL] Missing dependency: {exc}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
MODEL_DIR   = REPO_ROOT / "models" / "distilbert_pi"
RESULTS_DIR = REPO_ROOT / "results" / "distilbert_reproduction"

MAX_LENGTH = 256
BATCH_SIZE = 16


class PIDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return {
        "accuracy":  round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec), 4),
        "f1":        round(float(f1), 4),
    }


def main() -> None:
    print("=" * 70)
    print("ArgusX v9.2.1 -- DistilBERT Benchmark Reproduction")
    print("Branch: argusx-v9-adaptive-routing")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ---------------------------------------------------------
    # STEP 1 -- Load model & tokenizer
    # ---------------------------------------------------------
    log.info("STEP 1 -- Loading model from %s ...", MODEL_DIR)
    if not MODEL_DIR.exists():
        log.error("Model directory not found: %s", MODEL_DIR)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # STEP 2 -- Load DeepSet test split
    # ---------------------------------------------------------
    log.info("STEP 2 -- Loading deepset/prompt-injections [test split] ...")
    ds = hf_load_dataset("deepset/prompt-injections")
    df_test = [{"prompt": r["text"], "label": int(r["label"])} for r in ds["test"]]
    
    prompts = [r["prompt"] for r in df_test]
    labels  = [r["label"] for r in df_test]
    log.info("  Loaded DeepSet test split: %d samples", len(labels))

    enc = tokenizer(
        prompts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )
    test_ds = PIDataset(enc, labels)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # STEP 3 -- Inference
    # ---------------------------------------------------------
    log.info("STEP 3 -- Running inference ...")
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels   = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            
            y_pred.extend(preds)
            y_true.extend(batch_labels.tolist())

    # ---------------------------------------------------------
    # STEP 4 -- Compute metrics
    # ---------------------------------------------------------
    log.info("STEP 4 -- Computing metrics ...")
    metrics = _compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    report = classification_report(
        y_true, y_pred,
        target_names=["benign", "malicious"],
        digits=4,
    )

    cm_data = {
        "matrix": cm,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }

    # ---------------------------------------------------------
    # STEP 5 -- Export results
    # ---------------------------------------------------------
    log.info("STEP 5 -- Exporting results to %s ...", RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("ArgusX v9.2.1 -- DistilBERT Benchmark Reproduction\n")
        f.write("=" * 60 + "\n")
        f.write("DeepSet Official Test Split (N=116) -- Frozen Benchmark\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    with open(RESULTS_DIR / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(cm_data, f, indent=2, default=int)

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("REPRODUCTION RESULTS (DeepSet Test, N=116)")
    print("=" * 70)
    
    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"  F1 Score  : {metrics['f1'] * 100:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    
    print("\n--- Output Files ---")
    for fname in ("metrics.json", "classification_report.txt", "confusion_matrix.json"):
        p = RESULTS_DIR / fname
        kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {fname:<30}  {kb:>7.1f} KB  ->  {p}")

    print("\n✅ Reproducibility benchmark complete.")


if __name__ == "__main__":
    main()
