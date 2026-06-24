"""
ArgusX v9.2.0 -- DistilBERT Prompt Injection Classifier
========================================================
Branch  : argusx-v9-adaptive-routing
Script  : scripts/train_distilbert_pi.py

Pipeline
--------
  STEP 1  -- Load corpus       (train.csv / val.csv from data/pi_corpus/)
  STEP 2  -- Model             (distilbert-base-uncased, binary classification)
  STEP 3  -- Tokenisation      (max_length=256, padding, truncation)
  STEP 4  -- Training          (AdamW, lr=2e-5, bs=16, epochs=4, wd=0.01)
  STEP 5  -- Val metrics       (Accuracy, Precision, Recall, F1)
  STEP 6  -- Save model        (models/distilbert_pi/ via save_pretrained)
  STEP 7  -- DeepSet benchmark (frozen test split, 116 samples)
  STEP 8  -- Export results    (results/distilbert_pi/)

Constraints
-----------
  * No SBERT, no RF, no LOF, no routing changes.
  * Pure DistilBERT baseline to reproduce Dual-Stage paper methodology.
  * DeepSet official test split (116) is NEVER used during training.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
log = logging.getLogger("train_distilbert_pi")

# Suppress noisy HuggingFace / tokenizer warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Lazy imports (torch / transformers) -- checked after logging is ready
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
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
    print(
        f"\n[FATAL] Missing dependency: {exc}\n"
        "Install with:\n"
        "  pip install torch transformers datasets scikit-learn\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
CORPUS_DIR  = REPO_ROOT / "data" / "pi_corpus"
MODEL_DIR   = REPO_ROOT / "models" / "distilbert_pi"
RESULTS_DIR = REPO_ROOT / "results" / "distilbert_pi"

# ---------------------------------------------------------------------------
# Hyper-parameters (per spec)
# ---------------------------------------------------------------------------
MODEL_NAME   = "distilbert-base-uncased"
MAX_LENGTH   = 256
BATCH_SIZE   = 16
EPOCHS       = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_STATE = 42
PATIENCE     = 2          # early stopping: epochs without val-loss improvement

# ===========================================================================
# Dataset helper
# ===========================================================================

class PIDataset(Dataset):
    """PyTorch Dataset wrapping tokenised prompt-injection records."""

    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ===========================================================================
# STEP 1 -- Load corpus
# ===========================================================================

def step1_load() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("STEP 1 -- Loading corpus from %s ...", CORPUS_DIR)

    train_path = CORPUS_DIR / "train.csv"
    val_path   = CORPUS_DIR / "val.csv"

    for p in (train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {p}\n"
                "Run scripts/build_pi_corpus.py first."
            )

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)

    log.info("  train: %d rows | val: %d rows", len(df_train), len(df_val))
    log.info(
        "  train labels -- benign=%d  malicious=%d",
        (df_train["label"] == 0).sum(), (df_train["label"] == 1).sum(),
    )
    log.info(
        "  val   labels -- benign=%d  malicious=%d",
        (df_val["label"] == 0).sum(), (df_val["label"] == 1).sum(),
    )
    return df_train, df_val


# ===========================================================================
# STEP 2 -- Model
# ===========================================================================

def step2_load_model(device: torch.device) -> tuple:
    log.info("STEP 2 -- Loading model: %s ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model = model.to(device)
    log.info(
        "  Parameters: %s  |  Device: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
        device,
    )
    return tokenizer, model


# ===========================================================================
# STEP 3 -- Tokenisation
# ===========================================================================

def step3_tokenise(
    tokenizer,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[PIDataset, PIDataset]:
    log.info("STEP 3 -- Tokenising (max_length=%d) ...", MAX_LENGTH)

    def _encode(texts: list[str]) -> dict:
        return tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )

    train_enc = _encode(df_train["prompt"].tolist())
    val_enc   = _encode(df_val["prompt"].tolist())

    train_ds = PIDataset(train_enc, df_train["label"].tolist())
    val_ds   = PIDataset(val_enc,   df_val["label"].tolist())

    log.info("  Tokenisation complete.")
    return train_ds, val_ds


# ===========================================================================
# Metrics helper
# ===========================================================================

def _compute_metrics(y_true: list[int], y_pred: list[int], prefix: str = "") -> dict:
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    tag   = f"{prefix}_" if prefix else ""
    return {
        f"{tag}accuracy":  round(float(acc),  4),
        f"{tag}precision": round(float(prec), 4),
        f"{tag}recall":    round(float(rec),  4),
        f"{tag}f1":        round(float(f1),   4),
    }


def _predict(model, loader: DataLoader, device: torch.device) -> tuple[list, list]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


# ===========================================================================
# STEP 4 -- Training  |  STEP 5 -- Metrics
# ===========================================================================

def step4_train(
    model,
    train_ds: PIDataset,
    val_ds: PIDataset,
    device: torch.device,
) -> dict:
    log.info("STEP 4 -- Training (epochs=%d, lr=%s, bs=%d, wd=%s) ...",
             EPOCHS, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY)

    torch.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps    = len(train_loader) * EPOCHS
    warmup_steps   = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss   = float("inf")
    best_val_metrics: dict = {}
    no_improve_epochs = 0
    epoch_logs: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if step % 100 == 0:
                log.info(
                    "  Epoch %d/%d  step %d/%d  loss=%.4f",
                    epoch, EPOCHS, step, len(train_loader), epoch_loss / step,
                )

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validation pass ────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_gpu     = batch["labels"].to(device)
                labels_cpu     = batch["labels"]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_gpu,
                )
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels_cpu.tolist())

        avg_val_loss = val_loss / len(val_loader)
        metrics      = _compute_metrics(y_true, y_pred, prefix="val")
        elapsed      = time.time() - t0

        log.info(
            "  [Epoch %d/%d]  train_loss=%.4f  val_loss=%.4f  "
            "val_acc=%.4f  val_f1=%.4f  (%.1fs)",
            epoch, EPOCHS,
            avg_train_loss, avg_val_loss,
            metrics["val_accuracy"], metrics["val_f1"],
            elapsed,
        )

        epoch_logs.append({
            "epoch":          epoch,
            "train_loss":     round(avg_train_loss, 4),
            "val_loss":       round(avg_val_loss,   4),
            **metrics,
        })

        # ── Early stopping / best-model selection ──────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            best_val_metrics = {
                "best_epoch": epoch,
                "val_loss":   round(avg_val_loss, 4),
                **metrics,
            }
            no_improve_epochs = 0
            # Save a checkpoint of the best weights in memory
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info("  -> New best val_loss=%.4f saved.", best_val_loss)
        else:
            no_improve_epochs += 1
            log.info(
                "  -> No improvement (%d/%d patience).",
                no_improve_epochs, PATIENCE,
            )
            if no_improve_epochs >= PATIENCE:
                log.info("  Early stopping triggered at epoch %d.", epoch)
                break

    # Restore best weights before saving
    log.info("  Restoring best weights (epoch %d).", best_val_metrics["best_epoch"])
    model.load_state_dict(best_state)

    return {
        "epoch_logs":        epoch_logs,
        "best_val_metrics":  best_val_metrics,
    }


# ===========================================================================
# STEP 6 -- Save model
# ===========================================================================

def step6_save(model, tokenizer) -> None:
    log.info("STEP 6 -- Saving model to %s ...", MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    log.info("  Model saved.")


# ===========================================================================
# STEP 7 -- DeepSet benchmark (frozen test split, 116 samples)
# ===========================================================================

def step7_deepset_benchmark(model, tokenizer, device: torch.device) -> dict:
    log.info("STEP 7 -- DeepSet benchmark (frozen test split, N=116) ...")

    ds = hf_load_dataset("deepset/prompt-injections")
    df_test = pd.DataFrame([
        {"prompt": r["text"], "label": int(r["label"])}
        for r in ds["test"]
    ])
    log.info("  Loaded DeepSet test split: %d samples", len(df_test))

    enc = tokenizer(
        df_test["prompt"].tolist(),
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )
    test_ds     = PIDataset(enc, df_test["label"].tolist())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred = _predict(model, test_loader, device)

    metrics = _compute_metrics(y_true, y_pred, prefix="deepset_test")
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = (
        cm[0][0], cm[0][1],
        cm[1][0], cm[1][1],
    )

    log.info(
        "  DeepSet Test  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
        metrics["deepset_test_accuracy"],
        metrics["deepset_test_precision"],
        metrics["deepset_test_recall"],
        metrics["deepset_test_f1"],
    )
    log.info("  Confusion Matrix  TP=%d  FP=%d  TN=%d  FN=%d", tp, fp, tn, fn)

    report = classification_report(
        y_true, y_pred,
        target_names=["benign", "malicious"],
        digits=4,
    )

    return {
        "metrics":       metrics,
        "confusion_matrix": {
            "matrix": cm,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        },
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ===========================================================================
# STEP 8 -- Export results
# ===========================================================================

def step8_export(train_results: dict, deepset_results: dict) -> None:
    log.info("STEP 8 -- Exporting results to %s ...", RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── metrics.json ───────────────────────────────────────────────────────
    metrics_payload = {
        "model":          MODEL_NAME,
        "hyperparameters": {
            "max_length":    MAX_LENGTH,
            "batch_size":    BATCH_SIZE,
            "epochs_max":    EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay":  WEIGHT_DECAY,
            "random_state":  RANDOM_STATE,
            "early_stopping_patience": PATIENCE,
        },
        "training": {
            "epoch_logs": train_results["epoch_logs"],
        },
        "validation_best": train_results["best_val_metrics"],
        "deepset_test":    deepset_results["metrics"],
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, default=float)
    log.info("  metrics.json written.")

    # ── classification_report.txt ──────────────────────────────────────────
    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("ArgusX v9.2.0 -- DistilBERT Prompt Injection Classifier\n")
        f.write("=" * 60 + "\n")
        f.write("DeepSet Official Test Split (N=116) -- Frozen Benchmark\n")
        f.write("=" * 60 + "\n\n")
        f.write(deepset_results["classification_report"])
        f.write("\n\nValidation Best (SPML val):\n")
        f.write(json.dumps(train_results["best_val_metrics"], indent=2))
    log.info("  classification_report.txt written.")

    # ── confusion_matrix.json ──────────────────────────────────────────────
    with open(RESULTS_DIR / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(deepset_results["confusion_matrix"], f, indent=2, default=int)
    log.info("  confusion_matrix.json written.")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("ArgusX v9.2.0 -- DistilBERT Prompt Injection Classifier")
    print("Branch: argusx-v9-adaptive-routing")
    print("=" * 70)

    # Seed
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("  GPU: %s", torch.cuda.get_device_name(0))

    # STEP 1
    df_train, df_val = step1_load()

    # STEP 2
    tokenizer, model = step2_load_model(device)

    # STEP 3
    train_ds, val_ds = step3_tokenise(tokenizer, df_train, df_val)

    # STEP 4 + 5
    train_results = step4_train(model, train_ds, val_ds, device)

    # STEP 6
    step6_save(model, tokenizer)

    # STEP 7
    deepset_results = step7_deepset_benchmark(model, tokenizer, device)

    # STEP 8
    step8_export(train_results, deepset_results)

    # ── Final summary ───────────────────────────────────────────────────────
    bv = train_results["best_val_metrics"]
    dm = deepset_results["metrics"]
    cm = deepset_results["confusion_matrix"]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n--- Validation Best (SPML val, epoch {bv['best_epoch']}) ---")
    print(f"  Accuracy  : {bv['val_accuracy']:.4f}")
    print(f"  Precision : {bv['val_precision']:.4f}")
    print(f"  Recall    : {bv['val_recall']:.4f}")
    print(f"  F1        : {bv['val_f1']:.4f}")

    print("\n--- DeepSet Benchmark (frozen test, N=116) ---")
    print(f"  Accuracy  : {dm['deepset_test_accuracy']:.4f}")
    print(f"  Precision : {dm['deepset_test_precision']:.4f}")
    print(f"  Recall    : {dm['deepset_test_recall']:.4f}")
    print(f"  F1        : {dm['deepset_test_f1']:.4f}")
    print(f"  TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}")

    print("\n--- Output Files ---")
    for fname in ("metrics.json", "classification_report.txt", "confusion_matrix.json"):
        p = RESULTS_DIR / fname
        kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {fname:<30}  {kb:>7.1f} KB  ->  {p}")

    print(f"\n  Model saved to: {MODEL_DIR}")
    print("\nCorpus build complete. DeepSet official test split (116) was NEVER")
    print("used during training -- it is the frozen benchmark only.")
    print("\nNext step: RF integration / adaptive routing (not in scope here).")


if __name__ == "__main__":
    main()
