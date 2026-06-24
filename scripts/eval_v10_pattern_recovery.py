"""
ArgusX v10 — Pattern Recovery Benchmark
=========================================
Branch: argusx-v10-pattern-recovery
Base:   v9.6-calibrated-fusion

Evaluates the two new Pattern Engine rules (translation_bypass,
obfuscated_ignore) against the frozen DeepSet test split.

The pipeline used here mirrors the v9.6 calibrated fusion pipeline
but adds the new pattern rules. We do NOT retrain DistilBERT or RF.

Outputs:
    results/v10_pattern_recovery/metrics.json
    results/v10_pattern_recovery/classification_report.txt
    results/v10_pattern_recovery/confusion_matrix.json
"""

from __future__ import annotations

import io, json, logging, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("v10_pattern_benchmark")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.WARNING)

try:
    import joblib, torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset as hf_load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report,
    )
    from sklearn.calibration import CalibratedClassifierCV
except ImportError as exc:
    print(f"[FATAL] {exc}", file=sys.stderr)
    sys.exit(1)

# Paths
ROOT         = Path(__file__).resolve().parent.parent
MODEL_DIR    = ROOT / "models" / "distilbert_pi"
ARTIFACT_DIR = ROOT / "app" / "models" / "artifacts"
CORPUS_DIR   = ROOT / "data" / "pi_corpus"
OUT_DIR      = ROOT / "results" / "v10_pattern_recovery"

# v9.6 fusion weights (unchanged)
W_DB = 0.90
W_RF = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Step 1: Load models ─────────────────────────────────────────────────────

def step1_load():
    log.info("STEP 1 — Loading models ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    db_model  = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
    db_model.eval()
    log.info("  DistilBERT loaded from %s", MODEL_DIR)

    rf_model  = joblib.load(ARTIFACT_DIR / "behavioral_model.pkl")
    vectorizer = joblib.load(ARTIFACT_DIR / "vectorizer.pkl")
    log.info("  RF model + vectorizer loaded")
    return tokenizer, db_model, rf_model, vectorizer


# ─── Step 2: Load calibration data (val.csv) ─────────────────────────────────

def step2_load_val():
    log.info("STEP 2 — Loading val.csv ...")
    df = pd.read_csv(CORPUS_DIR / "val.csv")
    log.info("  val: %d samples", len(df))
    return df["prompt"].tolist(), df["label"].tolist()


# ─── Step 3: Compute raw probabilities on val ─────────────────────────────────

class _PromptDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        self.n = len(texts)

    def __len__(self): return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def _get_db_probs(texts, tokenizer, db_model):
    ds  = _PromptDataset(texts, tokenizer)
    dl  = DataLoader(ds, batch_size=32)
    probs = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = db_model(**batch).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def step3_get_val_probs(tokenizer, db_model, rf_model, vectorizer, val_texts):
    log.info("STEP 3 — Computing val probabilities ...")
    db_val = _get_db_probs(val_texts, tokenizer, db_model)
    X_val  = vectorizer.transform(val_texts)
    rf_val = rf_model.predict_proba(X_val)[:, 1]
    return db_val, rf_val


# ─── Step 4: Platt scaling on val ────────────────────────────────────────────

def step4_calibrate(db_val, rf_val, val_labels):
    log.info("STEP 4 — Calibrating probabilities (Platt) ...")
    platt_db = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt_db.fit(db_val.reshape(-1, 1), val_labels)
    platt_rf = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt_rf.fit(rf_val.reshape(-1, 1), val_labels)
    log.info("  Calibration complete")
    return platt_db, platt_rf


# ─── Step 5: Benchmark on DeepSet test split ──────────────────────────────────

def step5_benchmark(tokenizer, db_model, rf_model, vectorizer,
                    platt_db, platt_rf):
    log.info("STEP 5 — Loading DeepSet test split ...")
    ds          = hf_load_dataset("deepset/prompt-injections")
    test_texts  = [r["text"] for r in ds["test"]]
    test_labels = [int(r["label"]) for r in ds["test"]]
    log.info("  %d test samples", len(test_labels))

    # ── Raw probabilities ────────────────────────────────────────────────
    db_raw = _get_db_probs(test_texts, tokenizer, db_model)
    X_test = vectorizer.transform(test_texts)
    rf_raw = rf_model.predict_proba(X_test)[:, 1]

    # ── Calibrated probabilities (Platt) ─────────────────────────────────
    db_cal = platt_db.predict_proba(db_raw.reshape(-1, 1))[:, 1]
    rf_cal = platt_rf.predict_proba(rf_raw.reshape(-1, 1))[:, 1]

    # ── v9.6 Fusion (0.90 DB + 0.10 RF) ─────────────────────────────────
    fused_score = W_DB * db_cal + W_RF * rf_cal

    # ── Pattern Engine Layer ──────────────────────────────────────────────
    # Import here so module-level import has the updated rules
    sys.path.insert(0, str(ROOT))
    from app.detection.pattern_detector import PatternDetector
    pd_engine = PatternDetector()

    pattern_flags = []
    for text in test_texts:
        res = pd_engine.analyze(text)
        # If the pattern engine fires with weight ≥ 85 (our new rules),
        # mark as malicious (score > 0 and a named bypass rule hit).
        triggered = any(
            rule in res.matched_rules
            for rule in ("translation_bypass", "obfuscated_ignore")
        )
        pattern_flags.append(triggered)

    pattern_flags = np.array(pattern_flags)

    # ── v10 Decision: Pattern override then fusion threshold ──────────────
    # If pattern flags, predict malicious regardless of fusion score.
    # Otherwise fall back to the v9.6 calibrated fusion threshold of 0.5.
    v10_preds = np.where(pattern_flags, 1, (fused_score >= 0.5).astype(int))

    return test_labels, v10_preds.tolist(), fused_score, pattern_flags


# ─── Step 6: Metrics & export ─────────────────────────────────────────────────

def step6_export(test_labels, preds, fused_scores, pattern_flags):
    log.info("STEP 6 — Computing metrics ...")
    preds_arr = np.array(preds)
    labels    = np.array(test_labels)

    acc  = round(accuracy_score(labels, preds_arr), 4)
    prec = round(precision_score(labels, preds_arr), 4)
    rec  = round(recall_score(labels, preds_arr), 4)
    f1   = round(f1_score(labels, preds_arr), 4)

    tp = int(np.sum((preds_arr == 1) & (labels == 1)))
    fp = int(np.sum((preds_arr == 1) & (labels == 0)))
    tn = int(np.sum((preds_arr == 0) & (labels == 0)))
    fn = int(np.sum((preds_arr == 0) & (labels == 1)))

    log.info("  Accuracy:  %.4f", acc)
    log.info("  Precision: %.4f", prec)
    log.info("  Recall:    %.4f", rec)
    log.info("  F1:        %.4f", f1)
    log.info("  TP=%d  FP=%d  TN=%d  FN=%d", tp, fp, tn, fn)

    # Pattern rule stats
    pattern_hit_idx = [i for i, f in enumerate(pattern_flags) if f]
    log.info("  Pattern flags fired on %d samples: %s", len(pattern_hit_idx), pattern_hit_idx)

    report = classification_report(labels, preds_arr, target_names=["Benign", "Malicious"])
    log.info("\n%s", report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics = {
        "branch":    "argusx-v10-pattern-recovery",
        "base_tag":  "v9.6-calibrated-fusion",
        "dataset":   "deepset/prompt-injections",
        "split":     "test",
        "n_samples": len(test_labels),
        "fusion_weights": {"distilbert": W_DB, "rf": W_RF},
        "pattern_rules_added": ["translation_bypass", "obfuscated_ignore"],
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "delta_vs_v96": {
            "accuracy":  round(acc - 0.9655, 4),
            "precision": round(prec - 0.9667, 4),
            "recall":    round(rec - 0.9667, 4),
            "f1":        round(f1 - 0.9667, 4),
        },
        "pattern_flags_fired": len(pattern_hit_idx),
        "pattern_flagged_indices": pattern_hit_idx,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("  Saved metrics.json")

    # classification_report.txt
    with open(OUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"ArgusX v10 Pattern Recovery — DeepSet Benchmark\n")
        f.write(f"Branch: argusx-v10-pattern-recovery\n")
        f.write(f"Base:   v9.6-calibrated-fusion\n\n")
        f.write(report)
    log.info("  Saved classification_report.txt")

    # confusion_matrix.json
    cm = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    with open(OUT_DIR / "confusion_matrix.json", "w") as f:
        json.dump(cm, f, indent=2)
    log.info("  Saved confusion_matrix.json")

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ArgusX v10 — Pattern Recovery Benchmark")
    print("Branch: argusx-v10-pattern-recovery")
    print("Base:   v9.6-calibrated-fusion")
    print("=" * 70)

    tokenizer, db_model, rf_model, vectorizer = step1_load()
    val_texts, val_labels = step2_load_val()
    db_val, rf_val = step3_get_val_probs(tokenizer, db_model, rf_model, vectorizer, val_texts)
    platt_db, platt_rf = step4_calibrate(db_val, rf_val, val_labels)
    test_labels, preds, fused_scores, pattern_flags = step5_benchmark(
        tokenizer, db_model, rf_model, vectorizer, platt_db, platt_rf
    )
    metrics = step6_export(test_labels, preds, fused_scores, pattern_flags)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
    print(f"\n  Delta vs v9.6 F1: {metrics['delta_vs_v96']['f1']:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
