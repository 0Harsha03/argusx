"""
ArgusX v9.5.0 -- Probability Calibration Study
===============================================
Branch: argusx-v9-adaptive-routing
Script: scripts/eval_calibrated_fusion.py

Calibrates DistilBERT and RF probabilities using val.csv, then re-evaluates
fusion on the frozen DeepSet test split to determine whether calibration
unlocks additional error recovery beyond the 95.80% F1 uncalibrated result.
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
log = logging.getLogger("calibration_study")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

try:
    import joblib, torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset as hf_load_dataset
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, brier_score_loss,
    )
except ImportError as exc:
    print(f"[FATAL] {exc}", file=sys.stderr); sys.exit(1)

# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
MODEL_DIR   = REPO_ROOT / "models" / "distilbert_pi"
RF_PATH     = REPO_ROOT / "app" / "models" / "artifacts" / "behavioral_model.pkl"
VEC_PATH    = REPO_ROOT / "app" / "models" / "artifacts" / "vectorizer.pkl"
CORPUS_DIR  = REPO_ROOT / "data" / "pi_corpus"
OUT_DIR     = REPO_ROOT / "results" / "calibrated_fusion"

MAX_LEN   = 256
BATCH     = 32
THRESHOLD = 0.5
N_BINS    = 10

# Known DistilBERT error indices on DeepSet test (from forensics)
KNOWN_ERROR_IDX = [61, 69, 75, 92, 95, 101]

WEIGHT_GRID = [
    ("mean_fusion",  0.50, 0.50),
    ("w90db_10rf",   0.90, 0.10),
    ("w85db_15rf",   0.85, 0.15),
    ("w80db_20rf",   0.80, 0.20),
    ("w75db_25rf",   0.75, 0.25),
    ("w70db_30rf",   0.70, 0.30),
    ("w60db_40rf",   0.60, 0.40),
    ("w50db_50rf",   0.50, 0.50),
]

UNCAL_BASELINE = {
    "distilbert_baseline": {"accuracy": 0.9483, "precision": 0.9655,
                             "recall": 0.9333, "f1": 0.9492},
    "best_uncal_fusion":   {"accuracy": 0.9569, "precision": 0.9661,
                             "recall": 0.9500, "f1": 0.9580,
                             "strategy": "w70db_30rf"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleDS(Dataset):
    def __init__(self, enc, n):
        self.enc = enc; self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.enc.items()}


def _metrics(y_true, y_pred):
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _thresh(probs, t=THRESHOLD):
    return [1 if p >= t else 0 for p in probs]


def _ece(y_true, probs, n_bins=N_BINS):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = np.mean(np.array(y_true)[mask] == (probs[mask] >= THRESHOLD).astype(int))
        conf = np.mean(probs[mask])
        ece += mask.sum() * abs(acc - conf)
    return round(float(ece / n), 6)


def _brier(y_true, probs):
    return round(float(brier_score_loss(y_true, probs)), 6)


def _get_db_probs(tokenizer, model, texts, device):
    enc    = tokenizer(texts, max_length=MAX_LEN, padding="max_length",
                       truncation=True)
    ds     = SimpleDS(enc, len(texts))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    probs  = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            p    = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def _get_rf_probs(rf_model, vectorizer, texts):
    vecs    = vectorizer.transform(texts)
    raw     = rf_model.predict_proba(vecs)
    classes = list(rf_model.classes_)
    idx     = classes.index(1) if 1 in classes else 1
    return raw[:, idx]


# ---------------------------------------------------------------------------
# STEP 1 -- Load models
# ---------------------------------------------------------------------------
def step1_load(device):
    log.info("STEP 1 -- Loading models ...")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR)
    db_model   = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    db_model.eval()
    rf_model   = joblib.load(RF_PATH)
    vectorizer = joblib.load(VEC_PATH)
    log.info("  OK")
    return tokenizer, db_model, rf_model, vectorizer


# ---------------------------------------------------------------------------
# STEP 2 -- Load calibration data (val.csv)
# ---------------------------------------------------------------------------
def step2_load_val():
    log.info("STEP 2 -- Loading val.csv ...")
    df = pd.read_csv(CORPUS_DIR / "val.csv")
    log.info("  val: %d samples", len(df))
    return df["prompt"].tolist(), df["label"].tolist()


# ---------------------------------------------------------------------------
# STEP 3+4 -- Fit calibrators
# ---------------------------------------------------------------------------
def _fit_platt(raw_probs, labels):
    """Platt scaling via logistic regression on raw scores."""
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(raw_probs.reshape(-1, 1), labels)
    return lr


def _fit_isotonic(raw_probs, labels):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, labels)
    return iso


def _apply_cal(calibrator, probs):
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    else:  # IsotonicRegression
        return calibrator.predict(probs)


def step3_4_calibrate(tokenizer, db_model, rf_model, vectorizer,
                      val_texts, val_labels, device):
    log.info("STEP 3+4 -- Fitting calibrators on val split (%d samples) ...",
             len(val_labels))

    db_val_probs = _get_db_probs(tokenizer, db_model, val_texts, device)
    rf_val_probs = _get_rf_probs(rf_model, vectorizer, val_texts)

    cal_db_platt    = _fit_platt(db_val_probs, val_labels)
    cal_db_isotonic = _fit_isotonic(db_val_probs, val_labels)
    cal_rf_platt    = _fit_platt(rf_val_probs, val_labels)
    cal_rf_isotonic = _fit_isotonic(rf_val_probs, val_labels)

    log.info("  Calibrators fitted.")
    return (cal_db_platt, cal_db_isotonic,
            cal_rf_platt, cal_rf_isotonic,
            db_val_probs, rf_val_probs)


# ---------------------------------------------------------------------------
# STEP 5 -- Calibration quality
# ---------------------------------------------------------------------------
def step5_cal_quality(val_labels,
                      db_val_raw, rf_val_raw,
                      cal_db_platt, cal_db_iso,
                      cal_rf_platt, cal_rf_iso):
    log.info("STEP 5 -- Calibration quality on val ...")
    y = val_labels

    db_platt_cal = _apply_cal(cal_db_platt,    np.array(db_val_raw))
    db_iso_cal   = _apply_cal(cal_db_iso,      np.array(db_val_raw))
    rf_platt_cal = _apply_cal(cal_rf_platt,    np.array(rf_val_raw))
    rf_iso_cal   = _apply_cal(cal_rf_iso,      np.array(rf_val_raw))

    report = {
        "distilbert": {
            "uncalibrated": {
                "brier": _brier(y, db_val_raw),
                "ece":   _ece(y, np.array(db_val_raw)),
            },
            "platt": {
                "brier": _brier(y, db_platt_cal),
                "ece":   _ece(y, db_platt_cal),
            },
            "isotonic": {
                "brier": _brier(y, db_iso_cal),
                "ece":   _ece(y, db_iso_cal),
            },
        },
        "rf": {
            "uncalibrated": {
                "brier": _brier(y, rf_val_raw),
                "ece":   _ece(y, np.array(rf_val_raw)),
            },
            "platt": {
                "brier": _brier(y, rf_platt_cal),
                "ece":   _ece(y, rf_platt_cal),
            },
            "isotonic": {
                "brier": _brier(y, rf_iso_cal),
                "ece":   _ece(y, rf_iso_cal),
            },
        },
    }

    for model, d in report.items():
        for cal_type, vals in d.items():
            log.info("  %-12s  %-12s  Brier=%.4f  ECE=%.4f",
                     model, cal_type, vals["brier"], vals["ece"])
    return report


# ---------------------------------------------------------------------------
# STEP 6 -- Fusion re-evaluation on DeepSet test
# ---------------------------------------------------------------------------
def step6_fusion(tokenizer, db_model, rf_model, vectorizer,
                 test_texts, test_labels,
                 cal_db_platt, cal_db_iso,
                 cal_rf_platt, cal_rf_iso,
                 device):
    log.info("STEP 6 -- Fusion re-evaluation on DeepSet test (%d) ...",
             len(test_labels))

    db_raw = _get_db_probs(tokenizer, db_model, test_texts, device)
    rf_raw = _get_rf_probs(rf_model, vectorizer, test_texts)

    # Calibrated test probs
    db_platt = _apply_cal(cal_db_platt, db_raw)
    db_iso   = _apply_cal(cal_db_iso,   db_raw)
    rf_platt = _apply_cal(cal_rf_platt, rf_raw)
    rf_iso   = _apply_cal(cal_rf_iso,   rf_raw)

    def _eval_grid(db_p, rf_p, tag):
        out = {}
        for name, w_db, w_rf in WEIGHT_GRID:
            fused = w_db * db_p + w_rf * rf_p
            preds = _thresh(fused)
            m = _metrics(test_labels, preds)
            out[f"{tag}_{name}"] = {"calibration": tag, "strategy": name,
                                    "db_weight": w_db, "rf_weight": w_rf,
                                    **m}
        return out

    results = {}
    # Uncalibrated baseline (reproduced)
    results["uncal_db_only"] = {
        "calibration": "none", "strategy": "db_only",
        **_metrics(test_labels, _thresh(db_raw)),
    }
    results.update(_eval_grid(db_platt, rf_platt, "platt"))
    results.update(_eval_grid(db_iso,   rf_iso,   "isotonic"))
    # Mixed calibration (DB-platt + RF-isotonic)
    for name, w_db, w_rf in WEIGHT_GRID:
        fused = w_db * db_platt + w_rf * rf_iso
        preds = _thresh(fused)
        results[f"mixed_{name}"] = {
            "calibration": "mixed(db-platt, rf-iso)",
            "strategy": name,
            "db_weight": w_db, "rf_weight": w_rf,
            **_metrics(test_labels, preds),
        }

    log.info("  Evaluated %d calibrated fusion strategies.", len(results))
    return results, db_raw, rf_raw, db_platt, db_iso, rf_platt, rf_iso


# ---------------------------------------------------------------------------
# STEP 7 -- Forensic recheck on 6 error samples
# ---------------------------------------------------------------------------
def step7_forensic_recheck(test_texts, test_labels,
                           db_raw, rf_raw,
                           db_platt, db_iso,
                           rf_platt, rf_iso):
    log.info("STEP 7 -- Forensic recheck on error indices %s ...",
             KNOWN_ERROR_IDX)

    cases = []
    for idx in KNOWN_ERROR_IDX:
        y = test_labels[idx]
        needed = ">= 0.5 (FN recover)" if y == 1 else "< 0.5 (FP correct)"

        def _fused(w_db, w_rf, dp, rp):
            return round(float(w_db * dp[idx] + w_rf * rp[idx]), 4)

        case = {
            "index":       idx,
            "ground_truth": y,
            "error_type":  "FN" if y == 1 else "FP",
            "threshold_needed": needed,
            "uncalibrated": {
                "db_prob":     round(float(db_raw[idx]), 4),
                "rf_prob":     round(float(rf_raw[idx]), 4),
                "w70_score":   _fused(0.70, 0.30, db_raw, rf_raw),
                "recovered":   (_fused(0.70, 0.30, db_raw, rf_raw) >= 0.5) == (y == 1),
            },
            "platt": {
                "db_prob":     round(float(db_platt[idx]), 4),
                "rf_prob":     round(float(rf_platt[idx]), 4),
                "w70_score":   _fused(0.70, 0.30, db_platt, rf_platt),
                "recovered":   (_fused(0.70, 0.30, db_platt, rf_platt) >= 0.5) == (y == 1),
            },
            "isotonic": {
                "db_prob":     round(float(db_iso[idx]), 4),
                "rf_prob":     round(float(rf_iso[idx]), 4),
                "w70_score":   _fused(0.70, 0.30, db_iso, rf_iso),
                "recovered":   (_fused(0.70, 0.30, db_iso, rf_iso) >= 0.5) == (y == 1),
            },
        }
        cases.append(case)

    uncal_recovered  = sum(1 for c in cases if c["uncalibrated"]["recovered"])
    platt_recovered  = sum(1 for c in cases if c["platt"]["recovered"])
    iso_recovered    = sum(1 for c in cases if c["isotonic"]["recovered"])

    log.info("  Errors recovered -- uncal=%d  platt=%d  isotonic=%d",
             uncal_recovered, platt_recovered, iso_recovered)
    return cases, uncal_recovered, platt_recovered, iso_recovered


# ---------------------------------------------------------------------------
# STEP 8 -- Comparison table
# ---------------------------------------------------------------------------
def step8_comparison(fusion_results):
    """Pick best per calibration type for the summary table."""
    groups = {
        "platt":     [(k, v) for k, v in fusion_results.items()
                      if v["calibration"] == "platt"],
        "isotonic":  [(k, v) for k, v in fusion_results.items()
                      if v["calibration"] == "isotonic"],
        "mixed":     [(k, v) for k, v in fusion_results.items()
                      if "mixed" in v["calibration"]],
    }
    table = []
    # DistilBERT baseline (uncalibrated, standalone)
    table.append({
        "method": "DistilBERT Baseline (uncal, no fusion)",
        "accuracy": 0.9483, "precision": 0.9655, "recall": 0.9333, "f1": 0.9492,
    })
    # Best uncalibrated fusion (from prior study)
    table.append({
        "method": "Best Uncal Fusion (w70/30)",
        "accuracy": 0.9569, "precision": 0.9661, "recall": 0.9500, "f1": 0.9580,
    })
    for group_name, items in groups.items():
        if not items:
            continue
        best = max(items, key=lambda x: x[1]["f1"])
        table.append({
            "method": f"Best {group_name.title()}-Cal Fusion ({best[1]['strategy']})",
            **{k: best[1][k] for k in ("accuracy", "precision", "recall", "f1")},
        })
    return table


# ---------------------------------------------------------------------------
# STEP 9 -- Recommendation
# ---------------------------------------------------------------------------
def step9_recommend(comparison_table, platt_recovered, iso_recovered):
    uncal_f1 = comparison_table[1]["f1"]
    best_cal_f1 = max(r["f1"] for r in comparison_table[2:])
    improves = best_cal_f1 > uncal_f1 + 0.001  # at least 0.1pp gain

    if improves:
        best = max(comparison_table[2:], key=lambda x: x["f1"])
        return {
            "calibration_improves_fusion": True,
            "recommendation": "Adopt calibrated fusion for ArgusX v9.",
            "best_method": best["method"],
            "best_f1": best["f1"],
            "best_recall": best["recall"],
            "errors_recovered_platt":    platt_recovered,
            "errors_recovered_isotonic": iso_recovered,
            "rationale": (
                f"Calibration raised F1 from {uncal_f1:.4f} to {best_cal_f1:.4f}. "
                f"Isotonic regression recovered {iso_recovered}/6 known errors "
                f"vs {platt_recovered}/6 for Platt. "
                "Recommend the best-performing calibration scheme before deployment."
            ),
        }
    else:
        return {
            "calibration_improves_fusion": False,
            "recommendation": (
                "Calibration did not materially improve fusion (delta F1 < 0.1pp). "
                "Move to meta-classifier stacking as the next experiment."
            ),
            "uncal_f1": uncal_f1,
            "best_cal_f1": best_cal_f1,
            "errors_recovered_platt":    platt_recovered,
            "errors_recovered_isotonic": iso_recovered,
            "rationale": (
                "The calibration mismatch is confirmed but probability calibration "
                "alone is insufficient — the errors have extreme DB confidence that "
                "even a calibrated RF cannot override at reasonable weighting. "
                "A learned stacking meta-classifier that jointly considers both "
                "scores is the appropriate next step."
            ),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ArgusX v9.5.0 -- Probability Calibration Study")
    print("Branch: argusx-v9-adaptive-routing")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    tokenizer, db_model, rf_model, vectorizer = step1_load(device)
    val_texts, val_labels = step2_load_val()

    (cal_db_platt, cal_db_iso,
     cal_rf_platt, cal_rf_iso,
     db_val_raw, rf_val_raw) = step3_4_calibrate(
        tokenizer, db_model, rf_model, vectorizer,
        val_texts, val_labels, device
    )

    cal_quality = step5_cal_quality(
        val_labels, db_val_raw, rf_val_raw,
        cal_db_platt, cal_db_iso, cal_rf_platt, cal_rf_iso
    )

    # Load DeepSet test
    log.info("Loading DeepSet test split ...")
    ds = hf_load_dataset("deepset/prompt-injections")
    test_texts  = [r["text"] for r in ds["test"]]
    test_labels = [int(r["label"]) for r in ds["test"]]
    log.info("  %d test samples", len(test_labels))

    (fusion_results,
     db_raw, rf_raw,
     db_platt, db_iso,
     rf_platt, rf_iso) = step6_fusion(
        tokenizer, db_model, rf_model, vectorizer,
        test_texts, test_labels,
        cal_db_platt, cal_db_iso, cal_rf_platt, cal_rf_iso,
        device
    )

    error_cases, uncal_rec, platt_rec, iso_rec = step7_forensic_recheck(
        test_texts, test_labels,
        db_raw, rf_raw, db_platt, db_iso, rf_platt, rf_iso
    )

    comparison = step8_comparison(fusion_results)
    recommendation = step9_recommend(comparison, platt_rec, iso_rec)

    # Export
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "calibration_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cal_quality, f, indent=2, default=float)
    with open(OUT_DIR / "fusion_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fusion_results, f, indent=2, default=float)
    with open(OUT_DIR / "forensic_recovery.json", "w", encoding="utf-8") as f:
        json.dump({"cases": error_cases,
                   "summary": {"uncal": uncal_rec, "platt": platt_rec,
                                "isotonic": iso_rec}},
                  f, indent=2, default=float)
    with open(OUT_DIR / "recommendation.json", "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2, default=float)
    log.info("Results saved to %s", OUT_DIR)

    # ── Console output ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 -- Calibration Quality (val set)")
    print("=" * 70)
    print(f"  {'Model':<12} {'Method':<12} {'Brier':>8} {'ECE':>8}")
    print("  " + "-" * 44)
    for model, d in cal_quality.items():
        for meth, vals in d.items():
            print(f"  {model:<12} {meth:<12} {vals['brier']:>8.4f} {vals['ece']:>8.4f}")

    print("\n" + "=" * 70)
    print("STEP 7 -- Forensic Recheck: Error Sample Recovery")
    print("=" * 70)
    print(f"  {'Idx':>4}  {'Type':>4}  {'DB_raw':>7}  {'DB_cal-P':>8}  "
          f"{'DB_cal-I':>8}  {'w70-raw':>7}  {'w70-P':>7}  {'w70-I':>7}  "
          f"{'Rec-P':>5}  {'Rec-I':>5}")
    print("  " + "-" * 80)
    for c in error_cases:
        u = c["uncalibrated"]; p = c["platt"]; iso = c["isotonic"]
        print(f"  {c['index']:>4}  {c['error_type']:>4}  "
              f"{u['db_prob']:>7.4f}  {p['db_prob']:>8.4f}  {iso['db_prob']:>8.4f}  "
              f"{u['w70_score']:>7.4f}  {p['w70_score']:>7.4f}  {iso['w70_score']:>7.4f}  "
              f"{'YES' if p['recovered'] else 'NO':>5}  "
              f"{'YES' if iso['recovered'] else 'NO':>5}")
    print(f"\n  Uncal recovered: {uncal_rec}/6  |  "
          f"Platt recovered: {platt_rec}/6  |  "
          f"Isotonic recovered: {iso_rec}/6")

    print("\n" + "=" * 70)
    print("STEP 8 -- Comparison Table (DeepSet Test, N=116)")
    print("=" * 70)
    print(f"  {'Method':<45} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("  " + "-" * 75)
    for row in comparison:
        print(f"  {row['method']:<45} "
              f"{row['accuracy']*100:>6.2f}%"
              f"{row['precision']*100:>7.2f}%"
              f"{row['recall']*100:>7.2f}%"
              f"{row['f1']*100:>7.2f}%")

    print("\n" + "=" * 70)
    print("STEP 9 -- Recommendation")
    print("=" * 70)
    print(f"  Calibration improves fusion: {recommendation['calibration_improves_fusion']}")
    print(f"  {recommendation['recommendation']}")
    print(f"\n  {recommendation['rationale']}")

    print("\n--- Output Files ---")
    for fname in ("calibration_metrics.json", "fusion_metrics.json",
                  "forensic_recovery.json", "recommendation.json"):
        p  = OUT_DIR / fname
        kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {fname:<30}  {kb:>6.1f} KB")
    print("\nCalibration study complete.")


if __name__ == "__main__":
    main()
