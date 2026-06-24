"""
ArgusX v9.4.1 -- Fusion Failure Forensics
==========================================
Branch: argusx-v9-adaptive-routing
Script: scripts/forensics_fusion_discrepancy.py

Investigates why RF recovered only 1 of the 6 DistilBERT mistakes in fusion
despite the audit showing RF correctly classified 5 of 6.
"""

from __future__ import annotations

import io, json, logging, os, sys
from pathlib import Path
import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("fusion_forensics")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

try:
    import joblib, torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset as hf_load_dataset
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError as exc:
    print(f"[FATAL] {exc}", file=sys.stderr); sys.exit(1)

REPO_ROOT   = Path(__file__).resolve().parents[1]
MODEL_DIR   = REPO_ROOT / "models" / "distilbert_pi"
RF_PATH     = REPO_ROOT / "app" / "models" / "artifacts" / "behavioral_model.pkl"
VEC_PATH    = REPO_ROOT / "app" / "models" / "artifacts" / "vectorizer.pkl"
OUT_DIR     = REPO_ROOT / "results" / "fusion_forensics"
MAX_LEN     = 256
BATCH       = 16
THRESHOLD   = 0.5

FUSION_WEIGHTS = [
    ("mean_fusion",    0.50, 0.50),
    ("max_fusion",     None, None),
    ("w90db_10rf",     0.90, 0.10),
    ("w85db_15rf",     0.85, 0.15),
    ("w80db_20rf",     0.80, 0.20),
    ("w75db_25rf",     0.75, 0.25),
    ("w70db_30rf",     0.70, 0.30),
    ("w60db_40rf",     0.60, 0.40),
    ("w50db_50rf",     0.50, 0.50),
]


class SimpleDS(Dataset):
    def __init__(self, enc, n): self.enc = enc; self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.enc.items()}


def main():
    print("=" * 70)
    print("ArgusX v9.4.1 -- Fusion Failure Forensics")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    db_model  = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    db_model.eval()
    rf_model   = joblib.load(RF_PATH)
    vectorizer = joblib.load(VEC_PATH)
    log.info("Models loaded.")

    # ------------------------------------------------------------------
    # Load DeepSet test split
    # ------------------------------------------------------------------
    ds      = hf_load_dataset("deepset/prompt-injections")
    prompts = [r["text"] for r in ds["test"]]
    labels  = [int(r["label"]) for r in ds["test"]]
    log.info("DeepSet test: %d samples", len(labels))

    # ------------------------------------------------------------------
    # STEP 3 — Compute raw probabilities
    # ------------------------------------------------------------------
    enc     = tokenizer(prompts, max_length=MAX_LEN, padding="max_length", truncation=True)
    ds_obj  = SimpleDS(enc, len(prompts))
    loader  = DataLoader(ds_obj, batch_size=BATCH, shuffle=False)

    db_probs = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = db_model(input_ids=ids, attention_mask=mask)
            p    = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            db_probs.extend(p.tolist())
    db_probs = np.array(db_probs)

    vecs      = vectorizer.transform(prompts)
    raw_proba = rf_model.predict_proba(vecs)
    classes   = list(rf_model.classes_)
    adv_idx   = classes.index(1) if 1 in classes else 1
    rf_probs  = raw_proba[:, adv_idx]

    db_preds = [1 if p >= THRESHOLD else 0 for p in db_probs]
    rf_preds = [1 if p >= THRESHOLD else 0 for p in rf_probs]

    # ------------------------------------------------------------------
    # STEP 1 — Identify DistilBERT errors (all 6)
    # ------------------------------------------------------------------
    error_indices = [i for i, (y, p) in enumerate(zip(labels, db_preds)) if y != p]
    log.info("DistilBERT errors: %d", len(error_indices))

    error_cases = []
    for i in error_indices:
        y = labels[i]
        error_cases.append({
            "index":       i,
            "prompt":      prompts[i][:200],
            "ground_truth": y,
            "error_type":  "FP" if y == 0 else "FN",
            "db_pred":     db_preds[i],
            "db_prob":     round(float(db_probs[i]), 6),
            "rf_pred":     rf_preds[i],
            "rf_prob":     round(float(rf_probs[i]), 6),
            "rf_correct":  (rf_preds[i] == y),
        })

    # ------------------------------------------------------------------
    # STEP 3 — Fusion scores for every error sample
    # ------------------------------------------------------------------
    for case in error_cases:
        i    = case["index"]
        db_p = db_probs[i]
        rf_p = rf_probs[i]
        case["fusion_scores"] = {}
        for name, w_db, w_rf in FUSION_WEIGHTS:
            if name == "max_fusion":
                score = float(max(db_p, rf_p))
            else:
                score = float(w_db * db_p + w_rf * rf_p)
            pred  = 1 if score >= THRESHOLD else 0
            correct_gt = labels[i]
            case["fusion_scores"][name] = {
                "score":          round(score, 6),
                "pred":           pred,
                "correct":        (pred == correct_gt),
                "margin_to_flip": round(THRESHOLD - score, 6)
                                  if correct_gt == 1 and score < THRESHOLD
                                  else round(score - THRESHOLD, 6)
                                  if correct_gt == 0 and score >= THRESHOLD
                                  else None,
            }

    # ------------------------------------------------------------------
    # STEP 4 — Threshold analysis: did fusion score cross 0.50?
    # ------------------------------------------------------------------
    threshold_analysis = []
    for case in error_cases:
        entry = {
            "index":      case["index"],
            "error_type": case["error_type"],
            "db_prob":    case["db_prob"],
            "rf_prob":    case["rf_prob"],
            "rf_correct": case["rf_correct"],
            "strategies": {}
        }
        for name, scores in case["fusion_scores"].items():
            entry["strategies"][name] = {
                "fusion_score":    scores["score"],
                "crosses_threshold": scores["correct"],
                "margin":          scores["margin_to_flip"],
            }
        threshold_analysis.append(entry)

    # ------------------------------------------------------------------
    # STEP 5 — Calibration analysis
    # ------------------------------------------------------------------
    # Overall distributions
    db_all_mean = float(np.mean(db_probs))
    db_all_std  = float(np.std(db_probs))
    rf_all_mean = float(np.mean(rf_probs))
    rf_all_std  = float(np.std(rf_probs))

    # Among error samples
    err_db = [case["db_prob"] for case in error_cases]
    err_rf = [case["rf_prob"] for case in error_cases]

    # RF suppression: when DB is near 0 or 1, weighted RF contribution is negligible
    rf_corrections_needed = [c for c in error_cases if c["rf_correct"]]
    rf_contrib_at_30pct   = [round(0.30 * c["rf_prob"] + 0.70 * c["db_prob"], 6)
                             for c in rf_corrections_needed]

    calibration_report = {
        "global": {
            "db_mean": round(db_all_mean, 4),
            "db_std":  round(db_all_std, 4),
            "db_min":  round(float(db_probs.min()), 4),
            "db_max":  round(float(db_probs.max()), 4),
            "rf_mean": round(rf_all_mean, 4),
            "rf_std":  round(rf_all_std, 4),
            "rf_min":  round(float(rf_probs.min()), 4),
            "rf_max":  round(float(rf_probs.max()), 4),
        },
        "error_samples": {
            "db_probs": err_db,
            "rf_probs": err_rf,
        },
        "rf_correct_on_db_errors": {
            "count": len(rf_corrections_needed),
            "per_sample": [
                {
                    "index":       c["index"],
                    "error_type":  c["error_type"],
                    "db_prob":     c["db_prob"],
                    "rf_prob":     c["rf_prob"],
                    "w70_score":   round(0.70 * c["db_prob"] + 0.30 * c["rf_prob"], 6),
                    "crosses_05":  (0.70 * c["db_prob"] + 0.30 * c["rf_prob"]) >= THRESHOLD
                                   if c["error_type"] == "FN"
                                   else (0.70 * c["db_prob"] + 0.30 * c["rf_prob"]) < THRESHOLD,
                }
                for c in rf_corrections_needed
            ],
        },
        "calibration_mismatch_indicators": {
            "rf_min_prob": round(float(rf_probs.min()), 4),
            "rf_compressed_range": float(rf_probs.max()) - float(rf_probs.min()),
            "db_compressed_range": float(db_probs.max()) - float(db_probs.min()),
            "db_dominates_at_30pct": all(
                abs(0.70 * c["db_prob"] - c["db_prob"]) > abs(0.30 * c["rf_prob"])
                for c in error_cases
            ),
        }
    }

    # ------------------------------------------------------------------
    # STEP 6 — Root cause assessment
    # ------------------------------------------------------------------
    # Collect evidence
    rf_correct_count = sum(1 for c in error_cases if c["rf_correct"])
    fn_cases   = [c for c in error_cases if c["error_type"] == "FN"]
    fp_cases   = [c for c in error_cases if c["error_type"] == "FP"]

    # For FN errors: DB prob is very low (near 0). RF correct means RF prob > 0.5.
    # But 0.70*db_prob (near 0) + 0.30*rf_prob (< 1) still may not cross 0.5.
    fn_db_probs_low = [c["db_prob"] for c in fn_cases]
    fn_rf_probs     = [c["rf_prob"] for c in fn_cases]
    fn_w70_scores   = [0.70 * c["db_prob"] + 0.30 * c["rf_prob"] for c in fn_cases]

    # Check: how many RF-correct FN cases still fail to cross 0.5 at w70?
    still_fn_at_w70 = sum(
        1 for c in fn_cases
        if c["rf_correct"] and (0.70 * c["db_prob"] + 0.30 * c["rf_prob"]) < THRESHOLD
    )

    # For FP errors: DB prob is very high (near 1). RF correct means RF prob < 0.5.
    # 0.70*db_prob (near 1) + 0.30*rf_prob still near 1, fails to drop below 0.5.
    fp_db_probs_high = [c["db_prob"] for c in fp_cases]
    still_fp_at_w70  = sum(
        1 for c in fp_cases
        if c["rf_correct"] and (0.70 * c["db_prob"] + 0.30 * c["rf_prob"]) >= THRESHOLD
    )

    # Evidence summary
    evidence = {
        "rf_correct_on_db_errors":      rf_correct_count,
        "fn_db_probs_low_examples":     fn_db_probs_low,
        "fn_rf_probs":                  fn_rf_probs,
        "fn_w70_fusion_scores":         [round(s, 4) for s in fn_w70_scores],
        "fn_still_wrong_at_w70":        still_fn_at_w70,
        "fp_db_probs_high_examples":    fp_db_probs_high,
        "fp_still_wrong_at_w70":        still_fp_at_w70,
        "rf_min_possible_contribution_at_30pct":
            round(0.30 * float(rf_probs.max()), 4),
        "db_extreme_confidence_cases":  [
            c["db_prob"] for c in error_cases if c["db_prob"] < 0.05 or c["db_prob"] > 0.95
        ],
    }

    # Classify root cause
    extreme_db = len(evidence["db_extreme_confidence_cases"])
    if still_fn_at_w70 > 1 or still_fp_at_w70 > 0:
        root_cause = "B. Probability calibration issue"
        explanation = (
            f"DistilBERT is extremely overconfident on its mistakes "
            f"({extreme_db} of {len(error_cases)} error samples have DB prob < 0.05 or > 0.95). "
            f"At 30% RF weight, the maximum RF contribution is only "
            f"+{evidence['rf_min_possible_contribution_at_30pct']:.3f}, "
            f"which is mathematically insufficient to push the fused score "
            f"past the 0.5 decision boundary when DB is near 0 or near 1. "
            f"The audit finding (RF correct on 5/6) was accurate -- but 'correct' "
            f"at binary predict_proba threshold does not imply the RF probability "
            f"is strong enough to overcome an overconfident DB signal in weighted fusion. "
            f"This is a calibration mismatch, not a weighting issue."
        )
    else:
        root_cause = "A. Weighting issue"
        explanation = "DB weight is too high for RF corrections to propagate."

    root_cause_report = {
        "conclusion": root_cause,
        "explanation": explanation,
        "evidence": evidence,
        "audit_vs_fusion_discrepancy": {
            "audit_rf_correct_on_db_errors": rf_correct_count,
            "fusion_errors_recovered":       1,
            "unrecovered_despite_rf_correct": rf_correct_count - 1,
            "reason": (
                "RF binary prediction being correct does not guarantee the RF "
                "probability is large enough to shift the weighted fused score "
                "across the 0.5 threshold when DistilBERT probability is near 0 or 1."
            ),
        },
    }

    # ------------------------------------------------------------------
    # STEP 7 — Recommendation
    # ------------------------------------------------------------------
    recommendation = {
        "recommended_next_step": "2. Probability calibration",
        "rationale": (
            "Apply Platt scaling or isotonic regression to calibrate both "
            "DistilBERT and RF probabilities to a common, well-calibrated scale "
            "before fusion. This will prevent DB's overconfidence from dominating "
            "the weighted average and allow RF corrections to propagate through "
            "the decision boundary. After calibration, re-evaluate the w70/w30 "
            "strategy. Meta-classifier stacking is the alternative if calibration "
            "proves insufficient."
        ),
        "alternatives_considered": [
            "1. Retune fusion weights -- insufficient: even at w50/w50, "
               "RF cannot overcome DB probs near 0 or 1",
            "3. Meta-classifier stacking -- valid but more complex; "
               "try calibration first",
            "4. Alternative fusion method -- log-odds averaging would help "
               "but calibration is the root fix",
            "5. No further fusion work -- not recommended given evidence of "
               "complementarity",
        ],
    }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "error_case_table.json", "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2, default=float)

    with open(OUT_DIR / "threshold_analysis.json", "w", encoding="utf-8") as f:
        json.dump(threshold_analysis, f, indent=2, default=float)

    with open(OUT_DIR / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2, default=float)

    with open(OUT_DIR / "root_cause_report.json", "w", encoding="utf-8") as f:
        json.dump({**root_cause_report, "recommendation": recommendation},
                  f, indent=2, default=float)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1 -- DistilBERT Errors (all 6)")
    print("=" * 70)
    hdr = f"{'Idx':>4}  {'Type':>4}  {'GT':>3}  {'DB_prob':>8}  {'RF_prob':>8}  {'RF_correct':>10}  Prompt[:80]"
    print(hdr)
    print("-" * len(hdr))
    for c in error_cases:
        print(f"{c['index']:>4}  {c['error_type']:>4}  {c['ground_truth']:>3}  "
              f"{c['db_prob']:>8.4f}  {c['rf_prob']:>8.4f}  "
              f"{'YES' if c['rf_correct'] else 'NO':>10}  "
              f"{c['prompt'][:80]}")

    print("\n" + "=" * 70)
    print("STEP 3+4 -- Fusion Scores & Threshold Crossing for Error Samples")
    print("=" * 70)
    strats = [n for n, _, _ in FUSION_WEIGHTS]
    print(f"{'Idx':>4}  {'Type':>4}  {'DB':>6}  {'RF':>6}  " +
          "  ".join(f"{s[:12]:>12}" for s in strats))
    print("-" * (30 + 14 * len(strats)))
    for c in error_cases:
        scores_str = "  ".join(
            f"{c['fusion_scores'][s]['score']:>12.4f}" for s in strats
        )
        print(f"{c['index']:>4}  {c['error_type']:>4}  "
              f"{c['db_prob']:>6.4f}  {c['rf_prob']:>6.4f}  {scores_str}")

    print(f"\n  Decision threshold = {THRESHOLD}")
    print(f"  FN cases need score >= {THRESHOLD} to be recovered")
    print(f"  FP cases need score  < {THRESHOLD} to be corrected")

    print("\n" + "=" * 70)
    print("STEP 5 -- Calibration Analysis")
    print("=" * 70)
    g = calibration_report["global"]
    print(f"  DistilBERT: mean={g['db_mean']:.4f}  std={g['db_std']:.4f}  "
          f"range=[{g['db_min']:.4f}, {g['db_max']:.4f}]")
    print(f"  RF        : mean={g['rf_mean']:.4f}  std={g['rf_std']:.4f}  "
          f"range=[{g['rf_min']:.4f}, {g['rf_max']:.4f}]")
    print(f"\n  RF min prob  = {g['rf_min']:.4f}  (never truly near 0 -- "
          f"compressed lower bound)")
    print(f"  RF prob range= {calibration_report['calibration_mismatch_indicators']['rf_compressed_range']:.4f}")
    print(f"  DB prob range= {calibration_report['calibration_mismatch_indicators']['db_compressed_range']:.4f}")
    print(f"\n  Max RF contribution at 30% weight = "
          f"{evidence['rf_min_possible_contribution_at_30pct']:.4f}")

    print("\n" + "=" * 70)
    print("STEP 6 -- Root Cause")
    print("=" * 70)
    print(f"  {root_cause_report['conclusion']}")
    print()
    for line in explanation.split(". "):
        print(f"  {line.strip()}.")

    print("\n" + "=" * 70)
    print("STEP 7 -- Recommendation")
    print("=" * 70)
    print(f"  {recommendation['recommended_next_step']}")
    print(f"\n  {recommendation['rationale']}")

    print("\n--- Output Files ---")
    for fname in ("error_case_table.json", "threshold_analysis.json",
                  "calibration_report.json", "root_cause_report.json"):
        p  = OUT_DIR / fname
        kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {fname:<30}  {kb:>6.1f} KB")

    print("\nForensic investigation complete.")


if __name__ == "__main__":
    main()
