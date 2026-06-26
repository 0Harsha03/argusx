"""
ArgusX v12 — Phase 1: Calibration Artifact Generator
======================================================
One-time utility to generate and freeze the Platt scaling calibration
models for the validated v10 Prompt Injection inference pipeline.

These artifacts are FROZEN PRODUCTION ASSETS.
They must NOT be regenerated unless the validated calibration data or
models have been intentionally updated and reviewed.

Usage
-----
    # Safe (default): refuses to overwrite existing frozen artifacts
    python scripts/freeze_calibration.py

    # Force regeneration (requires explicit intent)
    python scripts/freeze_calibration.py --force

Artifacts produced
------------------
    app/models/artifacts/platt_db.pkl   — DistilBERT Platt calibrator
    app/models/artifacts/platt_rf.pkl   — RF Platt calibrator

Source procedure
----------------
    Calibration is reproduced exactly from eval_v10_pattern_recovery.py:
      - Dataset:   data/pi_corpus/val.csv  (1,592 samples, frozen)
      - Model:     LogisticRegression(solver='lbfgs', max_iter=1000)
      - No changes to methodology, data, or parameters.
"""

import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_v10_pattern_recovery import (
    step1_load,
    step2_load_val,
    step3_get_val_probs,
    step4_calibrate,
)

ARTIFACT_DIR = ROOT / "app" / "models" / "artifacts"
DB_PATH      = ARTIFACT_DIR / "platt_db.pkl"
RF_PATH      = ARTIFACT_DIR / "platt_rf.pkl"


def _check_existing(force: bool) -> None:
    """
    Refuse to overwrite existing calibration artifacts unless --force is set.
    Exits with a non-zero code if artifacts exist and force is False.
    """
    existing = [p for p in (DB_PATH, RF_PATH) if p.exists()]
    if not existing:
        return  # Nothing to guard against

    if force:
        print("WARNING: --force flag detected. Existing frozen calibration artifacts will be overwritten.")
        print(f"  Overwriting: {', '.join(p.name for p in existing)}")
        print()
        return

    # Default: hard stop
    print("=" * 70)
    print("ERROR: Frozen calibration artifacts already exist.")
    print("=" * 70)
    print()
    for p in existing:
        print(f"  {p}  ({p.stat().st_size} bytes)")
    print()
    print("These are FROZEN PRODUCTION ASSETS for ArgusX v12.")
    print("Overwriting them would break benchmark reproducibility.")
    print()
    print("If you intentionally need to regenerate them (e.g., after a")
    print("validated model or dataset update), run with --force:")
    print()
    print("    python scripts/freeze_calibration.py --force")
    print()
    sys.exit(1)


def main(force: bool = False) -> None:
    _check_existing(force)

    print("Loading models...")
    tokenizer, db_model, rf_model, vectorizer = step1_load()

    print("Loading val.csv...")
    val_texts, val_labels = step2_load_val()

    print("Computing val probabilities...")
    db_val, rf_val = step3_get_val_probs(
        tokenizer, db_model, rf_model, vectorizer, val_texts
    )

    print("Fitting Platt calibration models...")
    platt_db, platt_rf = step4_calibrate(db_val, rf_val, val_labels)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(platt_db, DB_PATH)
    joblib.dump(platt_rf, RF_PATH)

    print(f"Successfully saved {DB_PATH}  ({DB_PATH.stat().st_size} bytes)")
    print(f"Successfully saved {RF_PATH}  ({RF_PATH.stat().st_size} bytes)")
    print()
    print("Calibration artifacts are now frozen.")
    print("Commit platt_db.pkl and platt_rf.pkl to version control.")


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    main(force=force_flag)
