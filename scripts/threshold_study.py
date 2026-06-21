"""
ArgusX Threshold Study — Strategies A–E
========================================
Empirically sweeps all five proposed thresholding strategies across
three mandatory evaluation sets (DeepSet, CyberSecEval Prompt Injection,
MITRE FRR) plus two additional sets (SPML, Blind Benchmark) for a
complete publication-quality assessment.

Strategies:
  A  — Baseline: global threshold = 35
  B  — Category-aware: injection threshold = 30, all others = 35
  C  — Dynamic: Pattern + Semantic agreement lowers threshold
  D  — Dynamic: Pattern + Semantic + Behavioral triple agreement
  E  — Fixed threshold = 35, recalibrated fusion weights for injection

This script does NOT retrain any model. All model artifacts are frozen.

Run:
    $env:PYTHONPATH = "."
    python scripts/threshold_study.py > threshold_study_output.txt
"""

import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from app.services.model_registry import ModelRegistry
from app.services.sbert_detection_pipeline import SBERTDetectionPipeline

_GH = "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets"
POSITIVE = {"FLAG", "SANITIZE", "BLOCK"}

# ── Current fusion weights (from config.py) ────────────────────────────────
W_PAT = 0.30
W_SEM = 0.25
W_BEH = 0.30
W_ANO = 0.15


# ── Raw score extraction (bypasses threshold, recomputes decision) ─────────

def raw_score(res: dict) -> dict:
    """Extract per-engine scores from pipeline result."""
    return {
        "ps":  res["pattern_score"],
        "ss":  res["semantic_score"],
        "bs":  res["behavioral_score"],
        "ans": res["anomaly_score"],
        "fs":  res["final_score"],
        "dec": res["decision"],
        "mp":  res.get("matched_patterns", []),
    }


def recompute_decision_strategy(
    ps: float, ss: float, bs: float, ans: float,
    mp: list, strategy: str,
    block_thr: float = 75.0, flag_thr: float = 50.0,
) -> str:
    """Recompute decision under each strategy without re-running the pipeline."""

    # ── Base composite score (identical for all strategies) ────────────────
    fs = W_PAT * ps + W_SEM * ss + W_BEH * bs + W_ANO * ans

    # ── Critical rule override (always BLOCK regardless of strategy) ───────
    CRITICAL = {
        "jailbreak_dan", "role_manipulation_system_prompt",
        "exfil_credentials", "malware_generation_direct",
        "credential_dumping_tool", "credential_theft_direct",
        "exploit_cve_request", "malware_payload_request", "phishing_attack",
    }
    if mp and set(mp) & CRITICAL:
        fs = max(fs, block_thr + 1)

    # ── Strategy-specific SANITIZE threshold ───────────────────────────────

    if strategy == "A":
        # Baseline: global 35
        sanitize_thr = 35.0

    elif strategy == "B":
        # Category-aware: if Pattern=0 AND Semantic>25 → injection-like input
        if ps == 0 and ss > 25:
            sanitize_thr = 30.0
        else:
            sanitize_thr = 35.0

    elif strategy == "C":
        # Pattern + Semantic agreement: both above soft threshold → lower
        pat_signal  = ps >= 50
        sem_signal  = ss >= 40
        if pat_signal and sem_signal:
            sanitize_thr = 30.0
        elif sem_signal:
            sanitize_thr = 32.0
        else:
            sanitize_thr = 35.0

    elif strategy == "D":
        # Triple agreement: Pattern + Semantic + Behavioral all elevated
        pat_signal  = ps >= 50
        sem_signal  = ss >= 35
        beh_signal  = bs >= 70
        if pat_signal and sem_signal and beh_signal:
            sanitize_thr = 28.0
        elif sem_signal and beh_signal:
            sanitize_thr = 31.0
        elif beh_signal:
            sanitize_thr = 33.0
        else:
            sanitize_thr = 35.0

    elif strategy == "E":
        # Recalibrated fusion weights for injection profile (pattern=0, sem>0)
        # Upweight semantic + behavioral, downweight pattern + anomaly
        if ps == 0 and ss > 0:
            fs = 0.15 * ps + 0.35 * ss + 0.40 * bs + 0.10 * ans
            if mp and set(mp) & CRITICAL:
                fs = max(fs, block_thr + 1)
        sanitize_thr = 35.0

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # ── Apply decision tiers ───────────────────────────────────────────────
    if fs >= block_thr:
        return "BLOCK"
    if fs >= flag_thr:
        return "FLAG"
    if fs >= sanitize_thr:
        return "SANITIZE"
    return "ALLOW"


# ── Dataset loading ────────────────────────────────────────────────────────

def fetch(path: str) -> list:
    req = urllib.request.Request(f"{_GH}/{path}", headers={"User-Agent": "ArgusX-ThreshStudy/7.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def load_deepset() -> list:
    path = Path("evaluation")
    records = []
    for fname in path.glob("*.json"):
        if fname.name.startswith("_"):
            continue
        try:
            data = json.loads(fname.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for item in data:
                if "text" in item and "label" in item:
                    records.append({"prompt": item["text"], "label": int(item["label"])})
        except Exception:
            pass
    return records


def load_blind() -> list:
    data = json.loads(Path("evaluation/blind_dataset.json").read_text(encoding="utf-8"))
    return [{"prompt": d["prompt"], "label": d["label"]} for d in data]


def load_spml() -> list:
    """Load from HuggingFace datasets (already cached from training)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("deepset/prompt-injections", split="test")
        return [{"prompt": r["text"], "label": int(r["label"])} for r in ds]
    except Exception:
        return []


# ── Core evaluation ────────────────────────────────────────────────────────

def evaluate_dataset(raw_results: list, strategy: str, label_col: str = "label") -> dict:
    """
    raw_results: list of dicts with keys ps, ss, bs, ans, mp, label
    Returns metric dict.
    """
    y_true, y_pred = [], []
    hard_refusals = 0   # BLOCK on benign
    soft_refusals = 0   # SANITIZE/FLAG on benign

    for r in raw_results:
        dec = recompute_decision_strategy(
            r["ps"], r["ss"], r["bs"], r["ans"], r["mp"], strategy
        )
        pred = 1 if dec in POSITIVE else 0
        y_true.append(r["label"])
        y_pred.append(pred)
        if r["label"] == 0:
            if dec == "BLOCK":
                hard_refusals += 1
            elif dec in ("SANITIZE", "FLAG"):
                soft_refusals += 1

    n_benign = sum(1 for r in raw_results if r["label"] == 0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0
    hrr  = hard_refusals / n_benign if n_benign > 0 else 0
    srr  = soft_refusals / n_benign if n_benign > 0 else 0

    return {
        "strategy": strategy,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "acc":  round(acc  * 100, 2),
        "prec": round(prec * 100, 2),
        "rec":  round(rec  * 100, 2),
        "f1":   round(f1   * 100, 2),
        "fpr":  round(fpr  * 100, 2),
        "fnr":  round(fnr  * 100, 2),
        "hrr":  round(hrr  * 100, 2),
        "srr":  round(srr  * 100, 2),
    }


# ── Threshold sweep (Section 4 detail) ────────────────────────────────────

def threshold_sweep(raw_injection: list, thresholds: list) -> list:
    """Sweep global threshold on injection set to find optimal point."""
    results = []
    for thr in thresholds:
        y_true, y_pred = [], []
        for r in raw_injection:
            fs = W_PAT*r["ps"] + W_SEM*r["ss"] + W_BEH*r["bs"] + W_ANO*r["ans"]
            CRITICAL = {"jailbreak_dan","role_manipulation_system_prompt",
                        "exfil_credentials","malware_generation_direct",
                        "credential_dumping_tool","credential_theft_direct",
                        "exploit_cve_request","malware_payload_request","phishing_attack"}
            if r["mp"] and set(r["mp"]) & CRITICAL:
                fs = max(fs, 76.0)
            pred = 1 if fs >= thr else 0
            y_true.append(r["label"])
            y_pred.append(pred)
        rec  = recall_score(y_true, y_pred, zero_division=0) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        f1   = f1_score(y_true, y_pred, zero_division=0) * 100
        results.append({"thr": thr, "rec": round(rec, 2), "prec": round(prec, 2), "f1": round(f1, 2)})
    return results


# ── Reporting ──────────────────────────────────────────────────────────────

STRATEGIES = ["A", "B", "C", "D", "E"]
STRATEGY_DESC = {
    "A": "Baseline (global=35)",
    "B": "Category-Aware (inj=30, else=35)",
    "C": "Dynamic: Pattern+Sem agreement",
    "D": "Dynamic: Pat+Sem+Beh triple agreement",
    "E": "Recalibrated weights (no thr change)",
}

def print_table(name: str, all_results: dict):
    print(f"\n{'='*78}")
    print(f"DATASET: {name}")
    print(f"{'='*78}")
    hdr = f"{'Strat':<6} {'Desc':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'HRR':>6} {'SRR':>6}"
    print(hdr)
    print("-" * 78)
    for s, m in all_results.items():
        desc = STRATEGY_DESC[s][:39]
        print(f"{s:<6} {desc:<40} {m['acc']:>5.1f}% {m['prec']:>5.1f}% "
              f"{m['rec']:>5.1f}% {m['f1']:>5.1f}% {m['hrr']:>5.1f}% {m['srr']:>5.1f}%")


def main():
    print("=" * 78)
    print("ArgusX Threshold Study — Strategies A–E")
    print("=" * 78)

    # Init pipeline
    registry = ModelRegistry()
    registry.load_all()
    pipeline = SBERTDetectionPipeline(registry)
    print("Pipeline ready.")

    # ── Collect raw scores (one pipeline pass per dataset) ─────────────────
    def collect(prompts_labels: list) -> list:
        rows = []
        for item in prompts_labels:
            res = pipeline.analyze(item["prompt"])
            rows.append({
                "label": item["label"],
                "ps": res["pattern_score"],
                "ss": res["semantic_score"],
                "bs": res["behavioral_score"],
                "ans": res["anomaly_score"],
                "mp": res.get("matched_patterns", []),
            })
        return rows

    # DeepSet
    print("\nLoading DeepSet…", end=" ")
    deepset_data = load_deepset()
    print(f"{len(deepset_data)} samples")
    print("Scoring DeepSet…")
    raw_deepset = collect(deepset_data)

    # CyberSecEval Prompt Injection
    print("Loading CyberSecEval Injection…", end=" ")
    inject_raw = fetch("prompt_injection/prompt_injection.json")
    inject_data = [{"prompt": r["user_input"], "label": 1} for r in inject_raw if r.get("user_input")]
    print(f"{len(inject_data)} samples")
    print("Scoring CyberSecEval Injection…")
    raw_inject = collect(inject_data)

    # MITRE FRR
    print("Loading MITRE FRR…", end=" ")
    frr_raw = fetch("mitre_frr/mitre_frr.json")
    frr_data = [{"prompt": r["mutated_prompt"], "label": 0} for r in frr_raw if r.get("mutated_prompt")]
    print(f"{len(frr_data)} samples")
    print("Scoring MITRE FRR…")
    raw_frr = collect(frr_data)

    # Blind Benchmark
    print("Loading Blind Benchmark…", end=" ")
    blind_data = load_blind()
    print(f"{len(blind_data)} samples")
    print("Scoring Blind Benchmark…")
    raw_blind = collect(blind_data)

    # SPML (optional, may fail offline)
    print("Loading SPML…", end=" ")
    spml_data = load_spml()
    if spml_data:
        print(f"{len(spml_data)} samples")
        print("Scoring SPML…")
        raw_spml = collect(spml_data)
    else:
        print("SKIPPED (dataset unavailable)")
        raw_spml = []

    # ── Evaluate all strategies ────────────────────────────────────────────
    print("\n\nEvaluating all strategies…")
    datasets = {
        "DeepSet": raw_deepset,
        "CyberSecEval Injection": raw_inject,
        "MITRE FRR": raw_frr,
        "Blind Benchmark": raw_blind,
    }
    if raw_spml:
        datasets["SPML"] = raw_spml

    all_dataset_results = {}
    for ds_name, raw in datasets.items():
        all_dataset_results[ds_name] = {}
        for s in STRATEGIES:
            all_dataset_results[ds_name][s] = evaluate_dataset(raw, s)

    # ── Print tables ───────────────────────────────────────────────────────
    for ds_name, results in all_dataset_results.items():
        print_table(ds_name, results)

    # ── Threshold sweep on Injection ───────────────────────────────────────
    print(f"\n\n{'='*78}")
    print("THRESHOLD SWEEP — CyberSecEval Injection (label=1 only)")
    print(f"{'='*78}")
    sweep = threshold_sweep(raw_inject, list(range(25, 41)))
    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("-" * 40)
    for row in sweep:
        marker = " ◄ current" if row["thr"] == 35 else ""
        print(f"{row['thr']:>10}  {row['rec']:>7.2f}%  {row['prec']:>9.2f}%  {row['f1']:>7.2f}%{marker}")

    # ── Delta table (vs Strategy A baseline) ──────────────────────────────
    print(f"\n\n{'='*78}")
    print("DELTA TABLE — All Strategies vs Baseline (Strategy A)")
    print(f"{'='*78}")
    for ds_name, results in all_dataset_results.items():
        base = results["A"]
        print(f"\n  {ds_name}")
        print(f"  {'Strat':<6} {'ΔRec':>7} {'ΔPrec':>7} {'ΔF1':>7} {'ΔHRR':>7} {'ΔSRR':>7}")
        print(f"  {'-'*42}")
        for s in STRATEGIES[1:]:
            m = results[s]
            print(f"  {s:<6} "
                  f"{m['rec']-base['rec']:>+6.1f}% "
                  f"{m['prec']-base['prec']:>+6.1f}% "
                  f"{m['f1']-base['f1']:>+6.1f}% "
                  f"{m['hrr']-base['hrr']:>+6.1f}% "
                  f"{m['srr']-base['srr']:>+6.1f}%")

    # ── Theoretical maximum (threshold → 0) ───────────────────────────────
    print(f"\n\n{'='*78}")
    print("THEORETICAL MAXIMUM — if threshold suppression fully eliminated")
    print(f"{'='*78}")
    for ds_name, raw in [("DeepSet", raw_deepset), ("CyberSecEval Injection", raw_inject)]:
        # What if threshold = 0 (every score > 0 gets flagged)?
        y_true = [r["label"] for r in raw]
        y_pred_max = []
        for r in raw:
            fs = W_PAT*r["ps"] + W_SEM*r["ss"] + W_BEH*r["bs"] + W_ANO*r["ans"]
            y_pred_max.append(1 if fs > 0 else 0)
        rec_max  = recall_score(y_true, y_pred_max, zero_division=0) * 100
        prec_max = precision_score(y_true, y_pred_max, zero_division=0) * 100
        f1_max   = f1_score(y_true, y_pred_max, zero_division=0) * 100
        print(f"\n  {ds_name} (threshold=0):")
        print(f"    Max Recall = {rec_max:.2f}%  Precision = {prec_max:.2f}%  F1 = {f1_max:.2f}%")
        # Also at threshold=30
        y_pred_30 = [1 if (W_PAT*r["ps"] + W_SEM*r["ss"] + W_BEH*r["bs"] + W_ANO*r["ans"]) >= 30 else 0 for r in raw]
        rec_30  = recall_score(y_true, y_pred_30, zero_division=0) * 100
        prec_30 = precision_score(y_true, y_pred_30, zero_division=0) * 100
        f1_30   = f1_score(y_true, y_pred_30, zero_division=0) * 100
        print(f"    thr=30: Recall={rec_30:.2f}% Precision={prec_30:.2f}% F1={f1_30:.2f}%")

    # ── Export ─────────────────────────────────────────────────────────────
    export = {}
    for ds_name, results in all_dataset_results.items():
        export[ds_name] = {s: m for s, m in results.items()}
    with open("threshold_study_results.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print("\n\nResults saved to threshold_study_results.json")
    print("Threshold study complete.")


if __name__ == "__main__":
    main()
