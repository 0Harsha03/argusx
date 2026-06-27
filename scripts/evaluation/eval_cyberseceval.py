"""
ArgusX v7 — CyberSecEval Evaluation Suite
==========================================
Publication-quality evaluation of ArgusX v7 (SBERT) against five
CyberSecEval subsets from Meta's Purple Llama initiative.

Subsets evaluated:
  PRIMARY:
    mitre           — 1,000 ATT&CK-mapped offensive prompts (label=1)
    spear_phishing  —   856 multi-turn phishing scenarios (label=1)
    mitre_frr       —   750 benign security queries (label=0)

  OPTIONAL:
    prompt_injection —  251 direct/indirect injection prompts (label=1)
    interpreter      —  500 code interpreter abuse prompts (label=1)

Label logic:
  mitre, spear_phishing, prompt_injection, interpreter → ALL label=1
  mitre_frr → ALL label=0  (False Refusal Rate test set)

Decision mapping:
  BLOCK | SANITIZE | FLAG  → pred=1 (Threat Detected)
  ALLOW                   → pred=0 (Benign)

TP / FP / TN / FN:
  TP: label=1, pred=1  (attack correctly blocked)
  FP: label=0, pred=1  (benign security query incorrectly blocked)
  TN: label=0, pred=0  (benign query correctly allowed)
  FN: label=1, pred=0  (attack missed — allowed through)

Run:
    $env:PYTHONPATH = "."
    python scripts/eval_cyberseceval.py [--skip-optional]
"""

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from app.services.model_registry import ModelRegistry
from app.services.sbert_detection_pipeline import SBERTDetectionPipeline

pd.options.mode.chained_assignment = None

# ── GitHub raw base URL ────────────────────────────────────────────────────────
_GH_BASE = (
    "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main"
    "/CybersecurityBenchmarks/datasets"
)

POSITIVE_DECISIONS = {"FLAG", "SANITIZE", "BLOCK"}


# ── Subset Definitions ─────────────────────────────────────────────────────────
@dataclass
class Subset:
    name: str
    path: str
    label: int                          # 1 = malicious, 0 = benign
    prompt_field: str                   # JSON key containing the prompt text
    description: str
    expected_n: int
    attack_category: str
    results: dict = field(default_factory=dict)


SUBSETS = [
    Subset(
        name="MITRE (Offensive)",
        path="mitre/mitre_benchmark_100_per_category_with_augmentation.json",
        label=1,
        prompt_field="base_prompt",
        description="1,000 ATT&CK-mapped offensive cyber prompts (malicious)",
        expected_n=1000,
        attack_category="Malware / Exploitation / C2",
    ),
    Subset(
        name="Spear Phishing",
        path="spear_phishing/multiturn_phishing_challenges.json",
        label=1,
        prompt_field=None,              # complex nested structure — handled in loader
        description="856 multi-turn spear-phishing scenarios (malicious)",
        expected_n=856,
        attack_category="Phishing / Social Engineering",
    ),
    Subset(
        name="MITRE FRR (Benign)",
        path="mitre_frr/mitre_frr.json",
        label=0,
        prompt_field="mutated_prompt",
        description="750 benign cybersecurity education queries (False Refusal Rate)",
        expected_n=750,
        attack_category="Benign / Security Education",
    ),
    Subset(
        name="Prompt Injection",
        path="prompt_injection/prompt_injection.json",
        label=1,
        prompt_field="user_input",
        description="251 direct and indirect prompt injection attacks",
        expected_n=251,
        attack_category="Prompt Injection / System Prompt Extraction",
    ),
    Subset(
        name="Interpreter Abuse",
        path="interpreter/interpreter.json",
        label=1,
        prompt_field="mutated_prompt",
        description="500 code interpreter sandbox-escape prompts",
        expected_n=500,
        attack_category="RCE / Privilege Escalation",
    ),
]


# ── Data Loading ───────────────────────────────────────────────────────────────

def fetch_json(path: str) -> list:
    url = f"{_GH_BASE}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "ArgusX-Eval/7.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def extract_phishing_prompt(row: dict) -> Optional[str]:
    """
    Spear phishing rows are complex (multi-turn scenarios).
    We extract the final 'phishing_email' or 'challenge' as the prompt to evaluate.
    """
    # Try common field names
    for field in ("prompt", "challenge", "phishing_email",
                  "user_input", "message", "phishing_message"):
        if field in row and isinstance(row[field], str):
            return row[field]
    # Nested attack scenario
    for field in ("attack_scenario", "scenario"):
        if field in row and isinstance(row[field], dict):
            nested = row[field]
            for nf in ("prompt", "message", "content"):
                if nf in nested:
                    return str(nested[nf])
    # Last resort — serialize top-level string values
    parts = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 30]
    return parts[0] if parts else None


def load_subset(subset: Subset) -> pd.DataFrame:
    raw = fetch_json(subset.path)
    rows = []
    for item in raw:
        if subset.name == "Spear Phishing":
            prompt = extract_phishing_prompt(item)
        else:
            prompt = item.get(subset.prompt_field, "")
        if not prompt or not isinstance(prompt, str):
            continue
        prompt = prompt.strip()
        if not prompt:
            continue
        rows.append({
            "prompt":   prompt,
            "label":    subset.label,
            "category": subset.attack_category,
            "raw":      item,
        })
    df = pd.DataFrame(rows)
    return df


# ── Pipeline Evaluation ────────────────────────────────────────────────────────

def evaluate_subset(pipeline, df: pd.DataFrame, subset: Subset) -> dict:
    y_true, y_pred, records = [], [], []
    t0 = time.time()

    for _, row in df.iterrows():
        res = pipeline.analyze(row["prompt"])
        pred = 1 if res["decision"] in POSITIVE_DECISIONS else 0

        y_true.append(row["label"])
        y_pred.append(pred)
        records.append({
            "prompt":   row["prompt"][:120],
            "label":    row["label"],
            "category": row["category"],
            "ps":       res["pattern_score"],
            "ss":       res["semantic_score"],
            "bs":       res["behavioral_score"],
            "ans":      res["anomaly_score"],
            "fs":       res["final_score"],
            "dec":      res["decision"],
            "pred":     pred,
        })

    elapsed = time.time() - t0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Positive Rate
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # False Negative Rate
    frr  = fpr if subset.label == 0 else float("nan")  # False Refusal Rate only for FRR set

    # Top failures
    fns = sorted([r for r in records if r["label"] == 1 and r["pred"] == 0],
                 key=lambda x: x["fs"], reverse=True)[:25]
    fps = sorted([r for r in records if r["label"] == 0 and r["pred"] == 1],
                 key=lambda x: x["fs"], reverse=True)[:25]

    return {
        "subset":   subset.name,
        "n":        len(df),
        "label":    subset.label,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "acc":      round(acc  * 100, 2),
        "prec":     round(prec * 100, 2),
        "rec":      round(rec  * 100, 2),
        "f1":       round(f1   * 100, 2),
        "fpr":      round(fpr  * 100, 2),
        "fnr":      round(fnr  * 100, 2),
        "frr":      round(frr  * 100, 2) if not isinstance(frr, float) or not __import__("math").isnan(frr) else "N/A",
        "elapsed_s": round(elapsed, 1),
        "cms_per_sample": round(elapsed / len(df) * 1000, 1),
        "cm":       cm,
        "fns":      fns,
        "fps":      fps,
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_separator(char="=", width=78):
    print(char * width)

def print_results(results: list):
    print_separator()
    print("ArgusX v7 — CyberSecEval Publication Benchmark")
    print_separator()

    # ── Per-subset metrics ────────────────────────────────────────────────────
    print("\nTABLE 1: Detection Metrics per Subset (Threshold = 35.0)")
    print_separator("-")
    hdr = f"{'Subset':<26} {'N':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>7} {'FNR':>7}"
    print(hdr)
    print_separator("-")
    for r in results:
        frr_flag = "  ← FRR" if r["label"] == 0 else ""
        print(
            f"{r['subset']:<26} {r['n']:>5} "
            f"{r['acc']:>6.2f}% {r['prec']:>6.2f}% "
            f"{r['rec']:>6.2f}% {r['f1']:>6.2f}% "
            f"{r['fpr']:>6.2f}% {r['fnr']:>6.2f}%{frr_flag}"
        )

    # ── Confusion matrices ────────────────────────────────────────────────────
    print("\n\nTABLE 2: Confusion Matrices")
    print_separator("-")
    for r in results:
        print(f"\n  {r['subset']} (N={r['n']})")
        print(f"    Label semantics: {'label=1 → malicious' if r['label'] == 1 else 'label=0 → benign (FRR test)'}")
        print(f"    TN={r['TN']:4d}  FP={r['FP']:4d}")
        print(f"    FN={r['FN']:4d}  TP={r['TP']:4d}")
        print(f"    Matrix:\n{r['cm']}")

    # ── False Refusal Rate ────────────────────────────────────────────────────
    frr_results = [r for r in results if r["label"] == 0]
    if frr_results:
        print("\n\nTABLE 3: False Refusal Rate (Benign Subsets)")
        print_separator("-")
        for r in frr_results:
            print(f"  {r['subset']}: {r['FP']} of {r['n']} benign queries blocked → FRR = {r['fpr']:.2f}%")

    # ── Weighted aggregate ────────────────────────────────────────────────────
    malicious_res = [r for r in results if r["label"] == 1]
    if malicious_res:
        total_n  = sum(r["n"]  for r in malicious_res)
        total_tp = sum(r["TP"] for r in malicious_res)
        total_fn = sum(r["FN"] for r in malicious_res)
        macro_f1 = sum(r["f1"] for r in malicious_res) / len(malicious_res)
        macro_rec = sum(r["rec"] for r in malicious_res) / len(malicious_res)
        macro_prec = sum(r["prec"] for r in malicious_res) / len(malicious_res)
        print("\n\nTABLE 4: Weighted Aggregate (Malicious Subsets Only)")
        print_separator("-")
        print(f"  Total malicious prompts evaluated : {total_n}")
        print(f"  Total TPs (correctly blocked)     : {total_tp}")
        print(f"  Total FNs (missed attacks)        : {total_fn}")
        print(f"  Macro-avg Precision               : {macro_prec:.2f}%")
        print(f"  Macro-avg Recall                  : {macro_rec:.2f}%")
        print(f"  Macro-avg F1                      : {macro_f1:.2f}%")


def print_failure_analysis(results: list, top_n: int = 10):
    print_separator()
    print("FAILURE ANALYSIS")
    print_separator()
    for r in results:
        print(f"\n{r['subset']} — Top {top_n} FNs (missed attacks)")
        fns = r["fns"][:top_n]
        if not fns:
            print("  No False Negatives.")
        for i, row in enumerate(fns, 1):
            snippet = row["prompt"][:80].replace("\n", " ")
            print(f"  [{i:02d}] FS={row['fs']:.2f} "
                  f"Pat:{row['ps']:.0f} Sem:{row['ss']:.1f} "
                  f"Beh:{row['bs']:.1f} | {snippet}")

        print(f"\n{r['subset']} — Top {top_n} FPs (benign incorrectly blocked)")
        fps = r["fps"][:top_n]
        if not fps:
            print("  No False Positives.")
        for i, row in enumerate(fps, 1):
            snippet = row["prompt"][:80].replace("\n", " ")
            print(f"  [{i:02d}] FS={row['fs']:.2f} "
                  f"Pat:{row['ps']:.0f} Sem:{row['ss']:.1f} "
                  f"Beh:{row['bs']:.1f} | {snippet}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ArgusX v7 CyberSecEval Evaluation")
    parser.add_argument("--skip-optional", action="store_true",
                        help="Skip optional subsets (prompt_injection, interpreter)")
    parser.add_argument("--output", type=str, default="cyberseceval_results.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    # Select subsets
    primary_names   = {"MITRE (Offensive)", "Spear Phishing", "MITRE FRR (Benign)"}
    subsets_to_run  = [s for s in SUBSETS
                       if s.name in primary_names or not args.skip_optional]

    # Init pipeline
    print("Initialising ArgusX v7 SBERT pipeline…")
    registry = ModelRegistry()
    registry.load_all()
    pipeline = SBERTDetectionPipeline(registry)
    print("Pipeline ready.\n")

    all_results = []
    for subset in subsets_to_run:
        print(f"  Loading '{subset.name}'…", end=" ")
        try:
            df = load_subset(subset)
            print(f"{len(df)} samples")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        print(f"  Evaluating '{subset.name}'…", end=" ")
        result = evaluate_subset(pipeline, df, subset)
        all_results.append(result)
        print(f"done. F1={result['f1']}% Rec={result['rec']}% "
              f"FPR={result['fpr']}% [{result['elapsed_s']}s]")

    # Print tables
    print()
    print_results(all_results)
    print()
    print_failure_analysis(all_results)

    # Export
    export = []
    for r in all_results:
        e = {k: v for k, v in r.items() if k not in ("cm", "fns", "fps")}
        e["cm"] = r["cm"].tolist()
        export.append(e)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
