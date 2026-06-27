"""
ArgusX v7-SBERT Benchmark Evaluation
=====================================
Strict LODO evaluation: same test splits as train_eval_v6_lodo.py.
Uses the SAME v6.0 LODO model artifacts (RF + vectorizer + IF).
Only the SemanticAnalyzer is swapped to SBERT.

Comparisons produced:
  - v6.1-B (Config C, TF-IDF)  [hardcoded from prior execution]
  - v7-SBERT (Full Pipeline)
  - v7-SBERT (Config C — Pattern + SBERT Semantic + Behavioral)

Run:
    $env:PYTHONPATH="."; python scripts/eval_v7_sbert.py
"""

import json
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from app.services.model_registry import ModelRegistry
from app.services.sbert_detection_pipeline import SBERTDetectionPipeline

pd.options.mode.chained_assignment = None


# ── Hardcoded v6.1-B results at threshold=35.0 (from fusion study) ────────────
V61B_RESULTS = {
    "DeepSet Test":        {"acc": 68.10, "prec": 100.00, "rec": 38.33, "f1": 55.42},
    "SPML Test":           {"acc": 82.89, "prec": 100.00, "rec": 78.06, "f1": 87.68},
    "Internal Cyber Test": {"acc": 100.00,"prec": 100.00, "rec": 100.00,"f1": 100.00},
    "AdvBench Test":       {"acc": 91.30, "prec": 100.00, "rec": 89.47, "f1": 94.44},
}


# ── Dataset preparation (identical splits to train_eval_v6_lodo.py) ────────────

def load_internal_cyber():
    data = []
    files = [
        ('benign.json', 0), ('prompt_injection.json', 1),
        ('jailbreaks.json', 1), ('credential_theft.json', 1),
        ('malware_generation.json', 1), ('system_extraction.json', 1)
    ]
    for file, label in files:
        path = os.path.join('evaluation', file)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                for item in content:
                    data.append({'prompt': item['prompt'], 'label': label})
    return pd.DataFrame(data)

def split_70_15_15(df):
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val, test   = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    return train, val, test

def get_test_sets():
    print("  Loading DeepSet…")
    ds = load_dataset('deepset/prompt-injections')
    ds_test = pd.DataFrame([{'prompt': i['text'], 'label': i['label']} for i in ds['test']])

    print("  Loading SPML (3 000-sample subset)…")
    spml = load_dataset('reshabhs/SPML_Chatbot_Prompt_Injection', split='train')
    spml_df = pd.DataFrame(
        [{'prompt': i['User Prompt'], 'label': i['Prompt injection']} for i in spml]
    ).sample(3000, random_state=42)
    _, _, spml_test = split_70_15_15(spml_df)

    print("  Loading Internal Cyber / AdvBench (local JSON)…")
    local_df = load_internal_cyber().dropna(subset=['prompt']).drop_duplicates(subset=['prompt'])
    internal_df, adv_df = train_test_split(local_df, test_size=0.5, random_state=42,
                                           stratify=local_df['label'])
    _, _, int_test = split_70_15_15(internal_df)
    _, _, adv_test = split_70_15_15(adv_df)

    return {
        "DeepSet Test":        ds_test,
        "SPML Test":           spml_test,
        "Internal Cyber Test": int_test,
        "AdvBench Test":       adv_test,
    }


# ── Score override helpers (for Config-C without anomaly penalty) ───────────────

def score_config_c(r: dict) -> float:
    return (r['ps'] * 0.34) + (r['ss'] * 0.33) + (r['bs'] * 0.33)


# ── Core evaluation ─────────────────────────────────────────────────────────────

def evaluate(pipeline, dataset: pd.DataFrame, mode: str = "D", threshold: float = 35.0):
    """
    Run pipeline over dataset and return metrics.
    mode "D"  → use pipeline's real final_score  (full ArgusX fusion)
    mode "C"  → Pattern + SBERT Semantic + Behavioral only (no anomaly weight)
    """
    y_true, y_pred, raw = [], [], []

    for _, row in dataset.iterrows():
        res = pipeline.analyze(row['prompt'])
        r = {
            'prompt': row['prompt'],
            'label':  row['label'],
            'ps':  res['pattern_score'],
            'ss':  res['semantic_score'],
            'bs':  res['behavioral_score'],
            'ans': res['anomaly_score'],
            'fs':  res['final_score'],
            'dec': res['decision'],
        }
        raw.append(r)

        if mode == "D":
            pred = 1 if res['decision'] in ['FLAG', 'SANITIZE', 'BLOCK'] else 0
        else:  # Config C
            s = score_config_c(r)
            if res['pattern_score'] >= 80.0 and res['final_score'] >= 80.0:
                s = max(s, 80.0)
            pred = 1 if s >= threshold else 0

        y_true.append(row['label'])
        y_pred.append(pred)

    cm = confusion_matrix(y_true, y_pred)
    fns = sorted(
        [r for r, yt, yp in zip(raw, y_true, y_pred) if yt == 1 and yp == 0],
        key=lambda x: x['fs'], reverse=True
    )[:50]
    fps = sorted(
        [r for r, yt, yp in zip(raw, y_true, y_pred) if yt == 0 and yp == 1],
        key=lambda x: x['fs'], reverse=True
    )[:50]

    return {
        "acc":  accuracy_score(y_true, y_pred)   * 100,
        "prec": precision_score(y_true, y_pred, zero_division=0) * 100,
        "rec":  recall_score(y_true, y_pred, zero_division=0)    * 100,
        "f1":   f1_score(y_true, y_pred, zero_division=0)         * 100,
        "cm":   cm,
        "fns":  fns,
        "fps":  fps,
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("ArgusX v7-SBERT  —  Benchmark Evaluation")
    print("=" * 60)

    print("\n[1/4] Preparing test sets (LODO-identical splits)…")
    test_sets = get_test_sets()
    for k, v in test_sets.items():
        print(f"      {k}: {len(v)} samples")

    print("\n[2/4] Initialising SBERTDetectionPipeline (v6.0 LODO artifacts)…")
    registry = ModelRegistry()
    registry.load_all()
    pipeline = SBERTDetectionPipeline(registry)

    print("\n[3/4] Running evaluations…")
    results_d  = {}   # Full pipeline (Config D)
    results_c  = {}   # Config C (no anomaly weight)

    for name, df in test_sets.items():
        print(f"      Evaluating {name} (N={len(df)})…")
        results_d[name] = evaluate(pipeline, df, mode="D")
        results_c[name] = evaluate(pipeline, df, mode="C")

    # ── Comparison table ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON TABLE  (Threshold = 35.0)")
    print("=" * 80)
    header = f"{'Dataset':<22} | {'Metric':<6} | {'v6.1-B':>8} | {'v7-D':>8} | {'v7-C':>8} | {'Delta v6.1-B→v7-C':>18}"
    print(header)
    print("-" * 80)

    for name in test_sets:
        for metric in ['acc', 'prec', 'rec', 'f1']:
            b  = V61B_RESULTS[name][metric]
            d  = results_d[name][metric]
            c  = results_c[name][metric]
            delta = c - b
            print(f"{name:<22} | {metric.upper():<6} | {b:>7.2f}% | {d:>7.2f}% | {c:>7.2f}% | {delta:>+17.2f}%")

    # ── Confusion matrices ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONFUSION MATRICES  (v7 Full Pipeline — Config D)")
    print("=" * 60)
    for name in test_sets:
        cm = results_d[name]['cm']
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
        print(f"\n{name}:")
        print(f"  TN={tn}  FP={fp}")
        print(f"  FN={fn}  TP={tp}")
        print(f"  Matrix:\n{cm}")

    print("\n" + "=" * 60)
    print("CONFUSION MATRICES  (v7 Config C — Pattern + SBERT Sem + Beh)")
    print("=" * 60)
    for name in test_sets:
        cm = results_c[name]['cm']
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
        print(f"\n{name}:")
        print(f"  TN={tn}  FP={fp}")
        print(f"  FN={fn}  TP={tp}")
        print(f"  Matrix:\n{cm}")

    # ── Failure analysis ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS — Top 10 FNs per dataset  (v7 Config C)")
    print("=" * 60)
    for name in test_sets:
        fns = results_c[name]['fns']
        print(f"\n--- {name} ({len(fns)} total FNs) ---")
        for i, r in enumerate(fns[:10]):
            print(
                f"[FN {i+1:02d}] fs={r['fs']:.2f} "
                f"Pat:{r['ps']:.0f} Sem:{r['ss']:.1f} "
                f"Beh:{r['bs']:.1f} Ans:{r['ans']:.1f}"
            )
            print(f"         {r['prompt'][:90]}")

    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS — Top 10 FPs per dataset  (v7 Config C)")
    print("=" * 60)
    for name in test_sets:
        fps = results_c[name]['fps']
        print(f"\n--- {name} ({len(fps)} total FPs) ---")
        for i, r in enumerate(fps[:5]):
            print(
                f"[FP {i+1:02d}] fs={r['fs']:.2f} "
                f"Pat:{r['ps']:.0f} Sem:{r['ss']:.1f} "
                f"Beh:{r['bs']:.1f} Ans:{r['ans']:.1f}"
            )
            print(f"         {r['prompt'][:90]}")

    # ── Save raw results ────────────────────────────────────────────────────
    export = {}
    for name in test_sets:
        export[name] = {
            "config_D": {k: v for k, v in results_d[name].items() if k not in ('cm', 'fns', 'fps')},
            "config_C": {k: v for k, v in results_c[name].items() if k not in ('cm', 'fns', 'fps')},
        }
    with open("v7_sbert_results.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print("\nRaw results saved to v7_sbert_results.json")

    print("\n" + "=" * 60)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
