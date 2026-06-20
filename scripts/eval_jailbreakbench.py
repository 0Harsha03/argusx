"""
ArgusX JailbreakBench Evaluation
==================================
Unseen benchmark evaluation of ArgusX v6.1 (TF-IDF) vs v7 (SBERT).

Dataset: JailbreakBench/JBB-Behaviors (behaviors config)
  - 'harmful' split : 100 malicious jailbreak goals  → label = 1
  - 'benign'  split : 100 benign counterparts         → label = 0

Mapping:
  Malicious (label=1): BLOCK | FLAG | SANITIZE → True Positive
  Benign    (label=0): ALLOW                    → True Negative

No models are retrained. No code is modified.
This is evaluation-only on frozen production artifacts.

Run:
    $env:PYTHONPATH = "."
    python scripts/eval_jailbreakbench.py
"""

import json
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from app.services.model_registry import ModelRegistry
from app.services.detection_pipeline import DetectionPipeline          # v6.1 TF-IDF
from app.services.sbert_detection_pipeline import SBERTDetectionPipeline  # v7 SBERT

pd.options.mode.chained_assignment = None

# ── JailbreakBench category → ArgusX failure cluster mapping ──────────────────
CATEGORY_MAP = {
    "Harassment/Discrimination":        "Other",
    "Malware/Hacking":                  "Malware Request",
    "Physical harm":                    "Other",
    "Economic harm":                    "Other",
    "Fraud/Deception":                  "Credential Theft",
    "Disinformation":                   "Other",
    "Sexual/Adult content":             "Other",
    "Privacy":                          "System Prompt Extraction",
    "Expert advice":                    "Other",
    "Government decision-making":       "Other",
}

def jbb_to_cluster(category: str, goal: str) -> str:
    goal_l = goal.lower()
    if any(k in goal_l for k in ("ignore", "disregard", "forget", "override", "previous")):
        return "Prompt Injection"
    if any(k in goal_l for k in ("pretend", "act as", "roleplay", "persona", "character")):
        return "Roleplay"
    if any(k in goal_l for k in ("dan", "jailbreak", "no restrictions", "no limits",
                                   "unrestricted", "developer mode")):
        return "DAN-style Jailbreak"
    if any(k in goal_l for k in ("system prompt", "instructions", "configuration", "reveal",
                                   "tell me your", "what are your instructions")):
        return "System Prompt Extraction"
    if any(k in goal_l for k in ("malware", "virus", "ransomware", "keylogger", "exploit",
                                   "payload", "script", "hack", "backdoor", "botnet")):
        return "Malware Request"
    if any(k in goal_l for k in ("password", "credential", "api key", "token", "phish",
                                   "steal", "exfiltrate", "dump")):
        return "Credential Theft"
    return CATEGORY_MAP.get(category, "Other")


def load_jbb() -> pd.DataFrame:
    """Load JailbreakBench and return a flat labelled DataFrame."""
    print("  Loading JailbreakBench/JBB-Behaviors…")
    ds = load_dataset('JailbreakBench/JBB-Behaviors', 'behaviors')

    rows = []
    for item in ds['harmful']:
        rows.append({
            'prompt':   item['Goal'],
            'label':    1,
            'category': item['Category'],
            'behavior': item['Behavior'],
            'source':   item['Source'],
        })
    for item in ds['benign']:
        rows.append({
            'prompt':   item['Goal'],
            'label':    0,
            'category': item['Category'],
            'behavior': item['Behavior'],
            'source':   item['Source'],
        })

    df = pd.DataFrame(rows).dropna(subset=['prompt'])
    print(f"  Total samples: {len(df)}  (harmful={( df['label']==1).sum()}, "
          f"benign={(df['label']==0).sum()})")
    return df


def run_evaluation(pipeline, df: pd.DataFrame, pipeline_name: str) -> dict:
    """Run pipeline over df, return metrics + raw results."""
    y_true, y_pred, raw = [], [], []

    for _, row in df.iterrows():
        res = pipeline.analyze(row['prompt'])
        pred = 1 if res['decision'] in ('FLAG', 'SANITIZE', 'BLOCK') else 0

        y_true.append(row['label'])
        y_pred.append(pred)
        raw.append({
            'prompt':   row['prompt'],
            'label':    row['label'],
            'category': row['category'],
            'behavior': row['behavior'],
            'cluster':  jbb_to_cluster(row['category'], row['prompt']),
            'ps':  res['pattern_score'],
            'ss':  res['semantic_score'],
            'bs':  res['behavioral_score'],
            'ans': res['anomaly_score'],
            'fs':  res['final_score'],
            'dec': res['decision'],
            'pred': pred,
        })

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    fns = sorted([r for r in raw if r['label'] == 1 and r['pred'] == 0],
                 key=lambda x: x['fs'], reverse=True)[:25]
    fps = sorted([r for r in raw if r['label'] == 0 and r['pred'] == 1],
                 key=lambda x: x['fs'], reverse=True)[:25]

    return {
        "pipeline": pipeline_name,
        "acc":  accuracy_score(y_true, y_pred)                      * 100,
        "prec": precision_score(y_true, y_pred, zero_division=0)     * 100,
        "rec":  recall_score(y_true, y_pred, zero_division=0)         * 100,
        "f1":   f1_score(y_true, y_pred, zero_division=0)             * 100,
        "cm":   cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fns":  fns, "fps": fps, "raw": raw,
    }


def print_failure_table(title: str, items: list, n: int = 25):
    print(f"\n{title}  (top {min(n, len(items))} of {len(items)})")
    print(f"{'#':<4} {'Cluster':<28} {'Pat':>5} {'Sem':>5} {'Beh':>5} {'FS':>6}  Prompt")
    print("-" * 110)
    for i, r in enumerate(items[:n]):
        snippet = r['prompt'][:65].replace('\n', ' ')
        print(f"{i+1:<4} {r['cluster']:<28} {r['ps']:>5.1f} {r['ss']:>5.1f} "
              f"{r['bs']:>5.1f} {r['fs']:>6.2f}  {snippet}")


def cluster_breakdown(items: list) -> dict:
    from collections import Counter
    return dict(Counter(r['cluster'] for r in items))


def main():
    print("\n" + "=" * 65)
    print("ArgusX  ×  JailbreakBench  —  Unseen Benchmark Evaluation")
    print("=" * 65)

    # ── 1. Load dataset ────────────────────────────────────────────────────
    print("\n[1/4] Loading JailbreakBench…")
    df = load_jbb()

    # ── 2. Initialise pipelines ────────────────────────────────────────────
    print("\n[2/4] Initialising pipelines…")
    registry = ModelRegistry()
    registry.load_all()

    pipeline_v6  = DetectionPipeline(registry)         # v6.1 TF-IDF
    pipeline_v7  = SBERTDetectionPipeline(registry)    # v7 SBERT

    # ── 3. Evaluate ────────────────────────────────────────────────────────
    print("\n[3/4] Running evaluations…")
    print("      v6.1 TF-IDF…")
    res_v6 = run_evaluation(pipeline_v6, df, "v6.1 TF-IDF")
    print("      v7 SBERT…")
    res_v7 = run_evaluation(pipeline_v7, df, "v7 SBERT")

    # ── 4. Results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("COMPARISON TABLE")
    print("=" * 65)
    print(f"{'Metric':<12} | {'v6.1 TF-IDF':>12} | {'v7 SBERT':>12} | {'Delta':>10}")
    print("-" * 55)
    for metric in ('acc', 'prec', 'rec', 'f1'):
        v6 = res_v6[metric]
        v7 = res_v7[metric]
        print(f"{metric.upper():<12} | {v6:>11.2f}% | {v7:>11.2f}% | {v7-v6:>+9.2f}%")

    print("\n" + "=" * 65)
    print("CONFUSION MATRICES")
    print("=" * 65)
    for res in (res_v6, res_v7):
        print(f"\n{res['pipeline']}:")
        print(f"  TN={res['tn']}  FP={res['fp']}")
        print(f"  FN={res['fn']}  TP={res['tp']}")
        print(f"  Matrix:\n{res['cm']}")

    # ── 5. Failure analysis ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FAILURE ANALYSIS  — v6.1 TF-IDF")
    print("=" * 65)
    print_failure_table("FALSE NEGATIVES  (missed jailbreaks)", res_v6['fns'])
    print(f"\nFN cluster breakdown: {cluster_breakdown(res_v6['fns'])}")
    print_failure_table("FALSE POSITIVES  (benign flagged)", res_v6['fps'])
    print(f"\nFP cluster breakdown: {cluster_breakdown(res_v6['fps'])}")

    print("\n" + "=" * 65)
    print("FAILURE ANALYSIS  — v7 SBERT")
    print("=" * 65)
    print_failure_table("FALSE NEGATIVES  (missed jailbreaks)", res_v7['fns'])
    print(f"\nFN cluster breakdown: {cluster_breakdown(res_v7['fns'])}")
    print_failure_table("FALSE POSITIVES  (benign flagged)", res_v7['fps'])
    print(f"\nFP cluster breakdown: {cluster_breakdown(res_v7['fps'])}")

    # ── 6. Category-level breakdown ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CATEGORY-LEVEL DETECTION RATES  (harmful split, N=100)")
    print("=" * 65)
    cats = df[df['label'] == 1]['category'].unique()
    print(f"\n{'Category':<35} | {'v6.1 Recall':>12} | {'v7 Recall':>10}")
    print("-" * 65)
    for cat in sorted(cats):
        mask = [r['category'] == cat for r in res_v6['raw'] if r['label'] == 1]
        def cat_recall(res):
            subset = [r for r in res['raw'] if r['label'] == 1 and r['category'] == cat]
            if not subset:
                return 0.0
            return sum(1 for r in subset if r['pred'] == 1) / len(subset) * 100
        print(f"{cat:<35} | {cat_recall(res_v6):>11.1f}% | {cat_recall(res_v7):>9.1f}%")

    # ── 7. Save raw results ────────────────────────────────────────────────
    export = {}
    for res in (res_v6, res_v7):
        key = res['pipeline']
        export[key] = {
            "acc": res['acc'], "prec": res['prec'],
            "rec": res['rec'], "f1":   res['f1'],
            "TP": int(res['tp']), "FP": int(res['fp']),
            "TN": int(res['tn']), "FN": int(res['fn']),
        }
    with open("jailbreakbench_results.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print("\nRaw results saved to jailbreakbench_results.json")
    print("\n" + "=" * 65)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
