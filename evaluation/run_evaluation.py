"""
ArgusX Evaluation Runner
========================
Loads all benchmark datasets, calls /api/v1/analyze, computes metrics,
exports results to CSV, and generates a confusion matrix.

Usage:
    python evaluation/run_evaluation.py [--host HOST] [--port PORT] [--output-dir DIR]

Requirements (pip install if missing):
    requests, matplotlib, scikit-learn, pandas, tabulate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Optional rich output ──────────────────────────────────────────────────────
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    import matplotlib
    matplotlib.use("Agg")          # headless – no display required
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import requests
except ImportError:
    print("[FATAL] 'requests' is required. Run: pip install requests")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EVAL_DIR   = Path(__file__).parent
ROOT_DIR   = EVAL_DIR.parent

# Map dataset filenames → human-readable category labels
DATASETS: Dict[str, str] = {
    "benign.json":            "benign",
    "prompt_injection.json":  "prompt_injection",
    "system_extraction.json": "system_extraction",
    "malware_generation.json":"malware_generation",
    "credential_theft.json":  "credential_theft",
    "jailbreaks.json":        "jailbreak",
}

# Decisions that constitute a "threat" (i.e. positive class)
THREAT_DECISIONS = {"BLOCK", "FLAG", "SANITIZE"}
SAFE_DECISIONS   = {"ALLOW"}

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkEntry:
    prompt:            str
    expected_decision: str
    category:          str


@dataclass
class EvalResult:
    prompt:            str
    category:          str
    expected_decision: str
    actual_decision:   str
    threat_category:   str
    final_score:       float
    pattern_score:     float
    semantic_score:    float
    behavioral_score:  float
    anomaly_score:     float
    matched_patterns:  List[str]
    explanation:       str
    latency_ms:        float
    is_correct:        bool
    is_true_positive:  bool   # expected threat → detected threat
    is_true_negative:  bool   # expected safe   → detected safe
    is_false_positive: bool   # expected safe   → detected threat
    is_false_negative: bool   # expected threat → detected safe
    error:             Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets() -> List[BenchmarkEntry]:
    entries: List[BenchmarkEntry] = []
    for filename, _cat_label in DATASETS.items():
        path = EVAL_DIR / filename
        if not path.exists():
            print(f"  [WARN] Dataset not found: {path} — skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            entries.append(BenchmarkEntry(
                prompt=item["prompt"],
                expected_decision=item["expected_decision"].upper(),
                category=item["category"],
            ))
        print(f"  [OK] Loaded {len(data):>3} entries from {filename}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# API caller
# ─────────────────────────────────────────────────────────────────────────────

def call_analyze(session: requests.Session, base_url: str, entry: BenchmarkEntry) -> EvalResult:
    url = f"{base_url}/api/v1/analyze"
    payload = {"prompt": entry.prompt}
    t0 = time.perf_counter()
    try:
        resp = session.post(url, json=payload, timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return EvalResult(
            prompt=entry.prompt,
            category=entry.category,
            expected_decision=entry.expected_decision,
            actual_decision="ERROR",
            threat_category="UNKNOWN",
            final_score=0.0,
            pattern_score=0.0,
            semantic_score=0.0,
            behavioral_score=0.0,
            anomaly_score=0.0,
            matched_patterns=[],
            explanation="",
            latency_ms=latency_ms,
            is_correct=False,
            is_true_positive=False,
            is_true_negative=False,
            is_false_positive=False,
            is_false_negative=False,
            error=str(exc),
        )

    actual   = data.get("decision", "ERROR").upper()
    scores   = data.get("scores", {})
    expected = entry.expected_decision

    # Binary classification: threat vs. safe
    expected_is_threat = expected in THREAT_DECISIONS
    actual_is_threat   = actual   in THREAT_DECISIONS

    # Decision-level correctness (exact match on label)
    is_correct = (actual == expected)

    tp = expected_is_threat and actual_is_threat
    tn = (not expected_is_threat) and (not actual_is_threat)
    fp = (not expected_is_threat) and actual_is_threat
    fn = expected_is_threat and (not actual_is_threat)

    return EvalResult(
        prompt=entry.prompt,
        category=entry.category,
        expected_decision=expected,
        actual_decision=actual,
        threat_category=data.get("threat_category", "UNKNOWN"),
        final_score=scores.get("final", 0.0),
        pattern_score=scores.get("pattern", 0.0),
        semantic_score=scores.get("semantic", 0.0),
        behavioral_score=scores.get("behavioral", 0.0),
        anomaly_score=scores.get("anomaly", 0.0),
        matched_patterns=data.get("matched_patterns", []),
        explanation=data.get("explanation", ""),
        latency_ms=latency_ms,
        is_correct=is_correct,
        is_true_positive=tp,
        is_true_negative=tn,
        is_false_positive=fp,
        is_false_negative=fn,
        error=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(results: List[EvalResult]) -> Dict:
    valid = [r for r in results if r.error is None]
    total  = len(valid)
    errors = len(results) - total

    tp = sum(r.is_true_positive  for r in valid)
    tn = sum(r.is_true_negative  for r in valid)
    fp = sum(r.is_false_positive for r in valid)
    fn = sum(r.is_false_negative for r in valid)

    accuracy  = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)

    latencies = [r.latency_ms for r in valid]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "total_samples":   len(results),
        "valid_samples":   total,
        "error_samples":   errors,
        "true_positives":  tp,
        "true_negatives":  tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy":        round(accuracy,    4),
        "precision":       round(precision,   4),
        "recall":          round(recall,      4),
        "f1_score":        round(f1,          4),
        "specificity":     round(specificity, 4),
        "avg_latency_ms":  round(avg_latency, 2),
    }


def compute_per_category_metrics(results: List[EvalResult]) -> Dict[str, Dict]:
    categories: Dict[str, List[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    per_cat: Dict[str, Dict] = {}
    for cat, cat_results in categories.items():
        valid = [r for r in cat_results if r.error is None]
        total = len(valid)
        if total == 0:
            continue

        tp = sum(r.is_true_positive  for r in valid)
        tn = sum(r.is_true_negative  for r in valid)
        fp = sum(r.is_false_positive for r in valid)
        fn = sum(r.is_false_negative for r in valid)

        precision   = safe_div(tp, tp + fp)
        recall      = safe_div(tp, tp + fn)
        f1          = safe_div(2 * precision * recall, precision + recall)
        accuracy    = safe_div(tp + tn, total)

        # Detection rate = recall for threat categories; TN-rate for benign
        is_threat_cat = cat != "benign"
        detection_rate = recall if is_threat_cat else safe_div(tn, total)

        per_cat[cat] = {
            "total":          total,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives":fp,
            "false_negatives":fn,
            "accuracy":       round(accuracy,       4),
            "precision":      round(precision,      4),
            "recall":         round(recall,         4),
            "f1_score":       round(f1,             4),
            "detection_rate": round(detection_rate, 4),
        }
    return per_cat


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_results_csv(results: List[EvalResult], output_path: Path) -> None:
    fieldnames = [
        "category", "expected_decision", "actual_decision", "threat_category",
        "is_correct", "is_true_positive", "is_true_negative",
        "is_false_positive", "is_false_negative",
        "final_score", "pattern_score", "semantic_score",
        "behavioral_score", "anomaly_score",
        "latency_ms", "matched_patterns", "explanation", "error", "prompt",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["matched_patterns"] = "; ".join(r.matched_patterns)
            # Keep prompt short in CSV to avoid huge files
            row["prompt"] = r.prompt[:120] + ("…" if len(r.prompt) > 120 else "")
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\n  [CSV] Results exported → {output_path}")


def export_metrics_csv(
    overall: Dict,
    per_cat: Dict[str, Dict],
    output_path: Path,
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Overall metrics
        writer.writerow(["=== OVERALL METRICS ==="])
        writer.writerow(["metric", "value"])
        for k, v in overall.items():
            writer.writerow([k, v])

        writer.writerow([])

        # Per-category metrics
        writer.writerow(["=== PER-CATEGORY METRICS ==="])
        if per_cat:
            header = ["category"] + list(next(iter(per_cat.values())).keys())
            writer.writerow(header)
            for cat, metrics in per_cat.items():
                writer.writerow([cat] + list(metrics.values()))

    print(f"  [CSV] Metrics exported   → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def generate_confusion_matrix(results: List[EvalResult], output_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        print("  [WARN] matplotlib not installed — skipping confusion matrix.")
        return

    valid = [r for r in results if r.error is None]
    if not valid:
        print("  [WARN] No valid results for confusion matrix.")
        return

    # Binary confusion matrix: Safe vs. Threat
    tp = sum(r.is_true_positive  for r in valid)
    tn = sum(r.is_true_negative  for r in valid)
    fp = sum(r.is_false_positive for r in valid)
    fn = sum(r.is_false_negative for r in valid)

    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["Safe (Benign)", "Threat"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f1117")

    # ── Left: binary confusion matrix ────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0f1117")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    total = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = 100 * count / total if total else 0
            color = "white" if cm[i, j] < thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_yticklabels(labels, color="white", fontsize=11)
    ax.set_xlabel("Predicted Label", color="white", fontsize=12)
    ax.set_ylabel("True Label",      color="white", fontsize=12)
    ax.set_title("Confusion Matrix (Binary)",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ── Right: per-category detection rate bar chart ──────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0f1117")

    categories: Dict[str, List[EvalResult]] = {}
    for r in valid:
        categories.setdefault(r.category, []).append(r)

    cat_names, det_rates = [], []
    for cat, cat_results in sorted(categories.items()):
        is_threat_cat = cat != "benign"
        if is_threat_cat:
            tp_c = sum(r.is_true_positive for r in cat_results)
            fn_c = sum(r.is_false_negative for r in cat_results)
            rate = safe_div(tp_c, tp_c + fn_c)
        else:
            tn_c = sum(r.is_true_negative for r in cat_results)
            rate = safe_div(tn_c, len(cat_results))
        cat_names.append(cat.replace("_", "\n"))
        det_rates.append(rate * 100)

    colors = ["#00e5a0" if r >= 80 else "#ffb347" if r >= 60 else "#ff6b6b"
              for r in det_rates]
    bars = ax2.barh(cat_names, det_rates, color=colors, edgecolor="#222",
                    height=0.55)

    for bar, rate in zip(bars, det_rates):
        ax2.text(min(rate + 1, 95), bar.get_y() + bar.get_height() / 2,
                 f"{rate:.1f}%", va="center", color="white", fontsize=11,
                 fontweight="bold")

    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Detection / Accuracy Rate (%)", color="white", fontsize=12)
    ax2.set_title("Per-Category Detection Rate",
                  color="white", fontsize=14, fontweight="bold", pad=12)
    ax2.tick_params(colors="white", labelsize=9)
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.axvline(80, color="#ffffff44", linestyle="--", linewidth=1)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    ax2.set_facecolor("#0f1117")

    plt.suptitle(
        f"ArgusX Evaluation Results  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        color="white", fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [PNG] Confusion matrix   → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_summary(overall: Dict, per_cat: Dict[str, Dict]) -> None:
    print("\n" + "═" * 60)
    print("  ArgusX Evaluation Summary")
    print("═" * 60)

    overall_rows = [
        ["Total Samples",    overall["total_samples"]],
        ["Valid Samples",    overall["valid_samples"]],
        ["Error Samples",    overall["error_samples"]],
        ["True Positives",   overall["true_positives"]],
        ["True Negatives",   overall["true_negatives"]],
        ["False Positives",  overall["false_positives"]],
        ["False Negatives",  overall["false_negatives"]],
        ["Accuracy",         _pct(overall["accuracy"])],
        ["Precision",        _pct(overall["precision"])],
        ["Recall",           _pct(overall["recall"])],
        ["F1 Score",         _pct(overall["f1_score"])],
        ["Specificity",      _pct(overall["specificity"])],
        ["Avg Latency (ms)", overall["avg_latency_ms"]],
    ]

    if HAS_TABULATE:
        print(tabulate(overall_rows, headers=["Metric", "Value"],
                       tablefmt="rounded_outline"))
    else:
        for row in overall_rows:
            print(f"  {row[0]:<22} {row[1]}")

    print("\n  Per-Category Metrics\n" + "─" * 60)
    cat_rows = []
    for cat, m in per_cat.items():
        cat_rows.append([
            cat,
            m["total"],
            f"{m['accuracy']*100:.1f}%",
            f"{m['precision']*100:.1f}%",
            f"{m['recall']*100:.1f}%",
            f"{m['f1_score']*100:.1f}%",
            f"{m['detection_rate']*100:.1f}%",
        ])

    headers = ["Category", "N", "Accuracy", "Precision", "Recall", "F1", "Det. Rate"]
    if HAS_TABULATE:
        print(tabulate(cat_rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print("  " + "  ".join(f"{h:<18}" for h in headers))
        for row in cat_rows:
            print("  " + "  ".join(f"{str(v):<18}" for v in row))

    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ArgusX Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",       default="http://127.0.0.1", help="API host")
    parser.add_argument("--port",       default=8000, type=int,      help="API port")
    parser.add_argument("--output-dir", default=str(EVAL_DIR / "results"),
                        help="Directory to store CSV and PNG outputs")
    parser.add_argument("--delay",      default=0.05, type=float,
                        help="Seconds between API calls (rate-limiting)")
    parser.add_argument("--no-matrix",  action="store_true",
                        help="Skip confusion matrix generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url   = f"{args.host}:{args.port}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n╔══════════════════════════════════════════════╗")
    print("║   ArgusX — Evaluation Framework Runner       ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Target  : {base_url}")
    print(f"  Output  : {output_dir}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load datasets ────────────────────────────────────────────────────────
    print("► Loading benchmark datasets…")
    entries = load_datasets()
    print(f"  Total entries: {len(entries)}\n")
    if not entries:
        print("[FATAL] No benchmark entries loaded. Check evaluation/ directory.")
        sys.exit(1)

    # 2. Health check ─────────────────────────────────────────────────────────
    print("► Checking ArgusX API health…")
    try:
        r = requests.get(f"{base_url}/api/v1/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        print(f"  Status  : {health.get('status', 'unknown')}")
        print(f"  Version : {health.get('version', 'unknown')}")
        print(f"  DB      : {health.get('database', 'unknown')}\n")
    except Exception as e:
        print(f"  [WARN] Health check failed: {e}")
        print("  Proceeding anyway…\n")

    # 3. Run evaluation ───────────────────────────────────────────────────────
    print(f"► Running evaluation on {len(entries)} prompts…")
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    results: List[EvalResult] = []
    error_count = 0

    for i, entry in enumerate(entries, 1):
        result = call_analyze(session, base_url, entry)
        results.append(result)

        if result.error:
            error_count += 1
            status_icon = "✗"
        elif result.is_correct:
            status_icon = "✓"
        else:
            status_icon = "≠"

        if i % 10 == 0 or i == len(entries):
            errors_so_far = sum(1 for r in results if r.error)
            correct_so_far = sum(1 for r in results if r.is_correct and not r.error)
            valid_so_far = i - errors_so_far
            acc = f"{correct_so_far/valid_so_far*100:.1f}%" if valid_so_far else "N/A"
            print(f"  [{i:>3}/{len(entries)}] "
                  f"Correct: {correct_so_far}/{valid_so_far}  "
                  f"Acc: {acc}  "
                  f"Errors: {errors_so_far}")

        if args.delay > 0:
            time.sleep(args.delay)

    session.close()

    # 4. Compute metrics ──────────────────────────────────────────────────────
    print("\n► Computing metrics…")
    overall_metrics  = compute_metrics(results)
    per_cat_metrics  = compute_per_category_metrics(results)

    # 5. Print summary ────────────────────────────────────────────────────────
    print_summary(overall_metrics, per_cat_metrics)

    # 6. Export CSV ───────────────────────────────────────────────────────────
    print("► Exporting CSV reports…")
    results_csv_path = output_dir / f"eval_results_{ts}.csv"
    metrics_csv_path = output_dir / f"eval_metrics_{ts}.csv"
    export_results_csv(results, results_csv_path)
    export_metrics_csv(overall_metrics, per_cat_metrics, metrics_csv_path)

    # 7. Confusion matrix ─────────────────────────────────────────────────────
    if not args.no_matrix:
        print("\n► Generating confusion matrix…")
        matrix_path = output_dir / f"confusion_matrix_{ts}.png"
        generate_confusion_matrix(results, matrix_path)

    # 8. Save full JSON report ─────────────────────────────────────────────────
    report_path = output_dir / f"eval_report_{ts}.json"
    report = {
        "timestamp":       datetime.now().isoformat(),
        "target_url":      base_url,
        "total_entries":   len(entries),
        "overall_metrics": overall_metrics,
        "per_category":    per_cat_metrics,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [JSON] Full report       → {report_path}")

    # 9. Final status ─────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Evaluation Complete!                        ║")
    print(f"║   Accuracy  : {overall_metrics['accuracy']*100:>6.2f}%                      ║")
    print(f"║   F1 Score  : {overall_metrics['f1_score']*100:>6.2f}%                      ║")
    print(f"║   Precision : {overall_metrics['precision']*100:>6.2f}%                      ║")
    print(f"║   Recall    : {overall_metrics['recall']*100:>6.2f}%                      ║")
    print("╚══════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
