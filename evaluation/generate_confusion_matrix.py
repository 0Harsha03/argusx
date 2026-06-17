"""
ArgusX — Standalone Confusion Matrix Generator
===============================================
Re-generates confusion matrix visualizations from a previously exported
results CSV without needing to re-run the full evaluation.

Usage:
    python evaluation/generate_confusion_matrix.py --csv path/to/eval_results_*.csv

Requirements:
    pip install matplotlib pandas numpy
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("[FATAL] pandas is required: pip install pandas")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("[FATAL] matplotlib is required: pip install matplotlib")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

THREAT_DECISIONS = {"BLOCK", "FLAG", "SANITIZE"}

def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"true", "1", "yes"}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Binary Confusion Matrix (heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def plot_binary_confusion(df: pd.DataFrame, ax: plt.Axes) -> None:
    tp = int(df["is_true_positive"].apply(_to_bool).sum())
    tn = int(df["is_true_negative"].apply(_to_bool).sum())
    fp = int(df["is_false_positive"].apply(_to_bool).sum())
    fn = int(df["is_false_negative"].apply(_to_bool).sum())

    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    labels = ["Safe", "Threat"]

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    total = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            pct   = 100 * count / total if total else 0.0
            color = "white" if cm[i, j] < thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=color)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="white", fontsize=12)
    ax.set_yticklabels(labels, color="white", fontsize=12)
    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual",    color="white", fontsize=12)
    ax.set_title("Binary Confusion Matrix\n(Safe vs. Threat)",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-Category Detection Rate (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_category_rates(df: pd.DataFrame, ax: plt.Axes) -> None:
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    df["is_tp"] = df["is_true_positive"].apply(_to_bool)
    df["is_tn"] = df["is_true_negative"].apply(_to_bool)
    df["is_fn"] = df["is_false_negative"].apply(_to_bool)

    cat_data = []
    for cat, grp in df.groupby("category"):
        is_threat = cat != "benign"
        if is_threat:
            tp = grp["is_tp"].sum()
            fn = grp["is_fn"].sum()
            rate = safe_div(tp, tp + fn)
        else:
            tn = grp["is_tn"].sum()
            rate = safe_div(tn, len(grp))
        cat_data.append((cat.replace("_", "\n"), rate * 100))

    cat_data.sort(key=lambda x: x[1])
    names, rates = zip(*cat_data)

    colors = ["#00e5a0" if r >= 80 else "#ffb347" if r >= 60 else "#ff6b6b"
              for r in rates]
    bars = ax.barh(names, rates, color=colors, edgecolor="#222", height=0.55)

    for bar, rate in zip(bars, rates):
        ax.text(min(rate + 1.5, 97), bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va="center", color="white",
                fontsize=11, fontweight="bold")

    ax.set_xlim(0, 108)
    ax.set_xlabel("Detection / Accuracy Rate (%)", color="white", fontsize=12)
    ax.set_title("Per-Category Detection Rate",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.axvline(80, color="#ffffff44", linestyle="--", linewidth=1, label="80% threshold")

    legend_patches = [
        mpatches.Patch(color="#00e5a0", label="≥ 80% (Good)"),
        mpatches.Patch(color="#ffb347", label="60–80% (Fair)"),
        mpatches.Patch(color="#ff6b6b", label="< 60%  (Poor)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1a1d2e", edgecolor="#555",
              labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Score Distribution by Outcome (violin / box)
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    df["is_correct"] = df["is_correct"].apply(_to_bool)

    correct_scores   = pd.to_numeric(df.loc[df["is_correct"],  "final_score"], errors="coerce").dropna()
    incorrect_scores = pd.to_numeric(df.loc[~df["is_correct"], "final_score"], errors="coerce").dropna()

    data = [correct_scores.tolist(), incorrect_scores.tolist()]
    labels = ["Correct\nPredictions", "Incorrect\nPredictions"]
    colors = ["#00e5a0", "#ff6b6b"]

    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("white")
    parts["cmaxes"].set_color("#aaa")
    parts["cmins"].set_color("#aaa")
    parts["cbars"].set_color("#aaa")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylabel("Final Threat Score (0–100)", color="white", fontsize=11)
    ax.set_title("Score Distribution by Prediction Outcome",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_ylim(-5, 105)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.yaxis.set_tick_params(labelcolor="white")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Decision Distribution Stacked Bar
# ─────────────────────────────────────────────────────────────────────────────

def plot_decision_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    decisions = ["ALLOW", "FLAG", "SANITIZE", "BLOCK"]
    d_colors  = ["#00e5a0", "#ffb347", "#6ea8fe", "#ff6b6b"]

    cat_order = sorted(df["category"].unique())
    data: Dict[str, List[float]] = {d: [] for d in decisions}

    for cat in cat_order:
        sub = df[df["category"] == cat]
        total = len(sub)
        for d in decisions:
            count = (sub["actual_decision"] == d).sum()
            data[d].append(100 * count / total if total else 0)

    x = np.arange(len(cat_order))
    bottom = np.zeros(len(cat_order))
    for d, color in zip(decisions, d_colors):
        vals = np.array(data[d])
        bars = ax.bar(x, vals, bottom=bottom, color=color,
                      edgecolor="#111", width=0.6, label=d)
        for bar, val in zip(bars, vals):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%", ha="center", va="center",
                        color="black", fontsize=8, fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cat_order],
                       color="white", fontsize=8)
    ax.set_ylabel("Proportion of Decisions (%)", color="white", fontsize=11)
    ax.set_title("Decision Distribution by Category",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", facecolor="#1a1d2e",
              edgecolor="#555", labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(csv_path: Path, output_dir: Path) -> None:
    print(f"  Loading: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"  Rows: {len(df)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bg = "#0f1117"

    # ── Combined 4-panel figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor(bg)
    for ax in axes.flatten():
        ax.set_facecolor(bg)

    plot_binary_confusion(df,      axes[0, 0])
    plot_category_rates(df,        axes[0, 1])
    plot_score_distribution(df,    axes[1, 0])
    plot_decision_distribution(df, axes[1, 1])

    fig.suptitle(
        f"ArgusX — Evaluation Dashboard  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        color="white", fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=2.5)

    out_path = output_dir / f"eval_dashboard_{ts}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"  [OK] Dashboard saved → {out_path}")

    # ── Individual confusion matrix ───────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    fig2.patch.set_facecolor(bg)
    ax2.set_facecolor(bg)
    plot_binary_confusion(df, ax2)
    plt.tight_layout()
    cm_path = output_dir / f"confusion_matrix_{ts}.png"
    fig2.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(fig2)
    print(f"  [OK] Confusion matrix  → {cm_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArgusX Confusion Matrix / Dashboard Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to eval_results_*.csv exported by run_evaluation.py",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for plots (defaults to same dir as CSV)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"[FATAL] CSV not found: {csv_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   ArgusX — Confusion Matrix Generator         ║")
    print("╚══════════════════════════════════════════════╝\n")
    generate_all_plots(csv_path, output_dir)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
