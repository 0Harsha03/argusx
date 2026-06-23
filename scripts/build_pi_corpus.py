"""
ArgusX v9.1.0 — Prompt Injection Corpus Builder
================================================
Branch : argusx-v9-adaptive-routing
Script : scripts/build_pi_corpus.py

Pipeline
--------
  STEP 1  — Load datasets  (DeepSet TRAIN only + SPML full)
  STEP 2  — Schema normalisation  →  {prompt: str, label: int}
  STEP 3  — Preprocessing         (lowercase + NFKC; no stemming/removal)
  STEP 4  — Exact deduplication   (same-label: keep first; conflicting: remove all)
  STEP 5  — SPML stratified split (80 / 10 / 10, random_state=42)
  STEP 6  — Build final corpora   (DeepSet-Train + SPML-Train / SPML-Val / SPML-Test)
  STEP 7  — Export CSV artefacts  (data/pi_corpus/{train,val,test}.csv)
  STEP 8  — Statistics report     (data/pi_corpus/stats.json)

IMPORTANT
---------
  • DeepSet official TEST split (116 samples) is NEVER loaded into train/val/test.
  • JasperLS and Harelix are NOT used.
  • No stemming, lemmatisation, punctuation removal, or adversarial filtering.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Force UTF-8 on Windows stdout/stderr so em-dashes and emoji print cleanly
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("build_pi_corpus")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "pi_corpus"


# ===========================================================================
# STEP 1 — Load datasets
# ===========================================================================

def step1_load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DeepSet TRAIN split and SPML full dataset from HuggingFace."""
    log.info("STEP 1 — Loading datasets from HuggingFace…")

    # ── DeepSet (TRAIN only — test split is the frozen benchmark) ──────────
    log.info("  Loading deepset/prompt-injections [train split only]…")
    ds_deepset = load_dataset("deepset/prompt-injections")
    df_deepset_train = pd.DataFrame(ds_deepset["train"])
    log.info("  DeepSet train: %d samples", len(df_deepset_train))
    log.info("  DeepSet test (FROZEN — not loaded into pipeline): %d samples",
             len(ds_deepset["test"]))

    # ── SPML full dataset ───────────────────────────────────────────────────
    log.info("  Loading reshabhs/SPML_Chatbot_Prompt_Injection [full dataset]…")
    ds_spml = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")
    # The dataset may expose the data as "train" (the only available split)
    split_key = "train" if "train" in ds_spml else list(ds_spml.keys())[0]
    df_spml = pd.DataFrame(ds_spml[split_key])
    log.info("  SPML total: %d samples (split key: '%s')", len(df_spml), split_key)

    return df_deepset_train, df_spml


# ===========================================================================
# STEP 2 — Schema normalisation
# ===========================================================================

def step2_normalise(df_deepset: pd.DataFrame, df_spml: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map source column names to {prompt, label} and normalise label values
    to binary integers: malicious=1, benign=0.
    """
    log.info("STEP 2 — Schema normalisation…")

    # ── DeepSet: text → prompt, label → label ──────────────────────────────
    df_d = pd.DataFrame({
        "prompt": df_deepset["text"].astype(str),
        "label": df_deepset["label"].astype(int),
    })
    # Sanity-check: labels should already be 0/1
    assert df_d["label"].isin([0, 1]).all(), \
        f"Unexpected DeepSet labels: {df_d['label'].unique()}"
    log.info("  DeepSet: %d samples — labels %s", len(df_d), df_d["label"].value_counts().to_dict())

    # ── SPML: 'User Prompt' → prompt, 'Prompt Injection' → label ───────────
    # Resolve actual column names defensively (strip whitespace)
    col_map = {c.strip(): c for c in df_spml.columns}

    prompt_col = col_map.get("User Prompt")
    label_col  = col_map.get("Prompt Injection") or col_map.get("Prompt injection")

    if prompt_col is None or label_col is None:
        raise KeyError(
            f"Expected columns 'User Prompt' and 'Prompt Injection' in SPML. "
            f"Found: {list(df_spml.columns)}"
        )

    raw_labels = df_spml[label_col]
    # Labels may be integers (0/1) or strings ('0'/'1'/'yes'/'no' etc.)
    # Normalise to int via coercion; non-numeric → NaN
    int_labels = pd.to_numeric(raw_labels, errors="coerce")
    if int_labels.isna().any():
        # Fallback: treat truthy strings as 1, falsy as 0
        mapping = {"1": 1, "yes": 1, "true": 1, "0": 0, "no": 0, "false": 0}
        int_labels = raw_labels.astype(str).str.strip().str.lower().map(mapping)
        if int_labels.isna().any():
            raise ValueError(
                f"Cannot parse SPML labels. Unique raw values: {raw_labels.unique()[:20]}"
            )
    int_labels = int_labels.astype(int)

    df_s = pd.DataFrame({
        "prompt": df_spml[prompt_col].astype(str),
        "label": int_labels,
    })
    log.info("  SPML: %d samples — labels %s", len(df_s), df_s["label"].value_counts().to_dict())

    return df_d, df_s


# ===========================================================================
# STEP 3 — Preprocessing
# ===========================================================================

def _preprocess(text: str) -> str:
    """Lowercase + NFKC unicode normalisation. No stemming or removal."""
    return unicodedata.normalize("NFKC", text.lower())


def step3_preprocess(df_deepset: pd.DataFrame, df_spml: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("STEP 3 — Preprocessing (lowercase + NFKC)…")
    df_deepset = df_deepset.copy()
    df_spml    = df_spml.copy()
    df_deepset["prompt"] = df_deepset["prompt"].map(_preprocess)
    df_spml["prompt"]    = df_spml["prompt"].map(_preprocess)
    log.info("  Preprocessing complete.")
    return df_deepset, df_spml


# ===========================================================================
# STEP 4 — Deduplication
# ===========================================================================

def step4_deduplicate(
    df_deepset: pd.DataFrame,
    df_spml: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Exact dedup on the normalised prompt field, applied WITHIN each source
    independently before merging, then on the merged pool.

    Rules:
      1. Duplicate prompt + same label   → keep first occurrence.
      2. Duplicate prompt + diff labels  → remove ALL occurrences (label collision).

    Returns (df_deepset_deduped, df_spml_deduped, duplicates_removed, label_collisions_removed)
    """
    log.info("STEP 4 — Exact deduplication…")

    def _dedup(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, int, int]:
        n_before = len(df)
        # Identify prompts with conflicting labels
        label_per_prompt = df.groupby("prompt")["label"].nunique()
        collision_prompts = set(label_per_prompt[label_per_prompt > 1].index)
        n_collisions_rows = df["prompt"].isin(collision_prompts).sum()

        df_clean = df[~df["prompt"].isin(collision_prompts)].copy()
        # Drop exact duplicates (same prompt + same label), keeping first
        df_clean = df_clean.drop_duplicates(subset=["prompt"], keep="first")

        n_after = len(df_clean)
        dups_removed = (n_before - len(df[~df["prompt"].isin(collision_prompts)])) + \
                       (len(df[~df["prompt"].isin(collision_prompts)]) - n_after)
        cols_removed = n_collisions_rows

        log.info(
            "  [%s] before=%d | collision rows removed=%d | same-label dups removed=%d | after=%d",
            source_name, n_before, cols_removed, dups_removed, n_after,
        )
        return df_clean, dups_removed, cols_removed

    df_d, d_dups, d_coll = _dedup(df_deepset, "DeepSet-train")
    df_s, s_dups, s_coll = _dedup(df_spml,    "SPML")

    # Cross-source dedup: prompts in SPML that are already in DeepSet-train
    pre_cross = len(df_s)
    spml_in_deepset = df_s["prompt"].isin(set(df_d["prompt"]))
    df_s = df_s[~spml_in_deepset].copy()
    cross_removed = pre_cross - len(df_s)
    if cross_removed:
        log.info("  Cross-source overlap: %d SPML rows also in DeepSet-train — removed.", cross_removed)
        d_dups += cross_removed

    total_dups       = d_dups + s_dups + cross_removed
    total_collisions = d_coll + s_coll

    log.info("  Total duplicates_removed: %d", total_dups)
    log.info("  Total label_collisions_removed: %d", total_collisions)
    return df_d, df_s, total_dups, total_collisions


# ===========================================================================
# STEP 5 — SPML stratified split  (SPML only)
# ===========================================================================

def step5_split_spml(df_spml: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80 / 10 / 10 split of the SPML corpus.
    random_state=42, stratify=label.
    """
    log.info("STEP 5 — Stratified SPML split (80/10/10, seed=42)…")

    df_train_val, df_test = train_test_split(
        df_spml,
        test_size=0.10,
        random_state=42,
        stratify=df_spml["label"],
    )
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.10 / 0.90,      # 10% of original → ~11.11% of remainder
        random_state=42,
        stratify=df_train_val["label"],
    )

    log.info(
        "  SPML split -> train=%d | val=%d | test=%d",
        len(df_train), len(df_val), len(df_test),
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


# ===========================================================================
# STEP 6 — Build final corpora
# ===========================================================================

def step6_build_corpora(
    df_deepset: pd.DataFrame,
    df_spml_train: pd.DataFrame,
    df_spml_val: pd.DataFrame,
    df_spml_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Training  = DeepSet-train  +  SPML-train
    Validation = SPML-val
    Test       = SPML-test

    DeepSet official test split is NEVER included here.
    """
    log.info("STEP 6 — Building final corpora…")

    df_train = pd.concat([df_deepset, df_spml_train], ignore_index=True)
    df_val   = df_spml_val.copy()
    df_test  = df_spml_test.copy()

    log.info(
        "  Final corpora -> train=%d | val=%d | test=%d",
        len(df_train), len(df_val), len(df_test),
    )
    return df_train, df_val, df_test


# ===========================================================================
# STEP 7 — Export artefacts
# ===========================================================================

def step7_export(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
) -> None:
    log.info("STEP 7 — Exporting CSV artefacts to %s…", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train[["prompt", "label"]].to_csv(output_dir / "train.csv", index=False)
    df_val[["prompt", "label"]].to_csv(output_dir / "val.csv",   index=False)
    df_test[["prompt", "label"]].to_csv(output_dir / "test.csv",  index=False)

    log.info("  train.csv: %d rows", len(df_train))
    log.info("  val.csv:   %d rows", len(df_val))
    log.info("  test.csv:  %d rows", len(df_test))


# ===========================================================================
# STEP 8 — Statistics report
# ===========================================================================

def _class_dist(df: pd.DataFrame) -> dict:
    vc = df["label"].value_counts().to_dict()
    return {"benign": int(vc.get(0, 0)), "malicious": int(vc.get(1, 0))}


def step8_stats(
    deepset_train_count: int,
    spml_count: int,
    duplicates_removed: int,
    label_collisions_removed: int,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
) -> dict:
    log.info("STEP 8 — Generating stats.json…")

    stats = {
        "deepset_train_count":        deepset_train_count,
        "spml_count":                 spml_count,
        "duplicates_removed":         duplicates_removed,
        "label_collisions_removed":   label_collisions_removed,
        "final_train_count":          len(df_train),
        "final_validation_count":     len(df_val),
        "final_test_count":           len(df_test),
        "class_distribution_train":       _class_dist(df_train),
        "class_distribution_validation":  _class_dist(df_val),
        "class_distribution_test":        _class_dist(df_test),
    }

    def _json_default(obj):
        if hasattr(obj, "item"):   # numpy scalar -> Python native
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=_json_default)

    log.info("  stats.json written to %s", stats_path)
    return stats


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("ArgusX v9.1.0 — Prompt Injection Corpus Builder")
    print("Branch: argusx-v9-adaptive-routing")
    print("=" * 70)

    # STEP 1
    df_deepset_raw, df_spml_raw = step1_load()
    deepset_train_raw_count = len(df_deepset_raw)
    spml_raw_count          = len(df_spml_raw)

    # STEP 2
    df_deepset, df_spml = step2_normalise(df_deepset_raw, df_spml_raw)

    # STEP 3
    df_deepset, df_spml = step3_preprocess(df_deepset, df_spml)

    # STEP 4
    df_deepset, df_spml, dups_removed, collisions_removed = step4_deduplicate(df_deepset, df_spml)

    # STEP 5
    df_spml_train, df_spml_val, df_spml_test = step5_split_spml(df_spml)

    # STEP 6
    df_train, df_val, df_test = step6_build_corpora(df_deepset, df_spml_train, df_spml_val, df_spml_test)

    # STEP 7
    step7_export(df_train, df_val, df_test, OUTPUT_DIR)

    # STEP 8
    stats = step8_stats(
        deepset_train_count      = deepset_train_raw_count,
        spml_count               = spml_raw_count,
        duplicates_removed       = dups_removed,
        label_collisions_removed = collisions_removed,
        df_train                 = df_train,
        df_val                   = df_val,
        df_test                  = df_test,
        output_dir               = OUTPUT_DIR,
    )

    # ── Summary print ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CORPUS STATISTICS")
    print("=" * 70)
    print(json.dumps(stats, indent=2, default=int))

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    for fname in ("train.csv", "val.csv", "test.csv", "stats.json"):
        p = OUTPUT_DIR / fname
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {fname:<14}  {size_kb:>8.1f} KB  ->  {p}")

    print("\n✅ Corpus build complete.")
    print("   DeepSet official test split (116 samples) remains UNTOUCHED.")
    print("   Next step: DistilBERT fine-tuning is NOT in scope for this task.")


if __name__ == "__main__":
    main()
