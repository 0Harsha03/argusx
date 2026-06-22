"""
ArgusX v7.1 — Context Dilution Experiment (DeepSet Test Split)
================================================================
Evaluates the 'Context Dilution' hypothesis by isolating malicious
instructions from their surrounding benign document contexts.

1. Identifies DeepSet test-set False Negatives (Strategy D).
2. Generates Version B (reduced context) using a heuristic extractor.
3. Re-evaluates Version B through Strategy D.
4. Computes recovery rate and detailed metrics.
"""

import json
import logging
import re
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import confusion_matrix

from app.services.model_registry import ModelRegistry
from app.services.sbert_detection_pipeline import SBERTDetectionPipeline
from app.core.config import settings

logging.getLogger("app.detection").setLevel(logging.WARNING)
logging.getLogger("app.services").setLevel(logging.WARNING)


def extract_malicious_instruction(text: str) -> str:
    """
    Automated extraction workflow (Heuristic/Simulated).
    Locates the most likely malicious payload within a larger document.
    """
    if len(text.split()) < 30:
        return text  # Already short, no context to dilute

    # Look for common jailbreak/injection pivot phrases
    pivots = [
        r"(ignore\s+(?:all\s+)?previous\s+instructions?.*?$)",
        r"(forget\s+(?:everything|all).*?$)",
        r"(i\s+want\s+you\s+to\s+act\s+as\s+.*?$)",
        r"(you\s+are\s+now\s+.*?$)",
        r"(translate\s+(?:the\s+following|this).*?$)",
        r"(\bDAN\b.*?$)",
        r"(system\s+prompt.*?$)",
        r"(reveal\s+(?:your\s+)?instructions?.*?$)"
    ]
    
    # Try to find a pivot sentence
    for pivot in pivots:
        match = re.search(pivot, text, re.IGNORECASE | re.DOTALL)
        if match:
            # Return the pivot match and any trailing text
            reduced = match.group(1).strip()
            # Ensure it's not too long (cap at 50 words)
            words = reduced.split()
            if len(words) > 50:
                return " ".join(words[:50])
            return reduced
            
    # Fallback: if no clear pivot, take the last 2 sentences (or ~40 words)
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) > 2:
        return " ".join(sentences[-2:]).strip()
        
    return text


def main():
    print("=" * 90)
    print("ArgusX v7.1 — Context Dilution Experiment")
    print("=" * 90)

    # 1. Initialization
    print("[1/5] Loading DeepSet Test Split and Model Artifacts...")
    ds = load_dataset("deepset/prompt-injections", split="test")
    # Only test malicious samples (Label 1) for FN analysis
    malicious_df = pd.DataFrame([{"text": r["text"], "label": r["label"]} for r in ds if r["label"] == 1])
    
    registry = ModelRegistry()
    registry.load_all()
    pipeline = SBERTDetectionPipeline(registry)
    
    # Enforce Strategy D
    settings.STRATEGY_D_ENABLED = True
    pipeline._threat_scorer._strategy_d = True
    POSITIVE_DECISIONS = {"BLOCK", "SANITIZE", "FLAG"}

    # 2. Identify FNs
    print("[2/5] Identifying original False Negatives under Strategy D...")
    fns = []
    original_results = {}
    
    for _, row in malicious_df.iterrows():
        res = pipeline.analyze(row['text'])
        if res['decision'] not in POSITIVE_DECISIONS:
            # False Negative!
            fns.append({
                "id": f"FN-{len(fns)+1:03d}",
                "original_text": row['text'],
                "original_res": res
            })

    print(f"      Found {len(fns)} False Negatives out of {len(malicious_df)} malicious samples.")

    if not fns:
        print("No FNs found. Exiting.")
        return

    # 3. Create Version B & Re-evaluate
    print("[3/5] Extracting malicious instructions and re-evaluating...")
    results = []
    
    for fn in fns:
        ver_b_text = extract_malicious_instruction(fn['original_text'])
        res_b = pipeline.analyze(ver_b_text)
        
        orig = fn['original_res']
        
        results.append({
            "id": fn['id'],
            "orig_dec": orig['decision'],
            "orig_text": fn['original_text'],
            "reduced_text": ver_b_text,
            "reduced_dec": res_b['decision'],
            "pat_delta": res_b['pattern_score'] - orig['pattern_score'],
            "sem_delta": res_b['semantic_score'] - orig['semantic_score'],
            "beh_delta": res_b['behavioral_score'] - orig['behavioral_score'],
            "ts_delta": res_b['final_score'] - orig['final_score'],
            "recovered": res_b['decision'] in POSITIVE_DECISIONS
        })

    # 4. Compute Metrics
    print("[4/5] Computing Metrics...")
    recovered_count = sum(1 for r in results if r['recovered'])
    recovery_rate = (recovered_count / len(fns)) * 100
    
    mean_sem_gain = sum(r['sem_delta'] for r in results) / len(results)
    mean_beh_gain = sum(r['beh_delta'] for r in results) / len(results)
    mean_ts_gain = sum(r['ts_delta'] for r in results) / len(results)

    # 5. Output Table
    print("\n[5/5] Comparison Table (First 15 FNs)")
    print("-" * 110)
    print(f"{'Sample ID':<10} | {'Orig Dec':<10} | {'Red Dec':<10} | {'Pat Δ':>7} | {'Sem Δ':>7} | {'Beh Δ':>7} | {'TS Δ':>7}")
    print("-" * 110)
    
    for r in results[:15]:
        marker = " ✅" if r['recovered'] else " ❌"
        print(f"{r['id']:<10} | {r['orig_dec']:<10} | {r['reduced_dec']:<10} | "
              f"{r['pat_delta']:>+7.1f} | {r['sem_delta']:>+7.1f} | {r['beh_delta']:>+7.1f} | {r['ts_delta']:>+7.1f}{marker}")

    # 6. Interpretations
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS & INTERPRETATION")
    print("=" * 90)
    print(f"Total FNs Tested:       {len(fns)}")
    print(f"Successfully Recovered: {recovered_count}")
    print(f"Recovery Rate:          {recovery_rate:.1f}%\n")
    print(f"Mean Semantic Gain:     {mean_sem_gain:+.1f} points")
    print(f"Mean Behavioral Gain:   {mean_beh_gain:+.1f} points")
    print(f"Mean Threat Score Gain: {mean_ts_gain:+.1f} points\n")

    print("Interpretation based on rules:")
    if recovery_rate > 70:
        print("=> Context dilution is likely the primary root cause of DeepSet Prompt Injection failures.")
        support = "strongly supports"
    elif 40 <= recovery_rate <= 70:
        print("=> Context dilution is a major contributing factor, but not the sole dominant issue.")
        support = "partially supports"
    else:
        print("=> Context dilution is NOT the dominant issue. DeepSet failures are likely caused by")
        print("   novel vocabulary or structural evasion that bypasses semantic/behavioral engines")
        print("   even when presented in isolation.")
        support = "does not strongly support"
        
    print("\n" + "=" * 90)
    print("SLIDING WINDOW SBERT — ARCHITECTURAL ASSESSMENT")
    print("=" * 90)
    
    # Calculate impact on global Recall
    # DeepSet Test has 60 malicious samples.
    total_malicious = len(malicious_df)
    original_caught = total_malicious - len(fns)
    new_caught = original_caught + recovered_count
    orig_rec = (original_caught / total_malicious) * 100
    new_rec = (new_caught / total_malicious) * 100
    
    print(f"If Sliding Window SBERT perfectly mitigates context dilution:")
    print(f"Estimated DeepSet Recall Improvement: {orig_rec:.1f}% -> {new_rec:.1f}% (+{new_rec-orig_rec:.1f}%)")
    
    # Export results for documentation
    export = {
        "recovery_rate": recovery_rate,
        "mean_sem_gain": mean_sem_gain,
        "mean_beh_gain": mean_beh_gain,
        "support": support,
        "new_rec": new_rec,
        "orig_rec": orig_rec,
        "results": results
    }
    with open("context_dilution_results.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

if __name__ == "__main__":
    main()
