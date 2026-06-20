import pandas as pd
import requests
import time
from typing import List, Dict, Any
from datasets import load_dataset
from tabulate import tabulate

API_URL = "http://127.0.0.1:8000/api/v1/analyze"

def analyze_prompt(prompt: str) -> str:
    """Hits the local ArgusX API and returns the decision."""
    try:
        resp = requests.post(API_URL, json={"prompt": prompt}, timeout=10)
        resp.raise_for_status()
        return resp.json()["decision"]
    except Exception as e:
        print(f"Error on prompt '{prompt[:30]}...': {e}")
        return "ERROR"

def run_dataset_eval(name: str, prompts: List[Dict[str, str]]):
    """
    Evaluates a list of {"prompt": str, "expected": "THREAT"|"BENIGN"}
    Returns metrics.
    """
    total = len(prompts)
    tp = tn = fp = fn = 0
    false_negatives = []
    
    for i, item in enumerate(prompts):
        if i > 0 and i % 50 == 0:
            print(f"[{name}] Processed {i}/{total}...")
            
        p = item["prompt"]
        exp = item["expected"]
        actual = analyze_prompt(p)
        
        if actual == "ERROR":
            continue
            
        is_threat = (actual in ["BLOCK", "SANITIZE", "FLAG"])
        
        if exp == "THREAT":
            if is_threat:
                tp += 1
            else:
                fn += 1
                false_negatives.append((p, actual))
        else: # exp == "BENIGN"
            if not is_threat:
                tn += 1
            else:
                fp += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Dataset": name,
        "Total Samples": total,
        "Accuracy": f"{accuracy*100:.2f}%",
        "Precision": f"{precision*100:.2f}%",
        "Recall": f"{recall*100:.2f}%",
        "F1": f"{f1*100:.2f}%",
        "False Negatives": false_negatives
    }

def main():
    print("Loading AdvBench...")
    adv_df = pd.read_csv("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv")
    adv_prompts = [{"prompt": str(row["goal"]), "expected": "THREAT"} for _, row in adv_df.iterrows()]
    
    print("Loading JailbreakBench...")
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    jbb_prompts = []
    for item in jbb["harmful"]:
        jbb_prompts.append({"prompt": item["Goal"], "expected": "THREAT"})
    for item in jbb["benign"]:
        jbb_prompts.append({"prompt": item["Goal"], "expected": "BENIGN"})
        
    print(f"Total AdvBench prompts: {len(adv_prompts)}")
    print(f"Total JailbreakBench prompts: {len(jbb_prompts)}")
    
    results = []
    all_fns = {}
    
    # Run evaluations
    for ds_name, ds_prompts in [("AdvBench", adv_prompts), ("JailbreakBench", jbb_prompts)]:
        metrics = run_dataset_eval(ds_name, ds_prompts)
        all_fns[ds_name] = metrics.pop("False Negatives")
        results.append(metrics)
        
    print("\n\n=== COMPARISON TABLE ===")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    print("\n\n=== FALSE NEGATIVES ===")
    for ds_name, fns in all_fns.items():
        print(f"\n[{ds_name}] Total FNs: {len(fns)}")
        for i, (prompt, actual) in enumerate(fns[:15]): # Show up to 15 examples
            print(f"  {i+1}. {prompt}")
            print(f"     Actual: {actual}")
            
if __name__ == "__main__":
    main()
